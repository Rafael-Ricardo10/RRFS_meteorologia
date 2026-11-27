# modulos/agent_api.py - API para o agente de IA (RRFS Agente) consultar a meteorologia
"""
Endpoints (todos exigem login; use um usuário próprio para o agente, aprovado pelo admin):

  GET /api/agent/ping
      -> {success, username, is_admin}                       (testa o login do agente)

  GET /api/agent/weather?city=Curitiba[&country=BR][&lat=..&lon=..]
                        [&date=YYYY-MM-DD][&end_date=YYYY-MM-DD]
                        [&hour=15 | &hour_from=6&hour_to=11][&force=1]
      -> tudo que o sistema sabe sobre a cidade, para qualquer dia/horário:
         * local resolvido (com correção de digitação e alternativas se houver ambiguidade)
         * condição de AGORA (consenso multi-modelo)
         * hora a hora (consenso ponderado pelos pesos aprendidos, com correção de viés)
         * resumo diário (até 16 dias à frente)
         * probabilidades de eventos extremos (trovoada, chuva forte, granizo, tornado, vento)
         * datas passadas (até 1940) e fora da janela multi-modelo vêm direto do Open-Meteo,
           e isso é informado em "sources".

Registro no app.py:
    from modulos.agent_api import create_agent_blueprint
    app.register_blueprint(create_agent_blueprint({...}))
"""
import difflib
import json
import logging
import math
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import date as _date, datetime, timedelta, timezone
from functools import wraps

import pandas as pd
from flask import Blueprint, jsonify, request, session

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

FORECAST_HOURLY = ("temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,"
                   "precipitation_probability,weather_code,cloud_cover,pressure_msl,"
                   "wind_speed_10m,wind_direction_10m,wind_gusts_10m")
ARCHIVE_HOURLY = ("temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,"
                  "weather_code,cloud_cover,pressure_msl,wind_speed_10m,wind_direction_10m,wind_gusts_10m")
DAILY_VARS = ("weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,"
              "wind_speed_10m_max,wind_gusts_10m_max,sunrise,sunset")

# coluna horária do modelo -> variável de aprendizado (mesmo mapa do app principal)
HOURLY_CONS = {'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity',
               'wind_speed_10m': 'wind_speed', 'precipitation': 'precipitation',
               'precipitation_probability': 'precipitation'}

MAX_HOURLY_ROWS = 72
MAX_SPAN_DAYS = 16


# ════════════════════════════════════════════════════════
#  TEXTO / GEOCODIFICAÇÃO
# ════════════════════════════════════════════════════════

def _norm(text: str) -> str:
    t = unicodedata.normalize('NFD', text or '')
    t = ''.join(c for c in t if unicodedata.category(c) != 'Mn').lower()
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in t).split())


def _http_json(url: str, params: dict, timeout: int = 20) -> dict:
    full = url + '?' + urllib.parse.urlencode(params)
    req = urllib.request.Request(full, headers={'User-Agent': 'RRFS-Meteorologia/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def _geocode_raw(name: str, country_code: str = None, count: int = 10) -> list:
    params = {'name': name, 'count': count, 'language': 'pt', 'format': 'json'}
    if country_code:
        params['countryCode'] = country_code
    try:
        return _http_json(GEOCODE_URL, params, timeout=10).get('results') or []
    except Exception as e:
        logger.warning(f"Geocoding falhou para '{name}': {e}")
        return []


def resolve_city(raw_name: str, country: str = None) -> dict:
    """
    Resolve o nome digitado (mesmo com erro) para uma cidade real.
    -> {'found': bool, 'best': {...}, 'alternatives': [...], 'ambiguous': bool,
        'needs_confirmation': bool, 'matched_query': str}
    """
    raw_name = (raw_name or '').strip()
    if not raw_name:
        return {'found': False, 'alternatives': [], 'error': 'Cidade não informada'}

    # "Cidade, Estado/País" -> nome + dica
    parts = [p.strip() for p in raw_name.split(',') if p.strip()]
    name, hint = parts[0], _norm(' '.join(parts[1:]))
    if country:
        hint = (hint + ' ' + _norm(country)).strip()
    cc = country.strip().upper() if country and len(country.strip()) == 2 else None

    # variações a tentar: original, sem acento, e removendo palavras do fim
    variants, seen = [], set()
    words = name.split()
    for cand in (name, _norm(name), ' '.join(words[:-1]) if len(words) > 1 else '',
                 ' '.join(words[:2]) if len(words) > 2 else ''):
        if cand and _norm(cand) not in seen and len(cand) >= 2:
            seen.add(_norm(cand))
            variants.append(cand)

    results, used = [], ''
    for v in variants:
        results = _geocode_raw(v, cc)
        if not results and cc:
            results = _geocode_raw(v)            # país errado não pode impedir a busca
        if results:
            used = v
            break
    if not results:
        return {'found': False, 'alternatives': [], 'matched_query': '',
                'error': f"Cidade '{raw_name}' não encontrada"}

    target = _norm(name)
    scored = []
    for r in results:
        sim = difflib.SequenceMatcher(None, target, _norm(r.get('name', ''))).ratio()
        pop = r.get('population') or 0
        score = sim + min(math.log10(pop + 1) / 25.0, 0.3)
        if hint:
            hay = ' '.join([_norm(r.get('admin1', '')), _norm(r.get('country', '')),
                            _norm(r.get('country_code', ''))])
            if any(h in hay for h in hint.split()):
                score += 0.4
        scored.append((score, sim, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    def slim(item):
        score, sim, r = item
        return {'name': r.get('name'), 'region': r.get('admin1'), 'country': r.get('country'),
                'country_code': r.get('country_code'), 'latitude': r.get('latitude'),
                'longitude': r.get('longitude'), 'timezone': r.get('timezone'),
                'elevation': r.get('elevation'), 'population': r.get('population'),
                'similarity': round(sim, 2)}

    best = slim(scored[0])
    alts = [slim(s) for s in scored[1:4]]

    ambiguous = False
    if len(scored) > 1 and not hint:
        s1, s2 = scored[0], scored[1]
        same_name = _norm(s1[2].get('name', '')) == _norm(s2[2].get('name', ''))
        if same_name and s2[1] >= 0.9 and (s2[2].get('population') or 0) >= 0.4 * (s1[2].get('population') or 1):
            ambiguous = True
    return {'found': True, 'best': best, 'alternatives': alts, 'ambiguous': ambiguous,
            'needs_confirmation': best['similarity'] < 0.75, 'matched_query': used}


# ════════════════════════════════════════════════════════
#  OPEN-METEO DIRETO (datas fora da janela multi-modelo / passado)
# ════════════════════════════════════════════════════════

def fetch_direct_range(lat, lon, start: _date, end: _date, today: _date, describe) -> dict:
    """Hora a hora + diário de start..end direto do Open-Meteo (best match)."""
    if end > today + timedelta(days=MAX_SPAN_DAYS - 1):
        return {'error': f"Previsões vão até {today + timedelta(days=MAX_SPAN_DAYS - 1)} (16 dias à frente)."}

    note = None
    if start < today - timedelta(days=90):
        url, hourly_vars = ARCHIVE_URL, ARCHIVE_HOURLY
        last_ok = today - timedelta(days=6)
        if end > last_ok:
            end, note = last_ok, "Arquivo histórico só cobre até ~6 dias atrás; intervalo cortado."
        if start > end:
            return {'error': 'Intervalo sem dados históricos disponíveis.'}
        source = 'Open-Meteo (arquivo histórico / reanálise ERA5)'
    else:
        url, hourly_vars = FORECAST_URL, FORECAST_HOURLY
        source = 'Open-Meteo (best match, dado direto da API)'

    params = {'latitude': lat, 'longitude': lon, 'hourly': hourly_vars, 'daily': DAILY_VARS,
              'timezone': 'auto', 'start_date': start.isoformat(), 'end_date': end.isoformat()}
    try:
        data = _http_json(url, params, timeout=25)
    except urllib.error.HTTPError as e:
        return {'error': f"Open-Meteo HTTP {e.code} para {start}..{end}"}
    except Exception as e:
        return {'error': f"Open-Meteo indisponível: {e}"}

    hourly = []
    h = data.get('hourly') or {}
    times = h.get('time') or []
    for i, t in enumerate(times):
        row = {'time': t}
        for k, vals in h.items():
            if k != 'time' and i < len(vals) and vals[i] is not None:
                row[k] = vals[i]
        if 'weather_code' in row:
            row['weather_description'] = describe(row['weather_code'])[0]
        hourly.append(row)

    daily = []
    d = data.get('daily') or {}
    for i, day in enumerate(d.get('time') or []):
        def g(key):
            vals = d.get(key) or []
            return vals[i] if i < len(vals) else None
        code = g('weather_code')
        daily.append({'date': day, 'temp_max': g('temperature_2m_max'), 'temp_min': g('temperature_2m_min'),
                      'precipitation_sum': g('precipitation_sum'), 'wind_speed_max': g('wind_speed_10m_max'),
                      'wind_gusts_max': g('wind_gusts_10m_max'),
                      'weather_description': describe(code)[0] if code is not None else None,
                      'sunrise': g('sunrise'), 'sunset': g('sunset'), 'source': 'open-meteo'})
    return {'hourly': hourly, 'daily': daily, 'source': source, 'note': note,
            'timezone': data.get('timezone'), 'utc_offset_seconds': data.get('utc_offset_seconds')}


# ════════════════════════════════════════════════════════
#  CONSENSO HORÁRIO MULTI-MODELO
# ════════════════════════════════════════════════════════

def consensus_hourly(models: dict, weights: dict, temp_bias: float, weighted_series) -> pd.DataFrame:
    frames = []
    for md in (models or {}).values():
        df = md.get('hourly')
        if df is None or df.empty or 'time' not in df.columns:
            continue
        f = df.set_index('time')
        frames.append(f[~f.index.duplicated()])
    if not frames:
        return pd.DataFrame()

    big = pd.concat(frames)
    grp = big.groupby(level=0)
    mean = grp.mean(numeric_only=True)
    mean['models'] = grp.size()

    for col, var in HOURLY_CONS.items():
        if col not in mean.columns:
            continue
        try:
            s = weighted_series(models, col, (weights or {}).get(var, {}))
            if s is not None:
                mean[col] = pd.Series(s).reindex(mean.index).combine_first(mean[col])
        except Exception as e:
            logger.debug(f"weighted_series falhou para {col}: {e}")

    if 'weather_code' in big.columns:
        try:
            mean['weather_code'] = grp['weather_code'].agg(
                lambda x: x.dropna().mode().iloc[0] if x.notna().any() else float('nan'))
        except Exception:
            pass

    if temp_bias and 'temperature_2m' in mean.columns:
        mean['temperature_2m'] = mean['temperature_2m'] + temp_bias
    return mean


def _rows_from_consensus(cons: pd.DataFrame, dates: set, h_from: int, h_to: int, describe) -> list:
    rows = []
    for ts, r in cons.iterrows():
        day = ts.strftime('%Y-%m-%d')
        if day not in dates or not (h_from <= ts.hour <= h_to):
            continue
        row = {'time': ts.strftime('%Y-%m-%dT%H:%M')}
        for k, v in r.items():
            if pd.notna(v):
                row[k] = round(float(v), 1) if k != 'models' else int(v)
        if 'weather_code' in row:
            row['weather_description'] = describe(int(row['weather_code']))[0]
        rows.append(row)
    return rows


def _parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], '%Y-%m-%d').date()
    except ValueError:
        return None


def _int_arg(name, lo=0, hi=23):
    v = request.args.get(name)
    if v in (None, ''):
        return None
    try:
        return max(lo, min(hi, int(float(v))))
    except ValueError:
        return None


def _today_in_tz(tz_name: str) -> _date:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(tz_name)).date()
    except Exception:
        return (datetime.now(timezone.utc) - timedelta(hours=3)).date()


# ════════════════════════════════════════════════════════
#  BLUEPRINT
# ════════════════════════════════════════════════════════

def create_agent_blueprint(deps: dict) -> Blueprint:
    """
    deps (injetados pelo app.py): db, User, Location, cache_manager, data_processor,
    prediction_engine, get_climate_models (async), run_async, load_timeseries,
    fetch_and_process_weather (async), AdaptiveLearning, sanitize, get_weather_description, Config
    """
    bp = Blueprint('agent_api', __name__)
    db, User, Location = deps['db'], deps['User'], deps['Location']
    cache_manager, data_processor = deps['cache_manager'], deps['data_processor']
    prediction_engine = deps['prediction_engine']
    get_climate_models, run_async = deps['get_climate_models'], deps['run_async']
    load_timeseries, fetch_and_process = deps['load_timeseries'], deps['fetch_and_process_weather']
    AdaptiveLearning, sanitize = deps['AdaptiveLearning'], deps['sanitize']
    describe, Config = deps['get_weather_description'], deps['Config']
    geo_key, get_weather = deps['geo_key'], deps['get_weather']

    def agent_auth(f):
        """Como login_required, mas responde JSON (o agente não segue redirects de HTML)."""
        @wraps(f)
        def wrapper(*a, **kw):
            uid = session.get('user_id')
            if not uid:
                return jsonify({'success': False, 'error': 'Não autenticado'}), 401
            user = db.session.get(User, uid)
            if not user or not (user.approved or user.is_admin):
                return jsonify({'success': False, 'error': 'Usuário não aprovado'}), 403
            return f(*a, **kw)
        return wrapper

    @bp.route('/api/agent/ping', methods=['GET'])
    @agent_auth
    def ping():
        user = db.session.get(User, session['user_id'])
        return jsonify({'success': True, 'username': user.username, 'is_admin': bool(user.is_admin)})

    def find_or_create_location(geo: dict):
        uid = session['user_id']
        lat, lon = float(geo['latitude']), float(geo['longitude'])
        for loc in Location.query.filter_by(user_id=uid).all():
            if abs(loc.latitude - lat) < 0.05 and abs(loc.longitude - lon) < 0.05:
                return loc
        loc = Location(user_id=uid, city_name=geo.get('name') or 'Local', latitude=lat,
                       longitude=lon, country=geo.get('country') or '')
        db.session.add(loc)
        db.session.commit()
        return loc

    @bp.route('/api/agent/weather', methods=['GET'])
    @agent_auth
    def agent_weather():
        try:
            force = request.args.get('force') == '1'
            sources, notes = [], []

            # ── 1. Local ─────────────────────────────────────
            lat_arg, lon_arg = request.args.get('lat'), request.args.get('lon')
            city_arg = (request.args.get('city') or '').strip()
            resolved = None
            if lat_arg and lon_arg:
                geo = {'name': city_arg or f"{lat_arg},{lon_arg}", 'latitude': float(lat_arg),
                       'longitude': float(lon_arg), 'timezone': request.args.get('tz'), 'country': ''}
                resolved = {'query': city_arg, 'matched': geo['name'], 'alternatives': [],
                            'ambiguous': False, 'needs_confirmation': False}
            else:
                if not city_arg:
                    return jsonify({'success': False, 'error': 'Informe city (ou lat e lon).'}), 400
                res = resolve_city(city_arg, request.args.get('country'))
                if not res.get('found'):
                    return jsonify({'success': False, 'error': res.get('error', 'Cidade não encontrada'),
                                    'code': 'city_not_found', 'query': city_arg}), 404
                geo = res['best']
                resolved = {'query': city_arg, 'matched': geo['name'], 'alternatives': res['alternatives'],
                            'ambiguous': res['ambiguous'], 'needs_confirmation': res['needs_confirmation']}

            loc = find_or_create_location(geo)
            tz_name = geo.get('timezone') or 'UTC'
            today = _today_in_tz(tz_name)

            # ── 2. Período pedido ────────────────────────────
            start = _parse_date(request.args.get('date')) or today
            end = _parse_date(request.args.get('end_date')) or start
            if end < start:
                start, end = end, start
            if (end - start).days >= MAX_SPAN_DAYS:
                end = start + timedelta(days=MAX_SPAN_DAYS - 1)
                notes.append(f"Período limitado a {MAX_SPAN_DAYS} dias.")
            h = _int_arg('hour')
            h_from = h if h is not None else (_int_arg('hour_from') or 0)
            h_to = h if h is not None else (_int_arg('hour_to') if _int_arg('hour_to') is not None else 23)
            wanted = {(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)}

            # ── 3. Previsão DO SISTEMA (consenso ponderado + correção aprendida), com cache compartilhado ──
            key = geo_key(loc)
            ts = load_timeseries(loc, force)
            sys_hourly = (ts or {}).get('hourly') or []
            sys_daily = (ts or {}).get('daily') or []
            local_now = None
            if ts:
                try:
                    local_now = datetime.strptime(ts['local_now'], '%Y-%m-%d %H:%M')
                except Exception:
                    local_now = None
                tz_name = ts.get('timezone') or tz_name
                today = _parse_date(ts.get('today')) or today
            else:
                notes.append("Previsão do sistema indisponível agora; usando Open-Meteo direto.")

            # ── 4. Hora a hora ───────────────────────────────
            sys_dates = {r['time'][:10] for r in sys_hourly}
            from_sys = wanted & sys_dates
            missing = sorted(wanted - sys_dates)
            span_days = len(wanted)
            hourly, direct = [], None
            if from_sys:
                sources.append(f"Previsão do sistema RRFS: consenso de {(ts or {}).get('models_used', 0)} modelos "
                               "com pesos aprendidos e correção estatística deste local")
            if span_days <= 3 or h is not None:
                for r in sys_hourly:
                    hh = int(r['time'][11:13])
                    if r['time'][:10] in from_sys and h_from <= hh <= h_to:
                        hourly.append(r)
            else:
                notes.append("Período longo: hora a hora omitido (peça um dia específico para ver as horas).")

            direct_daily = []
            if missing:
                d0, d1 = _parse_date(missing[0]), _parse_date(missing[-1])
                direct = fetch_direct_range(loc.latitude, loc.longitude, d0, d1, today, describe)
                if direct.get('error'):
                    notes.append(direct['error'])
                else:
                    sources.append(direct['source'])
                    if direct.get('note'):
                        notes.append(direct['note'])
                    direct_daily = [x for x in direct['daily'] if x['date'] in wanted]
                    if span_days <= 3 or h is not None:
                        for r in direct['hourly']:
                            hh = int(r['time'][11:13])
                            if r['time'][:10] in missing and h_from <= hh <= h_to:
                                hourly.append(r)
            hourly.sort(key=lambda r: r['time'])
            if len(hourly) > MAX_HOURLY_ROWS:
                hourly = hourly[:MAX_HOURLY_ROWS]
                notes.append(f"Hora a hora limitado a {MAX_HOURLY_ROWS} linhas.")

            # condição de agora (valor do sistema)
            now_block = (ts or {}).get('now') if today.isoformat() in wanted else None

            # ── 5. Diário (sistema, até 16 dias; dias além do alcance dos modelos = best_match) ──
            daily = [d for d in sys_daily if d.get('date') in wanted]
            if daily:
                sources.append("Resumo diário do sistema (até 16 dias, com risco e tipo de tempestade por dia)")
            have = {d['date'] for d in daily}
            daily += [d for d in direct_daily if d['date'] not in have]
            daily.sort(key=lambda d: d['date'])

            # ── 6. Eventos extremos, radar e desempenho (só se o período inclui hoje) ──
            events, storm_outlook = None, None
            if today.isoformat() in wanted:
                try:
                    preds, _fresh = get_weather(loc, force)
                    if preds:
                        events = {k: v for k, v in preds.items()
                                  if v is None or isinstance(v, (int, float, str, bool))}
                        events['radar_validation'] = preds.get('radar_validation')
                        events['storm_skill'] = ((preds.get('skill') or {}).get('storm'))
                        storm_outlook = preds.get('storm_outlook')
                except Exception as e:
                    logger.warning(f"previsão de eventos indisponível: {e}")

            payload = {
                'success': True,
                'location': {'id': loc.id, 'name': geo.get('name'), 'region': geo.get('region'),
                             'country': geo.get('country'), 'latitude': loc.latitude,
                             'longitude': loc.longitude, 'timezone': tz_name,
                             'elevation': geo.get('elevation')},
                'resolved': resolved,
                'local_now': local_now.strftime('%Y-%m-%d %H:%M') if local_now else None,
                'today': today.isoformat(),
                'requested': {'start_date': start.isoformat(), 'end_date': end.isoformat(),
                              'hour_from': h_from, 'hour_to': h_to, 'exact_hour': h},
                'now': now_block, 'hourly': hourly, 'daily': daily, 'events': events,
                'storm_outlook': storm_outlook,
                'models_used': (ts or {}).get('models_used', 0), 'sources': sources, 'notes': notes,
                'generated_at': (ts or {}).get('generated_at') or datetime.now(timezone.utc).isoformat(),
            }
            return jsonify(sanitize(payload))

        except Exception as e:
            logger.exception(f"Erro em /api/agent/weather: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    return bp

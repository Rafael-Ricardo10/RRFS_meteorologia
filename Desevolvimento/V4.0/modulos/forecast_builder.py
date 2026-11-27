"""
Monta a previsão completa de um local a partir dos modelos + aprendizado:
agora, hora a hora (todos os dias dos modelos) e resumo diário (até 16 dias),
incluindo raios/tempestade por dia e a acurácia esperada de cada variável.
"""
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import Config
from modulos import neighborhood
from modulos.api_manager import get_weather_description
from modulos.prediction_engine import CTX_COLS, EVENT_KINDS, VAR_COLS

logger = logging.getLogger(__name__)

SEVERE_LABELS = {95: 'Tempestade', 96: 'Tempestade com Granizo', 99: 'Tempestade Severa com Granizo'}
RAIN_CODES = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}
NEAR_KINDS = ('lightning', 'thunderstorm', 'hail')    # eventos também medidos 'nas redondezas' (<= 30 km)
EXTRA_COLS = {  # coluna do Open-Meteo -> nome no resultado (média ponderada simples entre modelos)
    'apparent_temperature': 'apparent_temperature', 'dew_point_2m': 'dew_point',
    'visibility': 'visibility', 'uv_index': 'uv_index', 'cape': 'cape',
    'direct_radiation': 'direct_radiation', 'diffuse_radiation': 'diffuse_radiation',
    'cloud_cover_low': 'cloud_low', 'cloud_cover_mid': 'cloud_mid', 'cloud_cover_high': 'cloud_high',
    'lifted_index': 'lifted_index', 'freezing_level_height': 'freezing_level',
}


def _num(v, nd=1):
    try:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return None
        return round(float(v), nd)
    except (TypeError, ValueError):
        return None


def _vote_code(codes: np.ndarray, w: np.ndarray) -> Optional[int]:
    """Voto ponderado do weather_code; trovoada vence se >=40% do peso apontar para ela."""
    ok = ~np.isnan(codes)
    if not ok.any():
        return None
    tally = {}
    for c, wi in zip(codes[ok], w[ok]):
        tally[int(c)] = tally.get(int(c), 0.0) + wi
    total = sum(tally.values())
    severe = {c: v for c, v in tally.items() if c in SEVERE_LABELS}
    if severe and sum(severe.values()) / total >= 0.4:
        return max(severe, key=severe.get)
    return max(tally, key=tally.get)


def classify_storm(p: Dict[str, float], env: Dict = None) -> Dict:
    """
    p em %: lightning, thunderstorm, hail, extreme_wind, heavy_rain.
    env: cape (J/kg), shear06 (m/s), wind850 (m/s), lapse75 (°C/km) no horário de pico (qualquer um pode faltar).
    -> nível, tipo/ameaça principal, organização estimada da tempestade e lista de ameaças.
    Organização (critérios usados por meteorologistas): supercélula = CAPE alto + cisalhamento 0-6 km >= 18 m/s;
    sistema organizado (multicélula/linha) = cisalhamento >= 12 m/s; multicélula >= 8; abaixo disso, pulsada/isolada.
    """
    env = env or {}
    top = max(p.get('lightning', 0), p.get('thunderstorm', 0))
    level = 'ALTO' if top >= 70 else 'MODERADO' if top >= 45 else 'BAIXO' if top >= 20 else 'NENHUM'
    kind, org, note = None, None, None
    hazards = []
    if level != 'NENHUM':
        if p.get('lightning', 0) >= 40:
            hazards.append('raios')
        if p.get('hail', 0) >= 40:
            hazards.append('granizo')
        if p.get('extreme_wind', 0) >= 40:
            hazards.append('vento forte')
        if p.get('heavy_rain', 0) >= 40:
            hazards.append('chuva intensa')
        if p.get('hail', 0) >= 40:
            kind = 'Tempestade com granizo'
        elif p.get('extreme_wind', 0) >= 40:
            kind = 'Tempestade com vento forte'
        elif p.get('heavy_rain', 0) >= 40:
            kind = 'Tempestade com chuva intensa'
        else:
            kind = 'Tempestade com raios'

        cape, sh = env.get('cape'), env.get('shear06')
        if cape is not None and sh is not None and not (isinstance(cape, float) and math.isnan(cape)) \
                and not (isinstance(sh, float) and math.isnan(sh)):
            if cape >= 1500 and sh >= 18:
                org = 'Ambiente favorável a supercélula (tempestade organizada e de maior severidade)'
                note = ('Ambiente de supercélula: tornado não pode ser estimado com estes dados (faltam helicidade '
                        'e altura da base das nuvens). Acompanhe os alertas oficiais (Defesa Civil/INMET).')
            elif cape >= 800 and sh >= 12:
                strong = (env.get('wind850') or 0) >= 12 and p.get('extreme_wind', 0) >= 40
                org = ('Linha de instabilidade com vento forte (risco de rajadas danosas)' if strong
                       else 'Sistema organizado (multicélula / linha de instabilidade)')
            elif cape >= 500 and sh >= 8:
                org = 'Multicélula (várias células que se renovam)'
            elif cape >= 300:
                org = 'Tempestades isoladas/pulsadas (rápidas e localizadas)'
        elif cape is not None or sh is not None:
            org = None
    return {'level': level, 'type': kind, 'organization': org, 'hazards': hazards, 'tornado_note': note}


def _circular_mean_deg(wide: pd.DataFrame) -> pd.Series:
    rad = np.deg2rad(wide.to_numpy(dtype=float))
    s, c = np.nansum(np.sin(rad), axis=1), np.nansum(np.cos(rad), axis=1)
    deg = (np.rad2deg(np.arctan2(s, c)) + 360) % 360
    ok = np.isfinite(wide.to_numpy(dtype=float)).any(axis=1)
    return pd.Series(np.where(ok, deg, np.nan), index=wide.index)


def build_frame(processed: Dict, adaptive, engine, now_utc: datetime = None, ring: Dict = None) -> Optional[Dict]:
    """DataFrame horário (hora local) com o valor do SISTEMA e o consenso puro de cada variável."""
    if not processed:
        return None
    weights = adaptive.get_model_weights()
    bweights = adaptive.get_model_weights_by_bucket()      # peso por modelo x variável x antecedência
    first = next(iter(processed.values()))
    off = int(first['model_info'].get('utc_offset_seconds', 0) or 0)
    now_utc = now_utc or datetime.now(timezone.utc)
    now_local = now_utc.replace(tzinfo=None) + timedelta(seconds=off)

    wides = {var: engine.wide_frame(processed, col) for var, col in VAR_COLS.items()}
    idx = sorted(set().union(*[set(w.index) for w in wides.values() if not w.empty]))
    if not idx:
        return None
    idx = pd.DatetimeIndex(idx)
    hz = ((idx - pd.Timestamp(now_local)) / pd.Timedelta(hours=1)).to_numpy(dtype=float)
    lh = idx.hour.to_numpy()

    ctx = {}
    for k, col in CTX_COLS.items():
        w = engine.wide_frame(processed, col)
        ctx[k] = w.mean(axis=1).reindex(idx) if not w.empty else pd.Series(np.nan, index=idx)
    ctx_df = pd.DataFrame(ctx, index=idx)

    df = pd.DataFrame(index=idx)
    for var in VAR_COLS:
        w = wides[var].reindex(idx)
        if w.empty:
            continue
        ens = engine.combine_series({m: w[m] for m in w.columns}, weights.get(var, {}), var=var,
                                    bucket_weights={b: bweights[b].get(var, {}) for b in bweights}, hz=hz)
        ens = ens.reindex(idx)
        _, system = adaptive.correct_frame(var, w, ens, ctx_df, lh, hz)
        df[var] = system
        df[f'{var}_raw'] = ens.to_numpy()
        df[f'{var}_spread'] = w.std(axis=1, ddof=0).to_numpy()
        df[f'{var}_n'] = w.notna().sum(axis=1).to_numpy()

    pw = weights.get('temperature', {})
    for col, name in EXTRA_COLS.items():
        s = engine.weighted_series(processed, col, pw)
        if s is not None:
            df[name] = s.reindex(idx).to_numpy()

    try:
        env_df = engine.storm_env(processed, weights)
        for c in env_df.columns:
            df[c] = env_df[c].reindex(idx).to_numpy()
    except Exception as e:
        logger.warning(f"Ambiente de tempestade indisponível: {e}")

    wd = engine.wide_frame(processed, 'wind_direction_10m')
    if not wd.empty:
        df['wind_direction'] = _circular_mean_deg(wd.reindex(idx)).to_numpy()

    # probabilidade de chuva: a do modelo, se existir; senão a concordância entre modelos
    pp = engine.weighted_series(processed, 'precipitation_probability', weights.get('precipitation', {}))
    pw_wide = wides.get('precipitation', pd.DataFrame()).reindex(idx)
    agree = None
    if not pw_wide.empty:
        wts = np.array([weights.get('precipitation', {}).get(m, 1.0) for m in pw_wide.columns], dtype=float)
        wet = (pw_wide.to_numpy(dtype=float) >= 0.1).astype(float)
        ok = np.isfinite(pw_wide.to_numpy(dtype=float)).astype(float)
        den = ok @ wts
        with np.errstate(invalid='ignore', divide='ignore'):
            agree = pd.Series(np.where(den > 0, (np.nan_to_num(wet) * ok) @ wts / den * 100.0, np.nan), index=idx)
    if pp is not None:
        pp = pp.reindex(idx)
        df['rain_prob'] = pp.fillna(agree) if agree is not None else pp
    elif agree is not None:
        df['rain_prob'] = agree

    # códigos do tempo (voto ponderado) e eventos
    codes_w = engine.wide_frame(processed, 'weather_code').reindex(idx)
    if not codes_w.empty:
        wts = np.array([weights.get('precipitation', {}).get(m, 1.0) for m in codes_w.columns], dtype=float)
        arr = codes_w.to_numpy(dtype=float)
        df['weather_code'] = [_vote_code(arr[i], wts) if np.isfinite(arr[i]).any() else np.nan
                              for i in range(len(idx))]
    for kind in EVENT_KINDS:
        s = engine.event_series(processed, kind, weights)
        df[f'ev_{kind}'] = (s.reindex(idx).to_numpy() * 100.0) if s is not None else np.nan
    # risco "nas redondezas": local + anel de vizinhança + pontos a montante (sem anel = igual ao local)
    local = {k: np.nan_to_num(df[f'ev_{k}'].to_numpy(dtype=float)) for k in NEAR_KINDS if f'ev_{k}' in df}
    steer = (None, 25.0)
    try:
        steer = neighborhood.steering(df, int(np.abs(hz).argmin()))
        near = neighborhood.near_columns(ring, idx, steer, local)
    except Exception as e:
        logger.warning(f"Vizinhança indisponível: {e}")
        near = local
    for k, arr in near.items():
        df[f'ev_{k}_near'] = arr
    if 'weather_code' in df:
        c = df['weather_code'].to_numpy(dtype=float)
        storm = np.nan_to_num(df.get('ev_lightning', pd.Series(0, index=idx)).to_numpy(dtype=float)) >= 60
        hail = np.nan_to_num(df.get('ev_hail', pd.Series(0, index=idx)).to_numpy(dtype=float)) >= 60
        c = np.where(storm & ~(c >= 95), 95, c)
        c = np.where(hail & (c < 96), 96, c)
        df['weather_code'] = c
    return {'df': df, 'now_local': now_local, 'offset': off, 'weights': weights, 'steering': steer,
            'ring_points': len((ring or {}).get('scores', {})),
            'timezone': first['model_info'].get('timezone', 'UTC')}


def _hour_row(ts, r, now_local) -> Dict:
    code = r.get('weather_code')
    desc, icon = get_weather_description(None if pd.isna(code) else code)
    row = {'time': ts.strftime('%Y-%m-%dT%H:%M'), 'hour': ts.strftime('%H:%M'),
           'horizon_h': round((ts - pd.Timestamp(now_local)) / pd.Timedelta(hours=1), 1),
           'weather_description': desc, 'weather_icon': icon,
           'weather_code': None if pd.isna(code) else int(code)}
    for var in VAR_COLS:
        row[var] = _num(r.get(var), 2 if var == 'precipitation' else 1)
    row['wind_direction'] = _num(r.get('wind_direction'), 0)
    row['precipitation_probability'] = _num(r.get('rain_prob'), 0)
    for k in ('apparent_temperature', 'dew_point', 'uv_index'):
        row[k] = _num(r.get(k), 1)
    row['visibility_km'] = _num((r.get('visibility') or np.nan) / 1000.0, 1) if pd.notna(r.get('visibility')) else None
    row['cape'] = _num(r.get('cape'), 0)
    row['shear06'] = _num(r.get('shear06'), 1)
    row['freezing_level'] = _num(r.get('freezing_level'), 0)
    for kind in EVENT_KINDS:
        row[f'{kind}_pct'] = _num(r.get(f'ev_{kind}'), 0)
    for kind in NEAR_KINDS:
        row[f'{kind}_near_pct'] = _num(r.get(f'ev_{kind}_near'), 0)
    row['models'] = int(r.get('temperature_n')) if pd.notna(r.get('temperature_n')) else None
    return row


def build_daily(frame: Dict, adaptive, best16: Dict = None, sanitize=lambda x: x) -> list:
    df, now_local = frame['df'], frame['now_local']
    today = now_local.strftime('%Y-%m-%d')
    days = []
    dates = sorted(set(df.index.strftime('%Y-%m-%d')))
    for d in dates:
        sub = df[df.index.strftime('%Y-%m-%d') == d]
        mid = pd.Timestamp(d + ' 12:00')
        horizon = max(0.0, (mid - pd.Timestamp(now_local)) / pd.Timedelta(hours=1))
        temp, prec = sub.get('temperature'), sub.get('precipitation')
        gust = sub.get('wind_gusts')
        lead_days = max(0, (datetime.strptime(d, '%Y-%m-%d').date() - now_local.date()).days)
        probs, raws = {}, {}
        for kind in EVENT_KINDS:
            s = sub.get(f'ev_{kind}')
            raw = float(s.rolling(3, min_periods=1).mean().max()) if s is not None and s.notna().any() else 0.0
            raws[kind] = raw
            probs[kind] = adaptive.calibrate_event(kind, raw, lead_days) if kind in adaptive.EVENTS else raw
        for kind in NEAR_KINDS:
            s_ = sub.get(f'ev_{kind}_near')
            raw = float(s_.rolling(3, min_periods=1).mean().max()) if s_ is not None and s_.notna().any() else 0.0
            raws[f'{kind}_near'] = raw
            probs[f'{kind}_near'] = adaptive.calibrate_event(f'{kind}_near', raw, lead_days)
        ls = sub.get('ev_lightning')
        peak_ts = ls.idxmax() if ls is not None and ls.notna().any() else None
        env = {}
        for col, key in (('cape', 'cape'), ('shear06', 'shear06'), ('wind850', 'wind850'), ('lapse75', 'lapse75')):
            ser = sub.get(col)
            if ser is None or not ser.notna().any():
                continue
            env[key] = float(ser.loc[peak_ts]) if peak_ts is not None and pd.notna(ser.loc[peak_ts]) else float(ser.max())
        if 'cape' in sub and sub['cape'].notna().any():
            env['cape'] = max(env.get('cape', 0.0), float(sub['cape'].max()))
        storm = classify_storm(probs, env)
        peak_hour = peak_ts.strftime('%H:%M') if peak_ts is not None and probs['lightning'] >= 20 else None

        psum = float(prec.sum(skipna=True)) if prec is not None and prec.notna().any() else None
        gmax = float(gust.max()) if gust is not None and gust.notna().any() else None
        codes = sub['weather_code'].dropna().astype(int).tolist() if 'weather_code' in sub else []
        day_codes = [int(c) for c in sub.between_time('06:00', '21:00')['weather_code'].dropna()] or codes
        code = None
        if storm['level'] in ('MODERADO', 'ALTO'):
            code = 96 if probs['hail'] >= 50 else 95
        elif psum is not None and psum >= 1.0:
            rc = [c for c in day_codes if c in RAIN_CODES]
            code = max(set(rc), key=rc.count) if rc else 61
        elif day_codes:
            code = max(set(day_codes), key=day_codes.count)
        desc, icon = get_weather_description(code)

        severe, severe_label = False, None
        if code in SEVERE_LABELS:
            severe, severe_label = True, SEVERE_LABELS[code]
        elif gmax is not None and gmax >= 70:
            severe, severe_label = True, 'Vento Extremo'
        elif psum is not None and psum >= 50:
            severe, severe_label = True, 'Chuva Intensa'

        spread = sub['temperature_spread'].mean() if 'temperature_spread' in sub else 0.0
        n_models = int(round(sub['temperature_n'].mean())) if 'temperature_n' in sub and sub['temperature_n'].notna().any() else 0
        conf = float(min(1.0, 0.55 + 0.1 * n_models) * math.exp(-(spread or 0) / 3.0)) if n_models >= 2 else None
        acc = {v: adaptive.expected_accuracy(v, horizon)['hit_rate'] for v in
               ('temperature', 'precipitation', 'wind_speed', 'humidity', 'solar_radiation')}
        rp = sub.get('rain_prob')
        days.append({
            'date': d, 'is_today': d == today, 'horizon_h': round(horizon),
            'temp_max': _num(temp.max()) if temp is not None else None,
            'temp_min': _num(temp.min()) if temp is not None else None,
            'precipitation_sum': _num(psum, 1),
            'precipitation_probability_max': _num(rp.max(), 0) if rp is not None and rp.notna().any() else None,
            'wind_speed_max': _num(sub['wind_speed'].max()) if 'wind_speed' in sub else None,
            'wind_gusts_max': _num(gmax),
            'humidity_avg': _num(sub['humidity'].mean(), 0) if 'humidity' in sub else None,
            'cloud_avg': _num(sub['cloud_cover'].mean(), 0) if 'cloud_cover' in sub else None,
            'pressure_avg': _num(sub['pressure'].mean(), 1) if 'pressure' in sub else None,
            'solar_radiation_max': _num(sub['solar_radiation'].max(), 0) if 'solar_radiation' in sub else None,
            'solar_energy_kwh_m2': _num(sub['solar_radiation'].sum(skipna=True) / 1000.0, 2) if 'solar_radiation' in sub else None,
            'uv_index_max': _num(sub['uv_index'].max()) if 'uv_index' in sub else None,
            'weather_description': desc, 'weather_icon': icon,
            'severe': severe, 'severe_label': severe_label,
            'storm': {**storm, 'lightning_pct': _num(probs['lightning'], 0),
                      'thunderstorm_pct': _num(probs['thunderstorm'], 0), 'hail_pct': _num(probs['hail'], 0),
                      'lightning_near_pct': _num(probs['lightning_near'], 0),
                      'thunderstorm_near_pct': _num(probs['thunderstorm_near'], 0),
                      'hail_near_pct': _num(probs['hail_near'], 0),
                      'wind_pct': _num(probs['extreme_wind'], 0), 'heavy_rain_pct': _num(probs['heavy_rain'], 0),
                      'peak_hour': peak_hour, 'lead_days': lead_days, 'raw': {k: _num(v, 1) for k, v in raws.items()},
                      'cape_max': _num(env.get('cape'), 0), 'shear06': _num(env.get('shear06'), 1),
                      'lapse75': _num(env.get('lapse75'), 1)},
            'expected_accuracy': acc,
            'source': 'consenso', 'models_count': n_models, 'confidence': conf})

    # dias além da janela dos modelos (até 16): só best_match, sem aprendizado
    if best16:
        d16 = best16.get('daily') or {}
        times = d16.get('time') or []
        have = {x['date'] for x in days}

        def g(k, i):
            v = d16.get(k)
            return v[i] if v and i < len(v) else None
        for i, d in enumerate(times):
            if d in have:
                continue
            code = g('weather_code', i)
            desc, icon = get_weather_description(code)
            psum, gmax = g('precipitation_sum', i), g('wind_gusts_10m_max', i)
            sev = code is not None and int(code) in SEVERE_LABELS
            days.append({
                'date': d, 'is_today': False, 'horizon_h': None,
                'temp_max': _num(g('temperature_2m_max', i)), 'temp_min': _num(g('temperature_2m_min', i)),
                'precipitation_sum': _num(psum), 'precipitation_probability_max': _num(g('precipitation_probability_max', i), 0),
                'wind_speed_max': _num(g('wind_speed_10m_max', i)), 'wind_gusts_max': _num(gmax),
                'weather_description': desc, 'weather_icon': icon,
                'severe': sev, 'severe_label': SEVERE_LABELS.get(int(code)) if sev else None,
                'storm': {'level': 'INDICADO' if sev else 'NENHUM', 'type': SEVERE_LABELS.get(int(code)) if sev else None,
                          'lightning_pct': None},
                'expected_accuracy': {}, 'source': 'best_match', 'models_count': 1, 'confidence': None})
    days.sort(key=lambda x: x['date'])
    return days


def build_now(frame: Dict, adaptive, base: Dict) -> Dict:
    """Condições de AGORA (valor do sistema) + acerto histórico de cada variável."""
    df, now_local = frame['df'], frame['now_local']
    i = int(np.abs((df.index - pd.Timestamp(now_local)).to_numpy().astype('timedelta64[s]').astype(float)).argmin())
    r = df.iloc[i]
    row = _hour_row(df.index[i], r, now_local)
    nxt = df.iloc[i:i + 24]
    row['precipitation_24h'] = _num(nxt['precipitation'].sum(skipna=True), 1) if 'precipitation' in nxt else None
    row['precipitation_24h_prob'] = _num(nxt['rain_prob'].max(), 0) if 'rain_prob' in nxt else None
    row['raw'] = {v: _num(r.get(f'{v}_raw'), 2) for v in VAR_COLS}
    row['hit_rate'] = {}
    for var in VAR_COLS:
        e = adaptive.expected_accuracy(var, 1)
        row['hit_rate'][var] = {'rate': e['hit_rate'], 'mae': e['mae'], 'n': e['n'],
                                'tolerance': Config.VARIABLES[var]['tol'], 'unit': Config.VARIABLES[var]['unit'],
                                'label': Config.VARIABLES[var]['label']}
    row['confidence'] = {v: base.get(f'{v}_confidence') for v in VAR_COLS}
    row['events_near_raw'] = {f'{k}_near': _num(nxt[f'ev_{k}_near'].rolling(3, min_periods=1).mean().max(), 1)
                              for k in NEAR_KINDS if f'ev_{k}_near' in nxt}
    row['storm_next3h_near'] = _num(df.iloc[i:i + 3][[c for c in ('ev_lightning_near', 'ev_thunderstorm_near') if c in df.columns]].max().max(), 0)
    row['storm_next3h'] = _num(df.iloc[i:i + 3][[c for c in df.columns if c in ('ev_lightning', 'ev_thunderstorm')]].max().max(), 0)
    return row


def build_forecast(processed: Dict, adaptive, engine, base: Dict, best16: Dict = None,
                   now_utc: datetime = None, ring: Dict = None) -> Optional[Dict]:
    frame = build_frame(processed, adaptive, engine, now_utc, ring)
    if not frame:
        return None
    df, now_local = frame['df'], frame['now_local']
    hourly = [_hour_row(ts, r, now_local) for ts, r in df.iterrows()]
    daily = build_daily(frame, adaptive, best16)
    today = now_local.strftime('%Y-%m-%d')
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'timezone': frame['timezone'], 'local_now': now_local.strftime('%Y-%m-%d %H:%M'),
        'today': today, 'now': build_now(frame, adaptive, base),
        'hourly': hourly, 'hourly_today': [h for h in hourly if h['time'].startswith(today)],
        'daily': daily, 'days_available': len(daily), 'models_used': len(processed),
        'consensus': True,
        'neighborhood': {'points': frame.get('ring_points', 0), 'rings_km': list(Config.NEIGHBOR_RINGS_KM),
                         'steering_from_deg': (frame.get('steering') or (None, None))[0],
                         'steering_kmh': (frame.get('steering') or (None, None))[1]},
    }

"""
Aprendizado adaptativo por localização.

Ciclo (roda sozinho de hora em hora, mesmo sem ninguém logado):
 1. A cada rodada guardamos o que CADA modelo previu para +1h..+168h, mais o contexto
    atmosférico (nuvens, pressão, CAPE...) daquele instante  ->  "snapshot".
 2. Quando a hora prevista chega, comparamos com o valor observado (análise do Open-Meteo)
    e atualizamos erro, viés e taxa de acerto por modelo, por variável e por antecedência.
 3. Pesos dos modelos = 1/MAE² (suavizado): quem erra menos neste local pesa mais.
 4. MOS (correção estatística): uma regressão ridge recursiva (RLS) por variável e faixa de
    antecedência aprende o ERRO do consenso a partir de TODAS as variáveis que os modelos
    entregam (desvio de cada modelo, dispersão, nuvens, pressão, CAPE, umidade, hora do dia...).
 5. Honestidade: cada previsão é guardada em três versões (consenso puro, MOS e "sistema").
    O MOS só passa a valer quando, em previsões JÁ verificadas, ele erra menos que o consenso.
 6. Probabilidades de eventos extremos são recalibradas pela frequência realmente observada.
"""
import json
import math
import os
import threading
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import numpy as np
from config import Config

logger = logging.getLogger(__name__)
UTC = timezone.utc
_LOCK = threading.RLock()

VARS = tuple(Config.VARIABLES)
MODEL_ORDER = list(Config.CLIMATE_MODELS)
N_FEAT = 2 + len(MODEL_ORDER) + 8
BUCKET_NAMES = [b for b, _ in Config.HORIZON_BUCKETS]
BUCKET_EDGES = np.array([lim for _, lim in Config.HORIZON_BUCKETS[:-1]], dtype=float)
# escala do erro no "domínio" do aprendizado (chuva é aprendida em log1p(mm))
DSCALE = {v: (1.0 if v == 'precipitation' else float(c['scale'])) for v, c in Config.VARIABLES.items()}
CAP_FRAC = 0.6          # correção máxima = 60% da escala da variável
RIDGE = 10.0            # força do prior do RLS (quanto maior, mais lento p/ aprender)
FORGET = 0.998          # fator de esquecimento (aprende com o clima recente)
MOS_SHRINK_K = 15       # n/(n+K): poucas amostras -> correção pequena
GATE_MIN_N = 15         # amostras verificadas p/ liberar o MOS
GATE_RATIO = 0.97       # MOS precisa errar >=3% menos que o consenso
ALERT_P = 0.5            # a partir de 50% o sistema considera que "avisou" o evento
LEAD_MIN_N = 8           # amostras p/ calibrar probabilidades por antecedência
RADAR_CONFIRM_COV = 0.10 # cobertura de eco (raio ~33 km) que confirma o alerta
BOUNDS = {'humidity': (0, 100), 'cloud_cover': (0, 100), 'wind_speed': (0, None), 'wind_gusts': (0, None),
          'precipitation': (0, None), 'solar_radiation': (0, 1400), 'pressure': (850, 1100),
          'temperature': (-60, 60)}


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_hour(s: str) -> datetime:
    return datetime.strptime(s[:13], '%Y-%m-%dT%H').replace(tzinfo=UTC)


def _parse_iso(s: str) -> datetime:
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def lead_bucket(d) -> str:
    d = int(d)
    return '0' if d <= 0 else '1' if d == 1 else '2' if d == 2 else '3-6'


def bucket_of(h: float) -> str:
    return BUCKET_NAMES[int(np.searchsorted(BUCKET_EDGES, h, side='left'))]


def bucket_index(h) -> np.ndarray:
    return np.searchsorted(BUCKET_EDGES, np.asarray(h, dtype=float), side='left')


def to_domain(var: str, x):
    x = np.asarray(x, dtype=float)
    return np.log1p(np.clip(x, 0, None)) if var == 'precipitation' else x


def from_domain(var: str, x):
    x = np.asarray(x, dtype=float)
    return np.expm1(x) if var == 'precipitation' else x


def make_features(var: str, dev: np.ndarray, spread: np.ndarray, ctx: Dict, lh) -> np.ndarray:
    """Matriz de características (n x N_FEAT). dev = desvio de cada modelo p/ o consenso (domínio)."""
    n = dev.shape[0]
    ds = DSCALE[var]
    ang = 2 * np.pi * np.asarray(lh, dtype=float) / 24.0

    def c(key, fill):
        v = ctx.get(key)
        if v is None:
            return np.full(n, fill, dtype=float)
        return np.nan_to_num(np.asarray(v, dtype=float), nan=fill)

    cols = [np.ones(n), np.clip(np.asarray(spread, dtype=float) / ds, 0, 5)]
    cols += [np.clip(dev[:, j] / ds, -5, 5) for j in range(dev.shape[1])]
    cols += [c('cloud', 50) / 100.0, (c('press', 1013) - 1013) / 20.0, np.minimum(c('cape', 0), 4000) / 1000.0,
             c('hum', 60) / 100.0, c('temp', 20) / 30.0, c('wind', 10) / 20.0, np.sin(ang), np.cos(ang)]
    return np.column_stack(cols)


def entry_features(var: str, entry: Dict) -> Optional[np.ndarray]:
    """Características de UMA previsão guardada (snapshot). Retorna (1 x N_FEAT) ou None."""
    vals = entry.get('values', {})
    ens = (vals.get('ensemble') or {}).get(var)
    if ens is None:
        return None
    mv = np.full(len(MODEL_ORDER), np.nan)
    for j, m in enumerate(MODEL_ORDER):
        v = (vals.get(m) or {}).get(var)
        if v is not None:
            mv[j] = v
    dom = to_domain(var, mv)
    ens_d = float(to_domain(var, ens))
    dev = np.nan_to_num(dom - ens_d)[None, :]
    ok = np.isfinite(dom)
    spread = np.array([float(np.std(dom[ok])) if ok.sum() > 1 else 0.0])
    ctx = {k: np.array([v]) for k, v in (entry.get('ctx') or {}).items()}
    return make_features(var, dev, spread, ctx, [entry.get('lh', 12)])


class AdaptiveLearning:
    MIN_N = 5
    HORIZON_LIMIT = 48           # só horizontes <=48h alimentam pesos/viés por modelo
    SHRINK_K = 5
    BIAS_SHRINK = 10
    EVENT_SHRINK = 10
    PENDING_MAX = 1600
    VARS = VARS
    EVENTS = ('thunderstorm', 'heavy_rain', 'extreme_wind', 'lightning')
    CONF_SCALE = {'temperature': 3.0, 'humidity': 15.0, 'wind_speed': 10.0, 'wind_gusts': 15.0,
                  'pressure': 3.0, 'cloud_cover': 30.0, 'solar_radiation': 150.0}
    BIAS_CAP = {'temperature': 4.0, 'humidity': 15.0, 'wind_speed': 10.0}

    def __init__(self, location_id: int):
        self.location_id = location_id
        base = f'dados/processed/location_{location_id}'
        self.state_path = f'{base}_learning.json'
        self.pending_path = f'{base}_pending.json'
        self.state = self._default_state()
        self._pending: Optional[List[Dict]] = None
        self._load()

    # ------------------------------------------------------------ persistência
    @staticmethod
    def _default_state() -> Dict:
        return {'stats': {}, 'skill': {}, 'mos': {}, 'event_bins': {}, 'event_stats': {},
                'event_pending': [], 'event_daily_pending': [], 'event_lead': {},
                'radar_obs': [], 'radar_alerts_pending': [], 'radar_alerts': {'n': 0, 'confirmed': 0},
                'history': [], 'radar': {'n': 0, 'both': 0, 'model_only': 0,
                                                               'radar_only': 0, 'none': 0},
                'last_snapshot': None, 'verified_total': 0, 'pending_count': 0}

    @staticmethod
    def _read(path):
        with open(path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _write(path, obj):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(obj, f, default=str)
        os.replace(tmp, path)

    def _load(self):
        with _LOCK:
            try:
                if os.path.exists(self.state_path):
                    self.state.update(self._read(self.state_path))
            except Exception as e:
                logger.error(f"Erro ao carregar aprendizado: {e}")

    def _save(self):
        with _LOCK:
            try:
                if self._pending is not None:
                    self.state['pending_count'] = len(self._pending)
                    self._write(self.pending_path, self._pending)
                self._write(self.state_path, self.state)
            except Exception as e:
                logger.error(f"Erro ao salvar aprendizado: {e}")

    @property
    def pending(self) -> List[Dict]:
        if self._pending is None:
            try:
                self._pending = self._read(self.pending_path) if os.path.exists(self.pending_path) else []
            except Exception:
                self._pending = []
        return self._pending

    # ---------------------------------------------------------------- MOS (RLS)
    def _mos_slot(self, var: str, b: str) -> Dict:
        slot = self.state['mos'].setdefault(var, {}).get(b)
        if slot is None:
            slot = {'n': 0, 'w': [0.0] * N_FEAT, 'P': (np.eye(N_FEAT) / RIDGE).tolist()}
            self.state['mos'][var][b] = slot
        return slot

    def _mos_update(self, var: str, b: str, x: np.ndarray, y: float):
        slot = self._mos_slot(var, b)
        w, P = np.array(slot['w']), np.array(slot['P'])
        x = x.reshape(-1)
        Px = P @ x
        k = Px / (FORGET + x @ Px)
        w = w + k * (y - w @ x)
        P = (P - np.outer(k, Px)) / FORGET
        P = (P + P.T) / 2.0
        slot['w'], slot['P'], slot['n'] = w.tolist(), P.tolist(), slot['n'] + 1

    def _gate_open(self, var: str, b: str) -> bool:
        sk = self.state['skill'].get(var, {}).get(b, {})
        e, m = sk.get('ensemble'), sk.get('mos')
        return bool(e and m and m['n'] >= GATE_MIN_N and m['mae'] < GATE_RATIO * e['mae'])

    def correct_array(self, var: str, ens: np.ndarray, X: np.ndarray, bidx: np.ndarray):
        """Aplica o MOS. Retorna (mos, system) com o mesmo tamanho de ens."""
        ens = np.asarray(ens, dtype=float)
        mos = ens.copy()
        system = ens.copy()
        ens_d = to_domain(var, ens)
        ds = DSCALE[var]
        for bi, b in enumerate(BUCKET_NAMES):
            mask = (bidx == bi) & np.isfinite(ens)
            if not mask.any():
                continue
            slot = self.state['mos'].get(var, {}).get(b)
            if not slot or slot['n'] < self.MIN_N:
                continue
            w = np.array(slot['w'])
            shrink = slot['n'] / (slot['n'] + MOS_SHRINK_K)
            corr = np.clip((X[mask] @ w) * ds * shrink, -CAP_FRAC * ds, CAP_FRAC * ds)
            val = from_domain(var, ens_d[mask] + corr)
            lo, hi = BOUNDS.get(var, (None, None))
            val = np.clip(val, lo, hi) if (lo is not None or hi is not None) else val
            if var == 'precipitation':
                val = np.where(ens[mask] < 0.05, ens[mask], val)
            if var == 'solar_radiation':
                val = np.where(ens[mask] < 5, ens[mask], val)
            mos[mask] = val
            if self._gate_open(var, b):
                system[mask] = val
        return mos, system

    def correct_frame(self, var: str, wide, ens, ctx_df, lh, horizons):
        """
        Versão em lote p/ séries horárias. wide: DataFrame (hora x modelo); ens: Series; ctx_df: DataFrame
        com colunas cloud/press/cape/hum/temp/wind; lh: hora local; horizons: horas desde agora.
        Retorna (mos, system) como arrays.
        """
        dom = to_domain(var, wide.reindex(columns=MODEL_ORDER).to_numpy(dtype=float))
        ens_v = ens.to_numpy(dtype=float)
        ens_d = to_domain(var, ens_v)
        dev = np.nan_to_num(dom - ens_d[:, None])
        cnt = np.isfinite(dom).sum(axis=1)
        with np.errstate(invalid='ignore'):
            spread = np.where(cnt > 1, np.nanstd(np.where(np.isfinite(dom), dom, np.nan), axis=1), 0.0)
        spread = np.nan_to_num(spread)
        ctx = {k: ctx_df[k].to_numpy(dtype=float) for k in ctx_df.columns}
        X = make_features(var, dev, spread, ctx, lh)
        h = np.asarray(horizons, dtype=float)
        mos, system = self.correct_array(var, ens_v, X, bucket_index(np.clip(h, 0, None)))
        past = h < -0.5                                  # horas que já passaram: sem correção
        mos[past], system[past] = ens_v[past], ens_v[past]
        return mos, system

    # ---------------------------------------------------------------- snapshot
    def record_forecast_snapshot(self, entries: List[Dict], event_probs: Dict = None,
                                 min_interval_minutes: int = 50) -> bool:
        with _LOCK:
            now = _now()
            last = self.state.get('last_snapshot')
            if last and now - _parse_iso(last) < timedelta(minutes=min_interval_minutes):
                return False
            for e in entries:                            # adiciona as versões "mos" e "system"
                self._attach_corrections(e)
            self.pending.extend(entries)
            self._pending = self.pending[-self.PENDING_MAX:]
            if event_probs:
                probs = {k: round(float(v) / 100.0, 4) for k, v in event_probs.items() if k in self.EVENTS}
                self.state['event_pending'].append({
                    'issued': now.isoformat(), 'window_end': (now + timedelta(hours=24)).isoformat(),
                    'probs': probs})
                self.state['event_pending'] = self.state['event_pending'][-100:]
            self.state['last_snapshot'] = now.isoformat()
            self._save()
            return True

    def _attach_corrections(self, e: Dict):
        b = bucket_of(e['horizon'])
        bidx = np.array([BUCKET_NAMES.index(b)])
        mos_v, sys_v = {}, {}
        for var, ens in (e['values'].get('ensemble') or {}).items():
            x = entry_features(var, e)
            if x is None:
                continue
            m, s = self.correct_array(var, np.array([ens]), x, bidx)
            mos_v[var], sys_v[var] = round(float(m[0]), 3), round(float(s[0]), 3)
        e['values']['mos'], e['values']['system'] = mos_v, sys_v

    def has_due_verifications(self) -> bool:
        cutoff = _now() - timedelta(hours=1)
        try:
            if any(_parse_hour(e['target']) <= cutoff for e in self.pending):
                return True
            if any(_parse_iso(e['window_end']) <= cutoff for e in self.state['event_pending']):
                return True
            return any(self._day_end(e) <= cutoff for e in self.state['event_daily_pending'])
        except Exception:
            return False

    # ------------------------------------------------------------ verificação
    @staticmethod
    def is_hit(var: str, pred: float, real: float) -> bool:
        tol = Config.VARIABLES[var]['tol']
        if var == 'precipitation':
            wet_p, wet_r = pred >= 0.1, real >= 0.1
            if not wet_p and not wet_r:
                return True
            if wet_p and wet_r:
                return abs(pred - real) <= max(tol, 0.5 * real)
            return False
        return abs(pred - real) <= tol

    @staticmethod
    def _skip(var: str, pred: float, real: float) -> bool:
        return var == 'solar_radiation' and pred < 5 and real < 5     # noite não conta

    @staticmethod
    def _upd(s: Dict, err: float, hit: bool):
        s['n'] = s.get('n', 0) + 1
        a = max(1.0 / s['n'], 0.02)
        s['mae'] = s.get('mae', 0.0) + a * (abs(err) - s.get('mae', 0.0))
        s['bias'] = s.get('bias', 0.0) + a * (err - s.get('bias', 0.0))
        s['hits'] = s.get('hits', 0) + (1 if hit else 0)

    def verify_pending(self, obs: Dict) -> int:
        """obs = {'YYYY-MM-DDTHH:00': {temperature, humidity, ..., weather_code}}"""
        with _LOCK:
            now = _now()
            cutoff = now - timedelta(hours=1)
            keep, scored = [], 0
            for e in self.pending:
                try:
                    t = _parse_hour(e['target'])
                except Exception:
                    continue
                if t > cutoff:
                    keep.append(e)
                    continue
                o = obs.get(e['target'])
                if o is None:
                    if now - t < timedelta(days=4):
                        keep.append(e)
                    continue
                self._score_entry(e, o)
                scored += 1
            self._pending = keep
            self._verify_events(obs, now, cutoff)
            self._verify_daily_events(obs, now, cutoff)
            self.state['verified_total'] += scored
            self._save()
            if scored:
                logger.info(f"Aprendizado loc {self.location_id}: {scored} previsões verificadas")
            return scored

    def _score_entry(self, e: Dict, o: Dict):
        h = int(e.get('horizon', 0))
        b = bucket_of(h)
        sys_errs = {}
        for series, vals in e['values'].items():
            is_model = series not in ('ensemble', 'mos', 'system')
            for var in VARS:
                pred, real = vals.get(var), o.get(var)
                if pred is None or real is None:
                    continue
                pred, real = float(pred), float(real)
                if self._skip(var, pred, real):
                    continue
                err, hit = real - pred, self.is_hit(var, pred, real)
                if series in ('ensemble', 'mos', 'system'):
                    sk = self.state['skill'].setdefault(var, {}).setdefault(b, {}).setdefault(series, {})
                    self._upd(sk, err, hit)
                    if series == 'system':
                        sys_errs[var] = err
                if (is_model or series == 'ensemble') and h <= self.HORIZON_LIMIT:
                    st = self.state['stats'].setdefault(var, {}).setdefault(series, {})
                    self._upd(st, err, hit)
        # treina o MOS DEPOIS de medir (teste-depois-treino): a medição nunca viu o dado
        for var in VARS:
            real = o.get(var)
            ens = (e['values'].get('ensemble') or {}).get(var)
            if real is None or ens is None or self._skip(var, float(ens), float(real)):
                continue
            x = entry_features(var, e)
            if x is None:
                continue
            y = (float(to_domain(var, real)) - float(to_domain(var, ens))) / DSCALE[var]
            self._mos_update(var, b, x, float(np.clip(y, -3, 3)))
        if sys_errs:
            self.state['history'].append({
                'timestamp': e['target'] + ':00+00:00', 'event_type': 'verification', 'horizon': h,
                'prediction': e['values'].get('system', {}),
                'observation': {v: o.get(v) for v in VARS},
                'error': sys_errs, 'accuracy': self._acc(sys_errs)})
            self.state['history'] = self.state['history'][-300:]

    def _acc(self, errs: Dict[str, float]) -> float:
        accs = [max(0.0, 1 - abs(er) / Config.VARIABLES[v]['scale']) for v, er in errs.items()]
        return float(np.mean(accs)) if accs else 0.0

    @staticmethod
    def _event_outcomes(window: List[Dict]) -> Dict[str, bool]:
        codes = [o.get('weather_code') for o in window if o.get('weather_code') is not None]
        pr = [o.get('precipitation') or 0 for o in window]
        gusts = [o.get('wind_gusts') or 0 for o in window]
        storm = any(int(c) in (95, 96, 99) for c in codes)
        return {
            'thunderstorm': storm, 'lightning': storm,
            'heavy_rain': (max(pr) >= 7.5 if pr else False) or sum(pr) >= 30,
            'extreme_wind': (max(gusts) >= 60 if gusts else False),
        }

    def _verify_events(self, obs: Dict, now: datetime, cutoff: datetime):
        keep = []
        for ev in self.state['event_pending']:
            try:
                start, end = _parse_iso(ev['issued']), _parse_iso(ev['window_end'])
            except Exception:
                continue
            if end > cutoff:
                keep.append(ev)
                continue
            window = [o for k, o in obs.items() if start <= _parse_hour(k) <= end]
            if len(window) < 12:
                if now - end < timedelta(days=4):
                    keep.append(ev)
                continue
            outcomes = self._event_outcomes(window)
            for name, p in ev['probs'].items():
                hit = 1 if outcomes.get(name) else 0
                b = str(min(int(p * 5), 4))
                bin_ = self.state['event_bins'].setdefault(name, {}).setdefault(b, {'n': 0, 'hits': 0})
                bin_['n'] += 1
                bin_['hits'] += hit
                st = self.state['event_stats'].setdefault(name, {'n': 0, 'hits': 0, 'brier': 0.0})
                st['n'] += 1
                st['hits'] += hit
                st['brier'] += ((p - hit) ** 2 - st['brier']) / st['n']
        self.state['event_pending'] = keep

    # ------------------------------------------- tempestades: previsões de vários dias
    @staticmethod
    def _day_end(e: Dict) -> datetime:
        day = datetime.strptime(e['date'], '%Y-%m-%d').replace(tzinfo=UTC)
        return day - timedelta(seconds=int(e.get('offset', 0))) + timedelta(hours=24)

    def record_daily_events(self, daily: List[Dict], offset_seconds: int = 0):
        """
        Guarda, para cada dia futuro (hoje..+6), a probabilidade BRUTA de cada evento e a antecedência,
        para conferir depois com o que aconteceu: é isso que mede a taxa de acerto de tempestade
        a 1, 2, 3... dias de distância.
        """
        with _LOCK:
            now = _now()
            pend = self.state['event_daily_pending']
            have = {(e['date'], e['lead']) for e in pend}
            for d in daily or []:
                raw, h = (d.get('storm') or {}).get('raw'), d.get('horizon_h')
                if not raw or h is None:
                    continue
                lead = (d.get('storm') or {}).get('lead_days')
                if lead is None:
                    lead = int(h // 24)
                if lead > 6 or (d['date'], lead) in have:
                    continue
                probs = {k: round(float(raw[k]) / 100.0, 4) for k in self.EVENTS if raw.get(k) is not None}
                if probs:
                    pend.append({'date': d['date'], 'lead': lead, 'issued': now.isoformat(),
                                 'offset': int(offset_seconds), 'probs': probs})
            self.state['event_daily_pending'] = pend[-400:]
            self._save()

    def _score_lead(self, name: str, lead: int, p: float, hit: bool):
        s = self.state['event_lead'].setdefault(name, {}).setdefault(
            lead_bucket(lead), {'n': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'brier': 0.0, 'bins': {}})
        alert = p >= ALERT_P
        key = ('tp' if hit else 'fp') if alert else ('fn' if hit else 'tn')
        s[key] += 1
        s['n'] += 1
        s['brier'] += ((p - (1 if hit else 0)) ** 2 - s['brier']) / s['n']
        bn = s['bins'].setdefault(str(min(int(p * 5), 4)), {'n': 0, 'hits': 0})
        bn['n'] += 1
        bn['hits'] += 1 if hit else 0

    def _verify_daily_events(self, obs: Dict, now: datetime, cutoff: datetime):
        keep = []
        for e in self.state['event_daily_pending']:
            try:
                end = self._day_end(e)
            except Exception:
                continue
            if end > cutoff:
                keep.append(e)
                continue
            start = end - timedelta(hours=24)
            window = [o for k, o in obs.items() if start <= _parse_hour(k) < end]
            if len(window) < 20:
                if now - end < timedelta(days=4):
                    keep.append(e)
                continue
            outcomes = self._event_outcomes(window)
            for name, p in e['probs'].items():
                self._score_lead(name, e['lead'], p, bool(outcomes.get(name)))
        self.state['event_daily_pending'] = keep

    @staticmethod
    def _scores(s: Dict) -> Dict:
        tp, fp, fn, tn, n = (s.get(k, 0) for k in ('tp', 'fp', 'fn', 'tn', 'n'))
        r = lambda a, b: round(a / b, 3) if b else None
        return {'n': n, 'events': tp + fn, 'alerts': tp + fp,
                'pod': r(tp, tp + fn),            # % dos eventos reais que foram avisados
                'far': r(fp, tp + fp),            # % dos avisos que foram alarme falso
                'csi': r(tp, tp + fp + fn),       # índice crítico de sucesso
                'brier': round(s['brier'], 4) if n and 'brier' in s else None}

    def get_event_scores(self) -> Dict:
        """Por evento: desempenho geral e por antecedência (0, 1, 2 e 3-6 dias). Limiar de aviso: 50%."""
        out = {}
        for name, leads in self.state['event_lead'].items():
            tot = {'n': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'brier_sum': 0.0}
            per = {}
            for lb in ('0', '1', '2', '3-6'):
                s = leads.get(lb)
                if not s:
                    continue
                per[lb] = self._scores(s)
                for k in ('n', 'tp', 'fp', 'fn', 'tn'):
                    tot[k] += s[k]
                tot['brier_sum'] += s['brier'] * s['n']
            tot['brier'] = tot['brier_sum'] / tot['n'] if tot['n'] else 0.0
            out[name] = {'overall': self._scores(tot), 'by_lead': per}
        return out

    def get_storm_skill(self) -> Dict:
        scores = self.get_event_scores()
        ra = self.state['radar_alerts']
        summary = {k: {'pod': v['overall']['pod'], 'far': v['overall']['far'], 'dias_verificados': v['overall']['n']}
                   for k, v in scores.items() if k in ('thunderstorm', 'lightning')}
        return {'scores': scores, 'summary': summary,
                'radar_alerts': {'n': ra['n'], 'confirmed': ra['confirmed'],
                                 'rate': round(ra['confirmed'] / ra['n'], 3) if ra['n'] else None},
                'window24': self.get_local_extreme_patterns(),
                'alert_threshold_pct': int(ALERT_P * 100)}

    # ---------------------------------------- radar: observações e confirmação de alertas
    def record_radar_observation(self, radar_val: Dict, storm_next3h: float):
        """Chamado na atualização horária: guarda o eco do radar e confere alertas antigos (confirmação)."""
        with _LOCK:
            if not radar_val or not radar_val.get('available'):
                return
            now = _now()
            hour = now.strftime('%Y-%m-%dT%H')
            obs = self.state['radar_obs']
            if obs and obs[-1]['h'] == hour:
                return
            obs.append({'h': hour, 'c10': radar_val.get('coverage_10', 0), 'c30': radar_val.get('coverage_30', 0),
                        'echo': bool(radar_val.get('echo_local'))})
            self.state['radar_obs'] = obs[-96:]
            if (storm_next3h or 0) >= 50:
                self.state['radar_alerts_pending'].append(
                    {'issued': now.isoformat(), 'until': (now + timedelta(hours=3)).isoformat()})
            keep = []
            for a in self.state['radar_alerts_pending']:
                until = _parse_iso(a['until'])
                if until > now:
                    keep.append(a)
                    continue
                win = [o for o in self.state['radar_obs']
                       if a['issued'][:13] <= o['h'] <= a['until'][:13]]
                if len(win) < 2:
                    if now - until < timedelta(hours=6):
                        keep.append(a)
                    continue
                ra = self.state['radar_alerts']
                ra['n'] += 1
                ra['confirmed'] += 1 if any(o['c30'] >= RADAR_CONFIRM_COV for o in win) else 0
            self.state['radar_alerts_pending'] = keep[-50:]
            self._save()

    # ---------------------------------------------------------------- radar
    def record_radar_check(self, model_rain_now: bool, radar_echo: bool):
        """Validação dupla: modelo x radar (só para a hora atual)."""
        with _LOCK:
            r = self.state['radar']
            r['n'] += 1
            key = 'both' if (model_rain_now and radar_echo) else 'model_only' if model_rain_now \
                else 'radar_only' if radar_echo else 'none'
            r[key] += 1
            self._save()

    def get_radar_agreement(self) -> Dict:
        r = self.state['radar']
        n = r['n']
        return {'n': n, **{k: r[k] for k in ('both', 'model_only', 'radar_only', 'none')},
                'agreement': round((r['both'] + r['none']) / n, 3) if n else None}

    # ------------------------------------------------------- pesos / correções
    def get_model_weights(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for var, c in Config.VARIABLES.items():
            w = {}
            for model, s in self.state['stats'].get(var, {}).items():
                if model == 'ensemble':
                    continue
                n = s['n']
                mae_eff = (n * s['mae'] + self.SHRINK_K * c['prior']) / (n + self.SHRINK_K)
                w[model] = 1.0 / (mae_eff ** 2 + 1e-6)
            out[var] = w
        return out

    def get_bias(self, var: str) -> float:
        st = self.state['stats'].get(var, {}).get('ensemble')
        if not st or st['n'] < self.MIN_N:
            return 0.0
        corr = st['bias'] * st['n'] / (st['n'] + self.BIAS_SHRINK)
        cap = self.BIAS_CAP.get(var, 0.0)
        return float(max(-cap, min(cap, corr)))

    def calibrate_event(self, name: str, p_pct: float, lead_days=None) -> float:
        """Recalibra a probabilidade pela frequência realmente observada (por antecedência, se houver histórico)."""
        p = p_pct / 100.0
        if lead_days is not None:
            bins = self.state['event_lead'].get(name, {}).get(lead_bucket(lead_days), {}).get('bins', {})
            b = bins.get(str(min(int(p * 5), 4)))
            if b and b['n'] >= LEAD_MIN_N:
                p = (b['hits'] + self.EVENT_SHRINK * p) / (b['n'] + self.EVENT_SHRINK)
                return float(min(max(p, 0.0), 1.0) * 100.0)
        b = self.state['event_bins'].get(name, {}).get(str(min(int(p * 5), 4)))
        if b and b['n'] >= self.MIN_N:
            p = (b['hits'] + self.EVENT_SHRINK * p) / (b['n'] + self.EVENT_SHRINK)
        return float(min(max(p, 0.0), 1.0) * 100.0)

    def _series_stat(self, var: str, b: str, series: str) -> Optional[Dict]:
        s = self.state['skill'].get(var, {}).get(b, {}).get(series)
        return s if s and s.get('n') else None

    def expected_accuracy(self, var: str, horizon_h: float) -> Dict:
        """Acerto esperado de uma previsão com essa antecedência (histórico real do local)."""
        b = bucket_of(max(horizon_h, 0))
        s = self._series_stat(var, b, 'system') or self._series_stat(var, b, 'ensemble')
        if not s or s['n'] < self.MIN_N:
            return {'hit_rate': None, 'mae': None, 'n': s['n'] if s else 0, 'bucket': b}
        return {'hit_rate': round(s['hits'] / s['n'], 3), 'mae': round(s['mae'], 3), 'n': s['n'], 'bucket': b}

    def _confidence(self, var: str, spread: float, n_models: int, mean: Optional[float]) -> float:
        s = self._series_stat(var, '0-3h', 'system')
        mae = s['mae'] if (s and s['n'] >= self.MIN_N and var != 'precipitation') else 0.0
        err = math.sqrt((spread or 0.0) ** 2 + mae ** 2)
        scale = (0.5 * (mean or 0.0) + 3.0) if var == 'precipitation' else self.CONF_SCALE[var]
        nf = min(1.0, 0.55 + 0.1 * max(n_models, 1))
        return float(min(0.98, max(0.05, nf * math.exp(-err / scale))))

    def predict_with_learning(self, prediction: Dict) -> Dict:
        """Confiança por variável + calibração das probabilidades de eventos + resumo do aprendizado."""
        try:
            res = dict(prediction)
            for var in VARS:
                res[f'{var}_confidence'] = self._confidence(
                    var, res.get(f'{var}_spread', 0.0), res.get(f'{var}_n', 1), res.get(var))
            res['wind_confidence'] = res['wind_speed_confidence']      # nome antigo
            raw = res.get('event_probs_raw', {})
            for name in self.EVENTS:
                if name in raw:
                    res[f'{name}_probability'] = self.calibrate_event(name, raw[name])
            res['learning'] = self.get_learning_summary()
            return res
        except Exception as e:
            logger.error(f"Erro ao aplicar aprendizado: {e}")
            return prediction

    # ---------------------------------------------------------------- relatórios
    def get_skill_summary(self) -> Dict:
        """Taxa de acerto por variável (o que o dashboard mostra ao lado da previsão de agora)."""
        out = {}
        gains = []
        for var, c in Config.VARIABLES.items():
            buckets = {}
            for b in BUCKET_NAMES:
                sysd, ens = self._series_stat(var, b, 'system'), self._series_stat(var, b, 'ensemble')
                if not sysd and not ens:
                    continue
                use = sysd or ens
                buckets[b] = {
                    'n': use['n'], 'hit_rate': round(use['hits'] / use['n'], 3),
                    'mae': round(use['mae'], 3), 'bias': round(use['bias'], 3),
                    'ensemble_hit_rate': round(ens['hits'] / ens['n'], 3) if ens else None,
                    'ensemble_mae': round(ens['mae'], 3) if ens else None,
                    'correction_active': self._gate_open(var, b)}
                if sysd and ens and ens['mae'] > 0 and sysd['n'] >= self.MIN_N:
                    gains.append(1 - sysd['mae'] / ens['mae'])
            out[var] = {'label': c['label'], 'unit': c['unit'], 'tolerance': c['tol'],
                        'now': self.expected_accuracy(var, 1), 'buckets': buckets}
        return {'verified': self.state.get('verified_total', 0), 'pending': self.state.get('pending_count', 0),
                'vars': out, 'radar': self.get_radar_agreement(), 'storm': self.get_storm_skill(),
                'error_reduction_vs_consensus': round(float(np.mean(gains)), 3) if gains else None}

    def get_learning_summary(self) -> Dict:
        hits = []
        for var in VARS:
            e = self.expected_accuracy(var, 1)
            if e['hit_rate'] is not None:
                hits.append(e['hit_rate'])
        return {'verified': self.state.get('verified_total', 0), 'pending': self.state.get('pending_count', 0),
                'overall_accuracy': float(np.mean(hits)) if hits else None}

    def get_location_bias(self) -> Dict:
        out = {}
        for var in VARS:
            st = self.state['stats'].get(var, {}).get('ensemble')
            if st and st['n'] >= self.MIN_N:
                out[var] = {'mean_bias': float(st['bias']), 'mean_abs_error': float(st['mae']),
                            'samples': st['n']}
        return out

    def get_model_performance(self) -> Dict:
        out = {}
        w = self.get_model_weights()
        for var, models in self.state['stats'].items():
            for m, s in models.items():
                out.setdefault(m, {})[var] = {
                    'mae': round(s['mae'], 3), 'bias': round(s['bias'], 3), 'n': s['n'],
                    'hit_rate': round(s.get('hits', 0) / s['n'], 3) if s['n'] else None,
                    'weight': round(w.get(var, {}).get(m, 0), 4)}
        return out

    def get_accuracy_by_horizon(self) -> Dict:
        out = {}
        for b in BUCKET_NAMES:
            rates, maes, n = {}, {}, 0
            for var in VARS:
                s = self._series_stat(var, b, 'system') or self._series_stat(var, b, 'ensemble')
                if s:
                    rates[var], maes[var] = s['hits'] / s['n'], round(s['mae'], 3)
                    n = max(n, s['n'])
            if rates:
                out[b] = {'mean_accuracy': float(np.mean(list(rates.values()))),
                          'mae': maes, 'samples': n}
        return out

    def get_seasonal_pattern(self) -> Dict:
        hist = self.state['history']
        if len(hist) < 20:
            return {}
        seasons = {'summer': [], 'autumn': [], 'winter': [], 'spring': []}
        for h in hist:
            m = _parse_iso(h['timestamp']).month
            s = 'summer' if m in (12, 1, 2) else 'autumn' if m in (3, 4, 5) else \
                'winter' if m in (6, 7, 8) else 'spring'
            seasons[s].append(h)
        out = {}
        for s, items in seasons.items():
            temps = [i['observation'].get('temperature') for i in items
                     if i['observation'].get('temperature') is not None]
            prec = [i['observation'].get('precipitation') for i in items
                    if i['observation'].get('precipitation') is not None]
            if items:
                out[s] = {'avg_temperature': float(np.mean(temps)) if temps else None,
                          'avg_precipitation': float(np.mean(prec)) if prec else None,
                          'temp_std': float(np.std(temps)) if temps else None, 'samples': len(items)}
        return out

    def get_local_extreme_patterns(self) -> Dict:
        out = {}
        for name, st in self.state['event_stats'].items():
            out[name] = {'forecasts_verified': st['n'], 'events_observed': st['hits'],
                         'brier_score': round(st['brier'], 4)}
        return out

    def record_prediction_observation(self, prediction: Dict, observation: Dict, event_type: str):
        """Registro manual (endpoint /record-observation)."""
        with _LOCK:
            errs = {}
            for var in VARS:
                p, o = prediction.get(var), observation.get(var)
                if p is None or o is None:
                    continue
                errs[var] = float(o) - float(p)
                self._upd(self.state['stats'].setdefault(var, {}).setdefault('ensemble', {}),
                          errs[var], self.is_hit(var, float(p), float(o)))
            self.state['history'].append({
                'timestamp': _now().isoformat(), 'event_type': event_type, 'horizon': 0,
                'prediction': prediction, 'observation': observation, 'error': errs,
                'accuracy': self._acc(errs)})
            self.state['history'] = self.state['history'][-300:]
            self._save()

    def get_learning_report(self) -> Dict:
        s = self.get_learning_summary()
        return {
            'location_id': self.location_id,
            'samples': s['verified'],
            'pending_verifications': s['pending'],
            'overall_accuracy': s['overall_accuracy'],
            'skill': self.get_skill_summary(),
            'biases': self.get_location_bias(),
            'model_performance': self.get_model_performance(),
            'seasonal_patterns': self.get_seasonal_pattern(),
            'accuracy_by_horizon': self.get_accuracy_by_horizon(),
            'extreme_patterns': self.get_local_extreme_patterns(),
            'last_update': _now().isoformat(),
        }

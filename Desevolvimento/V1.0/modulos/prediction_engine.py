import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from config import Config

logger = logging.getLogger(__name__)

VAR_COLS = {
    'temperature': 'temperature_2m',
    'humidity': 'relative_humidity_2m',
    'wind_speed': 'wind_speed_10m',
    'precipitation': 'precipitation',
}
SNAPSHOT_HORIZONS = [1, 3, 6, 12, 24, 48, 72]   # horas à frente guardadas p/ verificação futura
EVENT_WINDOW_HOURS = 24                          # janela dos eventos extremos
SEVERE_CODES = [95, 96, 99]
HAIL_CODES = [96, 99]
EVENT_WEIGHT_VAR = {'thunderstorm': 'precipitation', 'heavy_rain': 'precipitation',
                    'hail': 'precipitation', 'extreme_wind': 'wind_speed'}


def _arr(df: pd.DataFrame, col: str, start: int, end: int) -> Optional[np.ndarray]:
    if col not in df.columns:
        return None
    a = df[col].iloc[start:end].to_numpy(dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return None
    return a


class PredictionEngine:
    """
    Consenso ponderado entre vários modelos. Os pesos vêm do aprendizado
    (AdaptiveLearning): modelos que erram menos naquela localização pesam mais.
    """

    def __init__(self):
        self.thresholds = Config.THRESHOLDS

    # ------------------------------------------------------------ utilitários
    @staticmethod
    def weighted_mean(pairs: List[Tuple[str, float]], wdict: Dict) -> Tuple[Optional[float], float, int]:
        """pairs=[(modelo, valor)] -> (média ponderada, desvio ponderado, n)"""
        if not pairs:
            return None, 0.0, 0
        default = float(np.mean(list(wdict.values()))) if wdict else 1.0
        w = np.array([wdict.get(m, default) for m, _ in pairs], dtype=float)
        x = np.array([v for _, v in pairs], dtype=float)
        w = w / w.sum()
        mean = float(w @ x)
        spread = float(np.sqrt(w @ ((x - mean) ** 2)))
        return mean, spread, len(pairs)

    def weighted_series(self, processed: Dict, col: str, wdict: Dict) -> Optional[pd.Series]:
        """Série temporal (índice = hora local) com a média ponderada entre modelos."""
        series = {}
        for m, md in processed.items():
            df = md['hourly']
            if col in df.columns and df[col].notna().any():
                series[m] = df.set_index('time')[col]
        if not series:
            return None
        wide = pd.DataFrame(series)
        default = float(np.mean(list(wdict.values()))) if wdict else 1.0
        w = np.array([wdict.get(m, default) for m in wide.columns], dtype=float)
        vals = wide.to_numpy(dtype=float)
        mask = ~np.isnan(vals)
        num = np.where(mask, vals, 0.0) @ w
        den = mask.astype(float) @ w
        with np.errstate(invalid='ignore', divide='ignore'):
            res = np.where(den > 0, num / den, np.nan)
        return pd.Series(res, index=wide.index)

    def _per_model_now(self, processed: Dict) -> Dict[str, Dict[str, float]]:
        per_model = {}
        for m, md in processed.items():
            df, i = md['hourly'], md['now_idx']
            vals = {}
            for var, col in VAR_COLS.items():
                if col not in df.columns:
                    continue
                if var == 'precipitation':   # acumulado das próximas 24h
                    w = df[col].iloc[i:i + 24]
                    if w.notna().any():
                        vals[var] = float(w.sum(skipna=True))
                else:
                    v = df[col].iloc[i]
                    if pd.notna(v):
                        vals[var] = float(v)
            if vals:
                per_model[m] = vals
        return per_model

    # -------------------------------------------------------- scores de eventos
    @staticmethod
    def _trigger(df, i, end, n):
        pp = _arr(df, 'precipitation_probability', i, end)
        if pp is not None:
            return np.clip(np.nan_to_num(pp) / 100.0, 0, 1)
        pr = _arr(df, 'precipitation', i, end)
        if pr is not None:
            return np.clip(np.nan_to_num(pr) / 1.0, 0, 1)
        return np.ones(n)

    def _score_thunderstorm(self, df, i, end):
        n = len(df.iloc[i:end])
        if n == 0:
            return None
        thr = self.thresholds['thunderstorm']
        parts = []
        cape = _arr(df, 'cape', i, end)
        if cape is not None:
            parts.append(np.clip((np.nan_to_num(cape) - 0.5 * thr['cape']) / (1.5 * thr['cape']), 0, 1))
        li = _arr(df, 'lifted_index', i, end)
        if li is not None:
            parts.append(np.clip(-np.nan_to_num(li, nan=5.0) / 6.0, 0, 1))
        code = _arr(df, 'weather_code', i, end)
        if not parts and code is None:
            return None
        code_score = (np.isin(np.nan_to_num(code, nan=-1), SEVERE_CODES).astype(float)
                      if code is not None else np.zeros(n))
        instab = np.mean(parts, axis=0) if parts else np.zeros(n)
        trig = self._trigger(df, i, end, n)
        return np.maximum(code_score, instab * (0.2 + 0.8 * trig))

    def _score_heavy_rain(self, df, i, end):
        pr = _arr(df, 'precipitation', i, end)
        if pr is None:
            return None
        pr = np.nan_to_num(pr)
        thr = self.thresholds['heavy_rain']['precipitation']
        hour_score = np.clip(pr / 15.0, 0, 1)                       # 7,5 mm/h ≈ 50%
        roll = pd.Series(pr).rolling(24, min_periods=1).sum().to_numpy()
        day_score = np.clip(roll / float(thr), 0, 1)                # 50 mm/24h = 100%
        score = np.maximum(hour_score, day_score)
        pp = _arr(df, 'precipitation_probability', i, end)
        if pp is not None:
            score = score * (0.6 + 0.4 * np.clip(np.nan_to_num(pp) / 100.0, 0, 1))
        return score

    def _score_hail(self, df, i, end):
        n = len(df.iloc[i:end])
        cape = _arr(df, 'cape', i, end)
        code = _arr(df, 'weather_code', i, end)
        if cape is None and code is None:
            return None
        thr = self.thresholds['hail']['cape']
        code_score = (np.isin(np.nan_to_num(code, nan=-1), HAIL_CODES).astype(float)
                      if code is not None else np.zeros(n))
        cape_score = (np.clip((np.nan_to_num(cape) - 0.5 * thr) / thr, 0, 1)
                      if cape is not None else np.zeros(n))
        fl = _arr(df, 'freezing_level_height', i, end)
        fl_factor = np.where(np.nan_to_num(fl, nan=0) > 4500, 0.7, 1.0) if fl is not None else 1.0
        trig = self._trigger(df, i, end, n)
        return np.maximum(code_score, cape_score * (0.2 + 0.8 * trig) * fl_factor)

    def _score_extreme_wind(self, df, i, end):
        g = _arr(df, 'wind_gusts_10m', i, end)
        if g is None:
            return None
        lo = self.thresholds['wind']['speed']           # 40 km/h: começa a pontuar
        return np.clip((np.nan_to_num(g) - lo) / 50.0, 0, 1)   # 90 km/h = 100%

    @staticmethod
    def _peak(score: np.ndarray) -> float:
        return float(pd.Series(score).rolling(3, min_periods=1).mean().max()) * 100.0

    def _event_prob(self, processed: Dict, kind: str, weights: Dict) -> float:
        fn = {'thunderstorm': self._score_thunderstorm, 'heavy_rain': self._score_heavy_rain,
              'hail': self._score_hail, 'extreme_wind': self._score_extreme_wind}[kind]
        pairs = []
        for m, md in processed.items():
            i = md['now_idx']
            sc = fn(md['hourly'], i, i + EVENT_WINDOW_HOURS)
            if sc is not None and len(sc):
                pairs.append((m, self._peak(sc)))
        mean, _, _ = self.weighted_mean(pairs, weights.get(EVENT_WEIGHT_VAR[kind], {}))
        return float(mean) if mean is not None else 0.0

    # compatibilidade com chamadas antigas
    def predict_thunderstorm_probability(self, processed, weights=None):
        return self._event_prob(processed, 'thunderstorm', weights or {})

    def predict_heavy_rain_probability(self, processed, weights=None):
        return self._event_prob(processed, 'heavy_rain', weights or {})

    def predict_hail_probability(self, processed, weights=None):
        return self._event_prob(processed, 'hail', weights or {})

    def predict_extreme_wind_probability(self, processed, weights=None):
        return self._event_prob(processed, 'extreme_wind', weights or {})

    # ------------------------------------------------------- previsão principal
    def compute_base_prediction(self, processed: Dict, weights: Dict = None) -> Dict:
        weights = weights or {}
        per_model = self._per_model_now(processed)
        if not per_model:
            return {}

        base = {}
        shares = {m: [] for m in per_model}
        for var in VAR_COLS:
            pairs = [(m, v[var]) for m, v in per_model.items() if var in v]
            mean, spread, n = self.weighted_mean(pairs, weights.get(var, {}))
            base[var] = mean
            base[f'{var}_spread'] = spread
            base[f'{var}_n'] = n
            if pairs:
                wd = weights.get(var, {})
                default = float(np.mean(list(wd.values()))) if wd else 1.0
                ws = np.array([wd.get(m, default) for m, _ in pairs], dtype=float)
                ws = ws / ws.sum()
                for (m, _), w in zip(pairs, ws):
                    shares[m].append(float(w))

        base['models_used'] = list(per_model)
        base['models_detail'] = {
            m: {**{k: round(v, 2) for k, v in vals.items()},
                'label': Config.MODEL_LABELS.get(m, m),
                'weight_pct': round(100 * float(np.mean(shares[m])), 1) if shares[m] else 0.0}
            for m, vals in per_model.items()
        }

        events = {k: self._event_prob(processed, k, weights)
                  for k in ('thunderstorm', 'heavy_rain', 'hail', 'extreme_wind')}
        base['event_probs_raw'] = events
        base['thunderstorm_probability'] = events['thunderstorm']
        base['heavy_rain_probability'] = events['heavy_rain']
        base['hail_probability'] = events['hail']
        base['extreme_wind_probability'] = events['extreme_wind']
        base['tornado_probability'] = 0.0   # não estimável com dados de superfície do Open-Meteo
        base['event_window_hours'] = EVENT_WINDOW_HOURS
        return base

    # ------------------------------------------------- snapshot p/ verificação
    def build_snapshot(self, processed: Dict, weights: Dict = None) -> List[Dict]:
        """Guarda o que cada modelo (e o consenso) previu para +1h..+72h, p/ comparar depois."""
        weights = weights or {}
        issued = datetime.now(timezone.utc).isoformat()
        entries = []
        for h in SNAPSHOT_HORIZONS:
            per_model, target = {}, None
            for m, md in processed.items():
                df = md['hourly']
                idx = md['now_idx'] + h
                if idx >= len(df):
                    continue
                row = df.iloc[idx]
                off = md['model_info'].get('utc_offset_seconds', 0)
                tgt = (row['time'] - timedelta(seconds=off)).strftime('%Y-%m-%dT%H:00')
                target = target or tgt
                vals = {var: round(float(row[col]), 3) for var, col in VAR_COLS.items()
                        if col in df.columns and pd.notna(row[col])}
                if vals:
                    per_model[m] = vals
            if not per_model or not target:
                continue
            ens = {}
            for var in VAR_COLS:
                pairs = [(m, v[var]) for m, v in per_model.items() if var in v]
                mean, _, _ = self.weighted_mean(pairs, weights.get(var, {}))
                if mean is not None:
                    ens[var] = round(mean, 3)
            entries.append({'issued': issued, 'target': target, 'horizon': h,
                            'values': {**per_model, 'ensemble': ens}})
        return entries

"""
Aprendizado adaptativo por localização (auto-calibração real).

Ciclo:
 1. A cada previsão guardamos o que CADA modelo previu para +1h..+72h (snapshot).
 2. Depois que essas horas passam, comparamos com o valor observado
    (análise do Open-Meteo) e atualizamos MAE e viés por modelo/variável.
 3. Pesos dos modelos = 1/MAE² (suavizado) -> modelos melhores no local pesam mais.
 4. Viés sistemático do consenso é corrigido (com encolhimento p/ poucas amostras).
 5. Probabilidades de eventos extremos são recalibradas pela frequência observada.
"""
import json
import math
import os
import threading
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)
UTC = timezone.utc
_LOCK = threading.RLock()


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_hour(s: str) -> datetime:
    return datetime.strptime(s[:13], '%Y-%m-%dT%H').replace(tzinfo=UTC)


def _parse_iso(s: str) -> datetime:
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=UTC)


class AdaptiveLearning:
    MIN_N = 5
    HORIZON_LIMIT = 24           # só horizontes <=24h alimentam pesos/viés
    SHRINK_K = 5                 # força do "prior" nos pesos
    BIAS_SHRINK = 10
    EVENT_SHRINK = 10
    PRIOR_MAE = {'temperature': 2.0, 'humidity': 10.0, 'wind_speed': 5.0, 'precipitation': 0.5}
    ACC_SCALE = {'temperature': 5.0, 'humidity': 30.0, 'wind_speed': 15.0, 'precipitation': 2.0}
    BIAS_CAP = {'temperature': 4.0, 'humidity': 15.0, 'wind_speed': 10.0}
    VARS = ('temperature', 'humidity', 'wind_speed', 'precipitation')
    EVENTS = ('thunderstorm', 'heavy_rain', 'extreme_wind')   # os que dá p/ verificar

    def __init__(self, location_id: int):
        self.location_id = location_id
        self.state_path = f'dados/processed/location_{location_id}_learning.json'
        self.state = self._default_state()
        self._load()

    # ------------------------------------------------------------ persistência
    @staticmethod
    def _default_state() -> Dict:
        return {'stats': {}, 'horizon_stats': {}, 'event_bins': {}, 'event_stats': {},
                'pending': [], 'event_pending': [], 'history': [],
                'last_snapshot': None, 'verified_total': 0}

    def _load(self):
        with _LOCK:
            try:
                if os.path.exists(self.state_path):
                    with open(self.state_path, 'r') as f:
                        self.state.update(json.load(f))
            except Exception as e:
                logger.error(f"Erro ao carregar aprendizado: {str(e)}")

    def _save(self):
        with _LOCK:
            try:
                os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
                tmp = self.state_path + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump(self.state, f, default=str)
                os.replace(tmp, self.state_path)
            except Exception as e:
                logger.error(f"Erro ao salvar aprendizado: {str(e)}")

    # ---------------------------------------------------------------- snapshot
    def record_forecast_snapshot(self, entries: List[Dict], event_probs: Dict = None,
                                 min_interval_hours: int = 3) -> bool:
        with _LOCK:
            now = _now()
            last = self.state.get('last_snapshot')
            if last and now - _parse_iso(last) < timedelta(hours=min_interval_hours):
                return False
            self.state['pending'].extend(entries)
            self.state['pending'] = self.state['pending'][-1500:]
            if event_probs:
                probs = {k: round(float(v) / 100.0, 4) for k, v in event_probs.items() if k in self.EVENTS}
                self.state['event_pending'].append({
                    'issued': now.isoformat(),
                    'window_end': (now + timedelta(hours=24)).isoformat(),
                    'probs': probs})
                self.state['event_pending'] = self.state['event_pending'][-100:]
            self.state['last_snapshot'] = now.isoformat()
            self._save()
            return True

    def has_due_verifications(self) -> bool:
        cutoff = _now() - timedelta(hours=1)
        try:
            if any(_parse_hour(e['target']) <= cutoff for e in self.state['pending']):
                return True
            return any(_parse_iso(e['window_end']) <= cutoff for e in self.state['event_pending'])
        except Exception:
            return False

    # ------------------------------------------------------------ verificação
    def verify_pending(self, obs: Dict) -> int:
        """obs = {'YYYY-MM-DDTHH:00': {temperature, humidity, wind_speed, precipitation, ...}}"""
        with _LOCK:
            now = _now()
            cutoff = now - timedelta(hours=1)
            keep, scored = [], 0
            for e in self.state['pending']:
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
            self.state['pending'] = keep
            self._verify_events(obs, now, cutoff)
            self.state['verified_total'] += scored
            self._save()
            if scored:
                logger.info(f"Aprendizado loc {self.location_id}: {scored} previsões verificadas")
            return scored

    def _update_stat(self, var: str, model: str, err: float):
        s = self.state['stats'].setdefault(var, {}).setdefault(model, {'n': 0, 'mae': 0.0, 'bias': 0.0})
        s['n'] += 1
        a = max(1.0 / s['n'], 0.05)
        s['mae'] += a * (abs(err) - s['mae'])
        s['bias'] += a * (err - s['bias'])

    @staticmethod
    def _bucket(h: int) -> str:
        return '0-6h' if h <= 6 else '6-24h' if h <= 24 else '24-48h' if h <= 48 else '48-72h'

    def _acc(self, errs: Dict[str, float]) -> float:
        accs = [max(0.0, 1 - abs(e) / self.ACC_SCALE[v]) for v, e in errs.items()]
        return float(np.mean(accs)) if accs else 0.0

    def _score_entry(self, e: Dict, o: Dict):
        h = int(e.get('horizon', 0))
        ens_errs = {}
        for model, vals in e['values'].items():
            for var in self.VARS:
                pred, real = vals.get(var), o.get(var)
                if pred is None or real is None:
                    continue
                err = float(real) - float(pred)
                if h <= self.HORIZON_LIMIT:
                    self._update_stat(var, model, err)
                if model == 'ensemble':
                    ens_errs[var] = err
                    hs = self.state['horizon_stats'].setdefault(self._bucket(h), {}).setdefault(
                        var, {'n': 0, 'mae': 0.0})
                    hs['n'] += 1
                    hs['mae'] += (abs(err) - hs['mae']) / hs['n']
        if ens_errs:
            self.state['history'].append({
                'timestamp': e['target'] + ':00+00:00', 'event_type': 'verification', 'horizon': h,
                'prediction': e['values'].get('ensemble', {}),
                'observation': {v: o.get(v) for v in self.VARS},
                'error': ens_errs, 'accuracy': self._acc(ens_errs)})
            self.state['history'] = self.state['history'][-500:]

    @staticmethod
    def _event_outcomes(window: List[Dict]) -> Dict[str, bool]:
        codes = [o.get('weather_code') for o in window if o.get('weather_code') is not None]
        pr = [o.get('precipitation') or 0 for o in window]
        gusts = [o.get('wind_gusts') or 0 for o in window]
        return {
            'thunderstorm': any(int(c) in (95, 96, 99) for c in codes),
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

    # ------------------------------------------------------- pesos / correções
    def get_model_weights(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for var, prior in self.PRIOR_MAE.items():
            w = {}
            for model, s in self.state['stats'].get(var, {}).items():
                if model == 'ensemble':
                    continue
                n = s['n']
                mae_eff = (n * s['mae'] + self.SHRINK_K * prior) / (n + self.SHRINK_K)
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

    def calibrate_event(self, name: str, p_pct: float) -> float:
        p = p_pct / 100.0
        b = self.state['event_bins'].get(name, {}).get(str(min(int(p * 5), 4)))
        if b and b['n'] >= self.MIN_N:
            p = (b['hits'] + self.EVENT_SHRINK * p) / (b['n'] + self.EVENT_SHRINK)
        return float(min(max(p, 0.0), 1.0) * 100.0)

    def _confidence(self, var: str, spread: float, n_models: int, mean: Optional[float]) -> float:
        st = self.state['stats'].get(var, {}).get('ensemble')
        mae = st['mae'] if (st and st['n'] >= self.MIN_N and var != 'precipitation') else 0.0
        err = math.sqrt((spread or 0.0) ** 2 + mae ** 2)
        scale = (0.5 * (mean or 0.0) + 3.0) if var == 'precipitation' else \
            {'temperature': 3.0, 'humidity': 15.0, 'wind_speed': 10.0}[var]
        nf = min(1.0, 0.55 + 0.1 * max(n_models, 1))
        return float(min(0.98, max(0.05, nf * math.exp(-err / scale))))

    def predict_with_learning(self, prediction: Dict) -> Dict:
        try:
            res = dict(prediction)
            for var in ('temperature', 'humidity', 'wind_speed'):
                v = res.get(var)
                if v is None:
                    continue
                corr = self.get_bias(var)
                v = v + corr
                if var == 'humidity':
                    v = min(100.0, max(0.0, v))
                if var == 'wind_speed':
                    v = max(0.0, v)
                res[var] = v
                res[f'{var}_bias_correction'] = round(corr, 3)

            for var, key in (('temperature', 'temperature_confidence'), ('humidity', 'humidity_confidence'),
                             ('precipitation', 'precipitation_confidence'), ('wind_speed', 'wind_confidence')):
                res[key] = self._confidence(var, res.get(f'{var}_spread', 0.0),
                                            res.get(f'{var}_n', 1), res.get(var))

            raw = res.get('event_probs_raw', {})
            for name, key in (('thunderstorm', 'thunderstorm_probability'),
                              ('heavy_rain', 'heavy_rain_probability'),
                              ('extreme_wind', 'extreme_wind_probability')):
                if name in raw:
                    res[key] = self.calibrate_event(name, raw[name])

            res['learning'] = self.get_learning_summary()
            return res
        except Exception as e:
            logger.error(f"Erro ao fazer previsão com aprendizado: {str(e)}")
            return prediction

    # ---------------------------------------------------------------- relatórios
    def get_learning_summary(self) -> Dict:
        errs = {}
        for var in self.VARS:
            st = self.state['stats'].get(var, {}).get('ensemble')
            if st and st['n'] >= self.MIN_N:
                errs[var] = st['mae']
        return {
            'verified': self.state.get('verified_total', 0),
            'pending': len(self.state.get('pending', [])),
            'overall_accuracy': self._acc(errs) if errs else None,
        }

    def get_location_bias(self) -> Dict:
        out = {}
        for var in self.VARS:
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
                out.setdefault(m, {})[var] = {'mae': round(s['mae'], 3), 'bias': round(s['bias'], 3),
                                              'n': s['n'], 'weight': round(w.get(var, {}).get(m, 0), 4)}
        return out

    def get_accuracy_by_horizon(self) -> Dict:
        out = {}
        for bucket, vs in self.state['horizon_stats'].items():
            accs = {v: max(0.0, 1 - s['mae'] / self.ACC_SCALE[v]) for v, s in vs.items()}
            out[bucket] = {'mean_accuracy': float(np.mean(list(accs.values()))) if accs else 0.0,
                           'mae': {v: round(s['mae'], 3) for v, s in vs.items()},
                           'samples': max((s['n'] for s in vs.values()), default=0)}
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
            for var in self.VARS:
                p, o = prediction.get(var), observation.get(var)
                if p is None or o is None:
                    continue
                errs[var] = float(o) - float(p)
                self._update_stat(var, 'ensemble', errs[var])
            self.state['history'].append({
                'timestamp': _now().isoformat(), 'event_type': event_type, 'horizon': 0,
                'prediction': prediction, 'observation': observation, 'error': errs,
                'accuracy': self._acc(errs)})
            self.state['history'] = self.state['history'][-500:]
            self._save()

    def get_learning_report(self) -> Dict:
        s = self.get_learning_summary()
        return {
            'location_id': self.location_id,
            'samples': s['verified'],
            'pending_verifications': s['pending'],
            'overall_accuracy': s['overall_accuracy'],
            'biases': self.get_location_bias(),
            'model_performance': self.get_model_performance(),
            'seasonal_patterns': self.get_seasonal_pattern(),
            'accuracy_by_horizon': self.get_accuracy_by_horizon(),
            'extreme_patterns': self.get_local_extreme_patterns(),
            'last_update': _now().isoformat(),
        }

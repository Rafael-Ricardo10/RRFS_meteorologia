"""
Vizinhança: olhar as regiões ao redor para prever e conferir o local.

Por quê: tempestade é pequena (poucos km). O modelo costuma acertar "vai haver tempestade na região" mas errar
"em qual quarteirão". Se a célula passa 15 km ao lado, a previsão para a região estava CERTA. Então:

 1) PREVER: além do local, o sistema baixa a previsão de pontos em anéis (30 e 60 km, 8 direções).
    - risco "nas redondezas" (<= 30 km) = máximo entre o local e os pontos do anel interno;
    - a montante: pontos que estão de onde o vento sopra e que, com a velocidade da corrente, chegam ao local
      nas próximas horas (a previsão do ponto vizinho horas antes "empresta" informação ao local).
 2) CONFERIR: as observações (análise do Open-Meteo nos pontos do anel, radar em dBZ e raios) são medidas em
    duas escalas: LOCAL (<= 10 km) e PERTO (<= 30 km). A probabilidade "nas redondezas" é calibrada pelo
    que aconteceu NA REGIÃO, não só em cima do ponto (por isso a tempestade que passa ao lado deixa de ser "erro").

Custo de API: UMA requisição multi-ponto por hora por local (16 pontos x poucas variáveis, 3 dias).
"""
import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)
EARTH_KM = 6371.0
KINDS = ('lightning', 'thunderstorm', 'hail', 'heavy_rain', 'extreme_wind')
RING_VARS = ['weather_code', 'precipitation', 'precipitation_probability', 'cape', 'lifted_index',
             'convective_inhibition', 'wind_gusts_10m', 'cloud_cover', 'freezing_level_height']
RING_VARS_FALLBACK = ['weather_code', 'precipitation', 'precipitation_probability', 'cape', 'wind_gusts_10m']
OBS_VARS = ['weather_code', 'precipitation', 'wind_gusts_10m']
UPWIND_CONE_DEG = 50.0
RING_WEIGHT = 0.90          # o ponto vizinho vem de um modelo só (sem consenso): peso um pouco menor
UPSTREAM_WEIGHT = 0.85


def destination(lat: float, lon: float, bearing_deg: float, dist_km: float) -> Tuple[float, float]:
    """Ponto a `dist_km` na direção `bearing_deg` (0 = norte, sentido horário)."""
    br, d = math.radians(bearing_deg), dist_km / EARTH_KM
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(br))
    l2 = l1 + math.atan2(math.sin(br) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return round(math.degrees(p2), 4), round((math.degrees(l2) + 540) % 360 - 180, 4)


def ring_points(lat: float, lon: float) -> List[Dict]:
    pts = []
    n = int(Config.NEIGHBOR_POINTS_PER_RING)
    for ri, km in enumerate(Config.NEIGHBOR_RINGS_KM):
        offset = 0.0 if ri % 2 == 0 else 180.0 / n          # anéis alternados: cobrem mais direções
        for i in range(n):
            brg = (360.0 / n) * i + offset
            la, lo = destination(lat, lon, brg, km)
            pts.append({'name': f'r{int(km)}_{int(round(brg)):03d}', 'lat': la, 'lon': lo,
                        'dist_km': float(km), 'bearing': float(brg)})
    return pts


async def _multi(api, session, params: Dict, label: str):
    status, data = await api._get_json(session, f"{api.openmeteo_url}/forecast", params, label, retries=1)
    if status == 200 and data is not None and not isinstance(data, list):
        data = [data]
    return status, data


async def fetch_ring(api, lat: float, lon: float) -> Dict:
    """Previsão horária (3 dias) dos pontos dos anéis, numa única requisição multi-ponto."""
    if not Config.NEIGHBOR_ENABLED:
        return {}
    pts = ring_points(lat, lon)
    base = {'latitude': ','.join(str(p['lat']) for p in pts), 'longitude': ','.join(str(p['lon']) for p in pts),
            'timezone': 'auto', 'forecast_days': 3}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40)) as session:
            status, data = await _multi(api, session, {**base, 'hourly': ','.join(RING_VARS)}, 'Open-Meteo[anel]')
            if status == 400:
                status, data = await _multi(api, session, {**base, 'hourly': ','.join(RING_VARS_FALLBACK)},
                                            'Open-Meteo[anel/básico]')
        if status != 200 or not data or len(data) != len(pts):
            return {}
        return {'points': pts, 'data': [{'hourly': d.get('hourly', {}), 'utc_offset_seconds': d.get('utc_offset_seconds', 0)}
                                        for d in data], 'fetched_at': datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"Erro ao buscar anéis de vizinhança: {e}")
        return {}


async def fetch_ring_observations(api, lat: float, lon: float, past_days: int = 4) -> Dict[str, Dict]:
    """
    'Verdade' em volta do local: análise do Open-Meteo no anel interno (tempo presente, hora a hora, em UTC).
    -> {'YYYY-MM-DDTHH:00': {ring_storm, ring_hail, ring_gust_max, ring_precip_max, ring_n}}
    """
    if not Config.NEIGHBOR_ENABLED:
        return {}
    pts = [p for p in ring_points(lat, lon) if p['dist_km'] <= Config.NEAR_RADIUS_KM]
    if not pts:
        return {}
    params = {'latitude': ','.join(str(p['lat']) for p in pts), 'longitude': ','.join(str(p['lon']) for p in pts),
              'hourly': ','.join(OBS_VARS), 'past_days': past_days, 'forecast_days': 1, 'timezone': 'GMT'}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=40)) as session:
            status, data = await _multi(api, session, params, 'Open-Meteo[anel/observações]')
        if status != 200 or not data:
            return {}
        now_key = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:00')
        out: Dict[str, Dict] = {}
        for d in data:
            h = d.get('hourly') or {}
            for i, t in enumerate(h.get('time', [])):
                key = t[:13] + ':00'
                if key > now_key:
                    continue
                o = out.setdefault(key, {'ring_storm': False, 'ring_hail': False, 'ring_gust_max': None,
                                         'ring_precip_max': None, 'ring_n': 0})
                code = (h.get('weather_code') or [None] * (i + 1))[i]
                gust = (h.get('wind_gusts_10m') or [None] * (i + 1))[i]
                prec = (h.get('precipitation') or [None] * (i + 1))[i]
                o['ring_n'] += 1
                if code is not None:
                    o['ring_storm'] = o['ring_storm'] or int(code) in (95, 96, 99)
                    o['ring_hail'] = o['ring_hail'] or int(code) in (96, 99)
                if gust is not None:
                    o['ring_gust_max'] = gust if o['ring_gust_max'] is None else max(o['ring_gust_max'], gust)
                if prec is not None:
                    o['ring_precip_max'] = prec if o['ring_precip_max'] is None else max(o['ring_precip_max'], prec)
        return out
    except Exception as e:
        logger.error(f"Erro ao buscar observações do anel: {e}")
        return {}


def merge_observations(obs: Dict[str, Dict], *sources: Dict[str, Dict]) -> Dict[str, Dict]:
    """Junta (por hora) radar, raios e anel às observações do ponto, sem criar horas novas."""
    for src in sources:
        for hour, extra in (src or {}).items():
            if hour in obs:
                obs[hour].update(extra)
    return obs


# ----------------------------------------------------------------------------- previsão: pontuações do anel
def ring_scores(ring_raw: Dict, engine) -> Optional[Dict]:
    """
    Pontuação (0-1) hora a hora de cada evento em cada ponto do anel, com as MESMAS regras que o local usa.
    -> {'index': DatetimeIndex, 'meta': {nome: (dist_km, bearing)}, 'scores': {nome: {tipo: array}}}
    """
    if not ring_raw or not ring_raw.get('data'):
        return None
    from modulos.data_processor import DataProcessor
    meta, scores, index = {}, {}, None
    for p, d in zip(ring_raw['points'], ring_raw['data']):
        try:
            df = DataProcessor._numeric_frame(d.get('hourly') or {})
            if df.empty:
                continue
            if index is None:
                index = pd.DatetimeIndex(df['time'])
            per = {}
            for kind in KINDS:
                sc = engine._scorer(kind)(df, 0, len(df))
                if sc is not None and len(sc):
                    per[kind] = np.nan_to_num(np.asarray(sc, dtype=float))
            if per:
                scores[p['name']], meta[p['name']] = per, (p['dist_km'], p['bearing'])
        except Exception as e:
            logger.warning(f"Ponto de vizinhança {p.get('name')} ignorado: {e}")
    if not scores or index is None:
        return None
    return {'index': index, 'meta': meta, 'scores': scores}


def steering(df: pd.DataFrame, now_idx: int) -> Tuple[Optional[float], float]:
    """(direção DE ONDE vem a corrente, velocidade km/h) das próximas 12 h, a partir do consenso local."""
    sl = df.iloc[max(0, now_idx):max(0, now_idx) + 12]
    frm = None
    if 'wind_direction' in sl and sl['wind_direction'].notna().any():
        rad = np.deg2rad(sl['wind_direction'].dropna().to_numpy(dtype=float))
        frm = float((np.rad2deg(np.arctan2(np.sin(rad).sum(), np.cos(rad).sum())) + 360) % 360)
    spd = None
    if 'wind850' in sl and sl['wind850'].notna().any():
        spd = float(sl['wind850'].median()) * 3.6
    elif 'wind_speed' in sl and sl['wind_speed'].notna().any():
        spd = float(sl['wind_speed'].median()) * 1.6
    return frm, float(np.clip(spd if spd is not None else 25.0, 15.0, 80.0))


def near_columns(rs: Optional[Dict], idx: pd.DatetimeIndex, steer: Tuple[Optional[float], float],
                 local: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Risco 'nas redondezas' (0-100) por hora para cada tipo de evento:
      max( local , RING_WEIGHT * máx(pontos do anel interno) , UPSTREAM_WEIGHT * pontos a montante defasados )
    `local` = séries 0-100 do consenso no local. Sem anel, devolve o próprio local.
    """
    out = {k: np.nan_to_num(np.asarray(v, dtype=float)) for k, v in local.items()}
    if not rs:
        return out
    ring_idx = rs['index']
    pos = ring_idx.get_indexer(idx)                      # -1 onde o anel não cobre a hora
    valid = pos >= 0
    from_deg, speed = steer
    for kind in out:
        best = out[kind].copy()
        for name, per in rs['scores'].items():
            arr = per.get(kind)
            if arr is None:
                continue
            dist, brg = rs['meta'][name]
            vals = np.zeros(len(idx))
            vals[valid] = arr[pos[valid]] * 100.0
            if dist <= Config.NEAR_RADIUS_KM:
                best = np.maximum(best, RING_WEIGHT * vals)
            if from_deg is not None:
                dev = abs(((brg - from_deg + 180.0) % 360.0) - 180.0)
                if dev <= UPWIND_CONE_DEG:             # o ponto está de onde o vento sopra
                    lag = int(round(dist / speed))
                    if 0 < lag < len(vals):
                        shifted = np.zeros(len(vals))
                        shifted[lag:] = vals[:-lag]       # o que estava lá há `lag` h chega aqui agora
                        best = np.maximum(best, UPSTREAM_WEIGHT * shifted)
        out[kind] = best
    return out

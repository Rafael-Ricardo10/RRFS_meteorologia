"""
Radar (RainViewer) -> validação das previsões e vigia de tempestades.

O que o serviço gratuito entrega (desde 01/2026): só quadros do PASSADO (10 em 10 min, ~2 h), zoom máx. 7,
paleta única "Universal Blue". Essa paleta é uma tabela fixa de dBZ -> cor (dBZ = índice - 32), então dá para
DECODIFICAR a refletividade de cada pixel (aproximadamente) e não só "tem eco / não tem eco":

    >= 15 dBZ  chuva fraca (eco)            >= 45 dBZ  chuva forte / núcleo convectivo forte
    >= 40 dBZ  núcleo convectivo            >= 55 dBZ  núcleo compatível com granizo (não confirma granizo)

LIMITES HONESTOS: é refletividade composta (sem velocidade/Doppler) -> NÃO detecta tornado nem rotação;
dBZ aqui é aproximado (mosaico de vários radares, sem calibração local); onde não há radar por perto não há eco.
Tudo aqui degrada com elegância: se a decodificação parecer errada (cores desconhecidas), o módulo volta ao
modo antigo (só cobertura de eco) e avisa em 'dbz_reliable': False.

Funções principais:
  sample_radar()   anéis de dBZ ao redor do local (2 / 10 / 30 / 60 km) e tendência vs. 30 min atrás
  motion_field()   movimento do eco (global e por célula), núcleo mais forte, ETA e distância de passagem
  assess()         cruza radar x modelos (confirmado / próximo / aguardando / ...)
  storm_level()    nível de tempestade pelo radar (+ raios, se houver): NENHUM / ATENÇÃO / ALERTA / SEVERO
  RadarLog         histórico de varreduras por local (serve de "verdade" p/ o aprendizado)
"""
import io
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from config import Config

logger = logging.getLogger(__name__)
TILE = 256
ZOOM = 7
GRID = 3                      # grade 3x3 de tiles (a mesma que o dashboard desenha)
HEADERS = {'User-Agent': 'RRFS-Meteorologia/1.0'}
NODATA = -128                 # pixel sem eco
DBZ = Config.RADAR_DBZ
LOCAL_KM = float(Config.LOCAL_RADIUS_KM)
NEAR_KM = float(Config.NEAR_RADIUS_KM)
REGION_KM = 60.0
POINT_KM = 2.0

# --------------------------------------------------------------------------- paleta Universal Blue
# (índice inicial, RGB inicial, índice final, RGB final); dBZ = índice - 32. Interpolação linear no trecho.
_SEGMENTS = [
    (22, (99, 97, 89), 37, (146, 136, 113)), (38, (158, 147, 117), 46, (222, 208, 151)),
    (47, (136, 221, 238), 52, (0, 163, 224)), (53, (0, 154, 213), 62, (0, 85, 136)),
    (63, (0, 81, 128), 66, (0, 71, 104)), (67, (255, 238, 0), 76, (255, 129, 0)),
    (77, (255, 68, 0), 81, (205, 13, 0)), (82, (193, 0, 0), 86, (93, 0, 0)),
    (87, (255, 170, 255), 96, (255, 78, 255)), (97, (255, 255, 255), 106, (255, 255, 255)),
    (107, (0, 255, 0), 127, (0, 255, 0)),
]


def _build_palette():
    dbz, rgb = [], []
    for a, ca, b, cb in _SEGMENTS:
        for i in range(a, b + 1):
            t = 0.0 if b == a else (i - a) / (b - a)
            dbz.append(i - 32)
            rgb.append([round(ca[k] + (cb[k] - ca[k]) * t) for k in range(3)])
    return np.array(dbz, dtype=np.int16), np.array(rgb, dtype=np.int32)


_PAL_DBZ, _PAL_RGB = _build_palette()


def rgba_to_dbz(rgba: np.ndarray) -> Tuple[np.ndarray, float]:
    """RGBA (tile) -> matriz int8 de dBZ (NODATA onde não há eco) e fração de cores reconhecidas (0-1)."""
    alpha = rgba[..., 3]
    out = np.full(alpha.shape, NODATA, dtype=np.int8)
    m = alpha > 20
    if not m.any():
        return out, 1.0
    rgb = rgba[..., :3][m].astype(np.int32)
    key = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    ur = np.stack([(uk >> 16) & 255, (uk >> 8) & 255, uk & 255], axis=1)
    d2 = ((ur[:, None, :] - _PAL_RGB[None, :, :]) ** 2).sum(axis=2)
    j = d2.argmin(axis=1)
    dist = np.sqrt(d2[np.arange(len(uk)), j])
    bad = dist > 45                                  # cor fora da paleta (anti-aliasing etc.)
    dbz_u = np.where(bad, 12, _PAL_DBZ[j]).astype(np.int8)   # há eco, mas intensidade desconhecida: fraco
    out[m] = dbz_u[inv]
    return out, float((~bad)[inv].mean())


def dbz_to_rain_rate(dbz: Optional[float]) -> Optional[float]:
    """Marshall-Palmer (Z = 200 R^1.6). Acima de ~53 dBZ o eco costuma ser granizo: limita em 100 mm/h."""
    if dbz is None:
        return None
    return round(min(100.0, (10 ** (dbz / 10.0) / 200.0) ** (1 / 1.6)), 1)


# --------------------------------------------------------------------------- download dos tiles
_RATE: deque = deque()
_RATE_LOCK = threading.Lock()
_TILE_CACHE: "OrderedDict" = OrderedDict()
_QUALITY: Dict[str, float] = {}
_TILE_LOCK = threading.Lock()
_TILE_CACHE_MAX = 200                    # 200 x 64 KB = ~13 MB


def _throttle():
    """Limite global de requisições por minuto (a API grátis aceita ~100/min por IP)."""
    limit = int(Config.RADAR_MAX_TILE_REQ_PER_MIN)
    while True:
        with _RATE_LOCK:
            now = time.time()
            while _RATE and now - _RATE[0] > 60:
                _RATE.popleft()
            if len(_RATE) < limit:
                _RATE.append(now)
                return
            wait = 60 - (now - _RATE[0]) + 0.05
        time.sleep(min(max(wait, 0.05), 5.0))


def _tile_float(lat: float, lon: float, z: int) -> Tuple[float, float]:
    n = 2.0 ** z
    xf = (lon + 180.0) / 360.0 * n
    yf = (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return xf, yf


def _tile_url(host: str, path: str, x: int, y: int) -> str:
    return f"{host}{path}/{TILE}/{ZOOM}/{x}/{y}/2/0_0.png"


def _fetch_tile(host: str, path: str, z: int, x: int, y: int) -> Optional[np.ndarray]:
    """Tile -> matriz int8 de dBZ (None se indisponível). Quadros do passado não mudam: cache por URL."""
    url = _tile_url(host, path, x, y)
    with _TILE_LOCK:
        hit = _TILE_CACHE.get(url)
        if hit is not None:
            _TILE_CACHE.move_to_end(url)
            return hit
    try:
        from PIL import Image
        _throttle()
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200:
            logger.warning(f"Radar tile HTTP {r.status_code}")
            return None
        arr, ok = rgba_to_dbz(np.array(Image.open(io.BytesIO(r.content)).convert('RGBA')))
        with _TILE_LOCK:
            _TILE_CACHE[url] = arr
            _QUALITY[url] = ok
            while len(_TILE_CACHE) > _TILE_CACHE_MAX:
                old, _ = _TILE_CACHE.popitem(last=False)
                _QUALITY.pop(old, None)
        return arr
    except Exception as e:
        logger.warning(f"Radar tile indisponível: {e}")
        return None


def _grid_dbz(host: str, path: str, cx: int, cy: int) -> Optional[Tuple[np.ndarray, float]]:
    """Grade 3x3 de tiles (768x768 px) centrada no tile (cx, cy) -> (dBZ int8, qualidade da decodificação)."""
    n = 2 ** ZOOM
    coords = [(cx + c - GRID // 2, cy + r - GRID // 2) for r in range(GRID) for c in range(GRID)]

    def get(t):
        if not 0 <= t[1] < n:
            return np.full((TILE, TILE), NODATA, dtype=np.int8)
        return _fetch_tile(host, path, ZOOM, t[0] % n, t[1])
    with ThreadPoolExecutor(max_workers=6) as ex:
        tiles = list(ex.map(get, coords))
    center = tiles[(GRID * GRID) // 2]
    if center is None or sum(t is not None for t in tiles) < 6:
        return None
    blank = np.full((TILE, TILE), NODATA, dtype=np.int8)
    t = [tt if tt is not None else blank for tt in tiles]
    grid = np.vstack([np.hstack(t[r * GRID:(r + 1) * GRID]) for r in range(GRID)])
    qs = [_QUALITY.get(_tile_url(host, path, c[0] % n, c[1])) for c in coords if 0 <= c[1] < n]
    qs = [q for q in qs if q is not None]
    return grid, (float(np.mean(qs)) if qs else 1.0)


def _geometry(lat: float, lon: float):
    xf, yf = _tile_float(lat, lon, ZOOM)
    cx, cy = int(xf), int(yf)
    km_px = 156543.03 * math.cos(math.radians(lat)) / (2 ** ZOOM) / 1000.0
    marker = ((xf - (cx - GRID // 2)) * TILE, (yf - (cy - GRID // 2)) * TILE)
    return cx, cy, km_px, marker


def _ring_stats(grid: np.ndarray, marker, km_px: float) -> Dict:
    """Máximo de dBZ e fração de eco (>=15 dBZ) dentro de cada raio ao redor do local."""
    mx, my = int(round(marker[0])), int(round(marker[1]))
    rmax = int(REGION_KM / km_px) + 1
    y0, y1 = max(0, my - rmax), min(grid.shape[0], my + rmax + 1)
    x0, x1 = max(0, mx - rmax), min(grid.shape[1], mx + rmax + 1)
    win = grid[y0:y1, x0:x1].astype(np.int16)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist = np.hypot(yy - marker[1], xx - marker[0]) * km_px
    out = {}
    for name, km in (('point', POINT_KM), ('local', LOCAL_KM), ('near', NEAR_KM), ('region', REGION_KM)):
        vals = win[dist <= km]
        has = vals > NODATA
        out[name] = {'max': int(vals.max()) if has.any() else None,
                     'cov': float((vals >= DBZ['echo']).mean()) if vals.size else 0.0}
    return out


def _stale(frames: List[Dict], max_age_s: int = 2400) -> bool:
    t = (frames[-1] or {}).get('time')
    return bool(t) and (time.time() - float(t)) > max_age_s


def sample_radar(radar_meta: Dict, lat: float, lon: float) -> Dict:
    """Mede o eco (dBZ) ao redor do ponto no quadro mais recente e ~30 min antes."""
    try:
        host = (radar_meta or {}).get('host')
        frames = ((radar_meta or {}).get('radar') or {}).get('past') or []
        if not host or not frames:
            return {'available': False}
        if _stale(frames):
            return {'available': False, 'stale': True}
        cx, cy, km_px, marker = _geometry(lat, lon)
        res = _grid_dbz(host, frames[-1]['path'], cx, cy)
        if res is None:
            return {'available': False}
        now, quality = res
        rings = _ring_stats(now, marker, km_px)
        reliable = quality >= 0.65
        prev_frame = frames[-4] if len(frames) >= 4 else frames[0]
        prev_res = _grid_dbz(host, prev_frame['path'], cx, cy)
        trend, trend_dbz = None, None
        if prev_res is not None:
            prings = _ring_stats(prev_res[0], marker, km_px)
            dcov = rings['near']['cov'] - prings['near']['cov']
            a, b = rings['near']['max'], prings['near']['max']
            trend_dbz = (a if a is not None else 0) - (b if b is not None else 0)
            if reliable and trend_dbz >= 5:
                trend = 'intensificando'
            elif dcov > 0.05:
                trend = 'crescendo'
            elif (reliable and trend_dbz <= -5) or dcov < -0.05:
                trend = 'diminuindo'
            else:
                trend = 'estável'
        return {
            'available': True, 'frame_time': frames[-1].get('time'),
            'dbz_reliable': bool(reliable), 'decode_quality': round(quality, 3),
            'echo_point': bool((rings['point']['max'] if rings['point']['max'] is not None else -99) >= DBZ['echo']),
            'dbz_point': rings['point']['max'], 'dbz_local': rings['local']['max'],
            'dbz_near': rings['near']['max'], 'dbz_region': rings['region']['max'],
            'coverage_10': round(rings['local']['cov'], 3), 'coverage_30': round(rings['near']['cov'], 3),
            'coverage_60': round(rings['region']['cov'], 3),
            'rain_rate_local_mmh': dbz_to_rain_rate(rings['local']['max']),
            'radius_km': {'near': LOCAL_KM, 'wide': NEAR_KM, 'region': REGION_KM},
            'trend': trend, 'trend_dbz': trend_dbz,
        }
    except Exception as e:
        logger.warning(f"Falha ao analisar radar: {e}")
        return {'available': False}


def _v(x, default=-99):
    return default if x is None else x


def assess(radar: Dict, storm_score_next3h: float, rain_model_now: bool) -> Dict:
    """Cruza radar x modelos. storm_score_next3h em % (0-100)."""
    if not radar or not radar.get('available'):
        return {'available': False, 'storm_status': 'indisponivel',
                'message': 'Radar indisponível no momento; usando só os modelos.'}
    reliable = bool(radar.get('dbz_reliable'))
    echo_local = radar['echo_point'] or radar['coverage_10'] >= 0.05
    wide = radar['coverage_30']
    d_near, d_reg = radar.get('dbz_near'), radar.get('dbz_region')
    if rain_model_now == echo_local:
        agreement = 'concordam'
    elif rain_model_now:
        agreement = 'modelo_chuva_sem_eco'
    else:
        agreement = 'eco_sem_chuva_prevista'

    core_near = reliable and _v(d_near) >= DBZ['convective']
    core_region = reliable and _v(d_reg) >= DBZ['convective']
    if storm_score_next3h >= 50:
        if core_near or (not reliable and wide >= 0.10):
            status = 'confirmado'
            msg = ('Os modelos indicam tempestade e o radar confirma'
                   + (f' núcleo de chuva forte (~{d_near} dBZ) a até {NEAR_KM:.0f} km.' if core_near
                      else ' chuva na região: alerta com apoio do radar.'))
        elif core_region:
            status = 'proximo'
            msg = (f'Os modelos indicam tempestade e o radar mostra núcleo forte (~{d_reg} dBZ) a até '
                   f'{REGION_KM:.0f} km, ainda fora do local: a tempestade pode passar ao lado. '
                   'Para a região, o alerta continua válido.')
        else:
            status = 'aguardando'
            msg = 'Os modelos indicam tempestade, mas o radar ainda não mostra chuva forte perto: o alerta segue só pelos modelos.'
    elif reliable and _v(d_near) >= DBZ['strong'] and storm_score_next3h < 30:
        status = 'radar_sem_previsao'
        msg = f'O radar mostra núcleo forte (~{d_near} dBZ) a até {NEAR_KM:.0f} km que os modelos não previam.'
    elif radar['coverage_10'] >= 0.2 and storm_score_next3h < 30:
        status = 'radar_sem_previsao'
        msg = 'O radar mostra chuva sobre o local que os modelos não previam.'
    else:
        status = 'sem_alerta'
        msg = 'Sem sinais de tempestade nos modelos e no radar.'
    if radar.get('trend') in ('crescendo', 'intensificando') and wide >= 0.05:
        msg += ' O eco está ' + ('se intensificando.' if radar['trend'] == 'intensificando' else 'aumentando.')
    keys = ('coverage_10', 'coverage_30', 'trend', 'radius_km', 'frame_time', 'dbz_point', 'dbz_local',
            'dbz_near', 'dbz_region', 'dbz_reliable', 'rain_rate_local_mmh')
    return {'available': True, 'echo_local': echo_local, 'agreement': agreement,
            'storm_status': status, 'message': msg, **{k: radar.get(k) for k in keys}}


# ======================================================================
#  MOVIMENTO DO ECO (direção/velocidade, núcleo mais forte, chegada ou "passa ao lado")
# ======================================================================
MAX_SHIFT = 45              # deslocamento máximo procurado em ~30 min (px; ~50 km => ~100 km/h)
MAX_SPEED_KMH = 120.0
MIN_CELL_PX = 128           # áreas menores que isso (~160 km²) são ruído
BLOCK = 4
TRACK_DBZ = 20              # eco usado p/ seguir o movimento (ignora garoa/ruído)
_MOTION_CACHE: Dict = {}
_MOTION_TTL = 300
_COMPASS = ['N', 'NE', 'L', 'SE', 'S', 'SO', 'O', 'NO']


def compass_pt(deg: float) -> str:
    return _COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


def _best_shift(now: np.ndarray, prev: np.ndarray, m: int):
    """Deslocamento (dy, dx) que leva o eco de `prev` até `now` (correlação cruzada por FFT)."""
    a, b = now.astype(np.float32), prev.astype(np.float32)
    na, nb = float(a.sum()), float(b.sum())
    if na < 1 or nb < 1:
        return 0, 0, 0.0
    H, W = a.shape
    sz = (H + m, W + m)
    r = np.fft.irfft2(np.conj(np.fft.rfft2(b, s=sz)) * np.fft.rfft2(a, s=sz), s=sz)
    rr = np.concatenate([r[-m:], r[:m + 1]], axis=0)
    rr = np.concatenate([rr[:, -m:], rr[:, :m + 1]], axis=1)
    iy, ix = np.unravel_index(int(np.argmax(rr)), rr.shape)
    return iy - m, ix - m, float(rr[iy, ix] / math.sqrt(na * nb))


def _label_cells(mask: np.ndarray):
    """Componentes conexos (8 vizinhos) em resolução reduzida -> células ordenadas por área."""
    H, W = mask.shape
    hb, wb = H // BLOCK, W // BLOCK
    ds = mask[:hb * BLOCK, :wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK).mean(axis=(1, 3)) > 0.3
    lab = np.zeros(ds.shape, dtype=np.int32)
    cells, cur = [], 0
    for y in range(hb):
        for x in range(wb):
            if ds[y, x] and lab[y, x] == 0:
                cur += 1
                q, pts = deque([(y, x)]), []
                lab[y, x] = cur
                while q:
                    yy, xx = q.popleft()
                    pts.append((yy, xx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = yy + dy, xx + dx
                            if 0 <= ny < hb and 0 <= nx < wb and ds[ny, nx] and lab[ny, nx] == 0:
                                lab[ny, nx] = cur
                                q.append((ny, nx))
                area = len(pts) * BLOCK * BLOCK
                if area >= MIN_CELL_PX:
                    ys, xs = zip(*pts)
                    cells.append({'id': cur, 'area_px': area,
                                  'cx': (np.mean(xs) + 0.5) * BLOCK, 'cy': (np.mean(ys) + 0.5) * BLOCK,
                                  'bbox': (min(ys) * BLOCK, (max(ys) + 1) * BLOCK, min(xs) * BLOCK, (max(xs) + 1) * BLOCK)})
    cells.sort(key=lambda c: c['area_px'], reverse=True)
    return lab, cells


def _eta_minutes(pos, vel, marker, radius_px, horizon=180.0):
    """Minutos até a borda da célula (círculo equivalente) alcançar o local, movimento constante."""
    rel = np.array(pos, dtype=float) - np.array(marker, dtype=float)
    v = np.array(vel, dtype=float)
    c = float(rel @ rel) - radius_px ** 2
    if c <= 0:
        return 0.0
    a = float(v @ v)
    if a < 1e-9:
        return None
    b = 2.0 * float(rel @ v)
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / (2 * a)
    return round(t, 0) if 0 < t <= horizon else None


def _closest_approach(pos, vel, marker, horizon=180.0):
    """(minutos até a maior aproximação, distância mínima em px) de um ponto que se move em linha reta."""
    rel = np.array(pos, dtype=float) - np.array(marker, dtype=float)
    v = np.array(vel, dtype=float)
    a = float(v @ v)
    if a < 1e-9:
        return None, float(np.hypot(*rel))
    t = -float(rel @ v) / a
    if t <= 0 or t > horizon:
        return None, float(np.hypot(*rel))
    return round(t, 0), float(np.hypot(*(rel + v * t)))


def motion_field(radar_meta: Dict, lat: float, lon: float) -> Dict:
    """
    Para onde o eco está indo (quadro mais recente x ~30 min antes). Devolve o movimento geral e o das
    maiores células (posição em pixels da grade 3x3 do dashboard, seta de 30 min), o núcleo mais forte de
    cada uma e, para cada célula: em quantos minutos ela alcança o local OU a que distância vai passar.
    """
    try:
        host = (radar_meta or {}).get('host')
        frames = ((radar_meta or {}).get('radar') or {}).get('past') or []
        if not host or len(frames) < 2 or _stale(frames):
            return {'available': False}
        f_now = frames[-1]
        f_prev = frames[-4] if len(frames) >= 4 else frames[0]
        dt_min = max(1.0, ((f_now.get('time') or 0) - (f_prev.get('time') or 0)) / 60.0)
        cx, cy, km_px, marker = _geometry(lat, lon)
        ck = (f_now['path'], cx, cy)
        hit = _MOTION_CACHE.get(ck)
        if hit and time.time() - hit[0] < _MOTION_TTL:
            return hit[1]

        g_now, g_prev = _grid_dbz(host, f_now['path'], cx, cy), _grid_dbz(host, f_prev['path'], cx, cy)
        if g_now is None or g_prev is None:
            return {'available': False}
        now_d, prev_d = g_now[0], g_prev[0]
        now_m, prev_m = now_d >= TRACK_DBZ, prev_d >= TRACK_DBZ
        out = {'available': True, 'dt_min': round(dt_min, 0), 'km_per_px': round(km_px, 3),
               'frame_time': f_now.get('time'), 'marker': {'x': round(marker[0], 1), 'y': round(marker[1], 1)},
               'global': None, 'cells': []}
        if now_m.mean() < 0.002 or prev_m.mean() < 0.002:
            _MOTION_CACHE[ck] = (time.time(), out)
            return out

        gdy, gdx, gscore = _best_shift(now_m, prev_m, MAX_SHIFT)

        def vec(dx, dy):
            spd = math.hypot(dx, dy) * km_px / dt_min * 60.0
            head = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0     # para ONDE vai (0=N, 90=L)
            return spd, head

        if gscore >= 0.25:
            spd, head = vec(gdx, gdy)
            if spd <= MAX_SPEED_KMH:
                out['global'] = {'speed_kmh': round(spd, 0), 'heading_deg': round(head, 0),
                                 'toward': compass_pt(head), 'stationary': spd < 5.0,
                                 'confidence': round(gscore, 2)}

        lab, cells = _label_cells(now_m)
        up = np.repeat(np.repeat(lab, BLOCK, axis=0), BLOCK, axis=1)
        H, W = now_m.shape
        up = np.pad(up, ((0, H - up.shape[0]), (0, W - up.shape[1])))
        for c in cells[:8]:
            y0, y1, x0, x1 = c['bbox']
            wy0, wy1 = max(0, y0 - MAX_SHIFT), min(H, y1 + MAX_SHIFT)
            wx0, wx1 = max(0, x0 - MAX_SHIFT), min(W, x1 + MAX_SHIFT)
            comp = (up[wy0:wy1, wx0:wx1] == c['id']) & now_m[wy0:wy1, wx0:wx1]
            dy, dx, sc = _best_shift(comp, prev_m[wy0:wy1, wx0:wx1], MAX_SHIFT)
            src = 'celula'
            if sc < 0.35:                                       # célula sem par claro: usa o movimento geral
                if gscore < 0.25:
                    continue
                dy, dx, sc, src = gdy, gdx, gscore, 'geral'
            spd, head = vec(dx, dy)
            if spd > MAX_SPEED_KMH:
                continue
            per_min = (dx / dt_min, dy / dt_min)
            req = math.sqrt(c['area_px'] / math.pi)
            eta = None if spd < 5.0 else _eta_minutes((c['cx'], c['cy']), per_min, marker, req + 6.0)

            # núcleo mais forte da célula (onde está o perigo de verdade) e sua passagem pelo local
            sub = np.where(up[y0:y1, x0:x1] == c['id'], now_d[y0:y1, x0:x1], NODATA)
            core = None
            if sub.size and int(sub.max()) > NODATA:
                mxd = int(sub.max())
                ys, xs = np.where(sub >= mxd - 3)
                kx, ky = float(xs.mean()) + x0, float(ys.mean()) + y0
                if spd < 5.0:
                    c_eta, c_miss_px = None, float(np.hypot(kx - marker[0], ky - marker[1]))
                else:
                    c_eta, c_miss_px = _closest_approach((kx, ky), per_min, marker)
                dist_km = float(np.hypot(kx - marker[0], ky - marker[1])) * km_px
                core = {'dbz': mxd, 'x': round(kx, 1), 'y': round(ky, 1), 'dist_km': round(dist_km, 1),
                        'closest_min': c_eta, 'miss_km': round(c_miss_px * km_px, 1),
                        'rain_rate_mmh': dbz_to_rain_rate(mxd)}
            out['cells'].append({
                'x': round(c['cx'], 1), 'y': round(c['cy'], 1),
                'area_km2': round(c['area_px'] * km_px ** 2, 0),
                'speed_kmh': round(spd, 0), 'heading_deg': round(head, 0), 'toward': compass_pt(head),
                'stationary': spd < 5.0, 'basis': src, 'confidence': round(sc, 2),
                'arrow': {'x2': round(c['cx'] + per_min[0] * 30, 1), 'y2': round(c['cy'] + per_min[1] * 30, 1)},
                'eta_min': eta, 'over_local': eta == 0.0,
                'max_dbz': core['dbz'] if core else None, 'core': core})
        _MOTION_CACHE[ck] = (time.time(), out)
        if len(_MOTION_CACHE) > 200:
            for k in sorted(_MOTION_CACHE, key=lambda k: _MOTION_CACHE[k][0])[:100]:
                _MOTION_CACHE.pop(k, None)
        return out
    except Exception as e:
        logger.warning(f"Falha ao calcular movimento do radar: {e}")
        return {'available': False}


def summarize_motion(motion: Dict) -> Dict:
    """Resumo em texto: movimento geral, célula que chega mais cedo e, se for o caso, 'passa ao lado'."""
    if not motion or not motion.get('available'):
        return {}
    g, cells = motion.get('global'), motion.get('cells') or []
    reach = sorted([c for c in cells if c.get('eta_min') is not None], key=lambda c: c['eta_min'])
    # núcleos fortes (>= 40 dBZ) que vão passar perto sem cobrir o local
    beside = sorted([c for c in cells if c.get('core') and c['core']['dbz'] >= DBZ['convective']
                     and c['core'].get('closest_min') is not None and c['core']['miss_km'] <= NEAR_KM
                     and c.get('eta_min') is None],
                    key=lambda c: c['core']['closest_min'])
    text, extra = None, {}
    if reach:
        c = reach[0]
        dbz = f" (núcleo ~{c['max_dbz']} dBZ)" if c.get('max_dbz') is not None else ''
        if c['eta_min'] == 0:
            text = f"Há eco de chuva{dbz} sobre o local, se deslocando para {c['toward']} a ~{c['speed_kmh']:.0f} km/h."
        else:
            text = (f"Célula de chuva{dbz} se aproximando: desloca-se para {c['toward']} a ~{c['speed_kmh']:.0f} km/h "
                    f"e deve alcançar o local em ~{c['eta_min']:.0f} min (extrapolação dos últimos 30 min).")
            k = c.get('core')
            if k and k['dbz'] >= DBZ['convective'] and k.get('closest_min') is not None and k['miss_km'] > LOCAL_KM:
                text += (f" O núcleo mais forte (~{k['dbz']} dBZ) deve passar a ~{k['miss_km']:.0f} km do local "
                         f"(daqui a ~{k['closest_min']:.0f} min): a chuva forte pode ficar ao lado.")
                extra = {'passes_beside': True, 'miss_km': k['miss_km'], 'closest_min': k['closest_min']}
    elif beside:
        c = beside[0]
        k = c['core']
        text = (f"Núcleo forte (~{k['dbz']} dBZ) se desloca para {c['toward']} e deve passar AO LADO do local, "
                f"a ~{k['miss_km']:.0f} km, daqui a ~{k['closest_min']:.0f} min: a chuva forte tende a ficar em volta, não em cima.")
        extra = {'passes_beside': True, 'miss_km': k['miss_km'], 'closest_min': k['closest_min']}
    elif g and not g.get('stationary') and cells:
        text = (f"O eco se desloca para {g['toward']} a ~{g['speed_kmh']:.0f} km/h; "
                "pelo movimento atual não deve atingir o local nas próximas 3 h.")
    elif g and g.get('stationary') and cells:
        text = "O eco está quase estacionário."
    top = reach[0] if reach else (beside[0] if beside else None)
    return {'toward': top['toward'] if top else (g or {}).get('toward'),
            'speed_kmh': top['speed_kmh'] if top else (g or {}).get('speed_kmh'),
            'eta_min': reach[0]['eta_min'] if reach else None,
            'max_dbz': (top or {}).get('max_dbz'), 'cells': len(cells), 'text': text, **extra}


# ======================================================================
#  NÍVEL DE TEMPESTADE PELO RADAR (+ raios)
# ======================================================================
def storm_level(sample: Dict, motion: Dict, lightning: Dict = None) -> Dict:
    """
    Regras práticas (heurísticas de acompanhamento, NÃO são alertas oficiais; consulte Defesa Civil/INMET):
      SEVERO : núcleo >= 55 dBZ (compatível com granizo) a <= 30 km
      ALERTA : núcleo >= 45 dBZ a <= 30 km | >= 3 raios a <= 25 km (30 min) | núcleo >= 50 dBZ chegando em <= 60 min
      ATENÇÃO: núcleo >= 40 dBZ a <= 60 km | núcleo >= 45 dBZ chegando em <= 120 min | >= 3 raios a <= 60 km
    Sem dBZ confiável o radar só entra como apoio (nunca sobe o nível sozinho).
    """
    lvl, reasons = 0, []
    s, lg = sample or {}, lightning or {}
    reliable = bool(s.get('available') and s.get('dbz_reliable'))
    d_near = s.get('dbz_near') if reliable else None
    d_reg = s.get('dbz_region') if reliable else None

    def up(n, why):
        nonlocal lvl
        lvl = max(lvl, n)
        reasons.append(why)
    if d_near is not None:
        if d_near >= DBZ['hail']:
            up(3, f'núcleo de ~{d_near} dBZ a até {NEAR_KM:.0f} km (compatível com granizo)')
        elif d_near >= DBZ['strong']:
            up(2, f'núcleo de ~{d_near} dBZ a até {NEAR_KM:.0f} km (chuva forte)')
    if d_reg is not None and d_reg >= DBZ['convective']:
        up(1, f'núcleo convectivo (~{d_reg} dBZ) a até {REGION_KM:.0f} km')
    for c in (motion or {}).get('cells') or []:
        k = c.get('core')
        if not k or not reliable:
            continue
        eta = k.get('closest_min')                      # chegada do NÚCLEO (não da borda da área de chuva)
        if eta is None or k['miss_km'] > NEAR_KM or k['dist_km'] > 80:
            continue
        if k['dbz'] >= DBZ['severe'] and eta <= 60:
            up(2, f"núcleo de ~{k['dbz']} dBZ chegando em ~{eta:.0f} min")
        elif k['dbz'] >= DBZ['strong'] and eta <= 120:
            up(1, f"núcleo de ~{k['dbz']} dBZ chegando em ~{eta:.0f} min")
    if lg.get('available'):
        if (lg.get('n25') or 0) >= 3:
            up(2, f"{lg['n25']} raios a até 25 km nos últimos 30 min")
        elif (lg.get('n60') or 0) >= 3:
            up(1, f"{lg['n60']} raios a até 60 km nos últimos 30 min")
    names = ['NENHUM', 'ATENÇÃO', 'ALERTA', 'SEVERO']
    return {'level': names[lvl], 'level_num': lvl, 'reasons': reasons,
            'hail_core': bool(d_near is not None and d_near >= DBZ['hail']),
            'tornado': 'não detectável: o radar gratuito não tem velocidade (Doppler)'}


# ======================================================================
#  HISTÓRICO DE VARREDURAS (verdade p/ o aprendizado)
# ======================================================================
_LOG_LOCK = threading.RLock()


class RadarLog:
    """Varreduras do radar por local, de 10 em 10 min: guardam o que realmente aconteceu em volta."""

    def __init__(self, key: str):
        self.path = f'dados/processed/radar_{key}.json'

    def _load(self) -> List[Dict]:
        try:
            with open(self.path, 'r') as f:
                return json.load(f)
        except Exception:
            return []

    def append(self, entry: Dict):
        with _LOG_LOCK:
            rows = self._load()
            if rows and entry.get('frame') is not None and rows[-1].get('frame') == entry.get('frame'):
                rows[-1] = {**rows[-1], **entry}                # mesmo quadro: só completa (ex.: raios)
            else:
                rows.append(entry)
            rows = rows[-int(Config.RADAR_LOG_HOURS * 6 + 12):]
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = f'{self.path}.{os.getpid()}.{threading.get_ident()}.tmp'
            with open(tmp, 'w') as f:
                json.dump(rows, f)
            os.replace(tmp, self.path)

    def max_recent(self, field: str, minutes: int = 60):
        """Máximo de um campo ('dn', 'cn'...) nas varreduras dos últimos `minutes` min (None se não houver)."""
        cut = time.time() - minutes * 60
        vals = [r.get(field) for r in self._load() if r.get('ok') and r.get('t', 0) >= cut and r.get(field) is not None]
        return max(vals) if vals else None

    def hourly(self) -> Dict[str, Dict]:
        """'YYYY-MM-DDTHH:00' (UTC) -> resumo da hora (usado p/ conferir previsões e eventos)."""
        out: Dict[str, Dict] = {}
        for r in self._load():
            if not r.get('ok'):
                continue
            h = datetime.fromtimestamp(r['t'], timezone.utc).strftime('%Y-%m-%dT%H:00')
            o = out.setdefault(h, {'radar_scans': 0, 'radar_dbz_local': None, 'radar_dbz_near': None,
                                   'radar_dbz_region': None, 'radar_wet': False,
                                   'radar_reliable': False, 'flashes_local': 0, 'flashes_near': 0})
            o['radar_scans'] += 1
            o['radar_reliable'] = o['radar_reliable'] or bool(r.get('rel'))
            for k_out, k_in in (('radar_dbz_local', 'dl'), ('radar_dbz_near', 'dn'), ('radar_dbz_region', 'dr')):
                v = r.get(k_in)
                if v is not None and (o[k_out] is None or v > o[k_out]):
                    o[k_out] = v
            o['radar_wet'] = o['radar_wet'] or _v(r.get('dl')) >= 20
            o['flashes_local'] += int(r.get('fl') or 0)
            o['flashes_near'] += int(r.get('fn') or 0)
        return out


def scan_location(radar_meta: Dict, lat: float, lon: float, lightning: Dict = None) -> Dict:
    """Varredura completa de um local (usada pelo vigia de 10 min): amostra + movimento + nível + registro."""
    sample = sample_radar(radar_meta, lat, lon)
    motion, msum = {'available': False}, {}
    if sample.get('available'):
        motion = motion_field(radar_meta, lat, lon)
        msum = summarize_motion(motion)
    sev = storm_level(sample, motion, lightning)
    entry = {'t': time.time(), 'frame': sample.get('frame_time'), 'ok': bool(sample.get('available')),
             'rel': bool(sample.get('dbz_reliable')), 'pt': sample.get('dbz_point'), 'dl': sample.get('dbz_local'),
             'dn': sample.get('dbz_near'), 'dr': sample.get('dbz_region'),
             'cl': sample.get('coverage_10'), 'cn': sample.get('coverage_30'),
             'fl': (lightning or {}).get('n10'), 'fn': (lightning or {}).get('n30')}
    return {'sample': sample, 'motion': motion, 'motion_summary': msum, 'severity': sev, 'entry': entry}


def selftest() -> Dict:
    """Confere a decodificação: cada cor da paleta deve voltar ao próprio dBZ (+-1)."""
    rgba = np.zeros((1, len(_PAL_DBZ), 4), dtype=np.uint8)
    rgba[0, :, :3] = _PAL_RGB.astype(np.uint8)
    rgba[0, :, 3] = 255
    dec, ok = rgba_to_dbz(rgba)
    err = np.abs(dec[0].astype(int) - _PAL_DBZ.astype(int))
    # cores repetidas na paleta (branco/verde) caem no 1º dBZ do trecho: não contam como erro
    return {'colors': int(len(_PAL_DBZ)), 'recognized': ok, 'max_err_dbz': int(err[_PAL_DBZ < 65].max())}

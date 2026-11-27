"""
Validação dupla pelo radar (RainViewer).

Limites reais do serviço gratuito (desde 01/2026): só imagens do PASSADO (10 em 10 min, ~2 h),
zoom máx. 7 e só a paleta "Universal Blue". Por isso o radar aqui serve para:
  * dizer se HÁ chuva/eco agora sobre o local (e num raio de ~11 km e ~33 km);
  * ver se o eco está crescendo ou diminuindo nos últimos 30 min;
  * comparar com o que os modelos dizem p/ a hora atual (concordância);
  * apoiar (ou não) um alerta de tempestade.
Ele NÃO mede intensidade (dBZ): a paleta única não permite isso com segurança.
"""
import io
import logging
import math
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)
TILE = 256
ZOOM = 7
HEADERS = {'User-Agent': 'RRFS-Meteorologia/1.0'}


def _tile_float(lat: float, lon: float, z: int) -> Tuple[float, float]:
    n = 2.0 ** z
    xf = (lon + 180.0) / 360.0 * n
    yf = (1.0 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n
    return xf, yf


def _fetch_tile(host: str, path: str, z: int, x: int, y: int) -> Optional[np.ndarray]:
    try:
        from PIL import Image
        url = f"{host}{path}/{TILE}/{z}/{x}/{y}/2/0_0.png"
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code != 200:
            logger.warning(f"Radar tile HTTP {r.status_code}")
            return None
        return np.array(Image.open(io.BytesIO(r.content)).convert('RGBA'))
    except Exception as e:
        logger.warning(f"Radar tile indisponível: {e}")
        return None


def _coverage(rgba: np.ndarray, px: int, py: int, r: int) -> float:
    h, w = rgba.shape[:2]
    y0, y1 = max(0, py - r), min(h, py + r + 1)
    x0, x1 = max(0, px - r), min(w, px + r + 1)
    win = rgba[y0:y1, x0:x1, 3]
    return float((win > 20).mean()) if win.size else 0.0


def _mosaic(host: str, path: str, x: int, y: int, px: int, py: int) -> Optional[np.ndarray]:
    """
    Mosaico 2x2 de tiles com o ponto bem dentro dele: assim o raio de ~33 km nunca é cortado quando
    o local fica perto da borda de um tile. Tile que não existir (polo) vira transparente.
    """
    n = 2 ** ZOOM
    ox = x - 1 if px < TILE // 2 else x
    oy = y - 1 if py < TILE // 2 else y
    coords = [(ox, oy), (ox + 1, oy), (ox, oy + 1), (ox + 1, oy + 1)]
    with ThreadPoolExecutor(max_workers=4) as ex:
        tiles = list(ex.map(lambda c: _fetch_tile(host, path, ZOOM, c[0] % n, c[1]) if 0 <= c[1] < n else
                            np.zeros((TILE, TILE, 4), dtype=np.uint8), coords))
    if tiles[(y - oy) * 2 + (x - ox)] is None:      # o tile do próprio ponto é essencial
        return None
    blank = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    t = [tt if tt is not None else blank for tt in tiles]
    return np.vstack([np.hstack([t[0], t[1]]), np.hstack([t[2], t[3]])]), ox, oy


def sample_radar(radar_meta: Dict, lat: float, lon: float) -> Dict:
    """Mede o eco de radar ao redor do ponto no quadro mais recente e ~30 min antes."""
    try:
        host = (radar_meta or {}).get('host')
        frames = ((radar_meta or {}).get('radar') or {}).get('past') or []
        if not host or not frames:
            return {'available': False}
        xf, yf = _tile_float(lat, lon, ZOOM)
        x, y = int(xf), int(yf)
        px, py = int((xf - x) * TILE), int((yf - y) * TILE)
        km_px = 156543.03 * math.cos(math.radians(lat)) / (2 ** ZOOM) / 1000.0

        res = _mosaic(host, frames[-1]['path'], x, y, px, py)
        if res is None:
            return {'available': False}
        now, ox, oy = res
        mx, my = (x - ox) * TILE + px, (y - oy) * TILE + py      # posição do ponto no mosaico
        prev_frame = frames[-4] if len(frames) >= 4 else frames[0]
        prev_res = _mosaic(host, prev_frame['path'], x, y, px, py)
        prev = prev_res[0] if prev_res else None

        r10, r30 = 10, 30
        out = {
            'available': True,
            'frame_time': frames[-1].get('time'),
            'echo_point': bool(_coverage(now, mx, my, 2) > 0),
            'coverage_10': round(_coverage(now, mx, my, r10), 3),
            'coverage_30': round(_coverage(now, mx, my, r30), 3),
            'radius_km': {'near': round(r10 * km_px, 1), 'wide': round(r30 * km_px, 1)},
            'trend': None,
        }
        if prev is not None and prev_res[1] == ox and prev_res[2] == oy:
            d = out['coverage_30'] - _coverage(prev, mx, my, r30)
            out['trend'] = 'crescendo' if d > 0.05 else 'diminuindo' if d < -0.05 else 'estável'
        return out
    except Exception as e:
        logger.warning(f"Falha ao analisar radar: {e}")
        return {'available': False}


def assess(radar: Dict, storm_score_next3h: float, rain_model_now: bool) -> Dict:
    """Cruza radar x modelos. storm_score_next3h em % (0-100)."""
    if not radar or not radar.get('available'):
        return {'available': False, 'storm_status': 'indisponivel',
                'message': 'Radar indisponível no momento; usando só os modelos.'}
    echo_local = radar['echo_point'] or radar['coverage_10'] >= 0.05
    wide = radar['coverage_30']
    if rain_model_now == echo_local:
        agreement = 'concordam'
    elif rain_model_now:
        agreement = 'modelo_chuva_sem_eco'
    else:
        agreement = 'eco_sem_chuva_prevista'

    if storm_score_next3h >= 50:
        if wide >= 0.10:
            status, msg = 'confirmado', 'Os modelos indicam tempestade e o radar mostra chuva na região: alerta com apoio do radar.'
        else:
            status, msg = 'aguardando', 'Os modelos indicam tempestade, mas o radar ainda não mostra chuva perto: o alerta segue só pelos modelos.'
    elif radar['coverage_10'] >= 0.2 and storm_score_next3h < 30:
        status, msg = 'radar_sem_previsao', 'O radar mostra chuva sobre o local que os modelos não previam.'
    else:
        status, msg = 'sem_alerta', 'Sem sinais de tempestade nos modelos e no radar.'
    if radar.get('trend') == 'crescendo' and wide >= 0.05:
        msg += ' O eco está aumentando.'
    return {'available': True, 'echo_local': echo_local, 'agreement': agreement,
            'storm_status': status, 'message': msg, **{k: radar[k] for k in
            ('coverage_10', 'coverage_30', 'trend', 'radius_km', 'frame_time')}}


# ======================================================================
#  MOVIMENTO DO ECO (direção/velocidade das células e chegada ao local)
# ======================================================================
GRID = 3                    # mesma grade 3x3 de tiles que o dashboard desenha
MAX_SHIFT = 45              # deslocamento máximo procurado em 30 min (px; ~50 km => ~100 km/h)
MAX_SPEED_KMH = 120.0
MIN_CELL_PX = 128           # células menores que isso (~160 km²) são tratadas como ruído
BLOCK = 4                   # rotulagem das células em resolução reduzida
_MOTION_CACHE: Dict = {}
_MOTION_TTL = 300
_COMPASS = ['N', 'NE', 'L', 'SE', 'S', 'SO', 'O', 'NO']


def compass_pt(deg: float) -> str:
    return _COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


def _grid_mask(host: str, path: str, cx: int, cy: int) -> Optional[np.ndarray]:
    """Máscara booleana (eco sim/não) da grade 3x3 de tiles centrada em (cx, cy)."""
    n = 2 ** ZOOM
    coords = [(cx + c - GRID // 2, cy + r - GRID // 2) for r in range(GRID) for c in range(GRID)]
    with ThreadPoolExecutor(max_workers=9) as ex:
        tiles = list(ex.map(lambda t: _fetch_tile(host, path, ZOOM, t[0] % n, t[1]) if 0 <= t[1] < n else
                            np.zeros((TILE, TILE, 4), dtype=np.uint8), coords))
    if sum(t is not None for t in tiles) < 5:
        return None
    blank = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    t = [tt if tt is not None else blank for tt in tiles]
    rows = [np.hstack(t[r * GRID:(r + 1) * GRID]) for r in range(GRID)]
    return np.vstack(rows)[:, :, 3] > 20


def _best_shift(now: np.ndarray, prev: np.ndarray, m: int):
    """
    Deslocamento (dy, dx) que leva o eco de `prev` até `now`, por correlação cruzada (FFT).
    Retorna (dy, dx, score) com score = sobreposição normalizada (0..1).
    """
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
    """Componentes conexos (8 vizinhos) em resolução reduzida -> lista de células ordenadas por área."""
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
    """Minutos até a borda da célula (círculo equivalente) alcançar o local, supondo movimento constante."""
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


def motion_field(radar_meta: Dict, lat: float, lon: float) -> Dict:
    """
    Mede para onde o eco do radar está indo comparando o quadro mais recente com o de ~30 min antes.
    Devolve o movimento geral, o das maiores células (posição em pixels da grade 3x3 do dashboard,
    seta de 30 min à frente) e, para cada uma, em quantos minutos ela alcança o local (extrapolação).
    """
    try:
        host = (radar_meta or {}).get('host')
        frames = ((radar_meta or {}).get('radar') or {}).get('past') or []
        if not host or len(frames) < 2:
            return {'available': False}
        f_now = frames[-1]
        f_prev = frames[-4] if len(frames) >= 4 else frames[0]
        dt_min = max(1.0, ((f_now.get('time') or 0) - (f_prev.get('time') or 0)) / 60.0)

        xf, yf = _tile_float(lat, lon, ZOOM)
        cx, cy = int(xf), int(yf)
        ck = (f_now['path'], cx, cy)
        hit = _MOTION_CACHE.get(ck)
        if hit and time.time() - hit[0] < _MOTION_TTL:
            return hit[1]

        km_px = 156543.03 * math.cos(math.radians(lat)) / (2 ** ZOOM) / 1000.0
        marker = ((xf - (cx - GRID // 2)) * TILE, (yf - (cy - GRID // 2)) * TILE)
        now_m = _grid_mask(host, f_now['path'], cx, cy)
        prev_m = _grid_mask(host, f_prev['path'], cx, cy)
        if now_m is None or prev_m is None:
            return {'available': False}

        out = {'available': True, 'dt_min': round(dt_min, 0), 'km_per_px': round(km_px, 3),
               'frame_time': f_now.get('time'), 'marker': {'x': round(marker[0], 1), 'y': round(marker[1], 1)},
               'global': None, 'cells': []}
        if now_m.mean() < 0.002 or prev_m.mean() < 0.002:       # quase sem eco: nada a seguir
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
        for c in cells[:6]:
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
            out['cells'].append({
                'x': round(c['cx'], 1), 'y': round(c['cy'], 1),
                'area_km2': round(c['area_px'] * km_px ** 2, 0),
                'speed_kmh': round(spd, 0), 'heading_deg': round(head, 0), 'toward': compass_pt(head),
                'stationary': spd < 5.0, 'basis': src, 'confidence': round(sc, 2),
                'arrow': {'x2': round(c['cx'] + per_min[0] * 30, 1), 'y2': round(c['cy'] + per_min[1] * 30, 1)},
                'eta_min': eta, 'over_local': eta == 0.0})
        _MOTION_CACHE[ck] = (time.time(), out)
        if len(_MOTION_CACHE) > 200:
            for k in sorted(_MOTION_CACHE, key=lambda k: _MOTION_CACHE[k][0])[:100]:
                _MOTION_CACHE.pop(k, None)
        return out
    except Exception as e:
        logger.warning(f"Falha ao calcular movimento do radar: {e}")
        return {'available': False}


def summarize_motion(motion: Dict) -> Dict:
    """Resumo em texto: movimento geral e a célula que chegará ao local mais cedo."""
    if not motion or not motion.get('available'):
        return {}
    g, cells = motion.get('global'), motion.get('cells') or []
    reach = sorted([c for c in cells if c.get('eta_min') is not None], key=lambda c: c['eta_min'])
    text = None
    if not cells and not g:
        text = None
    elif reach:
        c = reach[0]
        if c['eta_min'] == 0:
            text = f"Há eco de chuva sobre o local, se deslocando para {c['toward']} a ~{c['speed_kmh']:.0f} km/h."
        else:
            text = (f"Célula de chuva se aproximando: vem se deslocando para {c['toward']} a ~{c['speed_kmh']:.0f} km/h "
                    f"e deve alcançar o local em ~{c['eta_min']:.0f} min (extrapolação do movimento dos últimos 30 min).")
    elif g and not g.get('stationary') and cells:
        text = (f"O eco se desloca para {g['toward']} a ~{g['speed_kmh']:.0f} km/h; "
                "pelo movimento atual não deve atingir o local nas próximas 3 h.")
    elif g and g.get('stationary') and cells:
        text = "O eco está quase estacionário."
    return {'toward': (reach[0]['toward'] if reach else (g or {}).get('toward')),
            'speed_kmh': (reach[0]['speed_kmh'] if reach else (g or {}).get('speed_kmh')),
            'eta_min': reach[0]['eta_min'] if reach else None, 'cells': len(cells), 'text': text}

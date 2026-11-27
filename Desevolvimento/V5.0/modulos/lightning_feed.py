"""
Raios REAIS (verdade para validar tempestade): GOES-East GLM (Geostationary Lightning Mapper), da NOAA.
Dado aberto e gratuito no bucket público da AWS, arquivos de 20 s com cada "flash" (lat, lon, energia).
Cobre as Américas (inclui o Brasil). Atraso típico: 1-3 min.

 * EXPERIMENTAL: foi escrito conforme a documentação do produto GLM-L2-LCFA, mas não pôde ser testado aqui
   contra o bucket real. Se falhar, o sistema segue normalmente sem raios (ver /api/admin/lightning).
 * Requer:  pip install netCDF4
 * Só guarda flashes até 100 km dos locais cadastrados (o arquivo não cresce).
 * Fora da área do GOES-East o módulo se desliga sozinho para aquele local.

Uso: feed.set_sites([(lat, lon), ...]);  feed.update();  feed.counts(lat, lon);  feed.hourly_counts(lat, lon)
"""
import json
import logging
import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from config import Config

logger = logging.getLogger(__name__)
STORE = 'dados/lightning/flashes.json'
KEEP_RADIUS_KM = 100.0
PRODUCT = 'GLM-L2-LCFA'
HEADERS = {'User-Agent': 'RRFS-Meteorologia/1.0'}
_KEY_RE = re.compile(r'_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_')
_LOCK = threading.RLock()


def covered(lat: float, lon: float) -> bool:
    """Área aproximada do GOES-East (GLM)."""
    return -135.0 <= lon <= -20.0 and abs(lat) <= 52.0


def _haversine_km(lat, lon, lat0, lon0):
    p1, p2 = np.radians(lat), math.radians(lat0)
    dphi, dl = p1 - p2, np.radians(lon - lon0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * math.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _key_time(key: str) -> Optional[float]:
    m = _KEY_RE.search(key)
    if not m:
        return None
    y, doy, hh, mm, ss, tenth = (int(x) for x in m.groups())
    dt = datetime(y, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss + tenth / 10)
    return dt.timestamp()


def _read_flashes(content: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bytes de um arquivo GLM-L2-LCFA (.nc) -> (lat, lon, energia) dos flashes."""
    from netCDF4 import Dataset
    ds = Dataset('glm.nc', mode='r', memory=content)
    try:
        def get(name):
            v = ds.variables.get(name)
            return np.ma.filled(v[:].astype('float64'), np.nan) if v is not None else np.array([])
        return get('flash_lat'), get('flash_lon'), get('flash_energy')
    finally:
        ds.close()


class LightningFeed:
    def __init__(self):
        self.sites: List[Tuple[float, float]] = []
        self.state = {'cursor': None, 'flashes': [], 'covered': []}   # flashes: [t, lat, lon, energia]
        self.status = {'enabled': bool(Config.LIGHTNING_ENABLED), 'last_ok': None, 'last_error': None,
                       'files_total': 0, 'bucket': None, 'reason': None}
        self._load()
        if self.status['enabled']:
            try:
                import netCDF4  # noqa: F401
            except ImportError:
                self.status.update(enabled=False, reason='instale netCDF4 (pip install netCDF4)')
                logger.warning('Raios desligados: pip install netCDF4 para ativar o GLM')

    # ------------------------------------------------------------ persistência
    def _load(self):
        try:
            with open(STORE, 'r') as f:
                self.state.update(json.load(f))
        except Exception:
            pass

    def _save(self):
        try:
            os.makedirs(os.path.dirname(STORE), exist_ok=True)
            tmp = f'{STORE}.{os.getpid()}.{threading.get_ident()}.tmp'
            with open(tmp, 'w') as f:
                json.dump(self.state, f)
            os.replace(tmp, STORE)
        except Exception as e:
            logger.error(f'Erro ao salvar raios: {e}')

    def _cover(self, start: float, end: float):
        """Registra o intervalo em que o feed realmente funcionou (p/ não confundir 'sem dado' com 'sem raio')."""
        if end <= start:
            return
        cov = self.state.setdefault('covered', [])
        if cov and start <= cov[-1][1] + 60:
            cov[-1][1] = max(cov[-1][1], end)
        else:
            cov.append([start, end])
        cut = time.time() - 72 * 3600
        self.state['covered'] = [c for c in cov if c[1] >= cut][-200:]

    def set_sites(self, sites: List[Tuple[float, float]]):
        self.sites = [(float(a), float(b)) for a, b in sites if covered(float(a), float(b))]

    # ------------------------------------------------------------ download
    def _list_keys(self, bucket: str, when: datetime) -> List[str]:
        prefix = f"{PRODUCT}/{when:%Y}/{when.timetuple().tm_yday:03d}/{when:%H}/"
        r = requests.get(f'https://{bucket}.s3.amazonaws.com/', timeout=15, headers=HEADERS,
                         params={'list-type': 2, 'prefix': prefix, 'max-keys': 1000})
        r.raise_for_status()
        return re.findall(r'<Key>([^<]+)</Key>', r.text)

    def _download(self, bucket: str, key: str):
        try:
            r = requests.get(f'https://{bucket}.s3.amazonaws.com/{key}', timeout=30, headers=HEADERS)
            r.raise_for_status()
            return _key_time(key), r.content          # só baixa; netCDF4/HDF5 NÃO é thread-safe: lê em série
        except Exception as e:
            logger.debug(f'GLM {key}: {e}')
            return None

    def update(self, max_files: int = 200) -> Dict:
        """Baixa os arquivos novos desde a última chamada e guarda os flashes perto dos locais."""
        if not self.status['enabled'] or not self.sites:
            return dict(self.status)
        with _LOCK:
            now = datetime.now(timezone.utc)
            cursor = self.state.get('cursor') or (now - timedelta(minutes=40)).timestamp()
            cursor = max(cursor, (now - timedelta(minutes=90)).timestamp())
            last_err = None
            for bucket in Config.GLM_BUCKETS:
                try:
                    hours = {(now - timedelta(hours=h)).replace(minute=0, second=0, microsecond=0) for h in (0, 1, 2)}
                    keys = []
                    for h in sorted(hours):
                        keys += self._list_keys(bucket, h)
                    new = sorted((t, k) for k in keys if (t := _key_time(k)) and t > cursor and t <= now.timestamp())
                    new = new[-max_files:]
                    if not keys:
                        last_err = f'{bucket}: listagem vazia'
                        continue
                    got = 0
                    if new:
                        with ThreadPoolExecutor(max_workers=6) as ex:
                            results = list(ex.map(lambda tk: self._download(bucket, tk[1]), new))
                        rows = []
                        for res in results:
                            if not res or res[0] is None:
                                continue
                            try:
                                lat, lon, en = _read_flashes(res[1])
                            except Exception as e:
                                logger.debug(f'GLM ilegível: {e}')
                                continue
                            t = res[0]
                            if len(lat) == 0:
                                continue
                            keep = np.zeros(len(lat), dtype=bool)
                            for la0, lo0 in self.sites:
                                keep |= _haversine_km(lat, lon, la0, lo0) <= KEEP_RADIUS_KM
                            for i in np.where(keep)[0]:
                                rows.append([round(t, 1), round(float(lat[i]), 3), round(float(lon[i]), 3),
                                             round(float(en[i]) * 1e15, 2) if np.isfinite(en[i]) else None])
                        got = len(new)
                        self.state['flashes'].extend(rows)
                        self.state['cursor'] = new[-1][0]
                    elif self.state.get('cursor') is None:
                        self.state['cursor'] = now.timestamp()
                    self._cover(cursor, now.timestamp() - 180)
                    cut = (now - timedelta(hours=Config.LIGHTNING_KEEP_HOURS)).timestamp()
                    self.state['flashes'] = [f for f in self.state['flashes'] if f[0] >= cut][-60000:]
                    self._save()
                    self.status.update(last_ok=now.isoformat(), last_error=None, bucket=bucket,
                                       files_total=self.status['files_total'] + got)
                    return dict(self.status)
                except Exception as e:
                    last_err = f'{bucket}: {e}'
                    logger.warning(f'GLM indisponível ({last_err})')
            self.status['last_error'] = last_err
            return dict(self.status)

    # ------------------------------------------------------------ consultas
    def _fresh(self) -> bool:
        ok = self.status.get('last_ok')
        return bool(ok) and (datetime.now(timezone.utc) - datetime.fromisoformat(ok)).total_seconds() < 1800

    def counts(self, lat: float, lon: float, minutes: int = 30) -> Dict:
        """Raios nos últimos `minutes` ao redor do local (10 / 25 / 30 / 60 km)."""
        if not (self.status['enabled'] and covered(lat, lon) and self._fresh()):
            return {'available': False}
        cut = time.time() - minutes * 60
        fl = [f for f in self.state['flashes'] if f[0] >= cut]
        if not fl:
            return {'available': True, 'n10': 0, 'n25': 0, 'n30': 0, 'n60': 0, 'nearest_km': None}
        a = np.array([[f[1], f[2]] for f in fl])
        d = _haversine_km(a[:, 0], a[:, 1], lat, lon)
        return {'available': True, 'n10': int((d <= 10).sum()), 'n25': int((d <= 25).sum()),
                'n30': int((d <= 30).sum()), 'n60': int((d <= 60).sum()), 'nearest_km': round(float(d.min()), 1)}

    def hourly_counts(self, lat: float, lon: float) -> Dict[str, Dict]:
        """'YYYY-MM-DDTHH:00' (UTC) -> flashes a <=10 km e <=30 km, SÓ para horas totalmente cobertas pelo feed
        (verdade p/ conferir previsões de raios: hora sem cobertura não vira "não houve raio")."""
        if not (self.status['enabled'] and covered(lat, lon)):
            return {}
        out: Dict[str, Dict] = {}
        for a, b in self.state.get('covered', []):
            h = datetime.fromtimestamp(a, timezone.utc).replace(minute=0, second=0, microsecond=0)
            if h.timestamp() < a:
                h += timedelta(hours=1)
            while h.timestamp() + 3600 <= b:
                out[h.strftime('%Y-%m-%dT%H:00')] = {'flashes_local': 0, 'flashes_near': 0, 'glm_ok': True}
                h += timedelta(hours=1)
        for t, la, lo, _e in self.state['flashes']:
            d = float(_haversine_km(np.array([la]), np.array([lo]), lat, lon)[0])
            if d > 30:
                continue
            o = out.get(datetime.fromtimestamp(t, timezone.utc).strftime('%Y-%m-%dT%H:00'))
            if o is not None:
                o['flashes_near'] += 1
                if d <= 10:
                    o['flashes_local'] += 1
        return out

    def coverage_ok_until(self) -> Optional[float]:
        """Instante (epoch) até o qual temos dado de raios contínuo (p/ não confundir 'sem dado' com 'sem raio')."""
        ok = self.status.get('last_ok')
        return datetime.fromisoformat(ok).timestamp() if ok else None


feed = LightningFeed()

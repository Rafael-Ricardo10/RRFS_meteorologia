"""
Imagens de radar via RainViewer (gratuito, sem chave).
Solução para zoom: múltiplas resoluções e fallback inteligente.
"""
import requests
import logging
from datetime import datetime
from modulos.cache_manager import cache

logger = logging.getLogger('RRFS.Radar')

RAINVIEWER_API = 'https://api.rainviewer.com/public/weather-maps.json'

# RainViewer suporta tileSize 256 e 512
# zoom máximo REAL por tileSize:
# tileSize=512 → maxNativeZoom=6
# tileSize=256 → maxNativeZoom=8
RAINVIEWER_ZOOM_CONFIG = {
    'low':    {'tileSize': 512, 'maxNativeZoom': 6,  'zoomOffset': -1},
    'medium': {'tileSize': 256, 'maxNativeZoom': 8,  'zoomOffset':  0},
    'high':   {'tileSize': 256, 'maxNativeZoom': 8,  'zoomOffset':  0},
}


def buscar_frames_radar() -> dict | None:
    """Busca lista de frames de radar disponíveis."""
    cached = cache.get('radar', tipo='frames')
    if cached:
        return cached
    try:
        r = requests.get(RAINVIEWER_API, timeout=10)
        r.raise_for_status()
        dados = r.json()
        cache.set('radar', dados, tipo='frames')
        return dados
    except Exception as e:
        logger.error(f'RainViewer erro: {e}')
        return None


def obter_url_radar(lat: float, lon: float, zoom: int = 8,
                    tamanho: int = 512, cor: int = 4,
                    suavizar: int = 1, snow: int = 1) -> dict:
    """
    Retorna URLs das últimas 2h de radar e próximas 30min (nowcast).

    SOLUÇÃO ZOOM:
    - Fornece template de URL com {z}/{x}/{y} para Leaflet
    - Fornece config de zoom para cada nível
    - Leaflet gerencia os tiles automaticamente em qualquer zoom
    """
    frames = buscar_frames_radar()
    if not frames:
        return {}

    host       = frames.get('host', 'https://tilecache.rainviewer.com')
    radar_data = frames.get('radar', {})

    def url_template(path, tsize=512):
        """
        Template de URL para Leaflet — {z}/{x}/{y} são substituídos pelo Leaflet.
        tsize=512 com zoomOffset=-1 → zoom real do Leaflet 7 vira zoom 6 da API
        tsize=256 sem offset → zoom real = zoom da API
        """
        return (
            f"{host}{path}/{tsize}"
            f"/{{z}}/{{x}}/{{y}}/{cor}/{suavizar}_{snow}.png"
        )

    passado_raw = radar_data.get('past', [])
    nowcast_raw = radar_data.get('nowcast', [])

    # Frames com template 512px (zoom baixo/médio) e 256px (zoom alto)
    frames_512 = [url_template(f['path'], 512) for f in passado_raw]
    frames_256 = [url_template(f['path'], 256) for f in passado_raw]
    nowcast_512= [url_template(f['path'], 512) for f in nowcast_raw]
    nowcast_256= [url_template(f['path'], 256) for f in nowcast_raw]

    # Timestamps para label na interface
    timestamps = []
    for f in passado_raw:
        try:
            ts = datetime.utcfromtimestamp(f['time'])
            timestamps.append(ts.strftime('%H:%M'))
        except Exception:
            timestamps.append('--:--')

    timestamps_nowcast = []
    for f in nowcast_raw:
        try:
            ts = datetime.utcfromtimestamp(f['time'])
            timestamps_nowcast.append(ts.strftime('%H:%M'))
        except Exception:
            timestamps_nowcast.append('--:--')

    return {
        # Templates 512px — usar em zoom <= 7 (melhor performance)
        'passado':          frames_512,
        'nowcast':          nowcast_512,
        # Templates 256px — usar em zoom >= 8 (melhor resolução)
        'passado_256':      frames_256,
        'nowcast_256':      nowcast_256,
        # Timestamps legíveis
        'timestamps':       timestamps,
        'timestamps_nowcast': timestamps_nowcast,
        # Config de zoom para Leaflet
        'zoom_config': {
            # zoom Leaflet <= 7: tiles 512px, maxNativeZoom=6, zoomOffset=-1
            'low':  {
                'tileSize': 512, 'maxNativeZoom': 6,
                'zoomOffset': -1, 'minZoom': 1, 'maxZoom': 7,
            },
            # zoom Leaflet 8-18: tiles 256px, maxNativeZoom=8, zoomOffset=0
            'high': {
                'tileSize': 256, 'maxNativeZoom': 8,
                'zoomOffset': 0, 'minZoom': 8, 'maxZoom': 18,
            },
        },
        'ultimo':      frames_512[-1] if frames_512 else None,
        'host':        host,
        'atualizado':  datetime.now().isoformat(),
        'n_passado':   len(passado_raw),
        'n_nowcast':   len(nowcast_raw),
        'frames_raw':  radar_data,
    }


def url_radar_estatica(lat: float, lon: float, zoom: int = 8) -> str | None:
    """URL de imagem estática do radar para posição do usuário."""
    dados = buscar_frames_radar()
    if not dados:
        return None
    past = dados.get('radar', {}).get('past', [])
    if not past:
        return None
    return (
        f"https://www.rainviewer.com/map.html?"
        f"loc={lat},{lon},{zoom}&oFa=0&oC=1&oU=0&oCS=1&oF=0&oAP=1"
        f"&c=3&o=83&lm=1&layer=radar&sm=1&sn=1"
    )

"""
Imagens de radar via RainViewer (gratuito, sem chave).
"""
import requests
import logging
from datetime import datetime
from modulos.cache_manager import cache

logger = logging.getLogger('RRFS.Radar')

RAINVIEWER_API = 'https://api.rainviewer.com/public/weather-maps.json'


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
    Tiles: https://tilecache.rainviewer.com/v2/radar/{timestamp}/{size}/{z}/{x}/{y}/{color}/{options}.png
    """
    frames = buscar_frames_radar()
    if not frames:
        return {}

    host = frames.get('host', 'https://tilecache.rainviewer.com')
    radar_data = frames.get('radar', {})

    def url_tile(path):
        return (
            f"{host}{path}/{tamanho}"
            f"/{{z}}/{{x}}/{{y}}/{cor}/{suavizar}_{snow}.png"
        )

    passado    = [url_tile(f['path']) for f in radar_data.get('past', [])]
    nowcast    = [url_tile(f['path']) for f in radar_data.get('nowcast', [])]
    ultimo     = passado[-1] if passado else None

    return {
        'passado':  passado,
        'nowcast':  nowcast,
        'ultimo':   ultimo,
        'host':     host,
        'atualizado': datetime.now().isoformat(),
        'frames_raw': radar_data,
    }


def url_radar_estatica(lat: float, lon: float, zoom: int = 8) -> str | None:
    """URL de imagem estática do radar para posição do usuário."""
    dados = buscar_frames_radar()
    if not dados:
        return None
    host = dados.get('host', 'https://tilecache.rainviewer.com')
    past = dados.get('radar', {}).get('past', [])
    if not past:
        return None
    ultimo_path = past[-1]['path']
    # Imagem do último frame centrada na localização
    return (
        f"https://www.rainviewer.com/map.html?"
        f"loc={lat},{lon},{zoom}&oFa=0&oC=1&oU=0&oCS=1&oF=0&oAP=1"
        f"&c=3&o=83&lm=1&layer=radar&sm=1&sn=1"
    )

"""
Geocoding: cidade → coordenadas e coordenadas → cidade
Usa Nominatim (OpenStreetMap) - 100% gratuito.
"""
import requests
import logging
from modulos.cache_manager import cache

logger = logging.getLogger('RRFS.Geocoding')

NOMINATIM = 'https://nominatim.openstreetmap.org'
HEADERS    = {'User-Agent': 'RRFS-Meteorologia/1.0'}


def cidade_para_coords(cidade: str, estado: str = '', pais: str = '') -> dict | None:
    """Retorna {'lat', 'lon', 'display_name', 'cidade', 'estado', 'pais'}"""
    query = ', '.join(filter(None, [cidade, estado, pais]))
    cached = cache.get('geocoding', query=query)
    if cached:
        return cached

    try:
        r = requests.get(
            f'{NOMINATIM}/search',
            params={'q': query, 'format': 'json', 'limit': 1,
                    'addressdetails': 1},
            headers=HEADERS, timeout=10
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        item = data[0]
        addr = item.get('address', {})
        resultado = {
            'lat':          float(item['lat']),
            'lon':          float(item['lon']),
            'display_name': item['display_name'],
            'cidade':       addr.get('city') or addr.get('town') or
                            addr.get('village') or addr.get('municipality', cidade),
            'estado':       addr.get('state', estado),
            'pais':         addr.get('country', pais),
            'pais_codigo':  addr.get('country_code', '').upper(),
        }
        cache.set('geocoding', resultado, query=query)
        return resultado
    except Exception as e:
        logger.error(f'Geocoding erro: {e}')
        return None


def coords_para_cidade(lat: float, lon: float) -> dict | None:
    """Reverse geocoding."""
    cached = cache.get('geocoding', lat=round(lat, 4), lon=round(lon, 4))
    if cached:
        return cached

    try:
        r = requests.get(
            f'{NOMINATIM}/reverse',
            params={'lat': lat, 'lon': lon, 'format': 'json'},
            headers=HEADERS, timeout=10
        )
        r.raise_for_status()
        item = r.json()
        addr = item.get('address', {})
        resultado = {
            'lat':          lat,
            'lon':          lon,
            'display_name': item.get('display_name', ''),
            'cidade':       addr.get('city') or addr.get('town') or
                            addr.get('village') or addr.get('municipality', ''),
            'estado':       addr.get('state', ''),
            'pais':         addr.get('country', ''),
            'pais_codigo':  addr.get('country_code', '').upper(),
        }
        cache.set('geocoding', resultado,
                  lat=round(lat, 4), lon=round(lon, 4))
        return resultado
    except Exception as e:
        logger.error(f'Reverse geocoding erro: {e}')
        return None


def buscar_sugestoes(query: str) -> list:
    """Autocomplete para o campo de busca."""
    if len(query) < 3:
        return []
    cached = cache.get('geocoding', sugestao=query)
    if cached:
        return cached
    try:
        r = requests.get(
            f'{NOMINATIM}/search',
            params={'q': query, 'format': 'json', 'limit': 5,
                    'addressdetails': 1},
            headers=HEADERS, timeout=8
        )
        r.raise_for_status()
        data = r.json()
        sugestoes = []
        for item in data:
            addr = item.get('address', {})
            sugestoes.append({
                'lat':  float(item['lat']),
                'lon':  float(item['lon']),
                'nome': item['display_name'],
                'cidade':  addr.get('city') or addr.get('town') or
                           addr.get('village', ''),
                'estado':  addr.get('state', ''),
                'pais':    addr.get('country', ''),
            })
        cache.set('geocoding', sugestoes, ttl=3600, sugestao=query)
        return sugestoes
    except Exception as e:
        logger.error(f'Sugestões erro: {e}')
        return []

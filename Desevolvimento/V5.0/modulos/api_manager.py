import requests
import asyncio
import aiohttp
import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from config import Config

logger = logging.getLogger(__name__)

WEATHER_CODE_MAP = {
    0: ('Céu limpo', '☀️'), 1: ('Poucas nuvens', '🌤️'), 2: ('Parcialmente nublado', '⛅'),
    3: ('Nublado', '☁️'), 45: ('Nevoeiro', '🌫️'), 48: ('Nevoeiro com geada', '🌫️'),
    51: ('Garoa fraca', '🌦️'), 53: ('Garoa moderada', '🌦️'), 55: ('Garoa forte', '🌧️'),
    56: ('Garoa congelante fraca', '🌧️'), 57: ('Garoa congelante forte', '🌧️'),
    61: ('Chuva fraca', '🌦️'), 63: ('Chuva moderada', '🌧️'), 65: ('Chuva forte', '🌧️'),
    66: ('Chuva congelante fraca', '🌧️'), 67: ('Chuva congelante forte', '🌧️'),
    71: ('Neve fraca', '🌨️'), 73: ('Neve moderada', '🌨️'), 75: ('Neve forte', '❄️'),
    77: ('Grãos de neve', '❄️'), 80: ('Pancadas de chuva fracas', '🌦️'),
    81: ('Pancadas de chuva moderadas', '🌧️'), 82: ('Pancadas de chuva violentas', '⛈️'),
    85: ('Pancadas de neve fracas', '🌨️'), 86: ('Pancadas de neve fortes', '❄️'),
    95: ('Trovoada', '⛈️'), 96: ('Trovoada com granizo fraco', '⛈️'),
    99: ('Trovoada com granizo forte', '⛈️'),
}


def get_weather_description(code):
    if code is None:
        return ('Sem dados', '❓')
    try:
        return WEATHER_CODE_MAP.get(int(code), ('Desconhecido', '❓'))
    except (ValueError, TypeError):
        return ('Desconhecido', '❓')


# Todas as variáveis horárias úteis (o que o modelo não fornecer volta null e é ignorado)
HOURLY_FULL = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
    'precipitation', 'precipitation_probability', 'weather_code',
    'pressure_msl', 'surface_pressure', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid',
    'cloud_cover_high', 'visibility',
    'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation', 'direct_normal_irradiance',
    'uv_index', 'sunshine_duration',
    'cape', 'lifted_index', 'convective_inhibition', 'freezing_level_height', 'lightning_potential',
]
# Fallback 1: se algum nome for recusado (400), tenta o conjunto intermediário
HOURLY_MID = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature',
    'precipitation', 'precipitation_probability', 'weather_code', 'pressure_msl', 'cloud_cover',
    'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m',
    'shortwave_radiation', 'uv_index', 'cape', 'lifted_index',
]
# Fallback 2: conjunto básico
HOURLY_CORE = [
    'temperature_2m', 'relative_humidity_2m', 'precipitation', 'precipitation_probability',
    'weather_code', 'wind_speed_10m', 'wind_gusts_10m', 'pressure_msl', 'cape',
]
# Altitude (níveis de pressão): base do cisalhamento do vento e do gradiente térmico, que os
# meteorologistas usam para dizer se a tempestade será isolada, organizada ou supercélula.
# Pedido em uma requisição SEPARADA: se o modelo não tiver esses campos, o resto não é afetado.
UPPER_VARS = [
    'wind_speed_500hPa', 'wind_direction_500hPa', 'wind_speed_850hPa', 'wind_direction_850hPa',
    'wind_speed_925hPa', 'wind_direction_925hPa',
    'temperature_850hPa', 'temperature_700hPa', 'temperature_500hPa',
    'relative_humidity_850hPa', 'relative_humidity_700hPa',
]
DAILY_VARS = [
    'weather_code', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
    'precipitation_hours', 'precipitation_probability_max',
    'wind_speed_10m_max', 'wind_gusts_10m_max',
]


class APIManager:
    def __init__(self):
        self.openmeteo_url = Config.OPENMETEO_API
        self.cache = {}
        self.nominatim_headers = {'User-Agent': 'RRFS-Meteorologia/1.0 (contato@rrfs.local)'}

    # ------------------------------------------------------------------ HTTP
    async def _get_json(self, session, url, params, label, retries=2) -> Tuple[int, dict]:
        for attempt in range(retries + 1):
            try:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        return 200, await resp.json()
                    body = (await resp.text())[:200]
                    if resp.status in (429, 500, 502, 503, 504) and attempt < retries:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    logger.warning(f"{label}: HTTP {resp.status} - {body}")
                    return resp.status, None
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt < retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                logger.error(f"{label}: falha de rede ({e})")
                return 0, None
        return 0, None

    # ------------------------------------------------------- OPEN-METEO
    async def _fetch_one_model(self, session, sem, model, latitude, longitude):
        url = f"{self.openmeteo_url}/forecast"
        days = int(getattr(Config, 'FORECAST_DAYS', 16))      # modelos mais curtos devolvem null depois do alcance deles
        base = {
            'latitude': latitude, 'longitude': longitude, 'models': model,
            'daily': ','.join(DAILY_VARS), 'timezone': 'auto', 'forecast_days': days,
        }
        async with sem:
            status, data = 0, None
            for tier, hourly_vars in (('full', HOURLY_FULL), ('mid', HOURLY_MID), ('core', HOURLY_CORE)):
                status, data = await self._get_json(
                    session, url, {**base, 'hourly': ','.join(hourly_vars)},
                    f"Open-Meteo[{model}/{tier}]")
                if status != 400:      # 400 = variável não suportada -> tenta o conjunto menor
                    break
            if status == 400:          # último recurso: janela de 7 dias, conjunto básico
                status, data = await self._get_json(
                    session, url, {**base, 'forecast_days': 7, 'hourly': ','.join(HOURLY_CORE)},
                    f"Open-Meteo[{model}/core7]")

            if status == 200 and data and 'hourly' in data:       # altitude (opcional)
                ustatus, udata = await self._get_json(
                    session, url, {**{k: v for k, v in base.items() if k != 'daily'},
                                   'hourly': ','.join(UPPER_VARS)}, f"Open-Meteo[{model}/altitude]", retries=1)
                if ustatus == 200 and udata and 'hourly' in udata:
                    h, uh = data['hourly'], udata['hourly']
                    if uh.get('time') == h.get('time'):
                        for k, v in uh.items():
                            if k != 'time' and any(x is not None for x in v):
                                h[k] = v

        if status != 200 or not data or 'hourly' not in data:
            return model, None
        temps = data['hourly'].get('temperature_2m') or []
        if not any(t is not None for t in temps):
            logger.warning(f"Modelo {model} sem dados para esta localização - descartado")
            return model, None
        return model, data

    async def fetch_openmeteo_data(self, latitude: float, longitude: float,
                                   models: List[str] = None) -> Dict:
        """Busca em paralelo TODOS os modelos globais configurados."""
        models = models or Config.CLIMATE_MODELS
        sem = asyncio.Semaphore(5)
        timeout = aiohttp.ClientTimeout(total=30)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                results = await asyncio.gather(
                    *[self._fetch_one_model(session, sem, m, latitude, longitude) for m in models])
            data = {m: d for m, d in results if d}
            failed = [m for m, d in results if not d]
            logger.info(f"Modelos obtidos ({len(data)}/{len(models)}): {list(data)}"
                        + (f" | falharam: {failed}" if failed else ""))
            return data
        except Exception as e:
            logger.error(f"Erro na requisição Open-Meteo: {str(e)}")
            return {}

    async def fetch_detailed_forecast(self, latitude: float, longitude: float) -> Dict:
        """Previsão horária+diária de 16 dias (best_match), usada p/ estender além dos 7 dias."""
        params = {
            'latitude': latitude, 'longitude': longitude,
            'hourly': ','.join(['temperature_2m', 'precipitation_probability', 'precipitation',
                                'weather_code', 'wind_speed_10m', 'relative_humidity_2m',
                                'wind_gusts_10m', 'cape']),
            'daily': ','.join(DAILY_VARS),
            'timezone': 'auto', 'forecast_days': 16,
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                status, data = await self._get_json(
                    session, f"{self.openmeteo_url}/forecast", params, "Open-Meteo[best_match 16d]")
            return data if status == 200 and data else {}
        except Exception as e:
            logger.error(f"Erro na previsão detalhada: {str(e)}")
            return {}

    async def fetch_observations(self, latitude: float, longitude: float, past_days: int = 4) -> Dict:
        """
        Valores 'observados' das últimas horas (análise do Open-Meteo, em UTC), usados
        como verdade-terreno para o auto-aprendizado. A análise das horas que acabaram de
        passar é a "previsão de agora" (muito precisa) e serve de validação.
        Retorna {'YYYY-MM-DDTHH:00': {variável: valor, ..., 'wind_gusts', 'weather_code'}}.
        """
        cols = {v['col']: k for k, v in Config.VARIABLES.items()}
        cols['weather_code'] = 'weather_code'
        params = {
            'latitude': latitude, 'longitude': longitude,
            'hourly': ','.join(cols), 'past_days': past_days, 'forecast_days': 1, 'timezone': 'GMT',
        }
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                status, data = await self._get_json(
                    session, f"{self.openmeteo_url}/forecast", params, "Open-Meteo[observações]")
            if status != 200 or not data:
                return {}
            h = data.get('hourly', {})
            times = h.get('time', [])
            now_key = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:00')
            out = {}
            for i, t in enumerate(times):
                key = t[:13] + ':00'
                if key > now_key:
                    continue
                row = {}
                for col, name in cols.items():
                    series = h.get(col)
                    row[name] = series[i] if series and i < len(series) else None
                row['wind_gusts'] = row.get('wind_gusts')      # compatível c/ _event_outcomes
                out[key] = row
            return out
        except Exception as e:
            logger.error(f"Erro ao buscar observações: {str(e)}")
            return {}

    # --------------------------------------------------------- RAINVIEWER
    async def fetch_rainviewer_data(self, latitude: float = None, longitude: float = None) -> Dict:
        """Metadados globais (host + frames). A localização só é usada ao montar os tiles."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                status, data = await self._get_json(
                    session, "https://api.rainviewer.com/public/weather-maps.json", None, "RainViewer")
            return data if status == 200 and data else {}
        except Exception as e:
            logger.error(f"Erro RainViewer: {str(e)}")
            return {}

    @staticmethod
    def _deg2tile_float(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[float, float]:
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xf = (lon_deg + 180.0) / 360.0 * n
        yf = (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n
        return xf, yf

    def get_radar_tile_grid(self, latitude: float, longitude: float,
                            radar_meta: Dict, zoom: int = 7, grid_size: int = 3) -> Dict:
        """Grade de tiles (mapa base CARTO + radar RainViewer) com marcador na posição exata."""
        try:
            host = radar_meta.get('host')
            frames = (radar_meta.get('radar') or {}).get('past', [])
            if not host or not frames:
                logger.warning("Metadados de radar incompletos ou vazios")
                return {}

            last = frames[-1]
            frame_path, frame_time = last.get('path'), last.get('time')

            xf, yf = self._deg2tile_float(latitude, longitude, zoom)
            cx, cy = int(xf), int(yf)
            half = grid_size // 2
            n = 2 ** zoom
            tile_size = 256

            tiles = []
            for row_off in range(-half, half + 1):
                for col_off in range(-half, half + 1):
                    x = (cx + col_off) % n
                    y = cy + row_off
                    if y < 0 or y >= n:
                        continue
                    base_url = (Config.BASEMAP_TILE_URL
                                .replace('{s}', 'abcd'[(x + y) % 4])
                                .replace('{z}', str(zoom)).replace('{x}', str(x)).replace('{y}', str(y)))
                    tiles.append({
                        'row': row_off + half, 'col': col_off + half,
                        'base_url': base_url,
                        'radar_url': f"{host}{frame_path}/{tile_size}/{zoom}/{x}/{y}/2/1_1.png"
                    })

            # posição exata do ponto dentro da grade (em pixels)
            marker = {
                'x': round((xf - (cx - half)) * tile_size, 1),
                'y': round((yf - (cy - half)) * tile_size, 1),
            }
            frame_iso = (datetime.fromtimestamp(frame_time, tz=timezone.utc).isoformat()
                         if frame_time else None)

            return {
                'tiles': tiles, 'grid_size': grid_size, 'zoom': zoom, 'tile_size': tile_size,
                'marker': marker, 'frame_time': frame_time, 'frame_time_iso': frame_iso,
                'attribution': Config.BASEMAP_ATTRIBUTION,
            }
        except Exception as e:
            logger.error(f"Erro ao montar grade de radar: {str(e)}")
            return {}

    async def get_all_data(self, latitude: float, longitude: float) -> Dict:
        try:
            climate, radar = await asyncio.gather(
                self.fetch_openmeteo_data(latitude, longitude),
                self.fetch_rainviewer_data(latitude, longitude))
            return {'climate_models': climate, 'radar': radar,
                    'timestamp': datetime.now(timezone.utc).isoformat()}
        except Exception as e:
            logger.error(f"Erro ao obter todos os dados: {str(e)}")
            return {}

    # ---------------------------------------------------------- GEOCODING
    def reverse_geocode(self, latitude: float, longitude: float) -> Dict:
        try:
            response = requests.get("https://nominatim.openstreetmap.org/reverse",
                                    params={'lat': latitude, 'lon': longitude, 'format': 'json'},
                                    headers=self.nominatim_headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {'city': data.get('address', {}).get('city', 'Unknown'),
                        'country': data.get('address', {}).get('country', 'Unknown'),
                        'address': data.get('address', {})}
            logger.error(f"Erro reverse geocode: status {response.status_code}")
        except Exception as e:
            logger.error(f"Erro no reverse geocode: {str(e)}")
        return {}

    def geocode_city(self, city_name: str) -> List[Dict]:
        try:
            if not city_name or len(city_name.strip()) < 2:
                return []
            response = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={'q': city_name.strip(), 'format': 'json', 'limit': 6,
                        'addressdetails': 1, 'accept-language': 'pt-BR'},
                headers=self.nominatim_headers, timeout=10)
            logger.info(f"Geocode '{city_name}' -> status {response.status_code}")

            if response.status_code == 200:
                results = []
                for item in response.json():
                    address = item.get('address', {})
                    city = (address.get('city') or address.get('town') or address.get('village')
                            or address.get('municipality') or item.get('name')
                            or item.get('display_name', '').split(',')[0])
                    results.append({
                        'city': city, 'latitude': float(item.get('lat')),
                        'longitude': float(item.get('lon')),
                        'display_name': item.get('display_name'),
                        'country': address.get('country', ''), 'state': address.get('state', '')
                    })
                return results
            if response.status_code == 429:
                logger.warning("Rate limit atingido no Nominatim")
            else:
                logger.error(f"Erro no geocode: status {response.status_code}")
            return []
        except requests.exceptions.Timeout:
            logger.error("Timeout ao consultar Nominatim")
            return []
        except Exception as e:
            logger.error(f"Erro no geocode: {str(e)}")
            return []

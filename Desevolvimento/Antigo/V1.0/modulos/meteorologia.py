"""
Dados meteorológicos via Open-Meteo (gratuito, sem chave).
Tempo atual + previsão 7 dias + hourly detalhado.
"""
import requests
import logging
from datetime import datetime
from modulos.cache_manager import cache

logger = logging.getLogger('RRFS.Meteo')

BASE = 'https://api.open-meteo.com/v1'

# Variáveis horárias completas
HOURLY_VARS = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'apparent_temperature', 'precipitation_probability',
    'precipitation', 'rain', 'showers', 'snowfall', 'snow_depth',
    'weather_code', 'cloud_cover', 'cloud_cover_low',
    'cloud_cover_mid', 'cloud_cover_high',
    'visibility', 'wind_speed_10m', 'wind_speed_80m',
    'wind_direction_10m', 'wind_gusts_10m',
    'cape', 'lifted_index', 'convective_inhibition',
    'freezing_level_height', 'soil_temperature_0cm',
    'uv_index', 'is_day',
]

CURRENT_VARS = [
    'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
    'is_day', 'precipitation', 'rain', 'showers', 'snowfall',
    'weather_code', 'cloud_cover', 'wind_speed_10m',
    'wind_direction_10m', 'wind_gusts_10m', 'cape',
]

DAILY_VARS = [
    'weather_code', 'temperature_2m_max', 'temperature_2m_min',
    'apparent_temperature_max', 'apparent_temperature_min',
    'sunrise', 'sunset', 'uv_index_max',
    'precipitation_sum', 'rain_sum', 'showers_sum', 'snowfall_sum',
    'precipitation_hours', 'precipitation_probability_max',
    'wind_speed_10m_max', 'wind_gusts_10m_max', 'wind_direction_10m_dominant',
    'shortwave_radiation_sum',
]


def buscar_tempo_completo(lat: float, lon: float) -> dict | None:
    """Busca tempo atual + previsão horária + diária."""
    key_kw = dict(lat=round(lat, 3), lon=round(lon, 3))
    cached = cache.get('tempo_atual', **key_kw)
    if cached:
        return cached

    try:
        r = requests.get(
            f'{BASE}/forecast',
            params={
                'latitude':        lat,
                'longitude':       lon,
                'current':         ','.join(CURRENT_VARS),
                'hourly':          ','.join(HOURLY_VARS),
                'daily':           ','.join(DAILY_VARS),
                'timezone':        'auto',
                'forecast_days':   7,
                'wind_speed_unit': 'kmh',
            },
            timeout=15
        )
        r.raise_for_status()
        dados = r.json()
        dados['_atualizado'] = datetime.now().isoformat()
        cache.set('tempo_atual', dados, **key_kw)
        return dados
    except Exception as e:
        logger.error(f'Open-Meteo erro: {e}')
        return None


def buscar_ensemble(lat: float, lon: float) -> dict | None:
    """
    Busca 3 modelos numéricos via Open-Meteo Ensemble.
    Modelos: ICON, GFS, ECMWF.
    """
    key_kw = dict(lat=round(lat, 3), lon=round(lon, 3))
    cached = cache.get('modelos', **key_kw)
    if cached:
        return cached

    modelos_dados = {}
    modelos = ['icon_seamless', 'gfs_seamless', 'ecmwf_ifs025']

    for modelo in modelos:
        try:
            r = requests.get(
                f'{BASE}/forecast',
                params={
                    'latitude':      lat,
                    'longitude':     lon,
                    'hourly':        'temperature_2m,precipitation,wind_speed_10m,'
                                     'wind_gusts_10m,cape,precipitation_probability',
                    'models':        modelo,
                    'timezone':      'auto',
                    'forecast_days': 7,
                    'wind_speed_unit': 'kmh',
                },
                timeout=15
            )
            if r.status_code == 200:
                modelos_dados[modelo] = r.json()
        except Exception as e:
            logger.warning(f'Modelo {modelo} falhou: {e}')

    if not modelos_dados:
        return None

    resultado = {'modelos': modelos_dados, '_atualizado': datetime.now().isoformat()}
    cache.set('modelos', resultado, **key_kw)
    return resultado


def combinar_modelos(dados_ensemble: dict) -> dict:
    """Combina previsões dos 3 modelos (média ponderada)."""
    if not dados_ensemble:
        return {}

    modelos = dados_ensemble.get('modelos', {})
    if not modelos:
        return {}

    # Pesos: ECMWF=0.4, ICON=0.35, GFS=0.25
    pesos = {
        'ecmwf_ifs025': 0.40,
        'icon_seamless': 0.35,
        'gfs_seamless':  0.25,
    }

    vars_combinar = [
        'temperature_2m', 'precipitation', 'wind_speed_10m',
        'wind_gusts_10m', 'cape', 'precipitation_probability'
    ]

    combinado = {'times': [], 'vars': {v: [] for v in vars_combinar}}
    ref = next(iter(modelos.values()))

    if 'hourly' not in ref:
        return {}

    times = ref['hourly'].get('time', [])
    combinado['times'] = times

    for var in vars_combinar:
        n = len(times)
        soma_pesos = 0
        acumulado  = [0.0] * n

        for modelo, dados in modelos.items():
            peso  = pesos.get(modelo, 0.3)
            vals  = dados.get('hourly', {}).get(var, [None] * n)
            for i, v in enumerate(vals[:n]):
                if v is not None:
                    acumulado[i] += v * peso
            soma_pesos += peso

        if soma_pesos > 0:
            combinado['vars'][var] = [
                round(v / soma_pesos, 2) for v in acumulado
            ]

    return combinado


# Códigos WMO → descrição e ícone
WMO_CODES = {
    0:  ('Céu limpo',          '☀️'),
    1:  ('Predominantemente limpo', '🌤️'),
    2:  ('Parcialmente nublado','⛅'),
    3:  ('Nublado',            '☁️'),
    45: ('Nevoeiro',           '🌫️'),
    48: ('Nevoeiro com geada', '🌫️'),
    51: ('Garoa leve',         '🌦️'),
    53: ('Garoa moderada',     '🌦️'),
    55: ('Garoa intensa',      '🌧️'),
    61: ('Chuva leve',         '🌧️'),
    63: ('Chuva moderada',     '🌧️'),
    65: ('Chuva forte',        '🌧️'),
    71: ('Neve leve',          '🌨️'),
    73: ('Neve moderada',      '🌨️'),
    75: ('Neve intensa',       '❄️'),
    77: ('Grãos de neve',      '🌨️'),
    80: ('Pancadas leves',     '🌦️'),
    81: ('Pancadas moderadas', '🌧️'),
    82: ('Pancadas violentas', '⛈️'),
    85: ('Pancadas de neve',   '🌨️'),
    86: ('Pancadas de neve intensa','❄️'),
    95: ('Tempestade',         '⛈️'),
    96: ('Tempestade c/ granizo leve','⛈️'),
    99: ('Tempestade c/ granizo forte','⛈️'),
}


def decodificar_wmo(codigo: int) -> tuple:
    return WMO_CODES.get(codigo, ('Desconhecido', '❓'))


def hora_atual_indice(dados: dict) -> int:
    """Retorna o índice da hora atual nos dados horários."""
    try:
        times = dados['hourly']['time']
        agora = datetime.now().strftime('%Y-%m-%dT%H:00')
        for i, t in enumerate(times):
            if t == agora:
                return i
        return 0
    except Exception:
        return 0

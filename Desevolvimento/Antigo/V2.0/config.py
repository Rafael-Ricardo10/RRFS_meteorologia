import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Estrutura de pastas
PASTAS = {
    'dados':       BASE_DIR / 'dados',
    'cache':       BASE_DIR / 'dados' / 'cache',
    'radar':       BASE_DIR / 'dados' / 'radar',
    'usuarios':    BASE_DIR / 'dados' / 'usuarios',
    'alertas':     BASE_DIR / 'dados' / 'alertas',
    'calibracao':  BASE_DIR / 'dados' / 'calibracao',
    'logs':        BASE_DIR / 'dados' / 'logs',
}

for pasta in PASTAS.values():
    pasta.mkdir(parents=True, exist_ok=True)

# Flask
SECRET_KEY = os.environ.get('SECRET_KEY', 'rrfs_meteorologia_2024_chave_secreta')
HOST       = '0.0.0.0'
PORT       = 6800
DEBUG      = False

# Cache TTL (segundos)
CACHE_TTL = {
    'tempo_atual':    300,    # 5 min
    'previsao':       1800,   # 30 min
    'radar':          120,    # 2 min
    'alertas':        180,    # 3 min
    'geocoding':      86400,  # 24h
    'modelos':        3600,   # 1h
}

# Open-Meteo
OPENMETEO_URL        = 'https://api.open-meteo.com/v1'
OPENMETEO_MARINE_URL = 'https://marine-api.open-meteo.com/v1'

# Geocoding (Nominatim - OpenStreetMap, gratuito)
NOMINATIM_URL = 'https://nominatim.openstreetmap.org'

# Open-Meteo Air Quality
AIR_QUALITY_URL = 'https://air-quality-api.open-meteo.com/v1'

# RainViewer (radar - gratuito)
RAINVIEWER_URL = 'https://api.rainviewer.com/public/weather-maps.json'

# Open-Meteo Ensemble (modelos numéricos combinados)
ENSEMBLE_MODELS = [
    'icon_seamless',
    'gfs_seamless',
    'ecmwf_ifs025',
]

# Limiares de alerta
LIMIARES = {
    'vento_forte':        50,   # km/h
    'vento_tempestade':   80,   # km/h
    'vento_furacao':      119,  # km/h
    'chuva_moderada':     10,   # mm/h
    'chuva_forte':        25,   # mm/h
    'chuva_extrema':      50,   # mm/h
    'rajada_perigosa':    70,   # km/h
    'temp_alta':          38,   # °C
    'temp_baixa':        -5,    # °C
    'prob_raio':          40,   # %
    'neve':                1,   # mm/h
    'granizo':             1,   # mm/h
}

# Limites de memória
MAX_CACHE_ITEMS   = 500
MAX_CACHE_SIZE_MB = 100
MEMORY_LIMIT_MB   = 400

# Admin padrão
ADMIN_DEFAULT = {
    'username': 'admin',
    'password': 'admin',  # será hash
}

import os


class Config:
    # ------------------------------------------------------------------ Flask
    # Em produção defina SECRET_KEY e ADMIN_PASSWORD como variáveis de ambiente.
    SECRET_KEY = os.getenv('SECRET_KEY', 'rrfs-meteorologia-2024')
    DEBUG = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'

    # ------------------------------------------------------------------- APIs
    OPENMETEO_API = "https://api.open-meteo.com/v1"
    RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps"
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')

    # Mapa base do radar (CARTO, gratuito p/ uso não comercial, com atribuição).
    BASEMAP_TILE_URL = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
    BASEMAP_ATTRIBUTION = "© OpenStreetMap contributors © CARTO"

    # ------------------------------------------------------------------ Banco
    SQLALCHEMY_DATABASE_URI = 'sqlite:///rrfs_meteorologia.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ------------------------------------------------------------ Cache (seg.)
    CACHE_TIMEOUT = 3600
    FORECAST_CACHE_TIMEOUT = 3900   # 65 min: o agendador renova de hora em hora
    RADAR_CACHE_TIMEOUT = 600       # RainViewer atualiza a cada ~10 min
    CACHE_DIR = 'dados/cache'

    # --------------------------------------------------------- Agendador (1 h)
    SCHEDULER_ENABLED = os.getenv('SCHEDULER_ENABLED', '1') == '1'
    UPDATE_INTERVAL_MINUTES = 60
    SCHEDULER_FIRST_DELAY_SECONDS = 20
    SCHEDULER_PAUSE_BETWEEN_LOCATIONS = 1.0   # gentileza com a API gratuita

    # --------------------------------------------------------------- Previsão
    PREDICTION_HORIZON_HOURS = 168
    MIN_CONFIDENCE_THRESHOLD = 0.50

    # Modelos globais do Open-Meteo (modelos sem cobertura são descartados sozinhos).
    CLIMATE_MODELS = [
        'gfs_seamless',          # NOAA GFS
        'ecmwf_ifs025',          # ECMWF IFS
        'icon_seamless',         # DWD ICON
        'gem_seamless',          # Canadá GEM
        'meteofrance_seamless',  # Météo-France ARPEGE
        'ukmo_seamless',         # UK Met Office
        'jma_seamless',          # Japão JMA
        'cma_grapes_global',     # China CMA
        'bom_access_global',     # Austrália BOM
    ]
    MODEL_LABELS = {
        'gfs_seamless': 'GFS (NOAA)',
        'ecmwf_ifs025': 'ECMWF IFS',
        'icon_seamless': 'ICON (DWD)',
        'gem_seamless': 'GEM (Canadá)',
        'meteofrance_seamless': 'ARPEGE (Météo-France)',
        'ukmo_seamless': 'UKMO (Reino Unido)',
        'jma_seamless': 'JMA (Japão)',
        'cma_grapes_global': 'CMA GRAPES (China)',
        'bom_access_global': 'BOM ACCESS (Austrália)',
    }

    # ------------------------------------------------------------- Variáveis
    # Variáveis que o sistema prevê, verifica e aprende.
    #  col      coluna horária do Open-Meteo
    #  tol      tolerância p/ contar como "acerto" (mesma unidade da variável)
    #  scale    escala do erro (normaliza a acurácia e o aprendizado)
    #  prior    MAE esperado antes de haver histórico (suaviza os pesos)
    VARIABLES = {
        'temperature':     {'col': 'temperature_2m',       'label': 'Temperatura',      'unit': '°C',   'tol': 1.5,   'scale': 5.0,   'prior': 2.0},
        'humidity':        {'col': 'relative_humidity_2m', 'label': 'Umidade',          'unit': '%',    'tol': 10.0,  'scale': 30.0,  'prior': 10.0},
        'wind_speed':      {'col': 'wind_speed_10m',       'label': 'Vento',            'unit': 'km/h', 'tol': 5.0,   'scale': 15.0,  'prior': 5.0},
        'wind_gusts':      {'col': 'wind_gusts_10m',       'label': 'Rajadas',          'unit': 'km/h', 'tol': 8.0,   'scale': 25.0,  'prior': 8.0},
        'precipitation':   {'col': 'precipitation',        'label': 'Chuva (por hora)', 'unit': 'mm',   'tol': 0.5,   'scale': 2.0,   'prior': 0.5},
        'pressure':        {'col': 'pressure_msl',         'label': 'Pressão',          'unit': 'hPa',  'tol': 1.5,   'scale': 5.0,   'prior': 1.5},
        'cloud_cover':     {'col': 'cloud_cover',          'label': 'Nuvens',           'unit': '%',    'tol': 20.0,  'scale': 50.0,  'prior': 20.0},
        'solar_radiation': {'col': 'shortwave_radiation',  'label': 'Radiação solar',   'unit': 'W/m²', 'tol': 100.0, 'scale': 300.0, 'prior': 100.0},
    }

    # Faixas de antecedência (horas) usadas para medir/aprender separadamente.
    HORIZON_BUCKETS = [('0-3h', 3), ('3-12h', 12), ('12-48h', 48), ('48-168h', 10 ** 6)]
    SNAPSHOT_HORIZONS = [1, 3, 6, 12, 24, 48, 72, 96, 120, 144, 168]

    # ------------------------------------------------------------------ Admin
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD_DEFAULT = os.getenv('ADMIN_PASSWORD', 'admin')

    THRESHOLDS = {
        'thunderstorm': {'cape': 1000, 'shear': 15, 'lifted_index': -3},
        'heavy_rain': {'precipitation': 50},
        'wind': {'speed': 40},
        'hail': {'cape': 2000, 'temp_diff': 25},
        'lightning': {'cape': 800, 'lpi': 3.0},
    }


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

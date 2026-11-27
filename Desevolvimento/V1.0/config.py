import os


class Config:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'rrfs-meteorologia-2024')  # TROQUE em produção (variável de ambiente)
    DEBUG = False

    # APIs
    OPENMETEO_API = "https://api.open-meteo.com/v1"
    RAINVIEWER_API = "https://api.rainviewer.com/public/weather-maps"
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral')

    # Mapa base do radar.
    # O OpenStreetMap (tile.openstreetmap.org) BLOQUEIA (403) uso sem Referer/User-Agent
    # válido e uso pesado. Usamos o CARTO (gratuito p/ uso não comercial, com atribuição).
    # Para uso comercial troque por MapTiler/Stadia/etc. (com chave de API).
    BASEMAP_TILE_URL = "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
    BASEMAP_ATTRIBUTION = "© OpenStreetMap contributors © CARTO"

    # Banco
    SQLALCHEMY_DATABASE_URI = 'sqlite:///rrfs_meteorologia.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Cache (segundos)
    CACHE_TIMEOUT = 3600
    FORECAST_CACHE_TIMEOUT = 1800   # previsão multi-modelo
    RADAR_CACHE_TIMEOUT = 600       # RainViewer atualiza a cada ~10 min
    CACHE_DIR = 'dados/cache'

    # Previsão
    PREDICTION_HORIZON_HOURS = 168
    UPDATE_INTERVAL_MINUTES = 60
    MIN_CONFIDENCE_THRESHOLD = 0.50

    # Modelos globais do Open-Meteo (nomes VÁLIDOS da API).
    # Antes: 'ecmwf' e 'icon' não existem -> 400 -> sobrava só 1 modelo.
    # Modelos sem cobertura/erro são descartados automaticamente.
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

    # Admin
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD_DEFAULT = 'admin'  # TROQUE logo após o primeiro login

    THRESHOLDS = {
        'thunderstorm': {'cape': 1000, 'shear': 15, 'lifted_index': -3},
        'heavy_rain': {'precipitation': 50},
        'wind': {'speed': 40},
        'hail': {'cape': 2000, 'temp_diff': 25}
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

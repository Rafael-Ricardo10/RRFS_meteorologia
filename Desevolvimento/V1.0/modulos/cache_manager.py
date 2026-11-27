import os
import json
from datetime import datetime, timedelta
import logging
from config import Config

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self):
        self.cache_dir = Config.CACHE_DIR
        self.timeout = Config.CACHE_TIMEOUT
        os.makedirs(self.cache_dir, exist_ok=True)

    def get_cache_key(self, location_id: int, data_type: str) -> str:
        return f"{location_id}_{data_type}"

    def get_cache_file(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def get(self, location_id: int, data_type: str, ttl: int = None):
        """Recupera do cache se válido. ttl (s) sobrescreve o timeout padrão."""
        try:
            cache_key = self.get_cache_key(location_id, data_type)
            cache_file = self.get_cache_file(cache_key)

            if not os.path.exists(cache_file):
                return None

            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            timeout = ttl if ttl is not None else self.timeout
            timestamp = datetime.fromisoformat(cache_data.get('timestamp', ''))
            if datetime.utcnow() - timestamp > timedelta(seconds=timeout):
                logger.info(f"Cache expirado: {cache_key}")
                return None

            logger.info(f"Cache hit: {cache_key}")
            return json.dumps(cache_data.get('data'))

        except Exception as e:
            logger.error(f"Erro ao ler cache: {str(e)}")
            return None

    def set(self, location_id: int, data_type: str, data):
        try:
            cache_key = self.get_cache_key(location_id, data_type)
            cache_file = self.get_cache_file(cache_key)
            tmp = cache_file + '.tmp'

            with open(tmp, 'w') as f:
                json.dump({'timestamp': datetime.utcnow().isoformat(), 'data': data}, f)
            os.replace(tmp, cache_file)

            logger.info(f"Cache salvo: {cache_key}")
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {str(e)}")

    def clear_expired(self):
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.cache_dir, filename)
                    with open(filepath, 'r') as f:
                        cache_data = json.load(f)
                    timestamp = datetime.fromisoformat(cache_data.get('timestamp', ''))
                    if datetime.utcnow() - timestamp > timedelta(seconds=self.timeout):
                        os.remove(filepath)
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {str(e)}")

    def clear_all(self):
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info("Todo o cache foi limpo")
        except Exception as e:
            logger.error(f"Erro ao limpar cache: {str(e)}")

"""
Gerenciador de cache em memória com controle de tamanho e TTL.
Evita vazamento de memória e excesso de chamadas às APIs.
"""
import time
import threading
import json
import hashlib
import logging
import psutil
import os
from collections import OrderedDict
from config import CACHE_TTL, MAX_CACHE_ITEMS, MAX_CACHE_SIZE_MB, MEMORY_LIMIT_MB

logger = logging.getLogger('RRFS.Cache')


class CacheManager:
    def __init__(self):
        self._cache: OrderedDict = OrderedDict()
        self._lock  = threading.RLock()
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        # Limpeza automática a cada 60s
        self._iniciar_limpeza()

    # ── chave ────────────────────────────────────────────────────────────────
    @staticmethod
    def _chave(namespace: str, **kwargs) -> str:
        raw = namespace + json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    # ── get ──────────────────────────────────────────────────────────────────
    def get(self, namespace: str, **kwargs):
        chave = self._chave(namespace, **kwargs)
        with self._lock:
            item = self._cache.get(chave)
            if item is None:
                self._stats['misses'] += 1
                return None
            if time.time() > item['expires']:
                del self._cache[chave]
                self._stats['misses'] += 1
                return None
            # Move para o fim (LRU)
            self._cache.move_to_end(chave)
            self._stats['hits'] += 1
            return item['data']

    # ── set ──────────────────────────────────────────────────────────────────
    def set(self, namespace: str, data, ttl: int = None, **kwargs):
        chave = self._chave(namespace, **kwargs)
        ttl   = ttl or CACHE_TTL.get(namespace, 300)
        with self._lock:
            self._cache[chave] = {
                'data':    data,
                'expires': time.time() + ttl,
                'ns':      namespace,
            }
            self._cache.move_to_end(chave)
            self._verificar_limites()

    # ── delete ───────────────────────────────────────────────────────────────
    def delete(self, namespace: str, **kwargs):
        chave = self._chave(namespace, **kwargs)
        with self._lock:
            self._cache.pop(chave, None)

    def limpar_namespace(self, namespace: str):
        with self._lock:
            keys = [k for k, v in self._cache.items() if v['ns'] == namespace]
            for k in keys:
                del self._cache[k]

    # ── limites ──────────────────────────────────────────────────────────────
    def _verificar_limites(self):
        # Por quantidade
        while len(self._cache) > MAX_CACHE_ITEMS:
            self._cache.popitem(last=False)
            self._stats['evictions'] += 1
        # Por memória do processo
        proc = psutil.Process(os.getpid())
        mem_mb = proc.memory_info().rss / 1024 / 1024
        if mem_mb > MEMORY_LIMIT_MB:
            remover = len(self._cache) // 4
            for _ in range(remover):
                if self._cache:
                    self._cache.popitem(last=False)
                    self._stats['evictions'] += 1
            logger.warning(f'Memória alta ({mem_mb:.0f}MB) - cache reduzido')

    # ── limpeza automática ───────────────────────────────────────────────────
    def _limpar_expirados(self):
        agora = time.time()
        with self._lock:
            expirados = [k for k, v in self._cache.items() if agora > v['expires']]
            for k in expirados:
                del self._cache[k]
        if expirados:
            logger.debug(f'Cache: {len(expirados)} itens expirados removidos')

    def _iniciar_limpeza(self):
        def loop():
            while True:
                time.sleep(60)
                try:
                    self._limpar_expirados()
                except Exception as e:
                    logger.error(f'Erro limpeza cache: {e}')
        t = threading.Thread(target=loop, daemon=True)
        t.name = 'CacheCleaner'
        t.start()

    # ── stats ────────────────────────────────────────────────────────────────
    @property
    def stats(self):
        with self._lock:
            return {**self._stats, 'itens': len(self._cache)}


# Instância global
cache = CacheManager()

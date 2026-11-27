"""
Agendador: atualiza TODAS as localizações de hora em hora, mesmo sem ninguém logado.

 - roda numa thread em segundo plano (sem dependências extras);
 - se houver vários processos (ex.: gunicorn com vários workers), um arquivo de "batimento"
   garante que só um deles executa o trabalho; se o dono morrer, outro assume em ~2,5 min.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)
LOCK_FILE = 'dados/scheduler.lock'
TICK_SECONDS = 30
OWNER_TIMEOUT = 150


class HourlyUpdater:
    def __init__(self, job: Callable[[], dict], interval_minutes: int = 60, first_delay: int = 20):
        self.job = job
        self.interval = interval_minutes * 60
        self.next_run = time.time() + first_delay
        self.thread = None
        self._stop = threading.Event()
        self.status = {'running': False, 'last_start': None, 'last_end': None, 'last_duration_s': None,
                       'last_result': None, 'last_error': None, 'runs': 0, 'next_run': None, 'owner': False}

    # ---- arquivo de batimento (um único processo executa o trabalho)
    @staticmethod
    def _read_lock():
        try:
            with open(LOCK_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return None

    @staticmethod
    def _write_lock():
        try:
            os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
            tmp = LOCK_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'pid': os.getpid(), 'ts': time.time()}, f)
            os.replace(tmp, LOCK_FILE)
        except Exception as e:
            logger.debug(f"Falha ao gravar lock do agendador: {e}")

    def _am_owner(self) -> bool:
        owner = self._read_lock()
        if owner and owner.get('pid') != os.getpid() and time.time() - owner.get('ts', 0) < OWNER_TIMEOUT:
            return False
        self._write_lock()
        return True

    # ---- ciclo
    def _run_job(self):
        self.status.update(running=True, last_start=datetime.now(timezone.utc).isoformat())
        t0 = time.time()
        try:
            self.status['last_result'] = self.job()
            self.status['last_error'] = None
        except Exception as e:                       # o agendador nunca pode morrer
            logger.exception("Falha na atualização horária")
            self.status['last_error'] = str(e)
        finally:
            self.status.update(running=False, last_end=datetime.now(timezone.utc).isoformat(),
                               last_duration_s=round(time.time() - t0, 1), runs=self.status['runs'] + 1)
            self.next_run = t0 + self.interval
            self.status['next_run'] = datetime.fromtimestamp(self.next_run, timezone.utc).isoformat()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.status['owner'] = self._am_owner()
                if self.status['owner'] and time.time() >= self.next_run and not self.status['running']:
                    self._run_job()
            except Exception:
                logger.exception("Erro no laço do agendador")
            self._stop.wait(TICK_SECONDS)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.status['next_run'] = datetime.fromtimestamp(self.next_run, timezone.utc).isoformat()
        self.thread = threading.Thread(target=self._loop, name='rrfs-hourly-updater', daemon=True)
        self.thread.start()
        logger.info(f"Agendador iniciado: atualização a cada {self.interval // 60} min")

    def stop(self):
        self._stop.set()

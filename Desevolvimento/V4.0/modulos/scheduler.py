"""
Agendador: roda os ciclos em segundo plano, mesmo sem ninguém logado.

 - ciclo principal (30 min): reprocessa todos os locais (modelos em cache + radar + aprendizado);
 - vigia do radar/raios (10 min): leve, atualiza veredito e alertas de tempestade;
 - cada ciclo tem a PRÓPRIA thread (o vigia não espera o ciclo principal terminar);
 - se houver vários processos (ex.: gunicorn com vários workers), um arquivo de "batimento" garante que só um
   deles executa o trabalho; se o dono morrer, outro assume em ~2,5 min.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict

logger = logging.getLogger(__name__)
LOCK_FILE = 'dados/scheduler.lock'
TICK_SECONDS = 15
OWNER_TIMEOUT = 150


def _read_lock():
    try:
        with open(LOCK_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def _write_lock():
    try:
        os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
        tmp = f'{LOCK_FILE}.{os.getpid()}.{threading.get_ident()}.tmp'
        with open(tmp, 'w') as f:
            json.dump({'pid': os.getpid(), 'ts': time.time()}, f)
        os.replace(tmp, LOCK_FILE)
    except Exception as e:
        logger.debug(f"Falha ao gravar lock do agendador: {e}")


def am_owner() -> bool:
    owner = _read_lock()
    if owner and owner.get('pid') != os.getpid() and time.time() - owner.get('ts', 0) < OWNER_TIMEOUT:
        return False
    _write_lock()
    return True


class PeriodicJob:
    def __init__(self, name: str, job: Callable[[], dict], interval_s: int, first_delay: int):
        self.name, self.job, self.interval = name, job, interval_s
        self.next_run = time.time() + first_delay
        self.thread = None
        self._stop = threading.Event()
        self.status: Dict = {'running': False, 'last_start': None, 'last_end': None, 'last_duration_s': None,
                             'last_result': None, 'last_error': None, 'runs': 0, 'next_run': None,
                             'owner': False, 'interval_min': round(interval_s / 60, 1)}

    def _run(self):
        self.status.update(running=True, last_start=datetime.now(timezone.utc).isoformat())
        t0 = time.time()
        try:
            self.status['last_result'] = self.job()
            self.status['last_error'] = None
        except Exception as e:                       # o agendador nunca pode morrer
            logger.exception(f"Falha no ciclo '{self.name}'")
            self.status['last_error'] = str(e)
        finally:
            self.status.update(running=False, last_end=datetime.now(timezone.utc).isoformat(),
                               last_duration_s=round(time.time() - t0, 1), runs=self.status['runs'] + 1)
            # o próximo ciclo conta a partir do INÍCIO deste (cadência fixa); se demorou mais que o intervalo, roda já
            self.next_run = max(t0 + self.interval, time.time() + 5)
            self.status['next_run'] = datetime.fromtimestamp(self.next_run, timezone.utc).isoformat()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.status['owner'] = am_owner()
                if self.status['owner'] and time.time() >= self.next_run and not self.status['running']:
                    self._run()
            except Exception:
                logger.exception(f"Erro no laço do ciclo '{self.name}'")
            self._stop.wait(TICK_SECONDS)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.status['next_run'] = datetime.fromtimestamp(self.next_run, timezone.utc).isoformat()
        self.thread = threading.Thread(target=self._loop, name=f'rrfs-{self.name}', daemon=True)
        self.thread.start()
        logger.info(f"Ciclo '{self.name}' iniciado: a cada {self.interval / 60:g} min")

    def stop(self):
        self._stop.set()


class MultiScheduler:
    def __init__(self):
        self.jobs: Dict[str, PeriodicJob] = {}

    def add(self, name: str, job: Callable[[], dict], interval_minutes: float, first_delay: int = 20):
        self.jobs[name] = PeriodicJob(name, job, int(interval_minutes * 60), first_delay)
        return self.jobs[name]

    def start(self):
        for j in self.jobs.values():
            j.start()

    def stop(self):
        for j in self.jobs.values():
            j.stop()

    @property
    def status(self) -> Dict:
        return {name: j.status for name, j in self.jobs.items()}

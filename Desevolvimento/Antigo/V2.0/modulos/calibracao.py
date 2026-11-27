"""
Engine de calibração automática.
Ajusta limiares baseado no histórico de acertos/erros de previsão.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
import config

logger = logging.getLogger('RRFS.Calibracao')

CAL_DIR = config.PASTAS['calibracao']
LIMIARES_BASE = config.LIMIARES.copy()


class CalibracaoEngine:

    def _arq_local(self, lat: float, lon: float) -> Path:
        chave = f'{round(lat, 1)}_{round(lon, 1)}'.replace('-', 'n')
        return CAL_DIR / f'cal_{chave}.json'

    def obter_limiares(self, lat: float, lon: float) -> dict:
        """Retorna limiares calibrados para a localização."""
        arq = self._arq_local(lat, lon)
        if not arq.exists():
            return LIMIARES_BASE.copy()
        try:
            dados = json.loads(arq.read_text())
            limiares = LIMIARES_BASE.copy()
            limiares.update(dados.get('limiares_ajustados', {}))
            return limiares
        except Exception:
            return LIMIARES_BASE.copy()

    def registrar_resultado(self, lat: float, lon: float,
                            tipo_alerta: str, previsto: bool,
                            ocorreu: bool):
        """
        Registra acerto/erro e ajusta limiar.
        previsto=True, ocorreu=True  → acerto
        previsto=True, ocorreu=False → falso positivo → aumentar limiar
        previsto=False, ocorreu=True → falso negativo → diminuir limiar
        """
        arq = self._arq_local(lat, lon)
        dados = {}
        if arq.exists():
            try:
                dados = json.loads(arq.read_text())
            except Exception:
                dados = {}

        stats = dados.setdefault('stats', {})
        tipo_stats = stats.setdefault(tipo_alerta, {
            'acertos': 0, 'falsos_pos': 0, 'falsos_neg': 0, 'total': 0
        })

        tipo_stats['total'] += 1
        if previsto and ocorreu:
            tipo_stats['acertos'] += 1
        elif previsto and not ocorreu:
            tipo_stats['falsos_pos'] += 1
        elif not previsto and ocorreu:
            tipo_stats['falsos_neg'] += 1

        # Recalcula limiares após 10+ amostras
        if tipo_stats['total'] >= 10:
            dados.setdefault('limiares_ajustados', {})
            self._ajustar_limiar(tipo_alerta, tipo_stats,
                                 dados['limiares_ajustados'])

        dados['ultima_atualizacao'] = datetime.now().isoformat()
        arq.write_text(json.dumps(dados, ensure_ascii=False, indent=2))

    @staticmethod
    def _ajustar_limiar(tipo: str, stats: dict, limiares_ajustados: dict):
        """Ajuste conservador: máx ±20% do limiar base."""
        total = stats['total']
        if total == 0:
            return

        taxa_fp = stats['falsos_pos'] / total
        taxa_fn = stats['falsos_neg'] / total
        base    = LIMIARES_BASE

        mapa = {
            'raio':           'prob_raio',
            'tempestade':     'vento_forte',
            'vento_forte':    'vento_forte',
            'chuva_forte':    'chuva_forte',
            'chuva_extrema':  'chuva_extrema',
        }

        chave = mapa.get(tipo)
        if not chave or chave not in base:
            return

        atual = limiares_ajustados.get(chave, base[chave])

        if taxa_fp > 0.3:    # muitos falsos positivos → aumentar limiar
            novo = min(atual * 1.10, base[chave] * 1.20)
        elif taxa_fn > 0.3:  # muitos falsos negativos → diminuir limiar
            novo = max(atual * 0.90, base[chave] * 0.80)
        else:
            return

        limiares_ajustados[chave] = round(novo, 2)
        logger.info(f'Limiar {chave} ajustado: {atual:.1f} → {novo:.1f}')

    def obter_estatisticas(self, lat: float, lon: float) -> dict:
        arq = self._arq_local(lat, lon)
        if not arq.exists():
            return {}
        try:
            return json.loads(arq.read_text())
        except Exception:
            return {}

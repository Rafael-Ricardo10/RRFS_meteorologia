"""
Sistema de Precisão e Taxa de Acerto do RRFS.

Funciona em 3 etapas:
1. REGISTRO: quando o sistema faz uma previsão, registra
2. CONFIRMAÇÃO: após o tempo previsto, compara com radar/dados reais
3. CÁLCULO: atualiza taxa de acerto global e por categoria

Categorias de precisão:
- temperatura       (previsão de temperatura)
- tempestade        (previsão de tempestade)
- chuva_forte       (previsão de chuva forte)
- vento_forte       (previsão de vento)
- tornado           (condições de tornado)
- microburst        (condições de microburst)
- raio              (probabilidade de raios)
- direcao           (direção de sistemas)
- global            (média ponderada de todos)
"""
import json
import math
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
import config

logger = logging.getLogger('RRFS.Precisao')

PRECISAO_DIR = config.PASTAS.get('calibracao', Path('dados/calibracao'))
PRECISAO_FILE = PRECISAO_DIR / 'precisao_global.json'

# Pesos para a média global (categorias mais críticas têm mais peso)
PESOS_CATEGORIA = {
    'temperatura':  1.0,
    'tempestade':   2.0,
    'chuva_forte':  1.5,
    'vento_forte':  1.5,
    'tornado':      3.0,
    'microburst':   2.5,
    'raio':         2.0,
    'direcao':      1.5,
    'nowcast':      2.0,
}

# Tolerâncias para considerar "acerto"
TOLERANCIAS = {
    'temperatura': 2.0,    # ±2°C
    'direcao':    45.0,    # ±45° na direção
    'chuva_mm':    5.0,    # ±5mm
    'tempo_h':     1.0,    # ±1h no horário
}


class SistemaPrecisao:
    """
    Gerencia registro, confirmação e cálculo de taxa de acerto.
    Thread-safe com lock.
    """
    _lock = threading.RLock()

    def __init__(self):
        PRECISAO_DIR.mkdir(parents=True, exist_ok=True)
        self._dados = self._carregar()

    # ── Persistência ──────────────────────────────────────────────────────────

    def _carregar(self) -> dict:
        if not PRECISAO_FILE.exists():
            return self._estrutura_vazia()
        try:
            return json.loads(PRECISAO_FILE.read_text(encoding='utf-8'))
        except Exception:
            return self._estrutura_vazia()

    def _salvar(self):
        try:
            PRECISAO_FILE.write_text(
                json.dumps(self._dados, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            logger.error(f'Erro ao salvar precisão: {e}')

    @staticmethod
    def _estrutura_vazia() -> dict:
        cats = list(PESOS_CATEGORIA.keys())
        return {
            'global': {
                'acertos': 0, 'erros': 0, 'total': 0,
                'taxa': 0.0, 'ultima_atualizacao': None,
                'historico_diario': [],
            },
            'categorias': {
                c: {
                    'acertos': 0, 'erros': 0, 'total': 0,
                    'taxa': 0.0, 'confianca_media': 0.0,
                    'ultima_atualizacao': None,
                }
                for c in cats
            },
            'previsoes_pendentes': [],
            'historico_confirmacoes': [],
        }

    # ── Registro de Previsão ─────────────────────────────────────────────────

    def registrar_previsao(self, categoria: str, previsao: dict,
                           hora_prevista: str, lat: float, lon: float,
                           confianca: float = 1.0) -> str:
        """
        Registra uma previsão para confirmação futura.
        Retorna ID da previsão.
        """
        pid = f"{categoria}_{hora_prevista}_{lat:.2f}_{lon:.2f}"
        pid = pid.replace(':', '').replace('-', '').replace('.', '')

        entrada = {
            'id':            pid,
            'categoria':     categoria,
            'previsao':      previsao,
            'hora_prevista': hora_prevista,
            'lat':           lat,
            'lon':           lon,
            'confianca':     confianca,
            'registrado_em': datetime.now().isoformat(),
            'confirmado':    False,
            'expirado':      False,
        }

        with self._lock:
            # Evita duplicatas
            ids_existentes = {p['id'] for p in self._dados['previsoes_pendentes']}
            if pid not in ids_existentes:
                self._dados['previsoes_pendentes'].append(entrada)
                # Limpa pendentes antigos (>48h)
                self._limpar_pendentes_expirados()
                self._salvar()

        return pid

    def _limpar_pendentes_expirados(self):
        """Remove previsões pendentes mais antigas que 48h."""
        corte = (datetime.now() - timedelta(hours=48)).isoformat()
        antes = len(self._dados['previsoes_pendentes'])
        self._dados['previsoes_pendentes'] = [
            p for p in self._dados['previsoes_pendentes']
            if p['registrado_em'] >= corte or not p['confirmado']
        ]
        removidos = antes - len(self._dados['previsoes_pendentes'])
        if removidos > 0:
            logger.debug(f'Precisão: {removidos} previsões expiradas removidas')

    # ── Confirmação Automática ────────────────────────────────────────────────

    def confirmar_com_dados_reais(self, dados_atuais: dict,
                                   lat: float, lon: float):
        """
        Compara previsões pendentes com dados reais atuais.
        Chamado automaticamente a cada atualização de dados.
        """
        agora = datetime.now()
        with self._lock:
            pendentes = [
                p for p in self._dados['previsoes_pendentes']
                if not p['confirmado'] and not p['expirado']
            ]

        confirmados = []
        for prev in pendentes:
            try:
                hora_prev = datetime.fromisoformat(prev['hora_prevista'])
            except Exception:
                continue

            # Janela de confirmação: ±1h do horário previsto
            delta = abs((agora - hora_prev).total_seconds() / 3600)
            if delta > 2.0:
                # Ainda não chegou ou já passou demais
                if hora_prev < agora - timedelta(hours=3):
                    # Expirou sem confirmação — conta como erro
                    with self._lock:
                        prev['expirado'] = True
                    self._registrar_resultado(
                        prev['categoria'], False,
                        prev['confianca'], prev
                    )
                    confirmados.append(prev['id'])
                continue

            # Compara previsão com dados reais
            acertou = self._comparar_previsao(
                prev['categoria'], prev['previsao'], dados_atuais
            )

            self._registrar_resultado(
                prev['categoria'], acertou,
                prev['confianca'], prev
            )

            with self._lock:
                prev['confirmado']      = True
                prev['acertou']         = acertou
                prev['confirmado_em']   = agora.isoformat()
                prev['dados_reais_ref'] = str(dados_atuais.get('current', {}))[:200]
            confirmados.append(prev['id'])

            logger.info(
                f'Previsão {prev["categoria"]} confirmada: '
                f'{"✅ ACERTO" if acertou else "❌ ERRO"}'
            )

        if confirmados:
            with self._lock:
                self._atualizar_global()
                self._salvar()

    def _comparar_previsao(self, categoria: str,
                            previsao: dict, dados_reais: dict) -> bool:
        """Compara previsão com dados reais. Retorna True se acertou."""
        current = dados_reais.get('current', {})
        hourly  = dados_reais.get('hourly', {})

        try:
            if categoria == 'temperatura':
                t_prev  = previsao.get('temperatura')
                t_real  = current.get('temperature_2m')
                if t_prev is None or t_real is None:
                    return True  # sem dados suficientes → neutro
                return abs(t_prev - t_real) <= TOLERANCIAS['temperatura']

            elif categoria == 'tempestade':
                wmo_real = current.get('weather_code', 0)
                cape_real= current.get('cape', 0) or 0
                prev_sev = previsao.get('severidade', 'ALERTA')
                # Acerto: WMO indica tempestade OU CAPE alto
                return wmo_real >= 80 or cape_real >= 800

            elif categoria == 'chuva_forte':
                prec_prev= previsao.get('mm', 0)
                prec_real= current.get('precipitation', 0) or 0
                return prec_real >= (prec_prev * 0.5)  # aceitamos 50% do previsto

            elif categoria == 'vento_forte':
                gust_prev= previsao.get('gust', 0)
                gust_real= current.get('wind_gusts_10m', 0) or 0
                return gust_real >= (gust_prev * 0.6)

            elif categoria in ('tornado', 'microburst'):
                # Para tornado/microburst, confirmamos se CAPE alto E vento forte
                cape_real = current.get('cape', 0) or 0
                gust_real = current.get('wind_gusts_10m', 0) or 0
                return cape_real >= 1000 and gust_real >= 60

            elif categoria == 'raio':
                wmo_real  = current.get('weather_code', 0)
                cape_real = current.get('cape', 0) or 0
                return wmo_real >= 95 or cape_real >= 1200

            elif categoria == 'direcao':
                dir_prev = previsao.get('direcao')
                dir_real = current.get('wind_direction_10m')
                if dir_prev is None or dir_real is None:
                    return True
                diff = abs(dir_prev - dir_real)
                if diff > 180:
                    diff = 360 - diff
                return diff <= TOLERANCIAS['direcao']

            elif categoria == 'nowcast':
                # Nowcast confirmado se previsão de chuva bate com realidade
                prev_chuva = previsao.get('chuva', False)
                real_chuva = (current.get('precipitation', 0) or 0) > 0.5
                return prev_chuva == real_chuva

            else:
                return True  # categoria desconhecida → neutro

        except Exception as e:
            logger.warning(f'Erro ao comparar previsão {categoria}: {e}')
            return True  # dúvida → neutro

    # ── Registro de Resultado ─────────────────────────────────────────────────

    def _registrar_resultado(self, categoria: str, acertou: bool,
                              confianca: float, prev: dict):
        """Registra acerto/erro na categoria e no histórico."""
        with self._lock:
            cats = self._dados['categorias']
            if categoria not in cats:
                cats[categoria] = {
                    'acertos': 0, 'erros': 0, 'total': 0,
                    'taxa': 0.0, 'confianca_media': 0.0,
                    'ultima_atualizacao': None,
                }

            cat = cats[categoria]
            cat['total']   += 1
            if acertou:
                cat['acertos'] += 1
            else:
                cat['erros']   += 1

            # Confiança média com média móvel
            alfa = 0.1  # suavização exponencial
            cat['confianca_media'] = (
                (1 - alfa) * cat['confianca_media'] + alfa * confianca
            )

            # Taxa de acerto
            cat['taxa'] = round(
                cat['acertos'] / max(cat['total'], 1) * 100, 1
            )
            cat['ultima_atualizacao'] = datetime.now().isoformat()

            # Histórico
            self._dados['historico_confirmacoes'].append({
                'categoria': categoria,
                'acertou':   acertou,
                'confianca': confianca,
                'data':      datetime.now().isoformat(),
                'prev_id':   prev.get('id', ''),
            })

            # Mantém histórico de 90 dias
            corte = (datetime.now() - timedelta(days=90)).isoformat()
            self._dados['historico_confirmacoes'] = [
                h for h in self._dados['historico_confirmacoes']
                if h['data'] >= corte
            ]

    def _atualizar_global(self):
        """Recalcula taxa global ponderada."""
        cats = self._dados['categorias']
        soma_ponderada = 0.0
        soma_pesos     = 0.0

        for cat_nome, peso in PESOS_CATEGORIA.items():
            cat = cats.get(cat_nome, {})
            if cat.get('total', 0) >= 3:  # mínimo 3 amostras
                soma_ponderada += cat.get('taxa', 0.0) * peso
                soma_pesos     += peso

        if soma_pesos > 0:
            taxa_global = round(soma_ponderada / soma_pesos, 1)
        else:
            taxa_global = 0.0  # dados insuficientes

        g = self._dados['global']
        g['taxa'] = taxa_global
        # Total global = soma de todos os acertos
        total_a = sum(c.get('acertos', 0) for c in cats.values())
        total_e = sum(c.get('erros',   0) for c in cats.values())
        g['acertos'] = total_a
        g['erros']   = total_e
        g['total']   = total_a + total_e
        g['ultima_atualizacao'] = datetime.now().isoformat()

        # Histórico diário
        hoje = datetime.now().strftime('%Y-%m-%d')
        hist = g.get('historico_diario', [])
        if hist and hist[-1].get('data') == hoje:
            hist[-1]['taxa'] = taxa_global
        else:
            hist.append({'data': hoje, 'taxa': taxa_global})
            # Mantém 180 dias
            g['historico_diario'] = hist[-180:]

        logger.info(f'Taxa global atualizada: {taxa_global}%')

    # ── Consulta de Estatísticas ──────────────────────────────────────────────

    def obter_estatisticas(self) -> dict:
        """Retorna todas as estatísticas para a interface."""
        with self._lock:
            dados = json.loads(json.dumps(self._dados))  # deep copy

        cats = dados.get('categorias', {})
        global_ = dados.get('global', {})

        # Adiciona interpretação textual para a UI
        def interpretar_taxa(taxa, total):
            if total < 5:
                return {'texto': 'Dados insuficientes', 'cor': 'var(--text2)', 'emoji': '⏳'}
            if taxa >= 90:
                return {'texto': 'Excelente', 'cor': 'var(--success)', 'emoji': '🎯'}
            if taxa >= 75:
                return {'texto': 'Bom',       'cor': 'var(--info)',    'emoji': '✅'}
            if taxa >= 60:
                return {'texto': 'Moderado',  'cor': 'var(--warning)', 'emoji': '⚠️'}
            if taxa >= 40:
                return {'texto': 'Baixo',     'cor': 'var(--danger)',  'emoji': '❌'}
            return     {'texto': 'Calibrando','cor': 'var(--text2)',   'emoji': '🔧'}

        for cat_nome, cat in cats.items():
            cat['interpretacao'] = interpretar_taxa(
                cat.get('taxa', 0), cat.get('total', 0)
            )

        global_['interpretacao'] = interpretar_taxa(
            global_.get('taxa', 0), global_.get('total', 0)
        )

        # Quantas previsões pendentes de confirmação
        pendentes = [
            p for p in dados.get('previsoes_pendentes', [])
            if not p.get('confirmado') and not p.get('expirado')
        ]
        dados['n_pendentes'] = len(pendentes)
        dados['categorias_nomes'] = {
            'temperatura': 'Temperatura',
            'tempestade':  'Tempestades',
            'chuva_forte': 'Chuva Forte',
            'vento_forte': 'Vento Forte',
            'tornado':     'Tornado',
            'microburst':  'Microburst',
            'raio':        'Raios',
            'direcao':     'Direção de Sistemas',
            'nowcast':     'Nowcast (30min)',
        }

        return dados

    def registrar_previsao_direcao(self, lat: float, lon: float,
                                    direcao: float, velocidade: float,
                                    tipo_sistema: str, hora_chegada: str,
                                    confianca: float = 0.75):
        """Atalho para registrar previsão de direção de sistema."""
        return self.registrar_previsao(
            categoria='direcao',
            previsao={
                'direcao':        direcao,
                'velocidade':     velocidade,
                'tipo_sistema':   tipo_sistema,
            },
            hora_prevista=hora_chegada,
            lat=lat, lon=lon,
            confianca=confianca,
        )


# ── Instância global ──────────────────────────────────────────────────────────
sistema_precisao = SistemaPrecisao()

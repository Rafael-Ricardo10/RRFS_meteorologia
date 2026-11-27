"""
Sistema de alertas meteorológicos.
- Detecção baseada em dados Open-Meteo
- Integração com Open-Meteo Alerts (quando disponível)
- Calibração automática por histórico
"""
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from modulos.cache_manager import cache
from modulos.calibracao import CalibracaoEngine
from modulos.tornado_microburst import gerar_alertas_tornado_microburst
import config

logger = logging.getLogger('RRFS.Alertas')

ALERTAS_DIR = config.PASTAS['alertas']


class TipoAlerta:
    RAIO          = 'raio'
    TEMPESTADE    = 'tempestade'
    VENTO_FORTE   = 'vento_forte'
    CHUVA_FORTE   = 'chuva_forte'
    CHUVA_EXTREMA = 'chuva_extrema'
    NEVE          = 'neve'
    GRANIZO       = 'granizo'
    TEMP_ALTA     = 'temperatura_alta'
    TEMP_BAIXA    = 'temperatura_baixa'
    NEVOEIRO      = 'nevoeiro'
    FURACAO       = 'furacao'
    PERSONALIZADO = 'personalizado'


SEVERIDADE = {
    TipoAlerta.RAIO:          'PERIGO',
    TipoAlerta.TEMPESTADE:    'PERIGO',
    TipoAlerta.FURACAO:       'EXTREMO',
    TipoAlerta.CHUVA_EXTREMA: 'PERIGO',
    TipoAlerta.GRANIZO:       'PERIGO',
    TipoAlerta.VENTO_FORTE:   'ALERTA',
    TipoAlerta.CHUVA_FORTE:   'ALERTA',
    TipoAlerta.NEVE:          'ALERTA',
    TipoAlerta.TEMP_ALTA:     'ATENCAO',
    TipoAlerta.TEMP_BAIXA:    'ATENCAO',
    TipoAlerta.NEVOEIRO:      'ATENCAO',
    TipoAlerta.PERSONALIZADO: 'INFO',
}

ICONES = {
    'EXTREMO': '🚨', 'PERIGO': '⚠️', 'ALERTA': '🔔', 'ATENCAO': 'ℹ️',
}


class AnalisadorAlertas:
    def __init__(self):
        self.calibracao = CalibracaoEngine()

    def analisar(self, dados_meteo: dict, dados_ensemble: dict,
                 lat: float, lon: float) -> list:
        """Analisa todos os dados e retorna lista de alertas."""
        alertas = []

        if not dados_meteo:
            return alertas

        limiares = self.calibracao.obter_limiares(lat, lon)
        hourly   = dados_meteo.get('hourly', {})
        current  = dados_meteo.get('current', {})

        # Janela: próximas 24h
        from modulos.meteorologia import hora_atual_indice
        idx_atual = hora_atual_indice(dados_meteo)
        janela    = slice(idx_atual, idx_atual + 24)

        alertas += self._verificar_tempestade(hourly, janela, limiares)
        alertas += self._verificar_raios(hourly, janela, limiares)
        alertas += self._verificar_vento(hourly, janela, limiares)
        alertas += self._verificar_chuva(hourly, janela, limiares)
        alertas += self._verificar_neve_granizo(hourly, janela, limiares)
        alertas += self._verificar_temperatura(hourly, janela, limiares)
        alertas += self._verificar_nevoeiro(hourly, janela, limiares)

        # Validação cruzada com ensemble
        if dados_ensemble:
            alertas = self._validar_com_ensemble(alertas, dados_ensemble, idx_atual)

        # Remove duplicatas e ordena por severidade
        alertas = self._deduplicar(alertas)
        return alertas

    # ── verificações ─────────────────────────────────────────────────────────

    def _verificar_tempestade(self, hourly, janela, limiares) -> list:
        alertas = []
        cape_vals  = (hourly.get('cape', []) or [])[janela]
        li_vals    = (hourly.get('lifted_index', []) or [])[janela]
        prec_prob  = (hourly.get('precipitation_probability', []) or [])[janela]
        wmo_vals   = (hourly.get('weather_code', []) or [])[janela]
        gusts      = (hourly.get('wind_gusts_10m', []) or [])[janela]
        times      = (hourly.get('time', []) or [])[janela]

        for i, (cape, li, prob, wmo, gust) in enumerate(
                zip(cape_vals, li_vals or [None]*24,
                    prec_prob, wmo_vals, gusts)):

            if cape is None: continue
            hora = times[i] if i < len(times) else '?'

            # Furacão / ciclone extratropical
            if gust and gust >= limiares['vento_furacao']:
                alertas.append(self._criar_alerta(
                    TipoAlerta.FURACAO,
                    f'Rajadas de {gust:.0f} km/h previstas',
                    hora, {'gust': gust}
                ))
                continue

            # Tempestade severa
            if cape >= 2500 and (li is None or li <= -6):
                desc = f'CAPE {cape:.0f} J/kg - Tempestade SEVERA prevista'
                alertas.append(self._criar_alerta(
                    TipoAlerta.TEMPESTADE, desc, hora,
                    {'cape': cape, 'li': li}
                ))
            elif cape >= 1000 and (li is None or li <= -3):
                if prob and prob >= 50 or wmo in (95, 96, 99):
                    alertas.append(self._criar_alerta(
                        TipoAlerta.TEMPESTADE,
                        f'CAPE {cape:.0f} J/kg - Tempestade provável',
                        hora, {'cape': cape}
                    ))
            # Código WMO de tempestade
            elif wmo in (95, 96, 99):
                alertas.append(self._criar_alerta(
                    TipoAlerta.TEMPESTADE,
                    f'Tempestade prevista (código {wmo})',
                    hora, {'wmo': wmo}
                ))

        return alertas

    def _verificar_raios(self, hourly, janela, limiares) -> list:
        alertas = []
        cape_vals = (hourly.get('cape', []) or [])[janela]
        li_vals   = (hourly.get('lifted_index', []) or [])[janela]
        cin_vals  = (hourly.get('convective_inhibition', []) or [])[janela]
        times     = (hourly.get('time', []) or [])[janela]

        for i, (cape, li, cin) in enumerate(
                zip(cape_vals,
                    li_vals or [None]*24,
                    cin_vals or [None]*24)):

            if cape is None: continue
            hora = times[i] if i < len(times) else '?'

            prob_raio = self._calcular_prob_raio(cape, li, cin)
            if prob_raio >= limiares.get('prob_raio', 40):
                alertas.append(self._criar_alerta(
                    TipoAlerta.RAIO,
                    f'Probabilidade de raios: {prob_raio:.0f}% '
                    f'(CAPE={cape:.0f} J/kg)',
                    hora, {'prob': prob_raio, 'cape': cape}
                ))
        return alertas

    @staticmethod
    def _calcular_prob_raio(cape, li, cin) -> float:
        """Estima probabilidade de raios (0-100%)."""
        if cape is None or cape < 300:
            return 0.0
        prob = min(cape / 30, 80)          # max 80% pelo CAPE
        if li is not None and li < 0:
            prob += min(abs(li) * 3, 20)   # até +20% pelo LI
        if cin is not None and cin < -100:
            prob -= 15                     # CIN alto inibe
        return max(0.0, min(100.0, prob))

    def _verificar_vento(self, hourly, janela, limiares) -> list:
        alertas = []
        gusts = (hourly.get('wind_gusts_10m', []) or [])[janela]
        winds = (hourly.get('wind_speed_10m', []) or [])[janela]
        times = (hourly.get('time', []) or [])[janela]

        for i, (gust, wind) in enumerate(zip(gusts, winds)):
            if gust is None: continue
            hora = times[i] if i < len(times) else '?'
            if gust >= limiares['vento_furacao']:
                pass  # já detectado em tempestade
            elif gust >= limiares['vento_tempestade']:
                alertas.append(self._criar_alerta(
                    TipoAlerta.VENTO_FORTE,
                    f'Rajadas de {gust:.0f} km/h (tempestade de vento)',
                    hora, {'gust': gust, 'wind': wind}
                ))
            elif gust >= limiares['vento_forte']:
                alertas.append(self._criar_alerta(
                    TipoAlerta.VENTO_FORTE,
                    f'Rajadas de {gust:.0f} km/h previstas',
                    hora, {'gust': gust}
                ))
        return alertas

    def _verificar_chuva(self, hourly, janela, limiares) -> list:
        alertas = []
        precip = (hourly.get('precipitation', []) or [])[janela]
        rain   = (hourly.get('rain', []) or [])[janela]
        times  = (hourly.get('time', []) or [])[janela]

        for i, (p, r) in enumerate(zip(precip, rain)):
            if p is None: continue
            hora = times[i] if i < len(times) else '?'
            if p >= limiares['chuva_extrema']:
                alertas.append(self._criar_alerta(
                    TipoAlerta.CHUVA_EXTREMA,
                    f'Chuva EXTREMA: {p:.1f} mm/h prevista',
                    hora, {'mm': p}
                ))
            elif p >= limiares['chuva_forte']:
                alertas.append(self._criar_alerta(
                    TipoAlerta.CHUVA_FORTE,
                    f'Chuva forte: {p:.1f} mm/h prevista',
                    hora, {'mm': p}
                ))
        return alertas

    def _verificar_neve_granizo(self, hourly, janela, limiares) -> list:
        alertas = []
        snow    = (hourly.get('snowfall', []) or [])[janela]
        showers = (hourly.get('showers', []) or [])[janela]
        wmo     = (hourly.get('weather_code', []) or [])[janela]
        times   = (hourly.get('time', []) or [])[janela]

        for i, (s, sh, w) in enumerate(zip(snow, showers, wmo)):
            hora = times[i] if i < len(times) else '?'
            if s and s >= limiares.get('neve', 1):
                alertas.append(self._criar_alerta(
                    TipoAlerta.NEVE,
                    f'Neve: {s:.1f} mm prevista', hora, {'mm': s}
                ))
            if w in (96, 99):
                alertas.append(self._criar_alerta(
                    TipoAlerta.GRANIZO,
                    'Granizo previsto com tempestade', hora, {'wmo': w}
                ))
        return alertas

    def _verificar_temperatura(self, hourly, janela, limiares) -> list:
        alertas = []
        temps = (hourly.get('temperature_2m', []) or [])[janela]
        times = (hourly.get('time', []) or [])[janela]

        for i, t in enumerate(temps):
            if t is None: continue
            hora = times[i] if i < len(times) else '?'
            if t >= limiares['temp_alta']:
                alertas.append(self._criar_alerta(
                    TipoAlerta.TEMP_ALTA,
                    f'Temperatura alta: {t:.1f}°C', hora, {'temp': t}
                ))
            elif t <= limiares['temp_baixa']:
                alertas.append(self._criar_alerta(
                    TipoAlerta.TEMP_BAIXA,
                    f'Temperatura baixa: {t:.1f}°C', hora, {'temp': t}
                ))
        return alertas

    def _verificar_nevoeiro(self, hourly, janela, limiares) -> list:
        alertas = []
        vis  = (hourly.get('visibility', []) or [])[janela]
        wmo  = (hourly.get('weather_code', []) or [])[janela]
        times= (hourly.get('time', []) or [])[janela]
        for i, (v, w) in enumerate(zip(vis, wmo)):
            hora = times[i] if i < len(times) else '?'
            if w in (45, 48) or (v is not None and v < 1000):
                alertas.append(self._criar_alerta(
                    TipoAlerta.NEVOEIRO,
                    f'Nevoeiro - visibilidade {v}m' if v else 'Nevoeiro previsto',
                    hora, {'vis': v}
                ))
        return alertas

    # ── validação ensemble ───────────────────────────────────────────────────

    def _validar_com_ensemble(self, alertas: list, ensemble: dict,
                            idx_atual: int) -> list:
        """
        Remove alertas com baixa concordância entre modelos.
        Se ensemble não tiver dados suficientes, mantém o alerta
        com nota de confiança reduzida (não descarta).
        """
        modelos   = ensemble.get('modelos', {})
        n_modelos = len(modelos)

        # Se não há dados de ensemble suficientes,
        # mantém todos os alertas mas marca concordância reduzida
        if n_modelos < 2:
            for a in alertas:
                a['concordancia'] = 0.6
                a['n_modelos']    = max(n_modelos, 1)
            return alertas

        resultado = []
        for alerta in alertas:
            concordancia = self._contar_concordancia(alerta, modelos, idx_atual)
            alerta['concordancia'] = concordancia
            alerta['n_modelos']    = n_modelos

            sev     = alerta.get('severidade', 'ATENCAO')
            # Só descarta se tiver dados de ensemble E concordância muito baixa
            # Para CAPE, muitos modelos não retornam → não penaliza
            tem_dados_cape = any(
                'cape' in (d.get('hourly') or {})
                for d in modelos.values()
            )

            if not tem_dados_cape:
                # Sem dados CAPE no ensemble → mantém com concordância neutra
                alerta['concordancia'] = 0.65
                resultado.append(alerta)
                continue

            # Limiar de concordância por severidade
            min_conc = {
                'EXTREMO': 0.33,
                'PERIGO':  0.33,
                'ALERTA':  0.25,
                'ATENCAO': 0.20,
                'INFO':    0.10,
            }.get(sev, 0.25)

            if concordancia >= min_conc:
                resultado.append(alerta)
            else:
                logger.info(
                    f'Alerta {alerta["tipo"]} descartado '
                    f'(concordância: {concordancia:.0%}, min: {min_conc:.0%})'
                )

        return resultado

    def _contar_concordancia(self, alerta, modelos, idx) -> float:
        tipo = alerta['tipo']
        concordam = 0
        for dados in modelos.values():
            hourly = dados.get('hourly', {})
            if tipo in (TipoAlerta.TEMPESTADE, TipoAlerta.RAIO):
                cape = hourly.get('cape', [None] * (idx + 25))
                if idx < len(cape) and cape[idx] and cape[idx] >= 800:
                    concordam += 1
            elif tipo == TipoAlerta.VENTO_FORTE:
                gusts = hourly.get('wind_gusts_10m', [None] * (idx + 25))
                if idx < len(gusts) and gusts[idx] and gusts[idx] >= 50:
                    concordam += 1
            elif tipo in (TipoAlerta.CHUVA_FORTE, TipoAlerta.CHUVA_EXTREMA):
                prec = hourly.get('precipitation', [None] * (idx + 25))
                if idx < len(prec) and prec[idx] and prec[idx] >= 10:
                    concordam += 1
            else:
                concordam += 1  # sem dados ensemble para este tipo
        return concordam / max(len(modelos), 1)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _criar_alerta(tipo: str, descricao: str, hora: str,
                      dados_extras: dict = None) -> dict:
        sev   = SEVERIDADE.get(tipo, 'ATENCAO')
        icone = ICONES.get(sev, 'ℹ️')
        return {
            'id':          f'{tipo}_{hora}',
            'tipo':        tipo,
            'severidade':  sev,
            'icone':       icone,
            'descricao':   descricao,
            'hora':        hora,
            'dados':       dados_extras or {},
            'criado_em':   datetime.now().isoformat(),
            'concordancia': 1.0,
        }

    @staticmethod
    def _deduplicar(alertas: list) -> list:
        vistos = set()
        resultado = []
        ordem_sev = {'EXTREMO': 0, 'PERIGO': 1, 'ALERTA': 2, 'ATENCAO': 3, 'INFO': 4}
        alertas.sort(key=lambda a: ordem_sev.get(a['severidade'], 5))
        for a in alertas:
            chave = (a['tipo'], a['hora'][:13])  # agrupa por hora
            if chave not in vistos:
                vistos.add(chave)
                resultado.append(a)
        return resultado


def verificar_alertas_usuario(dados_meteo, dados_ensemble,
                               lat, lon, config_usuario: dict) -> list:
    """..."""
    analisador = AnalisadorAlertas()
    alertas = analisador.analisar(dados_meteo, dados_ensemble, lat, lon)

    # ── NOVO: Tornado e Microburst ────────────────────────────────────────
    alertas_tornado = gerar_alertas_tornado_microburst(dados_meteo, lat, lon)
    alertas.extend(alertas_tornado)

    # Alertas personalizados
    condicoes = config_usuario.get('condicoes_alerta', [])
    hourly = dados_meteo.get('hourly', {}) if dados_meteo else {}

    for cond in condicoes:
        var      = cond.get('variavel', '')
        valor    = cond.get('valor')
        operador = cond.get('operador', '>=')
        descricao = cond.get('descricao', f'{var} {operador} {valor}')

        vals = hourly.get(var, [])
        for i, v in enumerate(vals[:24]):
            if v is None: continue
            dispara = False
            if operador == '>='  and v >= valor: dispara = True
            elif operador == '<=' and v <= valor: dispara = True
            elif operador == '>'  and v > valor:  dispara = True
            elif operador == '<'  and v < valor:  dispara = True

            if dispara:
                hora = hourly.get('time', [''] * (i + 1))[i]
                alertas.append({
                    'id': f'custom_{var}_{hora}',
                    'tipo': TipoAlerta.PERSONALIZADO,
                    'severidade': 'INFO',
                    'icone': '🔔',
                    'descricao': f'[Alerta pessoal] {descricao}: {v:.1f}',
                    'hora': hora,
                    'dados': {'valor': v},
                    'criado_em': datetime.now().isoformat(),
                    'concordancia': 1.0,
                })
                break

    return alertas


def salvar_historico_alerta(username: str, alertas: list, confirmado: bool = False):
    """Salva histórico para calibração futura."""
    arq = ALERTAS_DIR / f'{username}_historico.json'
    historico = []
    if arq.exists():
        try:
            historico = json.loads(arq.read_text())
        except Exception:
            historico = []

    entrada = {
        'data': datetime.now().isoformat(),
        'alertas': alertas,
        'confirmado': confirmado,
    }
    historico.append(entrada)
    # Mantém apenas 180 dias
    corte = (datetime.now() - timedelta(days=180)).isoformat()
    historico = [h for h in historico if h['data'] >= corte]

    arq.write_text(json.dumps(historico, ensure_ascii=False, indent=2))


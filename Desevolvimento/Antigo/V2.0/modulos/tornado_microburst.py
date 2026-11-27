"""
Detecção de condições para Tornado e Microburst (Micro-explosão).
Baseado em índices termodinâmicos e cinemáticos.
"""
import logging
import math
from datetime import datetime

logger = logging.getLogger('RRFS.Tornado')


class DetectorTornadoMicroburst:
    """
    Detecta condições favoráveis para tornados e microbursts
    usando índices meteorológicos calculados a partir dos dados Open-Meteo.
    """

    # ── Limiares ──────────────────────────────────────────────────────────────
    LIMIARES_TORNADO = {
        'cape_min':       1500,   # J/kg mínimo para supercélula
        'cape_alto':      3000,   # J/kg para tornado significativo
        'li_max':        -5.0,    # LI muito negativo
        'shear_min':      15,     # m/s shear vertical estimado
        'ehi_risco':       1.0,   # EHI >= 1.0 = risco moderado
        'ehi_alto':        2.0,   # EHI >= 2.0 = risco alto
        'ehi_extremo':     3.0,   # EHI >= 3.0 = risco extremo
        'stp_risco':       0.5,   # STP >= 0.5 = condições possíveis
        'stp_alto':        1.0,   # STP >= 1.0 = condições favoráveis
        'stp_extremo':     3.0,   # STP >= 3.0 = risco muito alto
    }

    LIMIARES_MICROBURST = {
        'dcape_risco':     800,   # J/kg DCAPE para microburst
        'dcape_alto':     1200,   # J/kg DCAPE alto
        'shear_baixo_min': 20,    # km/h diferença vento baixo/alto
        'cape_umido_min':  500,   # J/kg mínimo
        'gust_resultado':  90,    # km/h rajada resultante estimada
    }

    def analisar_tornado(self, dados_meteo: dict,
                         idx_inicio: int = 0) -> list:
        """
        Analisa janela de 48h para condições favoráveis a tornado.
        Retorna lista de eventos com risco detectado.
        """
        h       = dados_meteo.get('hourly', {})
        times   = h.get('time', [])
        eventos = []

        cape_arr  = h.get('cape', [])
        li_arr    = h.get('lifted_index', [])
        cin_arr   = h.get('convective_inhibition', [])
        w80_arr   = h.get('wind_speed_80m', [])
        w10_arr   = h.get('wind_speed_10m', [])
        wd10_arr  = h.get('wind_direction_10m', [])
        gust_arr  = h.get('wind_gusts_10m', [])
        prec_arr  = h.get('precipitation', [])

        fim = min(idx_inicio + 48, len(times))

        for i in range(idx_inicio, fim):
            cape  = self._val(cape_arr,  i)
            li    = self._val(li_arr,    i)
            cin   = self._val(cin_arr,   i)
            w80   = self._val(w80_arr,   i)
            w10   = self._val(w10_arr,   i)
            wd10  = self._val(wd10_arr,  i)
            gust  = self._val(gust_arr,  i)

            if cape is None or cape < self.LIMIARES_TORNADO['cape_min']:
                continue

            # ── Shear vertical estimado ──────────────────────────────
            # Open-Meteo não fornece shear direto — estimamos pela
            # diferença entre vento a 80m e 10m
            shear_ms = 0.0
            if w80 is not None and w10 is not None:
                shear_ms = abs(w80 - w10) * (1000 / 28)  # normaliza ~0-1km

            # ── EHI (Energy Helicity Index) estimado ─────────────────
            # EHI = (CAPE × SRH) / 160000
            # SRH estimado pela magnitude do shear e direção
            srh_est = self._estimar_srh(w10, w80, wd10, shear_ms)
            ehi     = (cape * srh_est) / 160000.0 if srh_est > 0 else 0.0

            # ── STP (Significant Tornado Parameter) simplificado ─────
            # STP = (CAPE/1500) × (SRH/150) × (Shear/20) × ((2000-LCL)/1000)
            lcl_est = self._estimar_lcl(h, i)  # metros
            shear_norm = min(shear_ms / 20.0, 1.5)
            cape_norm  = min(cape / 1500.0, 3.0)
            srh_norm   = min(srh_est / 150.0, 2.0)
            lcl_norm   = max((2000 - lcl_est) / 1000.0, 0.0) if lcl_est else 1.0
            stp        = cape_norm * srh_norm * shear_norm * lcl_norm

            # ── Determina nível de risco ─────────────────────────────
            nivel, descricao = self._nivel_tornado(cape, li, ehi, stp, cin)

            if nivel is None:
                continue

            hora = times[i] if i < len(times) else ''

            eventos.append({
                'tipo':      'tornado',
                'nivel':     nivel,
                'hora':      hora,
                'indice':    i,
                'metricas': {
                    'cape':       round(cape, 0),
                    'li':         round(li, 2) if li else None,
                    'shear_ms':   round(shear_ms, 1),
                    'srh_est':    round(srh_est, 0),
                    'ehi':        round(ehi, 2),
                    'stp':        round(stp, 2),
                    'lcl_m':      round(lcl_est, 0) if lcl_est else None,
                    'cin':        round(cin, 0) if cin else None,
                    'gust_kmh':   round(gust, 0) if gust else None,
                },
                'descricao': descricao,
            })

        return self._agrupar_eventos(eventos, 'tornado')

    def analisar_microburst(self, dados_meteo: dict,
                             idx_inicio: int = 0) -> list:
        """
        Detecta condições para microburst (micro-explosão de vento).
        """
        h       = dados_meteo.get('hourly', {})
        times   = h.get('time', [])
        eventos = []

        cape_arr  = h.get('cape', [])
        temp_arr  = h.get('temperature_2m', [])
        td_arr    = h.get('dew_point_2m', [])
        prec_arr  = h.get('precipitation', [])
        gust_arr  = h.get('wind_gusts_10m', [])
        w10_arr   = h.get('wind_speed_10m', [])
        w80_arr   = h.get('wind_speed_80m', [])
        cloud_arr = h.get('cloud_cover', [])
        vis_arr   = h.get('visibility', [])

        fim = min(idx_inicio + 48, len(times))

        for i in range(idx_inicio, fim):
            cape  = self._val(cape_arr,  i)
            temp  = self._val(temp_arr,  i)
            td    = self._val(td_arr,    i)
            prec  = self._val(prec_arr,  i)
            gust  = self._val(gust_arr,  i)
            w10   = self._val(w10_arr,   i)
            w80   = self._val(w80_arr,   i)

            if cape is None or cape < self.LIMIARES_MICROBURST['cape_umido_min']:
                continue

            # ── DCAPE estimado ───────────────────────────────────────
            # Downdraft CAPE: energia disponível para corrente descendente
            # Estimamos com base na diferença temperatura/ponto orvalho
            # (camada seca em nível médio favorece microburst)
            dcape_est = self._estimar_dcape(cape, temp, td)

            # ── Shear baixo nível ────────────────────────────────────
            shear_baixo = 0.0
            if w80 is not None and w10 is not None:
                shear_baixo = abs(w80 - w10) * 3.6  # m/s → km/h

            # ── Indicadores adicionais ───────────────────────────────
            # Alta taxa de precip + CAPE alto = evaporação intensa =
            # corrente descendente acelerada
            prec_contrib = (prec or 0) * 20  # mm/h → contribuição

            # Seco em nível médio (diferença T-Td > 10°C)
            deficit_td = (temp - td) if (temp and td) else 0
            seco_medio = deficit_td > 8  # ar relativamente seco

            nivel, descricao = self._nivel_microburst(
                dcape_est, shear_baixo, cape, seco_medio, gust
            )

            if nivel is None:
                continue

            hora = times[i] if i < len(times) else ''

            # Rajada estimada do microburst
            gust_microburst = self._estimar_rajada_microburst(
                dcape_est, shear_baixo
            )

            eventos.append({
                'tipo':      'microburst',
                'nivel':     nivel,
                'hora':      hora,
                'indice':    i,
                'metricas': {
                    'cape':           round(cape, 0),
                    'dcape_est':      round(dcape_est, 0),
                    'shear_baixo':    round(shear_baixo, 1),
                    'deficit_td':     round(deficit_td, 1),
                    'seco_medio':     seco_medio,
                    'gust_atual':     round(gust, 0) if gust else None,
                    'gust_estimado':  round(gust_microburst, 0),
                    'prec':           round(prec, 1) if prec else 0,
                },
                'descricao': descricao,
            })

        return self._agrupar_eventos(eventos, 'microburst')

    # ── Helpers de cálculo ────────────────────────────────────────────────────

    @staticmethod
    def _val(arr, idx):
        try:
            v = arr[idx] if arr and idx < len(arr) else None
            return float(v) if v is not None else None
        except Exception:
            return None

    @staticmethod
    def _estimar_srh(w10, w80, wd10, shear_ms) -> float:
        """
        Estima Storm-Relative Helicity simplificada.
        SRH real requer perfil vertical completo — aqui usamos
        proxy baseado no shear e magnitude do vento.
        """
        if w10 is None or w80 is None:
            return 0.0
        # SRH proxy: shear × componente rotacional
        # Rotação favorecida quando vento vira com a altitude (veering)
        rotacao_fator = 1.0  # sem dado de direção em 80m, assumimos neutro
        srh_est = shear_ms * (w10 + w80) / 2 * rotacao_fator
        return max(0.0, min(srh_est, 500.0))

    @staticmethod
    def _estimar_lcl(h: dict, idx: int) -> float:
        """
        Estima altura da Base das Nuvens (LCL) em metros.
        LCL ≈ (T - Td) × 125 (fórmula empírica em metros)
        """
        try:
            temp_arr = h.get('temperature_2m', [])
            td_arr   = h.get('dew_point_2m', [])
            t  = temp_arr[idx] if idx < len(temp_arr) else None
            td = td_arr[idx]   if idx < len(td_arr)   else None
            if t is not None and td is not None:
                return max(0.0, (t - td) * 125.0)
            return 1000.0  # default
        except Exception:
            return 1000.0

    @staticmethod
    def _estimar_dcape(cape, temp, td) -> float:
        """
        Estima Downdraft CAPE (DCAPE).
        DCAPE alto = corrente descendente forte = microburst intenso.
        Proxy: CAPE × fator de secura da camada.
        """
        if cape is None:
            return 0.0
        deficit = max(0, (temp or 20) - (td or 15))
        # Camada seca amplia o DCAPE (evaporação resfria e acelera downdraft)
        fator_seco = min(1.0 + deficit / 20.0, 2.0)
        dcape = cape * 0.4 * fator_seco
        return max(0.0, dcape)

    @staticmethod
    def _estimar_rajada_microburst(dcape, shear) -> float:
        """
        Estima rajada máxima de microburst.
        Fórmula empírica: Vmax ≈ sqrt(2 × DCAPE) + shear/3.6
        """
        import math
        if dcape <= 0:
            return 0.0
        v_downdraft = math.sqrt(2.0 * dcape) * 0.8  # m/s
        v_total_ms  = v_downdraft + (shear / 3.6) * 0.3
        return v_total_ms * 3.6  # → km/h

    def _nivel_tornado(self, cape, li, ehi, stp, cin):
        """Classifica nível de risco para tornado."""
        L = self.LIMIARES_TORNADO

        # Supercélula possível mas CIN muito alto = improvável
        if cin is not None and cin < -200:
            return None, None

        if stp >= L['stp_extremo'] and ehi >= L['ehi_extremo']:
            return 'EXTREMO', (
                f'Condições EXCEPCIONAIS para tornado significativo. '
                f'STP={stp:.1f}, EHI={ehi:.1f}, CAPE={cape:.0f} J/kg. '
                f'Risco de tornado F2+ na região.'
            )
        elif stp >= L['stp_alto'] and ehi >= L['ehi_alto']:
            return 'ALTO', (
                f'Condições FAVORÁVEIS para tornado. '
                f'STP={stp:.1f}, EHI={ehi:.1f}, CAPE={cape:.0f} J/kg. '
                f'Supercélulas com potencial de tornado na região.'
            )
        elif stp >= L['stp_risco'] or ehi >= L['ehi_risco']:
            if cape >= L['cape_min']:
                return 'MODERADO', (
                    f'Condições favoráveis para supercélulas. '
                    f'STP={stp:.1f}, EHI={ehi:.1f}, CAPE={cape:.0f} J/kg. '
                    f'Tornado isolado não pode ser descartado.'
                )
        elif cape >= L['cape_alto'] and li is not None and li <= L['li_max']:
            return 'BAIXO', (
                f'Instabilidade extrema com CAPE={cape:.0f} J/kg, LI={li:.1f}. '
                f'Tempestades severas. Tornado muito isolado possível.'
            )

        return None, None

    def _nivel_microburst(self, dcape, shear, cape, seco, gust):
        """Classifica nível de risco para microburst."""
        L = self.LIMIARES_MICROBURST

        if dcape >= L['dcape_alto'] and seco:
            return 'ALTO', (
                f'Condições MUITO FAVORÁVEIS para micro-explosão (microburst). '
                f'DCAPE estimado={dcape:.0f} J/kg, shear={shear:.0f} km/h. '
                f'Rajadas extremas repentinas de até {self._estimar_rajada_microburst(dcape,shear):.0f} km/h possíveis.'
            )
        elif dcape >= L['dcape_risco']:
            return 'MODERADO', (
                f'Condições favoráveis para microburst. '
                f'DCAPE={dcape:.0f} J/kg. '
                f'Rajadas repentinas e intensas possíveis sob tempestades.'
            )
        elif dcape >= L['dcape_risco'] * 0.6 and cape > 1000 and seco:
            return 'BAIXO', (
                f'Algum potencial para correntes descendentes fortes. '
                f'CAPE={cape:.0f} J/kg com camada seca intermediária.'
            )

        return None, None

    @staticmethod
    def _agrupar_eventos(eventos: list, tipo: str) -> list:
        """
        Agrupa eventos próximos (janela de 6h) e mantém
        apenas o mais intenso do grupo para evitar spam.
        """
        if not eventos:
            return []

        ordemNivel = {'EXTREMO': 0, 'ALTO': 1, 'MODERADO': 2, 'BAIXO': 3}
        grupos     = []
        ultimo_idx = -99

        for ev in eventos:
            if ev['indice'] - ultimo_idx > 6:
                grupos.append(ev)
                ultimo_idx = ev['indice']
            else:
                # Mantém o mais intenso
                ultimo = grupos[-1]
                if ordemNivel.get(ev['nivel'], 9) < ordemNivel.get(ultimo['nivel'], 9):
                    grupos[-1] = ev
                    ultimo_idx = ev['indice']

        return grupos


# ── Instância global ──────────────────────────────────────────────────────────
detector_tornado = DetectorTornadoMicroburst()


def gerar_alertas_tornado_microburst(dados_meteo: dict,
                                      lat: float, lon: float) -> list:
    """
    Ponto de entrada principal.
    Retorna lista de alertas formatados para o sistema RRFS.
    """
    if not dados_meteo:
        return []

    from modulos.meteorologia import hora_atual_indice
    idx = hora_atual_indice(dados_meteo)

    alertas_out = []

    # ── Tornados ──────────────────────────────────────────────────────────────
    eventos_t = detector_tornado.analisar_tornado(dados_meteo, idx)
    for ev in eventos_t:
        sev_map = {'EXTREMO': 'EXTREMO', 'ALTO': 'PERIGO',
                   'MODERADO': 'ALERTA', 'BAIXO': 'ATENCAO'}
        ico_map = {'EXTREMO': '🌪️🚨', 'ALTO': '🌪️⚠️',
                   'MODERADO': '🌪️🔔', 'BAIXO': '🌪️ℹ️'}
        alertas_out.append({
            'id':          f'tornado_{ev["hora"]}',
            'tipo':        'tornado',
            'subtipo':     'tornado_condicoes',
            'severidade':  sev_map.get(ev['nivel'], 'ALERTA'),
            'icone':       ico_map.get(ev['nivel'], '🌪️'),
            'descricao':   ev['descricao'],
            'hora':        ev['hora'],
            'dados':       ev['metricas'],
            'nivel_risco': ev['nivel'],
            'concordancia': 0.85,
            'criado_em':   datetime.now().isoformat(),
        })

    # ── Microburst ────────────────────────────────────────────────────────────
    eventos_m = detector_tornado.analisar_microburst(dados_meteo, idx)
    for ev in eventos_m:
        sev_map = {'ALTO': 'PERIGO', 'MODERADO': 'ALERTA', 'BAIXO': 'ATENCAO'}
        alertas_out.append({
            'id':          f'microburst_{ev["hora"]}',
            'tipo':        'microburst',
            'subtipo':     'micro_explosao',
            'severidade':  sev_map.get(ev['nivel'], 'ALERTA'),
            'icone':       '💨🌪️',
            'descricao':   ev['descricao'],
            'hora':        ev['hora'],
            'dados':       ev['metricas'],
            'nivel_risco': ev['nivel'],
            'concordancia': 0.80,
            'criado_em':   datetime.now().isoformat(),
        })

    return alertas_out

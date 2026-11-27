"""
Motor de Explicação Meteorológica do RRFS.
Explica em linguagem simples E técnica por que o sistema
previu determinada condição ou emitiu um alerta.
"""
import logging
from datetime import datetime

logger = logging.getLogger('RRFS.Explicador')


# ═══════════════════════════════════════════════════════════════════
# BANCO DE CONHECIMENTO METEOROLÓGICO
# ═══════════════════════════════════════════════════════════════════

CONHECIMENTO = {

    'cape': {
        'nome':    'CAPE (Energia Potencial Convectiva Disponível)',
        'unidade': 'J/kg',
        'faixas': [
            (0,    300,  'fraco',    'atmosfera estável, baixo potencial convectivo'),
            (300,  1000, 'moderado', 'algum potencial convectivo, pancadas isoladas possíveis'),
            (1000, 2500, 'alto',     'bom potencial para tempestades, raios prováveis'),
            (2500, 4000, 'muito alto','tempestades severas prováveis, granizo e raios intensos'),
            (4000, 9999, 'extremo',  'condições para tempestades supercélulas e tornados'),
        ],
        'analogia': (
            'Pense no CAPE como a "gasolina" da tempestade. '
            'Quanto mais alto, mais energia disponível para o ar subir '
            'rapidamente e formar nuvens de tempestade gigantes (cumulonimbus).'
        ),
        'tecnico': (
            'O CAPE representa a integral vertical da força de empuxo positivo '
            'de uma parcela de ar desde o Nível de Convecção Livre (NCL) até o '
            'Nível de Equilíbrio (NE). Calculado como: '
            'CAPE = ∫(g × (Tv_parcela - Tv_ambiente) / Tv_ambiente) dz'
        ),
    },

    'lifted_index': {
        'nome':    'Índice de Levantamento (LI)',
        'unidade': '',
        'faixas': [
            (2,    99,  'estável',      'atmosfera estável, trovoadas improváveis'),
            (0,    2,   'ligeiramente', 'pequena instabilidade, trovoadas possíveis'),
            (-2,   0,   'instável',     'instabilidade moderada, trovoadas prováveis'),
            (-4,   -2,  'muito instável','tempestades severas prováveis'),
            (-99,  -4,  'extremamente', 'condições para tempestades violentas'),
        ],
        'analogia': (
            'O LI mede se uma "bolha" de ar quente consegue subir até 5km de altura. '
            'LI negativo = ar quente sobe facilmente = tempestade. '
            'Quanto mais negativo, mais violenta a tempestade tende a ser.'
        ),
        'tecnico': (
            'O LI é a diferença entre a temperatura do ambiente em 500hPa '
            'e a temperatura de uma parcela elevada adiabaticamente da superfície '
            'até 500hPa. LI = T_ambiente(500hPa) - T_parcela(500hPa). '
            'Valores negativos indicam instabilidade convectiva.'
        ),
    },

    'cin': {
        'nome':    'CIN (Inibição Convectiva)',
        'unidade': 'J/kg',
        'faixas': [
            (0,    -50,  'fraca',    'fácil de superar, convecção pode iniciar facilmente'),
            (-50,  -100, 'moderada', 'precisa de aquecimento diurno ou frente para superar'),
            (-100, -200, 'forte',    'difícil de superar, tempestades podem ser explosivas quando iniciam'),
            (-200, -9999,'muito forte','convecção muito improvável hoje'),
        ],
        'analogia': (
            'O CIN é como uma "tampa" na atmosfera que impede o ar quente de subir. '
            'Uma tampa forte impede trovoadas, mas se quebrar (ex: frente fria), '
            'a tempestade pode ser muito intensa pois a energia ficou acumulada.'
        ),
        'tecnico': (
            'O CIN é a integral da força de empuxo negativo abaixo do NCL. '
            'Representa a energia necessária para que uma parcela de superfície '
            'alcance o NCL. CIN forte com CAPE alto = "spring loaded atmosphere", '
            'condição clássica de tempestades severas no Brasil Central.'
        ),
    },

    'vento_jato': {
        'nome':    'Jato em Baixos Níveis',
        'unidade': 'km/h',
        'analogia': (
            'O jato em baixos níveis é como um "corredor de vento rápido" '
            'próximo ao chão (800-900m). Ele transporta calor e umidade do '
            'Pantanal/Amazônia para o Sudeste, sendo o "combustível" das '
            'tempestades noturnas no Brasil.'
        ),
        'tecnico': (
            'O Jato em Baixos Níveis da América do Sul (SALLJ) é um fenômeno '
            'típico que transporta umidade amazônica em direção ao sul. '
            'Velocidades >50 km/h em 850hPa indicam fluxo intenso de umidade '
            'e são associadas a Sistemas Convectivos de Mesoescala (SCM).'
        ),
    },

    'precipitacao': {
        'faixas': [
            (0,    0.5,  'sem chuva',      'precipitação desprezível'),
            (0.5,  2.5,  'chuva fraca',    'chuva leve ou garoa'),
            (2.5,  7.5,  'chuva moderada', 'chuva moderada'),
            (7.5,  15,   'chuva forte',    'chuva forte'),
            (15,   25,   'chuva muito forte','impactos no trânsito e alagamentos pontuais'),
            (25,   50,   'chuva intensa',  'risco de alagamentos e enxurradas'),
            (50,   9999, 'chuva extrema',  'risco grave de enchentes e deslizamentos'),
        ],
    },

    'vento': {
        'faixas': [
            (0,    20,  'calmo',         'vento fraco, sem impactos'),
            (20,   40,  'moderado',      'galhos se movem'),
            (40,   60,  'forte',         'dificuldade ao caminhar, galhos quebram'),
            (60,   80,  'muito forte',   'árvores tombam, danos a estruturas'),
            (80,   119, 'tempestade',    'danos significativos, risco de vida'),
            (119,  9999,'furacão/ciclone','destruição severa'),
        ],
    },

    'visibilidade': {
        'faixas': [
            (10000, 99999, 'excelente', 'visibilidade plena'),
            (4000,  10000, 'boa',       'boa visibilidade'),
            (1000,  4000,  'moderada',  'névoa ou chuva reduzindo visibilidade'),
            (200,   1000,  'ruim',      'nevoeiro, cuidado ao dirigir'),
            (0,     200,   'muito ruim','nevoeiro denso, perigo'),
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════
# GERADOR DE EXPLICAÇÕES
# ═══════════════════════════════════════════════════════════════════

class Explicador:

    @staticmethod
    def _classificar(valor, faixas):
        """Encontra a faixa correspondente ao valor."""
        for vmin, vmax, classe, desc in faixas:
            if vmin <= valor < vmax:
                return classe, desc
        return faixas[-1][2], faixas[-1][3]

    @staticmethod
    def _confianca_texto(concordancia: float, n_modelos: int) -> str:
        if n_modelos < 2:
            return 'baseado em 1 modelo (menor confiança)'
        pct = concordancia * 100
        if pct >= 90:
            return f'alta confiança — {n_modelos} de {n_modelos} modelos concordam ({pct:.0f}%)'
        elif pct >= 60:
            return f'confiança moderada — maioria dos modelos concorda ({pct:.0f}%)'
        else:
            return f'baixa confiança — modelos divergem ({pct:.0f}% de concordância)'

    # ── Explicação de Condição Atual ───────────────────────────────────────

    def explicar_condicao_atual(self, dados_meteo: dict) -> dict:
        """Explica as condições meteorológicas atuais."""
        current = dados_meteo.get('current', {})
        hourly  = dados_meteo.get('hourly', {})

        from modulos.meteorologia import hora_atual_indice, decodificar_wmo
        idx  = hora_atual_indice(dados_meteo)
        wmo  = current.get('weather_code', 0)
        desc_wmo, ico = decodificar_wmo(wmo)

        temp    = current.get('temperature_2m')
        sensacao= current.get('apparent_temperature')
        umidade = current.get('relative_humidity_2m')
        vento   = current.get('wind_speed_10m', 0)
        rajada  = current.get('wind_gusts_10m', 0)
        chuva   = current.get('precipitation', 0)
        cape_val= (hourly.get('cape') or [None] * (idx+1))[idx]
        nuvens  = current.get('cloud_cover', 0)

        # Monta explicação simples
        simples = []
        tecnico = []

        # Temperatura
        if temp is not None:
            dif_sens = round(sensacao - temp, 1) if sensacao else 0
            if dif_sens <= -3:
                simples.append(
                    f'A temperatura é {temp:.1f}°C, mas a sensação é de apenas '
                    f'{sensacao:.1f}°C — {abs(dif_sens):.0f}°C a menos por causa '
                    f'do vento e umidade.'
                )
                tecnico.append(
                    f'Temperatura de bulbo seco: {temp:.1f}°C. '
                    f'Temperatura aparente (índice Steadman): {sensacao:.1f}°C. '
                    f'Diferença de {abs(dif_sens):.1f}°C indica resfriamento eólico significativo.'
                )
            elif dif_sens >= 3:
                simples.append(
                    f'A temperatura é {temp:.1f}°C, mas a sensação é de '
                    f'{sensacao:.1f}°C — parece mais quente por causa da umidade alta.'
                )
                tecnico.append(
                    f'Temperatura: {temp:.1f}°C. Temperatura aparente: {sensacao:.1f}°C. '
                    f'Diferença positiva de {dif_sens:.1f}°C indica índice de calor elevado '
                    f'(efeito combinado de temperatura e umidade relativa de {umidade}%).'
                )
            else:
                simples.append(f'Temperatura de {temp:.1f}°C com sensação próxima à real.')

        # Umidade
        if umidade:
            if umidade >= 90:
                simples.append(
                    f'Umidade muito alta ({umidade}%) — ar abafado, '
                    f'chuva pode ocorrer a qualquer momento.'
                )
                tecnico.append(
                    f'Umidade relativa: {umidade}%. Próxima à saturação (100%). '
                    f'Ponto de orvalho elevado indica ar com grande quantidade '
                    f'de vapor d\'água disponível para precipitação.'
                )
            elif umidade <= 30:
                simples.append(
                    f'Umidade baixa ({umidade}%) — ar seco, '
                    f'hidratação e cuidado com queimadas.'
                )

        # Vento
        if vento > 20:
            _, desc_v = self._classificar(
                vento, CONHECIMENTO['vento']['faixas']
            )
            simples.append(
                f'Vento de {vento:.0f} km/h ({desc_v})'
                + (f' com rajadas de até {rajada:.0f} km/h.' if rajada > vento + 10 else '.')
            )
            tecnico.append(
                f'Velocidade do vento: {vento:.0f} km/h a 10m de altura. '
                + (f'Rajadas: {rajada:.0f} km/h. '
                   f'Razão rajada/vento médio: {rajada/vento:.1f}x '
                   f'(valores >1.5x indicam turbulência convectiva associada).'
                   if rajada > 0 else '')
            )

        # CAPE
        if cape_val and cape_val > 300:
            c_class, c_desc = self._classificar(
                cape_val, CONHECIMENTO['cape']['faixas']
            )
            simples.append(
                f'Há energia convectiva {c_class} na atmosfera '
                f'({cape_val:.0f} J/kg) — {c_desc}.'
            )
            tecnico.append(
                f'CAPE: {cape_val:.0f} J/kg. '
                + CONHECIMENTO['cape']['tecnico']
            )

        return {
            'simples': ' '.join(simples) or 'Condições normais no momento.',
            'tecnico': ' '.join(tecnico) or 'Sem anomalias técnicas relevantes.',
            'icone':   ico,
            'condicao': desc_wmo,
        }

    # ── Explicação de Previsão Horária ─────────────────────────────────────

    def explicar_hora(self, dados_meteo: dict, indice: int) -> dict:
        """Explica a previsão de uma hora específica."""
        h       = dados_meteo.get('hourly', {})
        times   = h.get('time', [])
        hora_str= times[indice][:16] if indice < len(times) else ''

        temp    = (h.get('temperature_2m')              or [])[indice]
        sensacao= (h.get('apparent_temperature')         or [])[indice]
        umidade = (h.get('relative_humidity_2m')         or [])[indice]
        prec    = (h.get('precipitation')                or [])[indice]
        prob    = (h.get('precipitation_probability')    or [])[indice]
        vento   = (h.get('wind_speed_10m')               or [])[indice]
        rajada  = (h.get('wind_gusts_10m')               or [])[indice]
        wmo     = (h.get('weather_code')                 or [])[indice]
        cape_v  = (h.get('cape')                         or [])[indice]
        li_v    = (h.get('lifted_index')                 or [])[indice]
        cin_v   = (h.get('convective_inhibition')        or [])[indice]
        vis     = (h.get('visibility')                   or [])[indice]
        uv      = (h.get('uv_index')                     or [])[indice]

        from modulos.meteorologia import decodificar_wmo
        desc_wmo, ico = decodificar_wmo(wmo or 0)

        partes_simples = []
        partes_tecnico = []

        # Temperatura
        if temp is not None:
            partes_simples.append(f'Temperatura prevista de {temp:.1f}°C')
            if sensacao and abs(sensacao - temp) >= 2:
                partes_simples[-1] += f' (sensação de {sensacao:.1f}°C)'
            partes_simples[-1] += '.'

        # Chuva
        if prob is not None and prob > 10:
            if prec and prec > 0:
                _, desc_p = self._classificar(
                    prec, CONHECIMENTO['precipitacao']['faixas']
                )
                partes_simples.append(
                    f'Chuva com {prob}% de probabilidade — {desc_p} '
                    f'({prec:.1f} mm esperados).'
                )
                partes_tecnico.append(
                    f'Probabilidade de precipitação: {prob}%. '
                    f'Acumulado previsto: {prec:.1f} mm/h. '
                    f'Classificação: {desc_p}.'
                )
            else:
                partes_simples.append(
                    f'Possibilidade de chuva de {prob}%, '
                    f'mas volume pequeno esperado.'
                )

        # Instabilidade
        if cape_v and cape_v > 500:
            c_class, c_desc = self._classificar(
                cape_v, CONHECIMENTO['cape']['faixas']
            )
            partes_simples.append(
                f'Instabilidade {c_class} na atmosfera: '
                f'há energia acumulada ({cape_v:.0f} J/kg) que pode '
                f'alimentar tempestades.'
            )
            partes_tecnico.append(
                f'CAPE: {cape_v:.0f} J/kg ({c_desc}). '
            )

            if li_v is not None:
                _, desc_li = self._classificar(
                    li_v, CONHECIMENTO['lifted_index']['faixas']
                )
                partes_simples.append(
                    f'O Índice de Levantamento ({li_v:.1f}) indica que '
                    f'o ar está {desc_li}.'
                )
                partes_tecnico.append(f'LI: {li_v:.1f}. ')

            if cin_v is not None and cin_v < -50:
                _, desc_cin = self._classificar(
                    cin_v, CONHECIMENTO['cin']['faixas']
                )
                partes_simples.append(
                    f'Há uma "tampa" atmosférica (CIN: {cin_v:.0f} J/kg) '
                    f'{desc_cin}. Se quebrar, a tempestade pode ser explosiva.'
                )
                partes_tecnico.append(f'CIN: {cin_v:.0f} J/kg. ')

        # Vento
        if vento and vento > 30:
            _, desc_v = self._classificar(
                vento, CONHECIMENTO['vento']['faixas']
            )
            partes_simples.append(
                f'Vento {desc_v}: {vento:.0f} km/h'
                + (f' com rajadas de {rajada:.0f} km/h.' if rajada else '.')
            )
            partes_tecnico.append(
                f'Vento previsto: {vento:.0f} km/h. '
                + (f'Rajadas: {rajada:.0f} km/h. ' if rajada else '')
            )

        # UV
        if uv and uv >= 8:
            partes_simples.append(
                f'Índice UV muito alto ({uv:.0f}) — proteção solar obrigatória.'
            )

        # Visibilidade
        if vis and vis < 4000:
            _, desc_vis = self._classificar(
                vis, CONHECIMENTO['visibilidade']['faixas']
            )
            partes_simples.append(
                f'Visibilidade {desc_vis}: {vis:.0f}m — cuidado ao dirigir.'
            )

        # Confiança do modelo
        partes_tecnico.append(
            f'Previsão gerada por combinação ponderada de modelos: '
            f'ECMWF IFS025 (40%), ICON Seamless (35%), GFS Seamless (25%).'
        )

        return {
            'hora':    hora_str,
            'icone':   ico,
            'condicao': desc_wmo,
            'simples': ' '.join(partes_simples) or f'Condições de {desc_wmo.lower()}.',
            'tecnico': ''.join(partes_tecnico) or 'Sem dados técnicos adicionais.',
        }

    # ── Explicação de Alerta ───────────────────────────────────────────────

    def explicar_alerta(self, alerta: dict, dados_meteo: dict,
                        dados_ensemble: dict = None) -> dict:
        """
        Gera explicação completa de por que um alerta foi emitido.
        Retorna {'simples': str, 'tecnico': str, 'fatores': list, 'confianca': str}
        """
        tipo        = alerta.get('tipo', '')
        dados_al    = alerta.get('dados', {})
        hora        = alerta.get('hora', '')
        concordancia= alerta.get('concordancia', 1.0)
        n_modelos   = alerta.get('n_modelos', 1)
        severidade  = alerta.get('severidade', 'ATENCAO')

        confianca_str = self._confianca_texto(concordancia, n_modelos)

        # Delega para método específico
        metodo = getattr(self, f'_explicar_{tipo}', self._explicar_generico)
        explicacao = metodo(dados_al, dados_meteo, hora)

        explicacao['confianca'] = confianca_str
        explicacao['severidade'] = severidade
        explicacao['tipo'] = tipo

        # Adiciona nota sobre janela de tempo
        if hora:
            try:
                dt_alerta = datetime.fromisoformat(hora)
                agora = datetime.now()
                delta_h = int((dt_alerta - agora).total_seconds() / 3600)
                if delta_h <= 0:
                    explicacao['janela'] = 'Condição atual ou muito próxima'
                elif delta_h <= 3:
                    explicacao['janela'] = f'Previsto para {delta_h}h a partir de agora'
                elif delta_h <= 12:
                    explicacao['janela'] = f'Previsto para esta {("manhã" if dt_alerta.hour < 12 else "tarde" if dt_alerta.hour < 18 else "noite")}'
                else:
                    explicacao['janela'] = f'Previsto para {dt_alerta.strftime("%d/%m às %H:%M")}'
            except Exception:
                explicacao['janela'] = hora[:16] if len(hora) >= 16 else hora

        return explicacao

    # ── Explicações específicas por tipo ───────────────────────────────────

    def _explicar_tempestade(self, dados: dict, meteo: dict, hora: str) -> dict:
        cape  = dados.get('cape', 0)
        li    = dados.get('li')
        wmo   = dados.get('wmo')

        fatores = []

        if cape >= 2500:
            fatores.append({
                'nome': f'CAPE Muito Alto: {cape:.0f} J/kg',
                'impacto': 'CRÍTICO',
                'simples': (
                    f'Há {cape:.0f} J/kg de energia convectiva disponível — '
                    f'equivale a combustível suficiente para tempestades severas. '
                    f'Para comparação: 1000 J/kg já produz trovoadas comuns.'
                ),
                'tecnico': (
                    f'CAPE de {cape:.0f} J/kg está na faixa de tempestades severas. '
                    f'Este valor está acima do percentil 90 climatológico para a região. '
                    f'Indica parcelas de ar com grande empuxo positivo, '
                    f'velocidades verticais potenciais > 70 m/s nas correntes ascendentes.'
                ),
            })
        elif cape >= 1000:
            fatores.append({
                'nome': f'CAPE Alto: {cape:.0f} J/kg',
                'impacto': 'ALTO',
                'simples': (
                    f'Energia convectiva de {cape:.0f} J/kg indica atmosfera '
                    f'carregada e propícia para tempestades com raios e rajadas.'
                ),
                'tecnico': f'CAPE: {cape:.0f} J/kg — faixa de instabilidade alta (1000-2500 J/kg).',
            })

        if li is not None and li <= -3:
            fatores.append({
                'nome': f'Índice de Levantamento Negativo: {li:.1f}',
                'impacto': 'ALTO' if li <= -4 else 'MODERADO',
                'simples': (
                    f'O Índice de Levantamento de {li:.1f} confirma que '
                    f'o ar quente próximo ao solo está muito mais instável que '
                    f'o ar acima. Isso "acelera" as correntes de ar que alimentam '
                    f'a tempestade para cima.'
                ),
                'tecnico': (
                    f'LI = {li:.1f}. Valores ≤ -3 indicam instabilidade severa. '
                    f'O LI representa a diferença de temperatura em 500hPa '
                    f'entre o ambiente e uma parcela elevada da superfície. '
                    f'LI negativo = parcela mais quente que o ambiente = empuxo positivo.'
                ),
            })

        if wmo in (95, 96, 99):
            fatores.append({
                'nome': f'Código de Tempo WMO: {wmo}',
                'impacto': 'CONFIRMAÇÃO',
                'simples': (
                    'O modelo meteorológico identificou diretamente '
                    'código de tempestade na previsão sinótica.'
                    + (' Com granizo.' if wmo in (96, 99) else '')
                ),
                'tecnico': (
                    f'Weather code WMO {wmo}: '
                    + ('thunderstorm with slight hail' if wmo == 96
                       else 'thunderstorm with heavy hail' if wmo == 99
                       else 'thunderstorm') + '. '
                    'Baseado na parametrização convectiva do modelo.'
                ),
            })

        simples = (
            f'O sistema detectou condições clássicas de tempestade para este horário. '
            f'Os principais indicadores são: '
            + ', '.join(f['nome'] for f in fatores) + '. '
            + 'Esses fatores combinados, quando confirmados por pelo menos 2 dos 3 modelos, '
            + 'disparam o alerta de tempestade.'
        )

        tecnico = (
            'Diagnóstico convectivo: '
            + ' | '.join(
                f"{f['nome']} [{f['impacto']}]" for f in fatores
            )
            + '. Método: análise multicritério com ponderação ECMWF/ICON/GFS. '
            + CONHECIMENTO['cape']['analogia']
        )

        return {
            'simples': simples,
            'tecnico': tecnico,
            'fatores': fatores,
        }

    def _explicar_raio(self, dados: dict, meteo: dict, hora: str) -> dict:
        prob  = dados.get('prob', 0)
        cape  = dados.get('cape', 0)

        fatores = [
            {
                'nome': f'Probabilidade de Raios: {prob:.0f}%',
                'impacto': 'ALTO' if prob >= 70 else 'MODERADO',
                'simples': (
                    f'O sistema calculou {prob:.0f}% de chance de raios '
                    f'com base na energia convectiva (CAPE: {cape:.0f} J/kg). '
                    f'Raios ocorrem quando correntes de ar dentro da nuvem '
                    f'criam separação de cargas elétricas.'
                ),
                'tecnico': (
                    f'Probabilidade estimada: {prob:.0f}%. '
                    f'Calculada por: prob = min(CAPE/30, 80) + |LI|×3 - ajuste_CIN. '
                    f'CAPE de {cape:.0f} J/kg indica correntes ascendentes '
                    f'suficientes para transporte de graupel e separação de carga '
                    f'elétrica dentro do cumulonimbus.'
                ),
            }
        ]

        simples = (
            f'Com {cape:.0f} J/kg de energia convectiva, '
            f'existe {prob:.0f}% de probabilidade de raios. '
            f'Raios são formados dentro de nuvens de tempestade quando '
            f'gelo e água interagem com as correntes de ar fortes, '
            f'criando eletricidade estática que se descarrega como raio.'
        )

        tecnico = (
            f'Diagnóstico de atividade elétrica: CAPE={cape:.0f} J/kg, '
            f'probabilidade={prob:.0f}%. '
            f'O mecanismo físico envolve colisões entre graupel '
            f'(gelo denso) e cristais de gelo na zona de temperatura mista '
            f'(-10°C a -25°C), gerando transferência de carga elétrica. '
            f'Campos elétricos > 100 kV/m resultam em descarga relâmpago.'
        )

        return {
            'simples': simples,
            'tecnico': tecnico,
            'fatores': fatores,
        }

    def _explicar_vento_forte(self, dados: dict, meteo: dict, hora: str) -> dict:
        gust = dados.get('gust', 0)
        wind = dados.get('wind', 0)

        _, desc_v = self._classificar(gust, CONHECIMENTO['vento']['faixas'])
        fatores = [
            {
                'nome': f'Rajada de {gust:.0f} km/h',
                'impacto': 'ALTO' if gust >= 80 else 'MODERADO',
                'simples': (
                    f'Rajadas de {gust:.0f} km/h previstas ({desc_v}). '
                    + ('Rajadas podem tombar árvores e causar danos a estruturas. '
                       if gust >= 70 else
                       'Dificuldade ao caminhar e galhos podem quebrar. ')
                ),
                'tecnico': (
                    f'Vento médio: {wind:.0f} km/h. '
                    f'Rajada máxima: {gust:.0f} km/h. '
                    f'Razão rajada/média: {gust/max(wind,1):.1f}x. '
                    f'Valores > 1.5x indicam turbulência convectiva (downdraft) '
                    f'ou gradiente de pressão associado a frente meteorológica.'
                ),
            }
        ]

        simples = (
            f'O modelo prevê rajadas de vento de {gust:.0f} km/h — {desc_v}. '
            f'Esse tipo de vento geralmente está associado a '
            + ('correntes descendentes de tempestades (downdrafts/microbursts).'
               if gust >= 70 else
               'sistemas de pressão ou passagem de frente fria.')
        )

        tecnico = (
            f'Diagnóstico de vento extremo: rajada {gust:.0f} km/h. '
            f'Mecanismo provável: '
            + ('outflow boundary de sistema convectivo (MCS) com '
               'evaporação de chuva em ar seco criando downdraft intenso. '
               if gust >= 70 else
               'forçante dinâmica por gradiente de pressão horizontal. ')
            + f'Extraído da variável wind_gusts_10m do modelo de superfície.'
        )

        return {
            'simples': simples,
            'tecnico': tecnico,
            'fatores': fatores,
        }

    def _explicar_chuva_forte(self, dados: dict, meteo: dict, hora: str) -> dict:
        mm = dados.get('mm', 0)
        _, desc_p = self._classificar(mm, CONHECIMENTO['precipitacao']['faixas'])

        fatores = [
            {
                'nome': f'Chuva de {mm:.1f} mm/h',
                'impacto': 'ALTO' if mm >= 25 else 'MODERADO',
                'simples': (
                    f'{mm:.1f} mm de chuva por hora é classificado como {desc_p}. '
                    + ('Em 1 hora com essa intensidade, podem ocorrer '
                       'alagamentos em áreas urbanas e enxurradas.'
                       if mm >= 25 else
                       'Trânsito pode ser afetado e visibilidade reduzida.')
                ),
                'tecnico': (
                    f'Taxa de precipitação: {mm:.1f} mm/h. '
                    f'Extraída de precipitation (convectiva + estratiforme). '
                    f'Para referência: 10mm/h já causa enxurradas em solos saturados, '
                    f'25mm/h é o limiar para chuva intensa pela OMM.'
                ),
            }
        ]

        simples = (
            f'Prevemos chuva de {mm:.1f} mm por hora — {desc_p}. '
            f'Para ter ideia: {mm:.0f} mm/h equivale a '
            f'{mm:.0f} litros de água por metro quadrado em uma hora.'
        )

        tecnico = (
            f'Precipitação prevista: {mm:.1f} mm/h. '
            f'Composta de rain ({mm:.1f}mm) + showers convectivos. '
            f'Limiar de alerta regional calibrado. '
            f'Risco de escoamento superficial em áreas impermeáveis '
            f'com i > 10mm/h (Equação Racional: Q = C × i × A).'
        )

        return {
            'simples': simples,
            'tecnico': tecnico,
            'fatores': fatores,
        }

    def _explicar_chuva_extrema(self, dados: dict, meteo: dict, hora: str) -> dict:
        mm = dados.get('mm', 0)
        fatores = [
            {
                'nome': f'Chuva EXTREMA: {mm:.1f} mm/h',
                'impacto': 'CRÍTICO',
                'simples': (
                    f'{mm:.1f} mm/h é uma chuva extremamente intensa — '
                    f'acima de 50mm/h qualquer área urbana entra em colapso '
                    f'de drenagem. Alto risco de enchentes e deslizamentos.'
                ),
                'tecnico': (
                    f'Taxa de precipitação: {mm:.1f} mm/h. '
                    f'Acima do percentil 99 para praticamente qualquer região. '
                    f'Período de retorno estimado > 25 anos. '
                    f'Risco iminente de flash floods e erosão de taludes.'
                ),
            }
        ]
        return {
            'simples': (
                f'Chuva extrema de {mm:.1f} mm/h prevista — isso é raro e perigoso. '
                f'Busque abrigo em local alto e seguro imediatamente.'
            ),
            'tecnico': (
                f'Precipitação convectiva extrema: {mm:.1f} mm/h. '
                f'Múscipalmente perigosa em bacias hidrográficas urbanas. '
                f'Limiar de alerta crítico definido em 50mm/h (COBRADE 1.3.2.1.3).'
            ),
            'fatores': fatores,
        }

    def _explicar_temperatura_alta(self, dados: dict, meteo: dict, hora: str) -> dict:
        temp = dados.get('temp', 0)
        fatores = [
            {
                'nome': f'Temperatura de {temp:.1f}°C',
                'impacto': 'ALTO' if temp >= 40 else 'MODERADO',
                'simples': (
                    f'Temperatura de {temp:.1f}°C é perigosa para a saúde. '
                    f'Risco de insolação e desidratação, especialmente '
                    f'para crianças, idosos e pessoas que trabalham ao ar livre.'
                ),
                'tecnico': (
                    f'Temperatura máxima prevista: {temp:.1f}°C. '
                    f'Índice de estresse térmico (WBGT estimado) em '
                    f'condições de sol pode ultrapassar 35°C. '
                    f'Protocolo de alerta ativado acima de {temp:.0f}°C '
                    f'conforme limiar calibrado para a região.'
                ),
            }
        ]
        return {
            'simples': (
                f'Temperatura muito alta prevista: {temp:.1f}°C. '
                f'Evite sol entre 10h-16h, beba água frequentemente '
                f'e não deixe crianças ou animais em carros.'
            ),
            'tecnico': (
                f'Temperatura: {temp:.1f}°C. '
                f'Verificar índice de calor (heat index) com umidade relativa. '
                f'Risco cardiovascular aumentado acima de 35°C WBGT.'
            ),
            'fatores': fatores,
        }

    def _explicar_temperatura_baixa(self, dados: dict, meteo: dict, hora: str) -> dict:
        temp = dados.get('temp', 0)
        return {
            'simples': (
                f'Temperatura baixa prevista: {temp:.1f}°C. '
                f'Agasalhe-se bem, especialmente crianças e idosos. '
                + ('Possibilidade de geada.' if temp <= 0 else '')
            ),
            'tecnico': (
                f'Temperatura mínima: {temp:.1f}°C. '
                + ('Ponto de congelamento atingido — geada possível '
                   'em superfícies expostas (radiação noturna).'
                   if temp <= 0 else
                   f'Abaixo do limiar de alerta frio ({temp:.0f}°C).')
            ),
            'fatores': [
                {
                    'nome': f'Temperatura de {temp:.1f}°C',
                    'impacto': 'ALTO' if temp <= 0 else 'MODERADO',
                    'simples': f'Temperatura muito baixa prevista.',
                    'tecnico': f'T2m: {temp:.1f}°C.',
                }
            ],
        }

    def _explicar_neve(self, dados: dict, meteo: dict, hora: str) -> dict:
        mm = dados.get('mm', 0)
        return {
            'simples': (
                f'Neve de {mm:.1f} mm prevista. Nas regiões Sul do Brasil '
                f'e regiões serranas, isso pode causar '
                f'fechamento de estradas e interrupção de serviços.'
            ),
            'tecnico': (
                f'Precipitação sólida (snowfall): {mm:.1f} mm equivalente de água. '
                f'Temperatura prevista abaixo de 0°C na camada de precipitação. '
                f'Isoterma de 0°C em nível suficientemente baixo '
                f'para neve atingir a superfície.'
            ),
            'fatores': [
                {
                    'nome': f'Neve: {mm:.1f} mm',
                    'impacto': 'ALTO',
                    'simples': 'Precipitação de neve prevista.',
                    'tecnico': f'Snowfall: {mm:.1f} mm.',
                }
            ],
        }

    def _explicar_granizo(self, dados: dict, meteo: dict, hora: str) -> dict:
        wmo = dados.get('wmo', 96)
        return {
            'simples': (
                'Granizo previsto junto com tempestade. '
                'O granizo forma-se quando gotas de água são carregadas '
                'para cima por correntes de ar dentro da nuvem, '
                'congelam e voltam a cair. '
                + ('Granizo grande — pode danificar veículos e plantações.'
                   if wmo == 99 else 'Granizo fino esperado.')
            ),
            'tecnico': (
                f'WMO code {wmo} (granizo {"intenso" if wmo == 99 else "leve"}). '
                f'Processo físico: updrafts > 25 m/s mantêm graupel em suspensão '
                f'por ciclos de acúmulo de gelo. CAPE elevado favorece '
                f'updrafts suficientes para granizo de diâmetro > 2cm.'
            ),
            'fatores': [
                {
                    'nome': 'Tempestade com granizo',
                    'impacto': 'CRÍTICO' if wmo == 99 else 'ALTO',
                    'simples': 'Granizo detectado nos modelos meteorológicos.',
                    'tecnico': f'WMO {wmo} — precipitação sólida convectiva.',
                }
            ],
        }

    def _explicar_nevoeiro(self, dados: dict, meteo: dict, hora: str) -> dict:
        vis = dados.get('vis', 500)
        return {
            'simples': (
                f'Nevoeiro previsto com visibilidade de apenas {vis}m. '
                f'Isso é menos que {'4 quarteirões de cidade' if vis < 400 else '5 quarteirões'}. '
                f'Redobre o cuidado ao dirigir, use faróis baixos.'
            ),
            'tecnico': (
                f'Visibilidade: {vis}m. Nevoeiro de radiação ou advecção. '
                f'Formado quando a temperatura do ar atinge o ponto de orvalho '
                f'próximo à superfície, condensando vapor d\'água. '
                f'Condição típica em noites claras após dias úmidos '
                f'(nevoeiro de radiação) ou chegada de massa de ar úmida.'
            ),
            'fatores': [
                {
                    'nome': f'Visibilidade de {vis}m',
                    'impacto': 'ALTO' if vis < 200 else 'MODERADO',
                    'simples': 'Nevoeiro reduzindo visibilidade.',
                    'tecnico': f'Vis: {vis}m.',
                }
            ],
        }

    def _explicar_furacao(self, dados: dict, meteo: dict, hora: str) -> dict:
        gust = dados.get('gust', 120)
        return {
            'simples': (
                f'ATENÇÃO MÁXIMA: Rajadas de {gust:.0f} km/h previstas — '
                f'equivalente a intensidade de furacão/ciclone extratropical. '
                f'Esses ventos derrubam árvores grandes, postes e estruturas. '
                f'Não saia de casa. Fique longe de janelas.'
            ),
            'tecnico': (
                f'Rajada extrema: {gust:.0f} km/h. '
                f'Escala Beaufort 12 (> 118 km/h). '
                f'Ciclone extratropical ou núcleo convectivo severo (supercélula). '
                f'Risco de destruição estrutural generalizada.'
            ),
            'fatores': [
                {
                    'nome': f'Rajada de {gust:.0f} km/h',
                    'impacto': 'EXTREMO',
                    'simples': 'Vento com força de furacão.',
                    'tecnico': f'Wind gust: {gust:.0f} km/h (Beaufort 12).',
                }
            ],
        }

    def _explicar_generico(self, dados: dict, meteo: dict, hora: str) -> dict:
        return {
            'simples': 'Condição meteorológica incomum detectada pelos modelos.',
            'tecnico': f'Dados: {dados}',
            'fatores': [],
        }

    # ── Explicação da Previsão Diária ─────────────────────────────────────

    def explicar_dia(self, dados_meteo: dict, indice_dia: int) -> dict:
        """Explica a previsão para um dia específico."""
        d = dados_meteo.get('daily', {})
        times = d.get('time', [])
        if indice_dia >= len(times):
            return {}

        dia_str = times[indice_dia]
        tmax    = (d.get('temperature_2m_max') or [])[indice_dia]
        tmin    = (d.get('temperature_2m_min') or [])[indice_dia]
        prob    = (d.get('precipitation_probability_max') or [])[indice_dia]
        prec    = (d.get('precipitation_sum') or [])[indice_dia]
        wmo     = (d.get('weather_code') or [])[indice_dia]
        vmax    = (d.get('wind_speed_10m_max') or [])[indice_dia]
        gmax    = (d.get('wind_gusts_10m_max') or [])[indice_dia]
        uv      = (d.get('uv_index_max') or [])[indice_dia]

        from modulos.meteorologia import decodificar_wmo
        desc_wmo, ico = decodificar_wmo(wmo or 0)

        # Confiança cai com o tempo
        try:
            dt_dia = datetime.fromisoformat(dia_str)
            delta_d = (dt_dia - datetime.now()).days
        except Exception:
            delta_d = 0

        confianca_pct = max(20, 95 - delta_d * 10)
        confianca_txt = (
            f'{confianca_pct}% de confiança para este dia '
            f'(previsões além de 5 dias têm confiança reduzida)'
        )

        partes = []
        partes.append(
            f'Previsão para {dia_str}: {desc_wmo}. '
            f'Temperatura entre {tmin:.0f if tmin else "--"}°C e '
            f'{tmax:.0f if tmax else "--"}°C.'
        )

        if prob and prob > 30:
            _, desc_p = self._classificar(
                prec or 0, CONHECIMENTO['precipitacao']['faixas']
            )
            partes.append(
                f'Chuva com {prob}% de probabilidade '
                f'({prec:.1f if prec else 0}mm acumulado — {desc_p}).'
            )

        if gmax and gmax >= 50:
            partes.append(f'Rajadas de vento de até {gmax:.0f} km/h.')

        if uv and uv >= 8:
            partes.append(f'Índice UV elevado ({uv:.0f}) — proteção solar necessária.')

        if delta_d >= 5:
            partes.append(
                '⚠️ Atenção: previsões acima de 5 dias têm precisão reduzida — '
                'considere como tendência, não certeza.'
            )

        return {
            'simples':    ' '.join(partes),
            'tecnico': (
                f'Dia {dia_str}. WMO {wmo}. '
                f'T2m max/min: {tmax}/{tmin}°C. '
                f'Prob. precip: {prob}%. Acum.: {prec}mm. '
                f'Wind max: {vmax} km/h. Gust max: {gmax} km/h. '
                f'UV max: {uv}. '
                f'Combinação ECMWF+ICON+GFS. {confianca_txt}.'
            ),
            'confianca': confianca_txt,
            'delta_dias': delta_d,
        }

    # ── Resumo Geral da Situação ──────────────────────────────────────────

    def resumo_situacao(self, dados_meteo: dict,
                        lista_alertas: list,
                        dados_ensemble: dict = None) -> dict:
        """
        Gera um resumo executivo da situação meteorológica.
        Explica o quadro geral para as próximas 24-48h.
        """
        if not dados_meteo:
            return {'simples': 'Sem dados disponíveis.', 'tecnico': ''}

        h = dados_meteo.get('hourly', {})
        from modulos.meteorologia import hora_atual_indice
        idx = hora_atual_indice(dados_meteo)

        # Coleta dados das próximas 48h
        cape_max = max(
            (v for v in (h.get('cape') or [])[idx:idx+48] if v), default=0
        )
        prec_max = max(
            (v for v in (h.get('precipitation') or [])[idx:idx+48] if v), default=0
        )
        gust_max = max(
            (v for v in (h.get('wind_gusts_10m') or [])[idx:idx+48] if v), default=0
        )
        prob_max = max(
            (v for v in (h.get('precipitation_probability') or [])[idx:idx+24] if v), default=0
        )

        n_alertas   = len(lista_alertas)
        alertas_sev = [a for a in lista_alertas if a.get('severidade') in ('PERIGO','EXTREMO')]

        # Determina tom geral
        if alertas_sev or cape_max >= 2500:
            tom = 'ATENÇÃO'
            emoji = '⚠️'
        elif n_alertas > 0 or cape_max >= 1000:
            tom = 'MONITORAMENTO'
            emoji = '🔔'
        else:
            tom = 'FAVORÁVEL'
            emoji = '✅'

        # Simples
        partes_s = [f'{emoji} Situação geral: {tom}.']

        if tom == 'FAVORÁVEL':
            partes_s.append(
                'As condições meteorológicas estão estáveis para as próximas 24-48h. '
                'Não há sinais significativos de tempestades ou fenômenos severos.'
            )
        else:
            if cape_max >= 1000:
                c_class, _ = self._classificar(
                    cape_max, CONHECIMENTO['cape']['faixas']
                )
                partes_s.append(
                    f'Há energia convectiva {c_class} na atmosfera '
                    f'(CAPE máx: {cape_max:.0f} J/kg) que pode gerar tempestades.'
                )
            if prec_max >= 10:
                partes_s.append(
                    f'Chuvas de até {prec_max:.0f} mm/h são possíveis.'
                )
            if gust_max >= 50:
                partes_s.append(
                    f'Rajadas de até {gust_max:.0f} km/h previstas.'
                )
            if n_alertas > 0:
                partes_s.append(
                    f'{n_alertas} alerta(s) emitido(s) para as próximas 24h.'
                )

        # Técnico
        n_modelos = len((dados_ensemble or {}).get('modelos', {}))
        partes_t = [
            f'Análise multicritério — {n_modelos or 1} modelo(s) numérico(s). '
            f'CAPE máx 48h: {cape_max:.0f} J/kg. '
            f'Precip máx: {prec_max:.1f} mm/h. '
            f'Rajada máx: {gust_max:.0f} km/h. '
            f'Prob. chuva máx 24h: {prob_max:.0f}%. '
            f'Total alertas: {n_alertas} ({len(alertas_sev)} severos).'
        ]

        return {
            'tom':     tom,
            'emoji':   emoji,
            'simples': ' '.join(partes_s),
            'tecnico': ' '.join(partes_t),
        }


# Instância global
explicador = Explicador()

"""
Dados meteorológicos via Open-Meteo (gratuito, sem chave).
Tempo atual + previsão 7 dias + hourly detalhado.
"""
import requests
import logging
from datetime import datetime
from modulos.cache_manager import cache

logger = logging.getLogger('RRFS.Meteo')

BASE = 'https://api.open-meteo.com/v1'

HOURLY_VARS = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'apparent_temperature', 'precipitation_probability',
    'precipitation', 'rain', 'showers', 'snowfall', 'snow_depth',
    'weather_code', 'cloud_cover', 'cloud_cover_low',
    'cloud_cover_mid', 'cloud_cover_high',
    'visibility', 'wind_speed_10m', 'wind_speed_80m',
    'wind_direction_10m', 'wind_gusts_10m',
    'cape', 'lifted_index', 'convective_inhibition',
    'freezing_level_height', 'soil_temperature_0cm',
    'uv_index', 'is_day',
]

CURRENT_VARS = [
    'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
    'is_day', 'precipitation', 'rain', 'showers', 'snowfall',
    'weather_code', 'cloud_cover', 'wind_speed_10m',
    'wind_direction_10m', 'wind_gusts_10m', 'cape',
]

DAILY_VARS = [
    'weather_code', 'temperature_2m_max', 'temperature_2m_min',
    'apparent_temperature_max', 'apparent_temperature_min',
    'sunrise', 'sunset', 'uv_index_max',
    'precipitation_sum', 'rain_sum', 'showers_sum', 'snowfall_sum',
    'precipitation_hours', 'precipitation_probability_max',
    'wind_speed_10m_max', 'wind_gusts_10m_max',
    'wind_direction_10m_dominant', 'shortwave_radiation_sum',
]

# Variáveis disponíveis em cada modelo do ensemble
# Nem todos os modelos retornam todas as variáveis
ENSEMBLE_VARS = [
    'temperature_2m',
    'precipitation',
    'wind_speed_10m',
    'wind_gusts_10m',
    'cape',
    'precipitation_probability',
]

# Metadados dos modelos
MODELOS_META = {
    'ecmwf_ifs025': {
        'nome':     'ECMWF IFS025',
        'sigla':    'ECMWF',
        'pais':     'Europa',
        'flag':     '🇪🇺',
        'peso':     0.40,
        'cor':      '#58a6ff',
        'resolucao':'0.25°',
        'ciclos':   '2× ao dia',
        'descricao':'European Centre for Medium-Range Weather Forecasts',
    },
    'icon_seamless': {
        'nome':     'ICON Seamless',
        'sigla':    'ICON',
        'pais':     'Alemanha',
        'flag':     '🇩🇪',
        'peso':     0.35,
        'cor':      '#d29922',
        'resolucao':'0.1°',
        'ciclos':   '4× ao dia',
        'descricao':'Icosahedral Nonhydrostatic Model — DWD Alemanha',
    },
    'gfs_seamless': {
        'nome':     'GFS Seamless',
        'sigla':    'GFS',
        'pais':     'EUA',
        'flag':     '🇺🇸',
        'peso':     0.25,
        'cor':      '#3fb950',
        'resolucao':'0.25°',
        'ciclos':   '4× ao dia',
        'descricao':'Global Forecast System — NOAA/NCEP USA',
    },
}


def buscar_tempo_completo(lat: float, lon: float) -> dict | None:
    """Busca tempo atual + previsão horária + diária."""
    key_kw = dict(lat=round(lat, 3), lon=round(lon, 3))
    cached = cache.get('tempo_atual', **key_kw)
    if cached:
        return cached
    try:
        r = requests.get(
            f'{BASE}/forecast',
            params={
                'latitude':        lat,
                'longitude':       lon,
                'current':         ','.join(CURRENT_VARS),
                'hourly':          ','.join(HOURLY_VARS),
                'daily':           ','.join(DAILY_VARS),
                'timezone':        'auto',
                'forecast_days':   7,
                'wind_speed_unit': 'kmh',
            },
            timeout=15
        )
        r.raise_for_status()
        dados = r.json()
        dados['_atualizado'] = datetime.now().isoformat()
        cache.set('tempo_atual', dados, **key_kw)
        return dados
    except Exception as e:
        logger.error(f'Open-Meteo erro: {e}')
        return None


def buscar_ensemble(lat: float, lon: float) -> dict | None:
    """
    Busca 3 modelos numéricos via Open-Meteo.
    Retorna dados reais de cada modelo separadamente.
    """
    key_kw = dict(lat=round(lat, 3), lon=round(lon, 3))
    cached = cache.get('modelos', **key_kw)
    if cached:
        return cached

    modelos_dados = {}

    for modelo in MODELOS_META.keys():
        try:
            r = requests.get(
                f'{BASE}/forecast',
                params={
                    'latitude':        lat,
                    'longitude':       lon,
                    'hourly':          ','.join(ENSEMBLE_VARS),
                    'models':          modelo,
                    'timezone':        'auto',
                    'forecast_days':   7,
                    'wind_speed_unit': 'kmh',
                },
                timeout=15
            )
            if r.status_code == 200:
                dados_modelo = r.json()
                dados_modelo['_modelo'] = modelo
                dados_modelo['_ok']     = True
                modelos_dados[modelo]   = dados_modelo
            else:
                logger.warning(
                    f'Modelo {modelo} retornou HTTP {r.status_code}'
                )
                modelos_dados[modelo] = {
                    '_modelo': modelo,
                    '_ok':     False,
                    '_erro':   f'HTTP {r.status_code}',
                }
        except Exception as e:
            logger.warning(f'Modelo {modelo} falhou: {e}')
            modelos_dados[modelo] = {
                '_modelo': modelo,
                '_ok':     False,
                '_erro':   str(e),
            }

    # Conta quantos tiveram sucesso
    n_ok = sum(1 for d in modelos_dados.values() if d.get('_ok'))
    if n_ok == 0:
        logger.error('Nenhum modelo ensemble respondeu')
        return None

    resultado = {
        'modelos':     modelos_dados,
        'n_ok':        n_ok,
        'n_total':     len(MODELOS_META),
        '_atualizado': datetime.now().isoformat(),
    }
    cache.set('modelos', resultado, **key_kw)
    return resultado


def extrair_valor_modelo(dados_modelo: dict, variavel: str,
                          idx: int) -> float | None:
    """
    Extrai o valor de uma variável em um índice horário
    de um modelo específico.
    Retorna None se não disponível.
    """
    if not dados_modelo or not dados_modelo.get('_ok'):
        return None
    try:
        hourly = dados_modelo.get('hourly', {})
        arr    = hourly.get(variavel, [])
        if idx < len(arr) and arr[idx] is not None:
            return float(arr[idx])
        return None
    except Exception:
        return None


def extrair_janela_modelo(dados_modelo: dict, variavel: str,
                           idx_inicio: int, n: int = 6) -> list:
    """
    Extrai uma janela de valores de uma variável.
    Retorna lista de valores (None onde não disponível).
    """
    if not dados_modelo or not dados_modelo.get('_ok'):
        return [None] * n
    try:
        hourly = dados_modelo.get('hourly', {})
        arr    = hourly.get(variavel, [])
        janela = arr[idx_inicio: idx_inicio + n]
        # Preenche com None se menor que n
        while len(janela) < n:
            janela.append(None)
        return [float(v) if v is not None else None for v in janela]
    except Exception:
        return [None] * n


def calcular_concordancia_real(ensemble: dict, variavel: str,
                                idx: int,
                                limiar: float | None = None) -> dict:
    """
    Calcula concordância REAL entre os 3 modelos para uma variável
    em um momento específico.

    Retorna:
    {
        'valores':     {modelo: valor},
        'media':       float,
        'desvio':      float,
        'concordancia': float (0-1),
        'todos_ok':    bool,
        'n_com_dados': int,
    }
    """
    if not ensemble:
        return {
            'valores': {}, 'media': None, 'desvio': None,
            'concordancia': 0.5, 'todos_ok': False, 'n_com_dados': 0,
        }

    modelos = ensemble.get('modelos', {})
    valores = {}

    for nome_modelo, dados_modelo in modelos.items():
        v = extrair_valor_modelo(dados_modelo, variavel, idx)
        if v is not None:
            valores[nome_modelo] = v

    n = len(valores)
    if n == 0:
        return {
            'valores': {}, 'media': None, 'desvio': None,
            'concordancia': 0.5, 'todos_ok': False, 'n_com_dados': 0,
        }

    vals  = list(valores.values())
    media = sum(vals) / n

    # Desvio padrão
    if n > 1:
        desvio = (sum((v - media) ** 2 for v in vals) / n) ** 0.5
    else:
        desvio = 0.0

    # Concordância: quanto menor o desvio relativo, maior a concordância
    if media != 0:
        cv = desvio / abs(media)      # coeficiente de variação
        concordancia = max(0.0, min(1.0, 1.0 - cv))
    else:
        concordancia = 1.0 if desvio < 0.1 else 0.5

    # Se tiver limiar: concordância baseada em quantos superam o limiar
    if limiar is not None:
        n_superam = sum(1 for v in vals if v >= limiar)
        concordancia_limiar = n_superam / len(MODELOS_META)
        # Média ponderada: 60% concordância de valor, 40% de limiar
        concordancia = 0.6 * concordancia + 0.4 * concordancia_limiar

    return {
        'valores':      valores,
        'media':        round(media, 2),
        'desvio':       round(desvio, 2),
        'concordancia': round(concordancia, 3),
        'todos_ok':     n == len(MODELOS_META),
        'n_com_dados':  n,
    }


def resumo_ensemble_atual(ensemble: dict, idx: int) -> dict:
    """
    Gera resumo REAL dos 3 modelos para o momento atual.
    Usado pela rota /api/ensemble/detalhado
    """
    if not ensemble:
        return {'disponivel': False, 'motivo': 'Ensemble não carregado'}

    modelos = ensemble.get('modelos', {})
    resultado = {
        'disponivel':  True,
        'n_ok':        ensemble.get('n_ok', 0),
        'n_total':     ensemble.get('n_total', 3),
        'atualizado':  ensemble.get('_atualizado', ''),
        'modelos_meta': MODELOS_META,
        'variaveis':   {},
        'concordancia_geral': 0.0,
    }

    # Variáveis a comparar com limiares para alerta
    vars_config = {
        'temperature_2m': {
            'nome':    'Temperatura',
            'unidade': '°C',
            'limiar':  None,
            'fmt':     lambda v: f'{v:.1f}',
        },
        'precipitation': {
            'nome':    'Precipitação',
            'unidade': 'mm/h',
            'limiar':  10.0,
            'fmt':     lambda v: f'{v:.1f}',
        },
        'wind_speed_10m': {
            'nome':    'Vento',
            'unidade': 'km/h',
            'limiar':  50.0,
            'fmt':     lambda v: f'{v:.0f}',
        },
        'wind_gusts_10m': {
            'nome':    'Rajada',
            'unidade': 'km/h',
            'limiar':  60.0,
            'fmt':     lambda v: f'{v:.0f}',
        },
        'cape': {
            'nome':    'CAPE',
            'unidade': 'J/kg',
            'limiar':  1000.0,
            'fmt':     lambda v: f'{v:.0f}',
        },
        'precipitation_probability': {
            'nome':    'Prob. Chuva',
            'unidade': '%',
            'limiar':  70.0,
            'fmt':     lambda v: f'{v:.0f}',
        },
    }

    concordancias = []

    for var_key, var_cfg in vars_config.items():
        # Extrai valores REAIS de cada modelo
        valores_reais = {}
        for modelo_key, dados_modelo in modelos.items():
            v = extrair_valor_modelo(dados_modelo, var_key, idx)
            if v is not None:
                valores_reais[modelo_key] = v

        if not valores_reais:
            resultado['variaveis'][var_key] = {
                'nome':        var_cfg['nome'],
                'unidade':     var_cfg['unidade'],
                'disponivel':  False,
                'valores':     {},
                'media':       None,
                'desvio':      None,
                'concordancia':0.5,
                'limiar':      var_cfg['limiar'],
            }
            continue

        # Calcula estatísticas reais
        vals  = list(valores_reais.values())
        n     = len(vals)
        media = sum(vals) / n
        desvio= (sum((v - media) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0

        # Concordância real
        if media != 0:
            cv           = desvio / abs(media)
            concordancia = max(0.0, min(1.0, 1.0 - cv))
        else:
            concordancia = 1.0 if desvio < 0.1 else 0.5

        # Limiar
        limiar_atingido = False
        n_superam       = 0
        if var_cfg['limiar'] is not None:
            n_superam       = sum(1 for v in vals if v >= var_cfg['limiar'])
            limiar_atingido = n_superam > 0
            concordancia_l  = n_superam / len(MODELOS_META)
            concordancia    = 0.6 * concordancia + 0.4 * concordancia_l

        concordancia = round(concordancia, 3)
        concordancias.append(concordancia)

        # Formata valores por modelo para exibição
        valores_formatados = {}
        for modelo_key, v in valores_reais.items():
            meta = MODELOS_META.get(modelo_key, {})
            valores_formatados[modelo_key] = {
                'valor':      round(v, 2),
                'formatado':  var_cfg['fmt'](v) + ' ' + var_cfg['unidade'],
                'supera_limiar': var_cfg['limiar'] is not None and v >= var_cfg['limiar'],
                'nome':       meta.get('nome', modelo_key),
                'sigla':      meta.get('sigla', modelo_key),
                'flag':       meta.get('flag', ''),
                'peso':       meta.get('peso', 0.33),
                'cor':        meta.get('cor', '#888'),
            }

        resultado['variaveis'][var_key] = {
            'nome':             var_cfg['nome'],
            'unidade':          var_cfg['unidade'],
            'disponivel':       True,
            'valores':          valores_formatados,
            'media':            round(media, 2),
            'media_formatada':  var_cfg['fmt'](media) + ' ' + var_cfg['unidade'],
            'desvio':           round(desvio, 2),
            'concordancia':     concordancia,
            'limiar':           var_cfg['limiar'],
            'limiar_atingido':  limiar_atingido,
            'n_superam_limiar': n_superam,
            'n_modelos':        n,
            'divergencia':      desvio > abs(media) * 0.15 if media != 0 else False,
        }

    # Concordância geral
    if concordancias:
        resultado['concordancia_geral'] = round(
            sum(concordancias) / len(concordancias), 3
        )

    return resultado


def combinar_modelos(dados_ensemble: dict) -> dict:
    """Combina previsões dos 3 modelos (média ponderada)."""
    if not dados_ensemble:
        return {}

    modelos = dados_ensemble.get('modelos', {})
    if not modelos:
        return {}

    pesos = {
        'ecmwf_ifs025':  0.40,
        'icon_seamless': 0.35,
        'gfs_seamless':  0.25,
    }

    vars_combinar = [
        'temperature_2m', 'precipitation', 'wind_speed_10m',
        'wind_gusts_10m', 'cape', 'precipitation_probability'
    ]

    combinado = {'times': [], 'vars': {v: [] for v in vars_combinar}}
    ref = next(
        (d for d in modelos.values() if d.get('_ok') and 'hourly' in d),
        None
    )

    if not ref:
        return {}

    times = ref['hourly'].get('time', [])
    combinado['times'] = times

    for var in vars_combinar:
        n          = len(times)
        acumulado  = [0.0] * n
        soma_pesos = 0.0

        for modelo, dados in modelos.items():
            if not dados.get('_ok'):
                continue
            peso = pesos.get(modelo, 0.33)
            vals = dados.get('hourly', {}).get(var, [None] * n)
            for i, v in enumerate(vals[:n]):
                if v is not None:
                    acumulado[i] += float(v) * peso
            soma_pesos += peso

        if soma_pesos > 0:
            combinado['vars'][var] = [
                round(v / soma_pesos, 2) for v in acumulado
            ]

    return combinado

def resumo_ensemble_por_hora(ensemble: dict, idx_hora: int) -> dict:
    """
    Retorna valores REAIS de cada modelo para uma hora específica futura.
    Usado para mostrar concordância real nos alertas.

    idx_hora: índice no array horário (pode ser hora futura)
    """
    if not ensemble:
        return {'disponivel': False}

    modelos = ensemble.get('modelos', {})

    # Variáveis relevantes para alertas
    vars_alerta = {
        'cape': {
            'nome': 'CAPE', 'unidade': 'J/kg',
            'limiares': {'baixo': 300, 'moderado': 1000,
                         'alto': 2500, 'extremo': 4000},
        },
        'precipitation': {
            'nome': 'Precipitação', 'unidade': 'mm/h',
            'limiares': {'baixo': 2, 'moderado': 10,
                         'alto': 25, 'extremo': 50},
        },
        'wind_gusts_10m': {
            'nome': 'Rajada', 'unidade': 'km/h',
            'limiares': {'baixo': 30, 'moderado': 50,
                         'alto': 80, 'extremo': 119},
        },
        'wind_speed_10m': {
            'nome': 'Vento', 'unidade': 'km/h',
            'limiares': {'baixo': 20, 'moderado': 40,
                         'alto': 60, 'extremo': 90},
        },
        'precipitation_probability': {
            'nome': 'Prob. Chuva', 'unidade': '%',
            'limiares': {'baixo': 20, 'moderado': 50,
                         'alto': 70, 'extremo': 90},
        },
    }

    resultado = {
        'disponivel': True,
        'idx_hora':   idx_hora,
        'variaveis':  {},
        'modelos_status': {},
        'concordancia_geral': 0.0,
    }

    concordancias = []

    # Status de cada modelo
    for modelo_key, dados_modelo in modelos.items():
        meta = MODELOS_META.get(modelo_key, {})
        ok   = dados_modelo.get('_ok', False)
        resultado['modelos_status'][modelo_key] = {
            'ok':    ok,
            'nome':  meta.get('nome',  modelo_key),
            'sigla': meta.get('sigla', modelo_key),
            'flag':  meta.get('flag',  ''),
            'peso':  meta.get('peso',  0.33),
            'cor':   meta.get('cor',   '#888'),
            'erro':  dados_modelo.get('_erro', ''),
        }

    for var_key, var_cfg in vars_alerta.items():
        valores_reais = {}

        for modelo_key, dados_modelo in modelos.items():
            if not dados_modelo.get('_ok'):
                continue
            v = extrair_valor_modelo(dados_modelo, var_key, idx_hora)
            if v is not None:
                valores_reais[modelo_key] = v

        if not valores_reais:
            resultado['variaveis'][var_key] = {
                'nome':        var_cfg['nome'],
                'unidade':     var_cfg['unidade'],
                'disponivel':  False,
                'valores':     {},
                'media':       None,
                'concordancia':0.5,
            }
            continue

        vals  = list(valores_reais.values())
        n     = len(vals)
        media = sum(vals) / n
        desvio= (sum((v - media) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0

        # Concordância por coeficiente de variação
        if media != 0:
            cv           = desvio / abs(media)
            concordancia = max(0.0, min(1.0, 1.0 - cv))
        else:
            concordancia = 1.0 if desvio < 0.5 else 0.5

        concordancias.append(concordancia)

        # Classifica cada modelo pelo limiar
        limiares = var_cfg['limiares']
        def classificar(v):
            if v >= limiares['extremo']:  return 'extremo',  '#ff6b6b'
            if v >= limiares['alto']:     return 'alto',     '#f85149'
            if v >= limiares['moderado']: return 'moderado', '#d29922'
            if v >= limiares['baixo']:    return 'baixo',    '#58a6ff'
            return 'normal', '#3fb950'

        valores_formatados = {}
        for modelo_key, v in valores_reais.items():
            meta   = MODELOS_META.get(modelo_key, {})
            classe, cor_val = classificar(v)
            fmt = f'{v:.0f}' if abs(v) >= 10 else f'{v:.1f}'
            valores_formatados[modelo_key] = {
                'valor':      round(v, 2),
                'formatado':  fmt + ' ' + var_cfg['unidade'],
                'classe':     classe,
                'cor':        cor_val,
                'nome':       meta.get('nome',  modelo_key),
                'sigla':      meta.get('sigla', modelo_key),
                'flag':       meta.get('flag',  ''),
                'peso':       meta.get('peso',  0.33),
                'cor_modelo': meta.get('cor',   '#888'),
            }

        # Verifica se modelos concordam na classificação
        classes = [v['classe'] for v in valores_formatados.values()]
        todos_mesma_classe = len(set(classes)) == 1

        resultado['variaveis'][var_key] = {
            'nome':             var_cfg['nome'],
            'unidade':          var_cfg['unidade'],
            'disponivel':       True,
            'valores':          valores_formatados,
            'media':            round(media, 2),
            'media_formatada':  (f'{media:.0f}' if abs(media) >= 10
                                 else f'{media:.1f}') + ' ' + var_cfg['unidade'],
            'desvio':           round(desvio, 2),
            'concordancia':     round(concordancia, 3),
            'todos_mesma_classe': todos_mesma_classe,
            'divergencia':      not todos_mesma_classe or desvio > abs(media) * 0.15,
            'limiares':         limiares,
        }

    if concordancias:
        resultado['concordancia_geral'] = round(
            sum(concordancias) / len(concordancias), 3
        )

    return resultado


# Códigos WMO
WMO_CODES = {
    0:  ('Céu limpo',               '☀️'),
    1:  ('Predominantemente limpo',  '🌤️'),
    2:  ('Parcialmente nublado',     '⛅'),
    3:  ('Nublado',                  '☁️'),
    45: ('Nevoeiro',                 '🌫️'),
    48: ('Nevoeiro com geada',       '🌫️'),
    51: ('Garoa leve',               '🌦️'),
    53: ('Garoa moderada',           '🌦️'),
    55: ('Garoa intensa',            '🌧️'),
    61: ('Chuva leve',               '🌧️'),
    63: ('Chuva moderada',           '🌧️'),
    65: ('Chuva forte',              '🌧️'),
    71: ('Neve leve',                '🌨️'),
    73: ('Neve moderada',            '🌨️'),
    75: ('Neve intensa',             '❄️'),
    77: ('Grãos de neve',            '🌨️'),
    80: ('Pancadas leves',           '🌦️'),
    81: ('Pancadas moderadas',       '🌧️'),
    82: ('Pancadas violentas',       '⛈️'),
    85: ('Pancadas de neve',         '🌨️'),
    86: ('Pancadas de neve intensa', '❄️'),
    95: ('Tempestade',               '⛈️'),
    96: ('Tempestade c/ granizo leve','⛈️'),
    99: ('Tempestade c/ granizo forte','⛈️'),
}


def decodificar_wmo(codigo: int) -> tuple:
    return WMO_CODES.get(codigo, ('Desconhecido', '❓'))


def hora_atual_indice(dados: dict) -> int:
    """Retorna o índice da hora atual nos dados horários."""
    try:
        times = dados['hourly']['time']
        agora = datetime.now().strftime('%Y-%m-%dT%H:00')
        for i, t in enumerate(times):
            if t == agora:
                return i
        return 0
    except Exception:
        return 0

"""
Ambiente favorável a fenômenos que NÃO dá para prever diretamente: microexplosão (downburst) e tornado.

Eles têm 1-4 km: menores que a grade dos modelos (11-25 km) e que o radar gratuito, e nenhum dado aberto aqui
tem velocidade do vento no radar (Doppler) nem helicidade. O que se pode fazer, e é o que meteorologistas fazem
na vigilância, é medir os INGREDIENTES do ambiente e dizer "o ambiente favorece". Isto é:

  * um índice 0-100 de ingredientes (parecido, em espírito, com os parâmetros compostos do SPC/NOAA);
  * NÃO é probabilidade, NÃO é previsão de que vai ocorrer, e NÃO é verificado pelo sistema (não há "verdade"
    aberta e confiável de tornado/microexplosão para conferir). Sempre rotulado assim na tela.
  * só aparece se também houver risco de tempestade (fator de convecção), senão ambiente favorável sozinho
    não significa nada.

Microexplosão: CAPE alto + ar seco em níveis médios (evaporação -> corrente descendente) + gradiente térmico
  forte + superfície quente.
Tornado (estilo "STP sem helicidade"): CAPE x base baixa das nuvens (LCL) x cisalhamento 0-6 km x
  cisalhamento de baixos níveis. Sem helicidade e sem CIN confiável: serve para destacar o ambiente, não o evento.
"""
import numpy as np
import pandas as pd

NOTE = ('Indicador de AMBIENTE (ingredientes), não é previsão de que o fenômeno ocorrerá e não é verificado pelo '
        'sistema: tornados e microexplosões são menores que a resolução dos modelos e do radar gratuito. '
        'Siga os alertas oficiais (Defesa Civil/INMET).')


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def level_of(pct) -> str:
    if pct is None or not np.isfinite(pct):
        return 'NENHUM'
    return 'ALTO' if pct >= 60 else 'MODERADO' if pct >= 35 else 'BAIXO' if pct >= 15 else 'NENHUM'


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Colunas do df (qualquer uma pode faltar): cape, temperature, dew_point, lapse75, rh700, shear06, shear01,
    ev_thunderstorm, ev_lightning. Devolve env_downburst e env_tornado (0-100, NaN onde faltam ingredientes).
    """
    n = len(df)
    nan = np.full(n, np.nan)

    def col(name):
        return df[name].to_numpy(dtype=float) if name in df else nan.copy()
    cape, T, Td = col('cape'), col('temperature'), col('dew_point')
    lapse, rh700, sh06, sh01 = col('lapse75'), col('rh700'), col('shear06'), col('shear01')
    thunder = np.fmax(np.nan_to_num(col('ev_thunderstorm')), np.nan_to_num(col('ev_lightning')))
    conv = _clip01(thunder / 50.0)                       # sem risco de tempestade, o ambiente não conta

    # microexplosão: precisa de CAPE; cada ingrediente que falta é ignorado (média dos disponíveis)
    cape_t = _clip01((cape - 1000.0) / 2000.0)
    parts, w = [], []
    for term, weight in ((_clip01((50.0 - rh700) / 30.0), 0.4), (_clip01((lapse - 6.0) / 2.5), 0.3),
                         (_clip01((T - 28.0) / 8.0), 0.3)):
        parts.append(np.nan_to_num(term)), w.append(np.where(np.isfinite(term), weight, 0.0))
    num = sum(p * ww for p, ww in zip(parts, w))
    den = sum(w)
    with np.errstate(invalid='ignore', divide='ignore'):
        ingredients = np.where(den > 0, num / den, np.nan)
    down = np.where(np.isfinite(cape), cape_t * ingredients, np.nan) * conv * 100.0

    # tornado: todos os 4 ingredientes são necessários
    lcl = 125.0 * (T - Td)                               # altura da base das nuvens (m), aproximação
    tor = (np.clip(cape / 2000.0, 0, 1.5) * _clip01((2000.0 - lcl) / 1000.0) *
           np.clip((sh06 - 12.0) / 12.0, 0, 1.5) * np.clip((sh01 - 6.0) / 8.0, 0, 1.5))
    tor = np.clip(tor, 0.0, 1.0) * conv * 100.0
    return pd.DataFrame({'env_downburst': down, 'env_tornado': tor}, index=df.index)


def summarize(df: pd.DataFrame, start: int, end: int) -> dict:
    """Máximo de cada índice na janela [start, end) e hora do pico."""
    out = {'note': NOTE}
    for key, col, label in (('downburst', 'env_downburst', 'Microexplosão / rajadas destrutivas'),
                            ('tornado', 'env_tornado', 'Tornado')):
        s = df[col].iloc[start:end] if col in df else None
        if s is None or not s.notna().any():
            out[key] = {'label': label, 'pct': None, 'level': 'NENHUM', 'available': False}
            continue
        pct = float(s.max())
        out[key] = {'label': label, 'pct': round(pct, 0), 'level': level_of(pct), 'available': True,
                    'peak_hour': s.idxmax().strftime('%H:%M') if pct >= 15 else None}
    return out

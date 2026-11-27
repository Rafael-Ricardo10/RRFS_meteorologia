import re
import hashlib
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IAAgent:
    def __init__(self):
        self.ollama_host = Config.OLLAMA_HOST
        self.ollama_url = f"{Config.OLLAMA_HOST}/api/generate"
        self.model = Config.OLLAMA_MODEL
        self.conversation_history = []
        self.last_error = None

    def set_model(self, model_name: str):
        """
        Atualiza o modelo LLM ativo em tempo real
        """
        self.model = model_name
        logger.info(f"Modelo IA atualizado para: {model_name}")

    def get_current_model(self) -> str:
        """
        Retorna o modelo atualmente em uso
        """
        return self.model

    def get_available_models(self) -> List[Dict]:
        """
        Lista todos os modelos instalados no Ollama
        """
        try:
            url = f"{self.ollama_host}/api/tags"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                models = []
                for m in data.get('models', []):
                    models.append({
                        'name': m.get('name'),
                        'size': m.get('size', 0),
                        'modified_at': m.get('modified_at', '')
                    })
                return models
            else:
                logger.error(f"Erro ao listar modelos Ollama: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"Erro ao conectar com Ollama: {str(e)}")
            return []

    def check_ollama_status(self) -> bool:
        """
        Verifica se o Ollama está rodando e acessível
        """
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _format_weather_context(self, predictions: Dict, radar_data: Dict = None) -> str:
        """
        Formata contexto meteorológico para análise
        """
        context = f"""
ANÁLISE METEOROLÓGICA
Timestamp: {datetime.utcnow().isoformat()}

PREVISÕES:
- Temperatura: {predictions.get('temperature', 'N/A')}°C (confiança: {predictions.get('temperature_confidence', 0)*100:.1f}%)
- Precipitação: {predictions.get('precipitation', 0):.1f}mm (confiança: {predictions.get('precipitation_confidence', 0)*100:.1f}%)
- Vento: {predictions.get('wind_speed', 0):.1f} km/h (confiança: {predictions.get('wind_confidence', 0)*100:.1f}%)
- Umidade: {predictions.get('humidity', 0):.1f}% (confiança: {predictions.get('humidity_confidence', 0)*100:.1f}%)

EVENTOS EXTREMOS:
- Probabilidade Tempestade: {predictions.get('thunderstorm_probability', 0):.1f}%
- Probabilidade Chuva Forte: {predictions.get('heavy_rain_probability', 0):.1f}%
- Probabilidade Granizo: {predictions.get('hail_probability', 0):.1f}%
- Probabilidade Raios: {predictions.get('lightning_probability') or 0:.1f}%
- Probabilidade Tornado: não estimável com os dados disponíveis
- Probabilidade Vento Extremo: {predictions.get('extreme_wind_probability', 0):.1f}%
"""

        if radar_data:
            context += f"\nVALIDAÇÃO POR RADAR: {json.dumps(radar_data, ensure_ascii=False)[:600]}"

        return context

    @staticmethod
    def analysis_signature(predictions: Dict, daily: List[Dict] = None) -> str:
        """
        Assinatura dos dados que alimentam a análise (valores arredondados).
        Se nada relevante mudou entre duas atualizações, a assinatura é a mesma e a resposta
        anterior do agente continua valendo (o LLM não é consultado de novo).
        """
        def r(v, step):
            return None if v is None else round(float(v) / step) * step
        p = predictions or {}
        sig = {
            'T': r(p.get('temperature'), 0.5), 'U': r(p.get('humidity'), 5),
            'R24': r(p.get('precipitation'), 1), 'W': r(p.get('wind_speed'), 3),
            'G': r(p.get('wind_gusts'), 5), 'P': r(p.get('pressure'), 1),
            'ev': {k: r(p.get(f'{k}_probability'), 10) for k in
                   ('lightning', 'thunderstorm', 'hail', 'extreme_wind', 'heavy_rain')},
            'radar': (p.get('radar_validation') or {}).get('storm_status'),
            'near': {k: r(p.get(f'{k}_near_probability'), 10) for k in ('lightning', 'thunderstorm', 'hail')},
            'sev': (p.get('storm_watch') or {}).get('level'),
            'days': [[d.get('date'), r(d.get('temp_max'), 1), r(d.get('temp_min'), 1),
                      r(d.get('precipitation_sum'), 2), (d.get('storm') or {}).get('level')]
                     for d in (daily or [])[:7]],
        }
        return hashlib.sha1(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def _format_forecast_context(self, location_name: str, predictions: Dict, daily: List[Dict],
                                 radar_data: Dict = None) -> str:
        p = predictions or {}

        def f(v, nd=0, unit=''):
            return 'n/d' if v is None else f"{float(v):.{nd}f}{unit}"
        lines = [f"LOCAL: {location_name}",
                 "CONDIÇÕES AGORA (previsão do sistema, já corrigida pelo aprendizado):",
                 f"- Temperatura {f(p.get('temperature'),1,'°C')}, umidade {f(p.get('humidity'),0,'%')}, "
                 f"vento {f(p.get('wind_speed'),0,' km/h')} (rajadas {f(p.get('wind_gusts'),0,' km/h')}), "
                 f"pressão {f(p.get('pressure'),0,' hPa')}, nuvens {f(p.get('cloud_cover'),0,'%')}",
                 f"- Chuva prevista nas próximas 24h: {f(p.get('precipitation'),1,' mm')}",
                 "RISCO DE EVENTOS (próximas 24h, probabilidade calibrada pelo histórico do local):",
                 f"- Raios {f(p.get('lightning_probability'),0,'%')}, tempestade {f(p.get('thunderstorm_probability'),0,'%')}, "
                 f"granizo {f(p.get('hail_probability'),0,'%')}, vento extremo {f(p.get('extreme_wind_probability'),0,'%')}, "
                 f"chuva forte {f(p.get('heavy_rain_probability'),0,'%')}. Tornado: não estimável com estes dados."]
        if p.get('lightning_near_probability') is not None:
            lines.append(f"- NAS REDONDEZAS (até ~30 km, a tempestade pode passar ao lado do ponto): raios {f(p.get('lightning_near_probability'),0,'%')}, "
                         f"tempestade {f(p.get('thunderstorm_near_probability'),0,'%')}, granizo {f(p.get('hail_near_probability'),0,'%')}")
        sw = p.get('storm_watch') or {}
        if sw.get('level') and sw.get('level') != 'NENHUM':
            lines.append(f"VIGIA DE TEMPESTADE (radar em dBZ + raios): nível {sw['level']} - " + '; '.join(sw.get('reasons') or []))
        rv = radar_data or {}
        if rv.get('available'):
            lines.append(f"RADAR: {rv.get('message')} (eco {f((rv.get('coverage_10') or 0)*100,0,'%')} perto, "
                         f"{f((rv.get('coverage_30') or 0)*100,0,'%')} na região; tendência: {rv.get('trend') or 'n/d'})")
            mo = rv.get('motion') or {}
            if mo.get('text'):
                lines.append(f"MOVIMENTO DO ECO NO RADAR: {mo['text']}")
        else:
            lines.append("RADAR: indisponível agora (alertas só pelos modelos).")
        lines.append("PRÓXIMOS DIAS:")
        for d in (daily or [])[:7]:
            st = d.get('storm') or {}
            storm = ''
            if st.get('level') not in (None, 'NENHUM'):
                storm = f" | tempestade {st.get('level')}: {st.get('type') or ''} {st.get('organization') or ''}".rstrip()
            lines.append(f"- {d.get('date')}: {f(d.get('temp_min'),0)}°–{f(d.get('temp_max'),0)}°C, "
                         f"chuva {f(d.get('precipitation_sum'),1,' mm')} ({f(d.get('precipitation_probability_max'),0,'%')}), "
                         f"rajadas até {f(d.get('wind_gusts_max'),0,' km/h')}, {d.get('weather_description') or ''}{storm}")
        sk = (p.get('skill') or {}).get('storm') or {}
        if sk:
            lines.append("DESEMPENHO DO SISTEMA EM TEMPESTADES (histórico verificado): "
                         + json.dumps(sk.get('summary') or {}, ensure_ascii=False)[:300])
        return '\n'.join(lines)

    def analyze_predictions(self, predictions: Dict, processed_data: Dict = None,
                            radar_data: Dict = None, daily: List[Dict] = None,
                            location_name: str = '') -> Dict:
        """
        Gera um boletim meteorológico em português a partir dos DADOS REAIS do sistema.
        (O LLM resume e interpreta; não inventa números nem substitui a previsão.)
        """
        try:
            context = self._format_forecast_context(location_name or 'local selecionado',
                                                    predictions, daily, radar_data)
            prompt = f"""Você é um meteorologista. Escreva em português do Brasil, de forma objetiva, usando SOMENTE os dados abaixo (não invente valores).

{context}

Escreva o boletim com estes tópicos curtos:
1. Situação agora e nas próximas horas
2. Tempestades: há risco? de que tipo (raios, granizo, vento, chuva forte) e em que dia/horário
3. Próximos dias (destaque mudanças importantes)
4. Grau de incerteza (cite onde a confiança é menor)
5. Recomendações práticas para a população"""

            response = self._query_ollama(prompt, max_tokens=700)
            if not response:
                return {'analysis': '', 'error': self.last_error or 'Sem resposta do modelo'}
            return {'analysis': response, 'timestamp': datetime.utcnow().isoformat(),
                    'model_used': self.model}

        except Exception as e:
            logger.error(f"Erro na análise IA: {str(e)}")
            return {'analysis': '', 'error': str(e)}

    def confirm_extreme_event(self, event_type: str, probability: float,
                             evidence: Dict) -> Dict:
        """
        Confirma eventos extremos com análise IA
        """
        try:
            prompt = f"""Evento Extremo: {event_type}
Probabilidade: {probability:.1f}%

Evidências:
{json.dumps(evidence, indent=2)}

Confirmação: Este evento extremo ocorrerá?
Responda com:
1. SIM/NÃO/TALVEZ
2. Justificativa técnica
3. Nível de certeza (0-100%)"""

            response = self._query_ollama(prompt)

            return {
                'event_type': event_type,
                'confirmed': bool(re.search(r'\bSIM\b', response.upper())),
                'analysis': response,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Erro na confirmação de evento: {str(e)}")
            return {'confirmed': False, 'error': str(e)}

    def calibrate_models(self, learning_data: List[Dict]) -> Dict:
        """
        Usa aprendizado acumulado para calibrar modelos
        """
        try:
            recent_accuracy = learning_data[-100:] if len(learning_data) > 100 else learning_data

            avg_accuracy = sum(d.get('accuracy', 0) for d in recent_accuracy) / len(recent_accuracy) if recent_accuracy else 0

            prompt = f"""Análise de Desempenho de Modelo
Acurácia Média Recente: {avg_accuracy*100:.1f}%
Eventos Analisados: {len(recent_accuracy)}

Dados de Aprendizado (últimos eventos):
{json.dumps(recent_accuracy[-5:], indent=2, default=str)}

Recomendações para melhoria:
1. Ajustes de parâmetros
2. Features a enfatizar
3. Estratégias de melhora
4. Próximos passos"""

            response = self._query_ollama(prompt)

            return {
                'calibration': response,
                'current_accuracy': avg_accuracy,
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Erro na calibração: {str(e)}")
            return {'error': str(e)}

    def _query_ollama(self, prompt: str, max_tokens: int = 700) -> str:
        """
        Consulta o Ollama SEM limite de tempo de resposta (servidor pequeno pode demorar minutos).
        Usa streaming: a conexão fica ativa enquanto o modelo gera e o texto é montado aos poucos.
        Só existe um limite para ESTABELECER a conexão (10 s), para não travar se o servidor estiver fora.
        """
        self.last_error = None
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "keep_alive": "30m",
                "options": {"temperature": 0.4, "top_p": 0.9, "num_predict": max_tokens},
            }
            parts = []
            # timeout=(conexão, leitura): leitura None = espera quanto for preciso
            with requests.post(self.ollama_url, json=payload, stream=True, timeout=(10, None)) as response:
                if response.status_code != 200:
                    self.last_error = f"Ollama respondeu HTTP {response.status_code}: {response.text[:200]}"
                    logger.error(self.last_error)
                    return ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get('error'):
                        self.last_error = str(chunk['error'])
                        logger.error(f"Erro Ollama: {self.last_error}")
                        return ""
                    parts.append(chunk.get('response', ''))
                    if chunk.get('done'):
                        break
            return ''.join(parts).strip()

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Erro ao consultar Ollama: {str(e)}")
            return ""

    def get_recommendations(self, location_name: str, predictions: Dict) -> str:
        """
        Gera recomendações para o usuário
        """
        try:
            prompt = f"""Você é um meteorologista. Gere recomendações para: {location_name}

Condições Previstas:
{json.dumps(predictions, indent=2)}

Forneça:
1. Recomendações de segurança
2. Atividades recomendadas/não recomendadas
3. Avisos importantes
Seja conciso e prático."""

            return self._query_ollama(prompt, max_tokens=500)

        except Exception as e:
            logger.error(f"Erro ao gerar recomendações: {str(e)}")
            return "Recomendações não disponíveis"

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
- Probabilidade Tornado: {predictions.get('tornado_probability', 0):.1f}%
- Probabilidade Vento Extremo: {predictions.get('extreme_wind_probability', 0):.1f}%
"""

        if radar_data:
            context += f"\nDADOS DE RADAR: {json.dumps(radar_data, indent=2)[:500]}..."

        return context

    def analyze_predictions(self, predictions: Dict, processed_data: Dict = None,
                           radar_data: Dict = None) -> Dict:
        """
        Usa LLM para analisar e confirmar previsões
        """
        try:
            context = self._format_weather_context(predictions, radar_data)

            prompt = f"""{context}

Analise as previsões acima e forneça:
1. Confirmação da precisão das previsões
2. Observações sobre concordância entre modelos
3. Indicação de possíveis eventos extremos confirmados
4. Recomendações de ajustes nas previsões
5. Nível geral de confiança na previsão (0-100%)

Seja conciso e técnico."""

            response = self._query_ollama(prompt)

            return {
                'analysis': response,
                'timestamp': datetime.utcnow().isoformat(),
                'model_used': self.model
            }

        except Exception as e:
            logger.error(f"Erro na análise IA: {str(e)}")
            return {
                'analysis': 'Análise não disponível',
                'error': str(e)
            }

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
                'confirmed': 'SIM' in response.upper(),
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

    def _query_ollama(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Consulta Ollama com prompt usando o modelo ativo
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_predict": max_tokens
            }

            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('response', 'Sem resposta')
            else:
                logger.error(f"Erro Ollama: {response.status_code}")
                return ""

        except Exception as e:
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

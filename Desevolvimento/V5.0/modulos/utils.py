import hashlib
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Utils:
    @staticmethod
    def hash_password(password: str) -> str:
        """
        Gera hash de senha
        """
        return hashlib.sha256(password.encode()).hexdigest()

    @staticmethod
    def verify_password(password: str, hash_pwd: str) -> bool:
        """
        Verifica senha com hash
        """
        return hashlib.sha256(password.encode()).hexdigest() == hash_pwd

    @staticmethod
    def format_prediction_time(hours: int) -> str:
        """
        Formata tempo de previsão
        """
        if hours < 24:
            return f"{hours} horas"
        else:
            days = hours // 24
            return f"{days} dias"

    @staticmethod
    def calculate_forecast_accuracy(actual: dict, predicted: dict) -> float:
        """
        Calcula acurácia de previsão
        """
        try:
            differences = []

            for key in actual:
                if key in predicted:
                    actual_val = actual[key]
                    predicted_val = predicted[key]

                    if actual_val is not None and predicted_val is not None:
                        diff = abs(actual_val - predicted_val)
                        differences.append(diff)

            if not differences:
                return 0.0

            # Usar erro médio percentual absoluto (MAPE)
            mape = sum(differences) / len(differences)
            accuracy = max(0, 1 - (mape / 100))

            return accuracy

        except Exception as e:
            logger.error(f"Erro ao calcular acurácia: {str(e)}")
            return 0.0

    @staticmethod
    def get_weather_alert_level(event_type: str, probability: float) -> str:
        """
        Determina nível de alerta baseado em probabilidade
        """
        if probability >= 80:
            return "CRÍTICO"
        elif probability >= 60:
            return "ALTO"
        elif probability >= 40:
            return "MODERADO"
        elif probability > 0:
            return "BAIXO"
        else:
            return "NENHUM"

    @staticmethod
    def celsius_to_fahrenheit(celsius: float) -> float:
        """
        Converte Celsius para Fahrenheit
        """
        return (celsius * 9/5) + 32

    @staticmethod
    def kmh_to_mph(kmh: float) -> float:
        """
        Converte km/h para mph
        """
        return kmh * 0.621371

    @staticmethod
    def mm_to_inches(mm: float) -> float:
        """
        Converte mm para polegadas
        """
        return mm / 25.4

    @staticmethod
    def get_compass_direction(degrees: int) -> str:
        """
        Converte graus em direção do vento
        """
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                     'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        index = round(degrees / 22.5) % 16
        return directions[index]

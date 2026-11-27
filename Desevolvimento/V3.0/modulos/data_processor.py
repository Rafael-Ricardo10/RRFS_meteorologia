import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        self.weather_codes = {
            0: 'Clear', 1: 'Mainly Clear', 2: 'Partly Cloudy', 3: 'Overcast',
            45: 'Foggy', 48: 'Depositing Rime Fog',
            51: 'Light Drizzle', 53: 'Moderate Drizzle', 55: 'Dense Drizzle',
            61: 'Slight Rain', 63: 'Moderate Rain', 65: 'Heavy Rain',
            71: 'Slight Snow', 73: 'Moderate Snow', 75: 'Heavy Snow', 77: 'Snow Grains',
            80: 'Slight Rain Showers', 81: 'Moderate Rain Showers', 82: 'Violent Rain Showers',
            85: 'Slight Snow Showers', 86: 'Heavy Snow Showers',
            95: 'Thunderstorm', 96: 'Thunderstorm with Slight Hail', 99: 'Thunderstorm with Heavy Hail'
        }

    def process_openmeteo_data(self, raw_data: Dict) -> Dict:
        """
        Converte o JSON de cada modelo em DataFrames numéricos e calcula
        'now_idx' = linha da hora atual (no fuso da localização).
        """
        processed = {}
        for model_name, model_data in (raw_data or {}).items():
            try:
                if not model_data or 'hourly' not in model_data:
                    continue
                offset = int(model_data.get('utc_offset_seconds', 0) or 0)
                hourly = self._process_hourly_data(model_data['hourly'])
                if hourly.empty:
                    continue
                processed[model_name] = {
                    'hourly': hourly,
                    'daily': self._process_daily_data(model_data.get('daily', {}) or {}),
                    'now_idx': self.current_index(hourly, offset),
                    'extremes': self._identify_extremes(hourly),
                    'model_info': {
                        'timezone': model_data.get('timezone', 'UTC'),
                        'utc_offset_seconds': offset,
                        'elevation': model_data.get('elevation'),
                    }
                }
            except Exception as e:
                logger.error(f"Erro ao processar modelo {model_name}: {str(e)}")
        return processed

    @staticmethod
    def _numeric_frame(data: Dict) -> pd.DataFrame:
        df = pd.DataFrame(data)
        if 'time' not in df.columns:
            return pd.DataFrame()
        df['time'] = pd.to_datetime(df['time'])
        for c in df.columns:
            if c != 'time':
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df

    def _process_hourly_data(self, hourly: Dict) -> pd.DataFrame:
        return self._numeric_frame(hourly)

    def _process_daily_data(self, daily: Dict) -> pd.DataFrame:
        return self._numeric_frame(daily)

    @staticmethod
    def current_index(df: pd.DataFrame, utc_offset_seconds: int) -> int:
        """Índice da hora mais próxima de 'agora' (a API devolve desde 00:00 de hoje)."""
        now_local = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=utc_offset_seconds)
        diffs = (df['time'] - now_local).abs()
        return int(diffs.values.argmin())

    @staticmethod
    def _identify_extremes(df: pd.DataFrame) -> Dict:
        def safe(col, fn):
            return float(getattr(df[col], fn)()) if col in df and df[col].notna().any() else None
        return {
            'max_temp': safe('temperature_2m', 'max'), 'min_temp': safe('temperature_2m', 'min'),
            'max_wind': safe('wind_gusts_10m', 'max'), 'max_precip': safe('precipitation', 'max'),
        }

    def compare_models(self, processed_data: Dict) -> Dict:
        comparison = {}
        for key, col in (('temperature', 'temperature_2m'), ('precipitation', 'precipitation')):
            vals = []
            for md in processed_data.values():
                df = md['hourly']
                i = md.get('now_idx', 0)
                if col in df.columns:
                    vals.append(df[col].iloc[i:i + 24].mean())
            vals = np.array([v for v in vals if pd.notna(v)])
            if len(vals):
                comparison[key] = {'mean': float(vals.mean()), 'std': float(vals.std()),
                                   'divergence': float(vals.max() - vals.min())}
        return comparison

    def calculate_weather_code_description(self, code: int) -> str:
        return self.weather_codes.get(code, f'Unknown ({code})')

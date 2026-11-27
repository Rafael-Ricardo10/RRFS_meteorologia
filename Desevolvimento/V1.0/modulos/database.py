from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import json

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    approved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Location(db.Model):
    __tablename__ = 'locations'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    city_name = db.Column(db.String(120), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    country = db.Column(db.String(120))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_favorite = db.Column(db.Boolean, default=False)

class WeatherData(db.Model):
    __tablename__ = 'weather_data'

    id = db.Column(db.Integer, primary_key=True)
    location_id = db.Column(db.Integer, db.ForeignKey('locations.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    model = db.Column(db.String(50), nullable=False)  # gfs, ecmwf, icon
    raw_data = db.Column(db.JSON, nullable=False)
    processed_data = db.Column(db.JSON)
    predictions = db.Column(db.JSON)

class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    location_id = db.Column(db.Integer, db.ForeignKey('locations.id'), nullable=False)
    timestamp_prediction = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Condições previstas
    temperature = db.Column(db.Float)
    temperature_confidence = db.Column(db.Float)

    humidity = db.Column(db.Float)
    humidity_confidence = db.Column(db.Float)

    precipitation = db.Column(db.Float)
    precipitation_confidence = db.Column(db.Float)

    wind_speed = db.Column(db.Float)
    wind_confidence = db.Column(db.Float)

    # Eventos extremos
    thunderstorm_probability = db.Column(db.Float, default=0.0)
    heavy_rain_probability = db.Column(db.Float, default=0.0)
    hail_probability = db.Column(db.Float, default=0.0)
    tornado_probability = db.Column(db.Float, default=0.0)
    extreme_wind_probability = db.Column(db.Float, default=0.0)

    # Análise IA
    ia_analysis = db.Column(db.Text)
    ia_confidence = db.Column(db.Float)
    ia_observations = db.Column(db.JSON)

    # Radar
    radar_confirmation = db.Column(db.Boolean, default=False)
    radar_data = db.Column(db.JSON)

class LearningData(db.Model):
    __tablename__ = 'learning_data'

    id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(50), nullable=False)
    features = db.Column(db.JSON, nullable=False)
    actual_outcome = db.Column(db.JSON, nullable=False)
    predicted_outcome = db.Column(db.JSON)
    accuracy = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class RadarCache(db.Model):
    __tablename__ = 'radar_cache'

    id = db.Column(db.Integer, primary_key=True)
    location_id = db.Column(db.Integer, db.ForeignKey('locations.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    radar_url = db.Column(db.String(500))
    radar_data = db.Column(db.JSON)
    last_updated = db.Column(db.DateTime)


class SystemSettings(db.Model):
    """
    Armazena configurações globais do sistema,
    como qual modelo LLM do Ollama está ativo
    """
    __tablename__ = 'system_settings'

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), unique=True, nullable=False)
    value = db.Column(db.String(500))
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @staticmethod
    def get_value(key, default=None):
        setting = SystemSettings.query.filter_by(key=key).first()
        return setting.value if setting else default

    @staticmethod
    def set_value(key, value):
        setting = SystemSettings.query.filter_by(key=key).first()
        if setting:
            setting.value = value
            setting.updated_at = datetime.utcnow()
        else:
            setting = SystemSettings(key=key, value=value)
            db.session.add(setting)
        db.session.commit()
        return setting

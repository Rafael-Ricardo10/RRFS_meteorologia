from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from modulos import db, User, Location, WeatherData, Prediction, LearningData, RadarCache, SystemSettings
from modulos import APIManager, DataProcessor, PredictionEngine, IAAgent, CacheManager, Utils
from modulos.api_manager import get_weather_description
from config import Config
from functools import wraps
from datetime import datetime, timedelta, timezone
from modulos.adaptive_learning import AdaptiveLearning
from modulos.forecast_builder import build_forecast
from modulos import radar_validator
from modulos.scheduler import HourlyUpdater
import asyncio
import queue
import shutil
import threading
import time
import logging
import math
import os
import json
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
if Config.SECRET_KEY == 'rrfs-meteorologia-2024':
    logging.getLogger(__name__).warning('SECRET_KEY padrão em uso: defina a variável de ambiente SECRET_KEY.')

db.init_app(app)
CORS(app)

api_manager = APIManager()
data_processor = DataProcessor()
prediction_engine = PredictionEngine()
ia_agent = IAAgent()
cache_manager = CacheManager()

for d in ('dados/cache', 'dados/models', 'dados/raw', 'dados/processed'):
    os.makedirs(d, exist_ok=True)

with app.app_context():
    db.create_all()

    admin = User.query.filter_by(username=Config.ADMIN_USERNAME).first()
    if not admin:
        admin = User(username=Config.ADMIN_USERNAME, email='admin@rrfs.local',
                     is_admin=True, approved=True)
        admin.set_password(Config.ADMIN_PASSWORD_DEFAULT)
        db.session.add(admin)
        db.session.commit()
        logger.info("Usuário admin criado")
        if Config.ADMIN_PASSWORD_DEFAULT == 'admin':
            logger.warning('Senha padrão do admin em uso: defina ADMIN_PASSWORD ou troque após o login.')

    saved_model = SystemSettings.get_value('ollama_model')
    if saved_model:
        ia_agent.set_model(saved_model)
        logger.info(f"Modelo LLM carregado das configurações: {saved_model}")
    else:
        SystemSettings.set_value('ollama_model', Config.OLLAMA_MODEL)
        logger.info(f"Modelo LLM padrão definido: {Config.OLLAMA_MODEL}")



# ==================== HELPERS ====================

def sanitize(obj):
    """Converte tipos numpy e NaN/Inf (JSON inválido) em tipos nativos / None."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def gkey(loc) -> str:
    """
    Chave do LOCAL (não do usuário): usuários que cadastram a mesma cidade compartilham cache,
    aprendizado e análise de IA, sem repetir busca nas APIs nem reprocessar.
    (~1 km de resolução: 2 casas decimais)
    """
    return f"g{float(loc.latitude):.2f}_{float(loc.longitude):.2f}"


_KEY_LOCKS, _KEY_LOCKS_GUARD = {}, threading.Lock()


def key_lock(key: str) -> threading.Lock:
    with _KEY_LOCKS_GUARD:
        return _KEY_LOCKS.setdefault(key, threading.Lock())


# O cache vale até a PRÓXIMA atualização horária (+10 min de folga), nunca menos que o configurado
TTL = max(int(getattr(Config, 'FORECAST_CACHE_TIMEOUT', 3600)), int(Config.UPDATE_INTERVAL_MINUTES) * 60 + 600)
# "Atualizar agora" repetido por vários usuários não deve martelar as APIs
FORCE_MIN_AGE = int(getattr(Config, 'FORCE_MIN_AGE_SECONDS', 300))


def migrate_learning_files():
    """Copia o aprendizado antigo (por id de localização) para a chave geográfica compartilhada."""
    try:
        for loc in Location.query.all():
            for suffix in ('learning', 'pending'):
                src = f'dados/processed/location_{loc.id}_{suffix}.json'
                dst = f'dados/processed/location_{gkey(loc)}_{suffix}.json'
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)
                    logger.info(f"Aprendizado migrado: {src} -> {dst}")
    except Exception as e:
        logger.error(f"Falha ao migrar aprendizado: {e}")


with app.app_context():
    migrate_learning_files()


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def wants_force():
    return request.args.get('force') == '1'


async def get_climate_models(latitude, longitude, location_id, force=False):  # location_id = chave do local (gkey)
    """Dados brutos de todos os modelos, com cache (compartilhado por forecast e timeseries)."""
    if not force:
        cached = cache_manager.get(location_id, 'climate_raw', ttl=TTL)
        if cached:
            return json.loads(cached)
    data = await api_manager.fetch_openmeteo_data(latitude, longitude)
    if data:
        cache_manager.set(location_id, 'climate_raw', data)
    return data


# ==================== DECORADORES ====================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = db.session.get(User, session['user_id'])
        if not user or not user.is_admin:
            return jsonify({'error': 'Acesso negado'}), 403
        return f(*args, **kwargs)
    return decorated_function


def approved_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        user = db.session.get(User, session['user_id'])
        if not user or not user.approved:
            return jsonify({'error': 'Usuário não aprovado'}), 403
        return f(*args, **kwargs)
    return decorated_function


# ==================== AUTENTICAÇÃO ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    try:
        data = request.json
        user = User.query.filter_by(username=data.get('username')).first()
        if not user or not user.check_password(data.get('password')):
            return jsonify({'error': 'Credenciais inválidas'}), 401
        if not user.approved and not user.is_admin:
            return jsonify({'error': 'Usuário aguardando aprovação do administrador'}), 403

        session['user_id'] = user.id
        user.last_login = datetime.now(timezone.utc)
        db.session.commit()
        logger.info(f"Usuário {user.username} autenticado")
        return jsonify({'success': True, 'is_admin': user.is_admin, 'username': user.username})
    except Exception as e:
        logger.error(f"Erro no login: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.json
        username, email, password = data.get('username'), data.get('email'), data.get('password')
        if not username or not email or not password or len(password) < 6:
            return jsonify({'error': 'Preencha usuário, e-mail e uma senha de pelo menos 6 caracteres'}), 400

        if User.query.filter_by(username=username).first():
            return jsonify({'error': 'Usuário já existe'}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email já registrado'}), 400

        new_user = User(username=username, email=email, approved=False)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        logger.info(f"Novo usuário registrado: {username}")
        return jsonify({'success': True, 'message': 'Registro concluído. Aguarde aprovação do administrador.'})
    except Exception as e:
        logger.error(f"Erro no registro: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


# ==================== ADMIN ====================

@app.route('/admin', methods=['GET'])
@admin_required
def admin_dashboard():
    return render_template('admin.html')


@app.route('/api/admin/pending-users', methods=['GET'])
@admin_required
def get_pending_users():
    try:
        pending = User.query.filter_by(approved=False, is_admin=False).all()
        return jsonify([{'id': u.id, 'username': u.username, 'email': u.email,
                         'created_at': u.created_at.isoformat()} for u in pending])
    except Exception as e:
        logger.error(f"Erro ao buscar usuários pendentes: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/approve-user/<int:user_id>', methods=['POST'])
@admin_required
def approve_user(user_id):
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'error': 'Usuário não encontrado'}), 404
        user.approved = True
        db.session.commit()
        logger.info(f"Usuário {user.username} aprovado")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Erro ao aprovar usuário: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/reject-user/<int:user_id>', methods=['POST'])
@admin_required
def reject_user(user_id):
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'error': 'Usuário não encontrado'}), 404
        username = user.username
        db.session.delete(user)
        db.session.commit()
        logger.info(f"Usuário {username} rejeitado")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Erro ao rejeitar usuário: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_all_users():
    try:
        return jsonify([{
            'id': u.id, 'username': u.username, 'email': u.email, 'is_admin': u.is_admin,
            'approved': u.approved, 'created_at': u.created_at.isoformat(),
            'last_login': u.last_login.isoformat() if u.last_login else None
        } for u in User.query.all()])
    except Exception as e:
        logger.error(f"Erro ao buscar usuários: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/delete-user/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    try:
        user = db.session.get(User, user_id)
        if not user:
            return jsonify({'error': 'Usuário não encontrado'}), 404
        if user.is_admin and user.id == session.get('user_id'):
            return jsonify({'error': 'Não pode deletar sua própria conta'}), 400
        username = user.username
        db.session.delete(user)
        db.session.commit()
        logger.info(f"Usuário {username} deletado")
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Erro ao deletar usuário: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def get_admin_stats():
    try:
        return jsonify({
            'total_users': User.query.count(),
            'approved_users': User.query.filter_by(approved=True).count(),
            'pending_users': User.query.filter_by(approved=False, is_admin=False).count(),
            'total_predictions': Prediction.query.count(),
            'total_locations': Location.query.count()
        })
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== CONFIGURAÇÃO DO LLM (OLLAMA) ====================

@app.route('/api/admin/ollama/status', methods=['GET'])
@admin_required
def get_ollama_status():
    try:
        return jsonify({'online': ia_agent.check_ollama_status(), 'host': Config.OLLAMA_HOST})
    except Exception as e:
        logger.error(f"Erro ao verificar status do Ollama: {str(e)}")
        return jsonify({'online': False, 'error': str(e)})


@app.route('/api/admin/ollama/models', methods=['GET'])
@admin_required
def get_ollama_models():
    try:
        models = ia_agent.get_available_models()
        if not models:
            return jsonify({'models': [], 'warning': 'Nenhum modelo encontrado. Verifique se o Ollama está rodando '
                                                     'e se há modelos instalados (ollama pull <modelo>)'})
        return jsonify({'models': models})
    except Exception as e:
        logger.error(f"Erro ao buscar modelos Ollama: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/ollama/current-model', methods=['GET'])
@admin_required
def get_current_ollama_model():
    try:
        return jsonify({'model': ia_agent.get_current_model()})
    except Exception as e:
        logger.error(f"Erro ao obter modelo atual: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/ollama/set-model', methods=['POST'])
@admin_required
def set_ollama_model():
    try:
        model_name = request.json.get('model')
        if not model_name:
            return jsonify({'error': 'Nome do modelo é obrigatório'}), 400
        SystemSettings.set_value('ollama_model', model_name)
        ia_agent.set_model(model_name)
        logger.info(f"Modelo Ollama alterado para: {model_name}")
        return jsonify({'success': True, 'model': model_name,
                        'message': f'Modelo alterado para {model_name} com sucesso'})
    except Exception as e:
        logger.error(f"Erro ao definir modelo: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/ollama/test-model', methods=['POST'])
@admin_required
def test_ollama_model():
    try:
        response = ia_agent._query_ollama("Responda apenas: 'Modelo funcionando corretamente!'", max_tokens=50)
        if response:
            return jsonify({'success': True, 'model': ia_agent.get_current_model(), 'response': response})
        return jsonify({'success': False,
                        'error': 'Modelo não respondeu. Verifique se está instalado corretamente.'}), 500
    except Exception as e:
        logger.error(f"Erro ao testar modelo: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== LOCALIZAÇÃO ====================

@app.route('/api/geocode', methods=['POST'])
@login_required
@approved_required
def geocode_city():
    try:
        city_name = request.json.get('city_name')
        if not city_name:
            return jsonify({'error': 'Nome da cidade é obrigatório'}), 400
        return jsonify(api_manager.geocode_city(city_name))
    except Exception as e:
        logger.error(f"Erro no geocode: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/locations', methods=['GET', 'POST'])
@login_required
@approved_required
def manage_locations():
    try:
        user_id = session.get('user_id')

        if request.method == 'GET':
            locations = Location.query.filter_by(user_id=user_id).all()
            return jsonify([{
                'id': l.id, 'city_name': l.city_name, 'latitude': l.latitude,
                'longitude': l.longitude, 'country': l.country,
                'is_favorite': l.is_favorite, 'created_at': l.created_at.isoformat()
            } for l in locations])

        data = request.json
        for ex in Location.query.filter_by(user_id=user_id).all():
            if abs(ex.latitude - float(data.get('latitude'))) < 0.02 and abs(ex.longitude - float(data.get('longitude'))) < 0.02:
                return jsonify({'success': True, 'location_id': ex.id})
        new_location = Location(user_id=user_id, city_name=data.get('city_name'),
                                latitude=data.get('latitude'), longitude=data.get('longitude'),
                                country=data.get('country', ''))
        db.session.add(new_location)
        db.session.commit()
        logger.info(f"Localização adicionada: {data.get('city_name')}")
        return jsonify({'success': True, 'location_id': new_location.id})

    except Exception as e:
        logger.error(f"Erro ao gerenciar localizações: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/locations/<int:location_id>', methods=['DELETE'])
@login_required
def delete_location(location_id):
    try:
        location = db.session.get(Location, location_id)
        if not location or location.user_id != session.get('user_id'):
            return jsonify({'error': 'Localização não encontrada'}), 404
        db.session.delete(location)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Erro ao deletar localização: {str(e)}")
        return jsonify({'error': str(e)}), 500


def _owned_location(location_id):
    location = db.session.get(Location, location_id)
    if not location or location.user_id != session.get('user_id'):
        return None
    return location


# ==================== PREVISÕES ====================

FULL_KEY = 'forecast_full'


async def get_best16(latitude, longitude, location_id, force=False):
    """Previsão best_match de 16 dias (só para os dias além da janela dos modelos)."""
    if not force:
        cached = cache_manager.get(location_id, 'best16', ttl=TTL)
        if cached:
            return json.loads(cached)
    data = await api_manager.fetch_detailed_forecast(latitude, longitude)
    if data:
        cache_manager.set(location_id, 'best16', data)
    return data


async def fetch_and_process_weather(latitude: float, longitude: float, location_id: int,
                                    force: bool = False, climate: dict = None,
                                    is_job: bool = False) -> dict:
    """
    Pipeline: multi-modelo -> verifica o que já passou (aprendizado) -> consenso ponderado ->
    snapshot p/ aprender depois -> correção estatística (MOS) -> eventos/raios -> validação por radar.
    """
    try:
        if climate:
            cache_manager.set(location_id, 'climate_raw', climate)
            radar_meta, best16 = await asyncio.gather(
                api_manager.fetch_rainviewer_data(), get_best16(latitude, longitude, location_id, True))
        else:
            climate, radar_meta, best16 = await asyncio.gather(
                get_climate_models(latitude, longitude, location_id, force),
                api_manager.fetch_rainviewer_data(), get_best16(latitude, longitude, location_id, force))
        if not climate:
            logger.warning("Nenhum dado climático obtido")
            return {}
        processed = data_processor.process_openmeteo_data(climate)
        if not processed:
            return {}

        adaptive = AdaptiveLearning(location_id)          # location_id aqui é a chave do local (gkey)
        if adaptive.has_due_verifications():           # 1) aprende com o que já aconteceu
            obs = await api_manager.fetch_observations(latitude, longitude)
            if obs:
                adaptive.verify_pending(obs)

        weights = adaptive.get_model_weights()
        base = prediction_engine.compute_base_prediction(processed, weights)
        if not base:
            return {}

        now_utc = datetime.now(timezone.utc)           # 2) guarda a previsão p/ verificar depois
        far_ok = adaptive.state.get('last_snapshot') is None or now_utc.hour % 6 == 0
        adaptive.record_forecast_snapshot(prediction_engine.build_snapshot(processed, weights, far_ok),
                                          base.get('event_probs_raw'))

        enhanced = adaptive.predict_with_learning(base)
        forecast = build_forecast(processed, adaptive, prediction_engine, enhanced, best16 or None)
        if not forecast:
            return {}
        now = forecast['now']
        for var in Config.VARIABLES:                   # valores finais = os do sistema
            if now.get(var) is not None:
                enhanced[var] = now[var]
        enhanced['precipitation_hour'] = now.get('precipitation')
        enhanced['precipitation'] = now.get('precipitation_24h')
        for k in ('apparent_temperature', 'dew_point', 'uv_index', 'visibility_km', 'wind_direction', 'cape'):
            enhanced[k] = now.get(k)
        enhanced['hit_rate'] = now.get('hit_rate')

        # previsões de tempestade p/ cada dia futuro ficam guardadas p/ medir o acerto depois (por antecedência)
        first_md = next(iter(processed.values()))
        adaptive.record_daily_events(forecast.get('daily'), first_md['model_info'].get('utc_offset_seconds', 0))
        enhanced['storm_outlook'] = [
            {'date': d['date'], 'level': (d.get('storm') or {}).get('level'),
             'type': (d.get('storm') or {}).get('type'), 'organization': (d.get('storm') or {}).get('organization'),
             'hazards': (d.get('storm') or {}).get('hazards'), 'lightning_pct': (d.get('storm') or {}).get('lightning_pct'),
             'hail_pct': (d.get('storm') or {}).get('hail_pct'), 'wind_pct': (d.get('storm') or {}).get('wind_pct'),
             'heavy_rain_pct': (d.get('storm') or {}).get('heavy_rain_pct'),
             'peak_hour': (d.get('storm') or {}).get('peak_hour'),
             'cape_max': (d.get('storm') or {}).get('cape_max'), 'shear06': (d.get('storm') or {}).get('shear06')}
            for d in (forecast.get('daily') or [])[:10]]

        # 3) validação dupla pelo radar
        radar_val = {'available': False}
        if radar_meta:
            loop = asyncio.get_running_loop()
            sample = await loop.run_in_executor(None, radar_validator.sample_radar, radar_meta, latitude, longitude)
            rain_now = (now.get('precipitation') or 0) >= 0.1
            radar_val = radar_validator.assess(sample, now.get('storm_next3h') or 0, rain_now)
            if radar_val.get('available'):
                motion = await loop.run_in_executor(None, radar_validator.motion_field, radar_meta, latitude, longitude)
                msum = radar_validator.summarize_motion(motion)
                if msum:
                    radar_val['motion'] = msum
                    if msum.get('text'):
                        radar_val['message'] = f"{radar_val['message']} {msum['text']}"
            if is_job and radar_val.get('available'):
                adaptive.record_radar_check(rain_now, radar_val['echo_local'])
                adaptive.record_radar_observation(radar_val, now.get('storm_next3h') or 0)
        enhanced['radar_validation'] = radar_val
        enhanced['radar_available'] = bool(radar_val.get('available'))
        enhanced['radar_confirmed'] = radar_val.get('storm_status') == 'confirmado'

        enhanced['adapted'] = True
        enhanced['models_total'] = len(processed)
        enhanced['skill'] = adaptive.get_skill_summary()
        enhanced['generated_at'] = forecast['generated_at']
        enhanced['update_interval_minutes'] = Config.UPDATE_INTERVAL_MINUTES
        enhanced['ia_analysis'] = ''
        enhanced['ia_pending'] = True
        enhanced = sanitize(enhanced)
        forecast = sanitize(forecast)

        cache_manager.set(location_id, 'weather_data', {'predictions': enhanced,
                                                        'timestamp': forecast['generated_at']})
        cache_manager.set(location_id, FULL_KEY, forecast)
        return {'processed': processed, 'predictions': enhanced, 'radar': radar_meta, 'forecast': forecast,
                'timestamp': forecast['generated_at']}
    except Exception as e:
        logger.exception(f"Erro ao buscar e processar dados: {str(e)}")
        return {}


def store_prediction_row(location_id, p, radar_meta):
    db.session.add(Prediction(
        location_id=location_id, timestamp_prediction=datetime.now(timezone.utc) + timedelta(hours=1),
        temperature=p.get('temperature'), temperature_confidence=p.get('temperature_confidence', 0),
        humidity=p.get('humidity'), humidity_confidence=p.get('humidity_confidence', 0),
        precipitation=p.get('precipitation'), precipitation_confidence=p.get('precipitation_confidence', 0),
        wind_speed=p.get('wind_speed'), wind_confidence=p.get('wind_confidence', 0),
        thunderstorm_probability=p.get('thunderstorm_probability', 0),
        heavy_rain_probability=p.get('heavy_rain_probability', 0),
        hail_probability=p.get('hail_probability', 0), tornado_probability=0,
        extreme_wind_probability=p.get('extreme_wind_probability', 0),
        ia_analysis='', ia_observations=None,
        radar_confirmation=bool(p.get('radar_confirmed')), radar_data=sanitize(radar_meta)))
    db.session.commit()


# ---------- atualização horária em segundo plano ----------
def update_all_locations() -> dict:
    """
    Roda de hora em hora: atualiza, verifica e aprende para TODOS os locais cadastrados.
    Locais iguais (mesma cidade em vários usuários) são processados UMA vez; cada usuário recebe seu registro.
    Ao final de cada local, a análise do agente de IA é enfileirada: quando o usuário abrir a página já estará pronta.
    """
    done, failed = 0, 0
    with app.app_context():
        groups = {}
        for loc in Location.query.all():
            groups.setdefault(gkey(loc), []).append(loc)
        for key, locs in groups.items():
            first = locs[0]
            try:
                with key_lock(key):
                    climate = run_async(api_manager.fetch_openmeteo_data(first.latitude, first.longitude))
                    res = run_async(fetch_and_process_weather(first.latitude, first.longitude, key,
                                                              climate=climate or None, is_job=True))
                if res:
                    for loc in locs:
                        store_prediction_row(loc.id, res['predictions'], res.get('radar'))
                    request_ia(key, res['predictions'], (res.get('forecast') or {}).get('daily'), first.city_name)
                    done += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                db.session.rollback()
                logger.error(f"Atualização horária falhou p/ local {key}: {e}")
            time.sleep(Config.SCHEDULER_PAUSE_BETWEEN_LOCATIONS)
        try:
            cache_manager.clear_expired()
        except Exception:
            pass
    logger.info(f"Atualização horária: {done} locais ok, {failed} falhas")
    return {'updated': done, 'failed': failed}


# ---------- agente de IA: fila em segundo plano (um LLM pequeno responde devagar e isso é normal) ----------
IA_Q = queue.Queue()
IA_LOCK = threading.Lock()
IA_PENDING = {}      # chave do local -> assinatura em andamento
IA_ERR = {}          # chave do local -> (assinatura, instante, mensagem)
IA_RETRY_SECONDS = 120


def request_ia(key, predictions, daily, name=''):
    """
    Regra do cache da IA:
      * mesma assinatura (nenhuma variável relevante mudou) -> reaproveita a resposta anterior, sem consultar o LLM;
      * assinatura diferente (passou a hora e os dados mudaram) -> consulta o LLM UMA vez e todos os usuários
        do local compartilham a resposta;
    Retorna o estado atual: ready | pending | error (com o texto antigo em 'stale_analysis' enquanto atualiza).
    """
    if not predictions:
        return {'status': 'error', 'analysis': '', 'error': 'Previsão ainda não disponível'}
    sig = ia_agent.analysis_signature(predictions, daily)
    cached = cache_manager.peek(key, 'ia_analysis') or {}
    if cached.get('analysis') and cached.get('signature') == sig:
        return {'status': 'ready', **cached}
    stale = cached.get('analysis', '')
    with IA_LOCK:
        err = IA_ERR.get(key)
        if IA_PENDING.get(key) == sig:
            return {'status': 'pending', 'analysis': '', 'stale_analysis': stale}
        if err and err[0] == sig and time.time() - err[1] < IA_RETRY_SECONDS:
            return {'status': 'error', 'analysis': '', 'stale_analysis': stale, 'error': err[2]}
        IA_PENDING[key] = sig
        IA_Q.put((key, sig, predictions, daily, name))
    return {'status': 'pending', 'analysis': '', 'stale_analysis': stale}


def _ia_worker():
    while True:
        key, sig, predictions, daily, name = IA_Q.get()
        try:
            res = ia_agent.analyze_predictions(predictions, None, predictions.get('radar_validation'),
                                               daily, name)
            text = res.get('analysis')
            if text:
                cache_manager.set(key, 'ia_analysis', {'analysis': text, 'model_used': res.get('model_used'),
                                                       'timestamp': res.get('timestamp'), 'signature': sig})
                IA_ERR.pop(key, None)
            else:
                IA_ERR[key] = (sig, time.time(), res.get('error') or 'O modelo não respondeu')
        except Exception as e:
            logger.exception("Falha no worker de IA")
            IA_ERR[key] = (sig, time.time(), str(e))
        finally:
            with IA_LOCK:
                if IA_PENDING.get(key) == sig:
                    IA_PENDING.pop(key, None)


def start_ia_worker():
    if not getattr(start_ia_worker, 'started', False):
        threading.Thread(target=_ia_worker, name='rrfs-ia-worker', daemon=True).start()
        start_ia_worker.started = True


updater = HourlyUpdater(update_all_locations, Config.UPDATE_INTERVAL_MINUTES,
                        Config.SCHEDULER_FIRST_DELAY_SECONDS)


@app.route('/api/admin/scheduler', methods=['GET'])
@admin_required
def scheduler_status():
    return jsonify(updater.status)


def get_weather(location, force=False):
    """
    Previsão do local com cache COMPARTILHADO entre usuários e trava (duas pessoas abrindo a mesma cidade
    ao mesmo tempo geram uma única busca/processamento). -> (predictions, resultado_novo | None)
    """
    key = gkey(location)
    with key_lock(key):
        if force:
            age = cache_manager.age_seconds(key, 'weather_data')
            if age is not None and age < FORCE_MIN_AGE:      # acabou de atualizar: não repete
                force = False
        if not force:
            cached = cache_manager.get(key, 'weather_data', ttl=TTL)
            if cached:
                return json.loads(cached).get('predictions', {}), None
        result = run_async(fetch_and_process_weather(location.latitude, location.longitude, key, force))
        if not result:
            return None, None
        request_ia(key, result['predictions'], (result.get('forecast') or {}).get('daily'), location.city_name)
        return result['predictions'], result


@app.route('/api/forecast/<int:location_id>', methods=['GET'])
@login_required
@approved_required
def get_forecast(location_id):
    try:
        location = _owned_location(location_id)
        if not location:
            return jsonify({'error': 'Localização não encontrada'}), 404
        preds, result = get_weather(location, wants_force())
        if preds is None:
            return jsonify({'error': 'Erro ao buscar previsão'}), 500
        if result:
            store_prediction_row(location_id, result['predictions'], result.get('radar'))
        return jsonify(preds)
    except Exception as e:
        db.session.rollback()
        logger.exception(f"Erro ao obter previsão: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/<int:location_id>/ia-analysis', methods=['GET'])
@login_required
@approved_required
def get_ia_analysis(location_id):
    """
    Estado da análise do agente de IA. Nunca bloqueia: se o LLM ainda está gerando, devolve status
    'pending' (com o texto anterior, se houver) e o dashboard consulta de novo até ficar pronto.
    """
    try:
        location = _owned_location(location_id)
        if not location:
            return jsonify({'error': 'Localização não encontrada'}), 404
        key = gkey(location)
        pred_cached = cache_manager.peek(key, 'weather_data')
        if not pred_cached:
            return jsonify({'status': 'pending', 'analysis': '', 'error': 'Previsão ainda não disponível'})
        daily = (cache_manager.peek(key, FULL_KEY) or {}).get('daily')
        out = request_ia(key, pred_cached.get('predictions', {}), daily, location.city_name)
        return jsonify(out)
    except Exception as e:
        logger.exception(f"Erro na análise IA: {str(e)}")
        return jsonify({'status': 'error', 'analysis': '', 'error': str(e)}), 500


def load_timeseries(location, force=False):
    """Previsão completa (hora a hora + 16 dias). Usado pela API web e pelo agente de IA."""
    key = gkey(location)
    if not force:
        cached = cache_manager.get(key, FULL_KEY, ttl=TTL)
        if cached:
            return json.loads(cached)
    preds, result = get_weather(location, force)
    if result:
        return result.get('forecast')
    cached = cache_manager.get(key, FULL_KEY, ttl=TTL)
    return json.loads(cached) if cached else None


@app.route('/api/forecast/<int:location_id>/timeseries', methods=['GET'])
@login_required
@approved_required
def get_forecast_timeseries(location_id):
    try:
        location = _owned_location(location_id)
        if not location:
            return jsonify({'error': 'Localização não encontrada'}), 404
        fc = load_timeseries(location, wants_force())
        if not fc:
            return jsonify({'error': 'Erro ao buscar previsão detalhada'}), 500
        return jsonify({k: v for k, v in fc.items() if k != 'hourly'})
    except Exception as e:
        logger.exception(f"Erro ao obter previsão temporal: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/<int:location_id>/day/<date_str>', methods=['GET'])
@login_required
@approved_required
def get_forecast_day(location_id, date_str):
    """Todas as horas e variáveis de um dia (ao clicar no dia no dashboard)."""
    try:
        location = _owned_location(location_id)
        if not location:
            return jsonify({'error': 'Localização não encontrada'}), 404
        fc = load_timeseries(location)
        if not fc:
            return jsonify({'error': 'Previsão indisponível'}), 500
        day = next((d for d in fc['daily'] if d['date'] == date_str), None)
        if not day:
            return jsonify({'error': 'Data fora do período previsto'}), 404
        rows = [h for h in fc['hourly'] if h['time'].startswith(date_str)]
        source = 'consenso'
        if not rows:                                    # além dos modelos: best_match (16 dias)
            source = 'best_match'
            cached = cache_manager.get(gkey(location), 'best16', ttl=TTL)
            h = (json.loads(cached) if cached else {}).get('hourly') or {}
            for i, t in enumerate(h.get('time', [])):
                if not t.startswith(date_str):
                    continue
                g = lambda k: (h.get(k) or [None] * (i + 1))[i]
                desc, icon = get_weather_description(g('weather_code'))
                rows.append({'time': t, 'hour': t[11:16], 'temperature': g('temperature_2m'),
                             'precipitation': g('precipitation'), 'precipitation_probability': g('precipitation_probability'),
                             'wind_speed': g('wind_speed_10m'), 'humidity': g('relative_humidity_2m'),
                             'weather_description': desc, 'weather_icon': icon})
        return jsonify(sanitize({'date': date_str, 'source': source, 'day': day, 'hourly': rows,
                                 'timezone': fc.get('timezone')}))
    except Exception as e:
        logger.exception(f"Erro ao obter dia: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/radar/<int:location_id>', methods=['GET'])
@login_required
@approved_required
def get_radar_image(location_id):
    try:
        location = _owned_location(location_id)
        if not location:
            return jsonify({'error': 'Localização não encontrada'}), 404
        if not wants_force():
            cached = cache_manager.get(gkey(location), 'radar_grid', ttl=Config.RADAR_CACHE_TIMEOUT)
            if cached:
                return jsonify(json.loads(cached))
        radar_meta = run_async(api_manager.fetch_rainviewer_data())
        if not radar_meta:
            return jsonify({'error': 'Dados de radar indisponíveis no momento'}), 503
        grid = api_manager.get_radar_tile_grid(location.latitude, location.longitude, radar_meta)
        if not grid:
            return jsonify({'error': 'Não foi possível montar a imagem de radar'}), 503
        loop_motion = radar_validator.motion_field(radar_meta, location.latitude, location.longitude)
        grid['motion'] = sanitize(loop_motion)
        grid['motion_summary'] = sanitize(radar_validator.summarize_motion(loop_motion))
        cache_manager.set(gkey(location), 'radar_grid', grid)
        return jsonify(grid)
    except Exception as e:
        logger.exception(f"Erro ao obter radar: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/<int:location_id>/extended', methods=['GET'])
@login_required
@approved_required
def get_extended_forecast(location_id):
    try:
        if not _owned_location(location_id):
            return jsonify({'error': 'Localização não encontrada'}), 404
        predictions = Prediction.query.filter_by(location_id=location_id) \
            .order_by(Prediction.timestamp_prediction.desc()).limit(168).all()
        return jsonify([{
            'timestamp': p.timestamp_prediction.isoformat(), 'temperature': p.temperature,
            'humidity': p.humidity, 'precipitation': p.precipitation, 'wind_speed': p.wind_speed,
            'thunderstorm': p.thunderstorm_probability, 'heavy_rain': p.heavy_rain_probability,
            'hail': p.hail_probability
        } for p in predictions])
    except Exception as e:
        logger.error(f"Erro ao obter previsão estendida: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== APRENDIZADO ADAPTATIVO ====================

@app.route('/api/learning/<int:location_id>/report', methods=['GET'])
@login_required
@approved_required
def get_learning_report(location_id):
    try:
        if not _owned_location(location_id):
            return jsonify({'error': 'Localização não encontrada'}), 404
        return jsonify(sanitize(AdaptiveLearning(gkey(_owned_location(location_id))).get_learning_report()))
    except Exception as e:
        logger.error(f"Erro ao obter relatório de aprendizado: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/learning/<int:location_id>/skill', methods=['GET'])
@login_required
@approved_required
def get_skill(location_id):
    try:
        if not _owned_location(location_id):
            return jsonify({'error': 'Localização não encontrada'}), 404
        ad = AdaptiveLearning(gkey(_owned_location(location_id)))
        return jsonify(sanitize({'skill': ad.get_skill_summary(), 'models': ad.get_model_performance(),
                                 'weights': ad.get_model_weights()}))
    except Exception as e:
        logger.error(f"Erro ao obter acurácia: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/learning/<int:location_id>/biases', methods=['GET'])
@login_required
@approved_required
def get_location_biases(location_id):
    try:
        if not _owned_location(location_id):
            return jsonify({'error': 'Localização não encontrada'}), 404
        return jsonify(sanitize(AdaptiveLearning(gkey(_owned_location(location_id))).get_location_bias()))
    except Exception as e:
        logger.error(f"Erro ao obter vieses: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/learning/<int:location_id>/patterns', methods=['GET'])
@login_required
@approved_required
def get_location_patterns(location_id):
    try:
        if not _owned_location(location_id):
            return jsonify({'error': 'Localização não encontrada'}), 404
        adaptive = AdaptiveLearning(gkey(_owned_location(location_id)))
        return jsonify(sanitize({
            'seasonal': adaptive.get_seasonal_pattern(),
            'extremes': adaptive.get_local_extreme_patterns(),
            'accuracy_horizons': adaptive.get_accuracy_by_horizon(),
            'model_performance': adaptive.get_model_performance()
        }))
    except Exception as e:
        logger.error(f"Erro ao obter padrões: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/learning/<int:location_id>/record-observation', methods=['POST'])
@login_required
@approved_required
def record_observation(location_id):
    try:
        if not _owned_location(location_id):
            return jsonify({'error': 'Localização não encontrada'}), 404
        data = request.json
        AdaptiveLearning(gkey(_owned_location(location_id))).record_prediction_observation(
            prediction=data.get('prediction', {}), observation=data.get('observation', {}),
            event_type=data.get('event_type', 'general'))
        return jsonify({'success': True, 'message': 'Observação registrada'})
    except Exception as e:
        logger.error(f"Erro ao registrar observação: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== PÁGINAS ====================

@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/dashboard')
@login_required
@approved_required
def dashboard():
    return render_template('dashboard.html')


# ==================== API PARA O AGENTE DE IA ====================
# O agente (RRFS Agente) faz login com um usuário próprio e consulta a meteorologia por aqui.
from modulos.agent_api import create_agent_blueprint

app.register_blueprint(create_agent_blueprint({
    'db': db, 'User': User, 'Location': Location,
    'cache_manager': cache_manager, 'data_processor': data_processor,
    'prediction_engine': prediction_engine,
    'get_climate_models': get_climate_models, 'run_async': run_async,
    'load_timeseries': load_timeseries,
    'fetch_and_process_weather': fetch_and_process_weather,
    'AdaptiveLearning': AdaptiveLearning, 'sanitize': sanitize,
    'get_weather_description': get_weather_description, 'Config': Config,
    'geo_key': gkey, 'get_weather': get_weather,
}))


# Agendador de 1 em 1 hora (com o reloader do Flask só o processo-filho inicia)
if Config.SCHEDULER_ENABLED and (not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true'):
    updater.start()

if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
    start_ia_worker()


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

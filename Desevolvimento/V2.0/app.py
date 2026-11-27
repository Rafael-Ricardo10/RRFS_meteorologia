from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
from modulos import db, User, Location, WeatherData, Prediction, LearningData, RadarCache, SystemSettings
from modulos import APIManager, DataProcessor, PredictionEngine, IAAgent, CacheManager, Utils
from modulos.api_manager import get_weather_description
from config import Config
from functools import wraps
from datetime import datetime, timedelta, timezone
from modulos.adaptive_learning import AdaptiveLearning
import asyncio
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


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def wants_force():
    return request.args.get('force') == '1'


async def get_climate_models(latitude, longitude, location_id, force=False):
    """Dados brutos de todos os modelos, com cache (compartilhado por forecast e timeseries)."""
    if not force:
        cached = cache_manager.get(location_id, 'climate_raw', ttl=Config.FORECAST_CACHE_TIMEOUT)
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

async def fetch_and_process_weather(latitude: float, longitude: float, location_id: int,
                                    force: bool = False) -> dict:
    """
    Pipeline: multi-modelo -> verificação do que já passou (aprendizado) ->
    consenso ponderado -> correção de viés -> calibração de eventos extremos.
    """
    try:
        climate, radar_meta = await asyncio.gather(
            get_climate_models(latitude, longitude, location_id, force),
            api_manager.fetch_rainviewer_data())

        if not climate:
            logger.warning("Nenhum dado climático obtido")
            return {}

        processed = data_processor.process_openmeteo_data(climate)
        if not processed:
            return {}

        adaptive = AdaptiveLearning(location_id)

        # 1) aprende com previsões passadas cujo horário já chegou
        if adaptive.has_due_verifications():
            obs = await api_manager.fetch_observations(latitude, longitude)
            if obs:
                adaptive.verify_pending(obs)

        # 2) consenso ponderado pelos pesos aprendidos
        weights = adaptive.get_model_weights()
        base = prediction_engine.compute_base_prediction(processed, weights)
        if not base:
            return {}

        # 3) guarda o que cada modelo previu p/ aprender depois
        adaptive.record_forecast_snapshot(prediction_engine.build_snapshot(processed, weights),
                                          base.get('event_probs_raw'))

        # 4) correção de viés + calibração de probabilidades + confiança
        enhanced = adaptive.predict_with_learning(base)
        enhanced['adapted'] = True
        enhanced['models_total'] = len(processed)
        enhanced['radar_available'] = bool((radar_meta or {}).get('radar', {}).get('past')) if radar_meta else False
        enhanced['ia_analysis'] = ''
        enhanced['ia_pending'] = True     # a análise do LLM é carregada depois (não bloqueia a tela)
        enhanced = sanitize(enhanced)

        cache_manager.set(location_id, 'weather_data', {
            'predictions': enhanced,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })

        return {'processed': processed, 'predictions': enhanced, 'radar': radar_meta,
                'timestamp': datetime.now(timezone.utc).isoformat()}

    except Exception as e:
        logger.exception(f"Erro ao buscar e processar dados: {str(e)}")
        return {}


@app.route('/api/forecast/<int:location_id>', methods=['GET'])
@login_required
@approved_required
def get_forecast(location_id):
    try:
        location = _owned_location(location_id)
        if not location:
            return jsonify({'error': 'Localização não encontrada'}), 404

        force = wants_force()
        if not force:
            cached = cache_manager.get(location_id, 'weather_data', ttl=Config.FORECAST_CACHE_TIMEOUT)
            if cached:
                return jsonify(json.loads(cached).get('predictions', {}))

        result = run_async(fetch_and_process_weather(location.latitude, location.longitude,
                                                     location_id, force))
        if not result:
            return jsonify({'error': 'Erro ao buscar previsão'}), 500

        p = result['predictions']
        prediction = Prediction(
            location_id=location_id,
            timestamp_prediction=datetime.now(timezone.utc) + timedelta(hours=1),
            temperature=p.get('temperature'),
            temperature_confidence=p.get('temperature_confidence', 0),
            humidity=p.get('humidity'),
            humidity_confidence=p.get('humidity_confidence', 0),
            precipitation=p.get('precipitation'),
            precipitation_confidence=p.get('precipitation_confidence', 0),
            wind_speed=p.get('wind_speed'),
            wind_confidence=p.get('wind_confidence', 0),
            thunderstorm_probability=p.get('thunderstorm_probability', 0),
            heavy_rain_probability=p.get('heavy_rain_probability', 0),
            hail_probability=p.get('hail_probability', 0),
            tornado_probability=p.get('tornado_probability', 0),
            extreme_wind_probability=p.get('extreme_wind_probability', 0),
            ia_analysis='',
            ia_observations=None,
            radar_confirmation=p.get('radar_available', False),
            radar_data=sanitize(result.get('radar'))
        )
        db.session.add(prediction)
        db.session.commit()

        return jsonify(p)

    except Exception as e:
        logger.exception(f"Erro ao obter previsão: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/forecast/<int:location_id>/ia-analysis', methods=['GET'])
@login_required
@approved_required
def get_ia_analysis(location_id):
    """Análise do LLM (Ollama) em endpoint separado: pode demorar e não deve travar o dashboard."""
    try:
        if not _owned_location(location_id):
            return jsonify({'error': 'Localização não encontrada'}), 404

        if not wants_force():
            cached = cache_manager.get(location_id, 'ia_analysis', ttl=Config.FORECAST_CACHE_TIMEOUT)
            if cached:
                return jsonify(json.loads(cached))

        pred_cached = cache_manager.get(location_id, 'weather_data', ttl=Config.FORECAST_CACHE_TIMEOUT)
        if not pred_cached:
            return jsonify({'analysis': '', 'error': 'Previsão ainda não disponível'}), 404

        predictions = json.loads(pred_cached).get('predictions', {})
        radar_cached = cache_manager.get(location_id, 'radar_grid', ttl=Config.RADAR_CACHE_TIMEOUT)
        radar_info = {'frame_time_iso': json.loads(radar_cached).get('frame_time_iso')} if radar_cached else None

        result = ia_agent.analyze_predictions(predictions, None, radar_info)
        out = {'analysis': result.get('analysis', '') or '',
               'model_used': result.get('model_used'), 'timestamp': result.get('timestamp')}
        if out['analysis'] and out['analysis'] != 'Análise não disponível':
            cache_manager.set(location_id, 'ia_analysis', out)
        else:
            out['analysis'] = ''
        return jsonify(out)

    except Exception as e:
        logger.exception(f"Erro na análise IA: {str(e)}")
        return jsonify({'analysis': '', 'error': str(e)}), 500


# Códigos de tempo do Open-Meteo considerados eventos severos
SEVERE_THUNDERSTORM_LABELS = {95: 'Tempestade', 96: 'Tempestade com Granizo',
                              99: 'Tempestade Severa com Granizo'}


def classify_day_severity(code, precip_sum, wind_gusts_max):
    try:
        if code is not None and int(code) in SEVERE_THUNDERSTORM_LABELS:
            return True, SEVERE_THUNDERSTORM_LABELS[int(code)]
    except (ValueError, TypeError):
        pass
    if wind_gusts_max is not None and wind_gusts_max >= 70:
        return True, 'Vento Extremo'
    if precip_sum is not None and precip_sum >= 50:
        return True, 'Chuva Intensa'
    return False, None


HOURLY_CONS = {'temperature_2m': 'temperature', 'relative_humidity_2m': 'humidity',
               'wind_speed_10m': 'wind_speed', 'precipitation': 'precipitation',
               'precipitation_probability': 'precipitation'}


def _vote_code(pairs, wdict):
    """Voto ponderado do weather_code; trovoada pesa se >=40% dos modelos apontarem."""
    tally = {}
    default = float(np.mean(list(wdict.values()))) if wdict else 1.0
    for m, c in pairs:
        tally[int(c)] = tally.get(int(c), 0.0) + wdict.get(m, default)
    if not tally:
        return None
    total = sum(tally.values())
    severe = {c: v for c, v in tally.items() if c in SEVERE_THUNDERSTORM_LABELS}
    if severe and sum(severe.values()) / total >= 0.4:
        return max(severe, key=severe.get)
    return max(tally, key=tally.get)


def process_detailed_forecast(raw_data: dict, models: dict = None, weights: dict = None,
                              temp_bias: float = 0.0) -> dict:
    """
    Hora a hora de hoje + diário (até 16 dias). Onde há dados multi-modelo (7 dias)
    usa o CONSENSO ponderado; além disso usa o best_match.
    """
    models, weights = models or {}, weights or {}
    hourly = raw_data.get('hourly', {}) or {}
    daily = raw_data.get('daily', {}) or {}
    hourly_times = hourly.get('time', []) or []
    daily_times = daily.get('time', []) or []
    today_str = daily_times[0] if daily_times else datetime.now().strftime('%Y-%m-%d')

    def safe_list(key, source, length):
        values = source.get(key)
        return [None] * length if values is None else values

    cons = {}
    for col, var in HOURLY_CONS.items():
        s = prediction_engine.weighted_series(models, col, weights.get(var, {}))
        if s is not None:
            cons[col] = s

    def pick(col, ts, fallback):
        s = cons.get(col)
        if s is not None:
            v = s.get(ts)
            if v is not None and not pd.isna(v):
                return float(v)
        return fallback

    h_len = len(hourly_times)
    h_temp = safe_list('temperature_2m', hourly, h_len)
    h_pprob = safe_list('precipitation_probability', hourly, h_len)
    h_precip = safe_list('precipitation', hourly, h_len)
    h_wind = safe_list('wind_speed_10m', hourly, h_len)
    h_hum = safe_list('relative_humidity_2m', hourly, h_len)
    h_code = safe_list('weather_code', hourly, h_len)

    hourly_today = []
    for i, t in enumerate(hourly_times):
        if not t.startswith(today_str):
            continue
        ts = pd.Timestamp(t)
        desc, icon = get_weather_description(h_code[i])
        temp = pick('temperature_2m', ts, h_temp[i])
        if temp is not None:
            temp += temp_bias
        hourly_today.append({
            'time': t, 'hour': t.split('T')[1][:5] if 'T' in t else t,
            'temperature': temp,
            'precipitation_probability': pick('precipitation_probability', ts, h_pprob[i]),
            'precipitation': pick('precipitation', ts, h_precip[i]),
            'wind_speed': pick('wind_speed_10m', ts, h_wind[i]),
            'humidity': pick('relative_humidity_2m', ts, h_hum[i]),
            'weather_description': desc, 'weather_icon': icon
        })

    model_daily = {}
    for m, md in models.items():
        ddf = md.get('daily')
        if ddf is not None and len(ddf) and 'time' in ddf.columns:
            model_daily[m] = ddf.assign(date=ddf['time'].dt.strftime('%Y-%m-%d')).set_index('date')

    def day_pairs(d, col):
        out = []
        for m, ddf in model_daily.items():
            if col in ddf.columns and d in ddf.index:
                v = ddf.at[d, col]
                if pd.notna(v):
                    out.append((m, float(v)))
        return out

    d_len = len(daily_times)
    d_tmax = safe_list('temperature_2m_max', daily, d_len)
    d_tmin = safe_list('temperature_2m_min', daily, d_len)
    d_psum = safe_list('precipitation_sum', daily, d_len)
    d_pprob = safe_list('precipitation_probability_max', daily, d_len)
    d_wmax = safe_list('wind_speed_10m_max', daily, d_len)
    d_gmax = safe_list('wind_gusts_10m_max', daily, d_len)
    d_code = safe_list('weather_code', daily, d_len)

    daily_list = []
    for i, d in enumerate(daily_times):
        _, tspread, n_models = prediction_engine.weighted_mean(
            day_pairs(d, 'temperature_2m_max'), weights.get('temperature', {}))
        use_cons = n_models >= 2

        def val(col, var, base):
            if use_cons:
                m_, _, _ = prediction_engine.weighted_mean(day_pairs(d, col), weights.get(var, {}))
                if m_ is not None:
                    return m_
            return base

        t_max = val('temperature_2m_max', 'temperature', d_tmax[i])
        t_min = val('temperature_2m_min', 'temperature', d_tmin[i])
        if t_max is not None:
            t_max += temp_bias
        if t_min is not None:
            t_min += temp_bias
        p_sum = val('precipitation_sum', 'precipitation', d_psum[i])
        p_prob = val('precipitation_probability_max', 'precipitation', d_pprob[i])
        w_max = val('wind_speed_10m_max', 'wind_speed', d_wmax[i])
        g_max = val('wind_gusts_10m_max', 'wind_speed', d_gmax[i])

        code = d_code[i]
        if use_cons:
            voted = _vote_code(day_pairs(d, 'weather_code'), weights.get('precipitation', {}))
            if voted is not None:
                code = voted

        desc, icon = get_weather_description(code)
        is_severe, severe_label = classify_day_severity(code, p_sum, g_max)
        confidence = (float(min(1.0, 0.55 + 0.1 * n_models) * math.exp(-tspread / 3.0))
                      if use_cons else None)

        daily_list.append({
            'date': d, 'is_today': (d == today_str),
            'temp_max': t_max, 'temp_min': t_min,
            'precipitation_sum': p_sum, 'precipitation_probability_max': p_prob,
            'wind_speed_max': w_max, 'wind_gusts_max': g_max,
            'weather_description': desc, 'weather_icon': icon,
            'severe': is_severe, 'severe_label': severe_label,
            'source': 'consenso' if use_cons else 'best_match',
            'models_count': n_models, 'confidence': confidence
        })

    return sanitize({
        'hourly_today': hourly_today, 'daily': daily_list,
        'timezone': raw_data.get('timezone', 'auto'),
        'days_available': len(daily_list),
        'models_used': len(model_daily), 'consensus': bool(model_daily)
    })


def load_timeseries(location, force=False):
    """Hora a hora de hoje + diário (16 dias) com consenso. Usado pela API web e pelo agente de IA."""
    location_id = location.id
    if not force:
        cached = cache_manager.get(location_id, 'timeseries_data', ttl=Config.FORECAST_CACHE_TIMEOUT)
        if cached:
            return json.loads(cached)

    async def load():
        return await asyncio.gather(
            api_manager.fetch_detailed_forecast(location.latitude, location.longitude),
            get_climate_models(location.latitude, location.longitude, location_id, force))

    best, climate = run_async(load())
    if not best:
        return None

    models = data_processor.process_openmeteo_data(climate) if climate else {}
    adaptive = AdaptiveLearning(location_id)
    processed = process_detailed_forecast(best, models, adaptive.get_model_weights(),
                                          adaptive.get_bias('temperature'))
    cache_manager.set(location_id, 'timeseries_data', processed)
    return processed


@app.route('/api/forecast/<int:location_id>/timeseries', methods=['GET'])
@login_required
@approved_required
def get_forecast_timeseries(location_id):
    try:
        location = _owned_location(location_id)
        if not location:
            return jsonify({'error': 'Localização não encontrada'}), 404

        processed = load_timeseries(location, wants_force())
        if not processed:
            return jsonify({'error': 'Erro ao buscar previsão detalhada'}), 500
        return jsonify(processed)

    except Exception as e:
        logger.exception(f"Erro ao obter previsão temporal: {str(e)}")
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
            cached = cache_manager.get(location_id, 'radar_grid', ttl=Config.RADAR_CACHE_TIMEOUT)
            if cached:
                return jsonify(json.loads(cached))

        radar_meta = run_async(api_manager.fetch_rainviewer_data())
        if not radar_meta:
            return jsonify({'error': 'Dados de radar indisponíveis no momento'}), 503

        grid = api_manager.get_radar_tile_grid(location.latitude, location.longitude, radar_meta)
        if not grid:
            return jsonify({'error': 'Não foi possível montar a imagem de radar'}), 503

        cache_manager.set(location_id, 'radar_grid', grid)
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
        return jsonify(sanitize(AdaptiveLearning(location_id).get_learning_report()))
    except Exception as e:
        logger.error(f"Erro ao obter relatório de aprendizado: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/learning/<int:location_id>/biases', methods=['GET'])
@login_required
@approved_required
def get_location_biases(location_id):
    try:
        if not _owned_location(location_id):
            return jsonify({'error': 'Localização não encontrada'}), 404
        return jsonify(sanitize(AdaptiveLearning(location_id).get_location_bias()))
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
        adaptive = AdaptiveLearning(location_id)
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
        AdaptiveLearning(location_id).record_prediction_observation(
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
}))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

"""
RRFS Meteorologia — Arquivo principal
Flask app + rotas + jobs background
"""
import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

import psutil
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, jsonify)
from flask_compress import Compress

import config
from modulos import auth, geocoding, meteorologia, radar, alertas, calibracao
from modulos.cache_manager import cache
from modulos.explicador import explicador
from modulos.precisao import sistema_precisao

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = config.PASTAS['logs']
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'rrfs.log', encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger('RRFS')

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates')
app.secret_key = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024
Compress(app)

# ── Inicialização ─────────────────────────────────────────────────────────────
auth.inicializar_admin()
analisador  = alertas.AnalisadorAlertas()
_inicio_srv = time.time()

WMO_MAP = {k: list(v) for k, v in meteorologia.WMO_CODES.items()}

# ── Registro de localizações monitoradas em background ────────────────────────
_locais_monitorados: dict = {}
_lock_locais = threading.RLock()


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _chave_local(lat: float, lon: float) -> str:
    """Chave de cache compartilhada por localização (arredonda 0.1°)."""
    return f"{round(lat, 1)}_{round(lon, 1)}"


def registrar_local_monitorado(username: str, lat: float, lon: float):
    """Registra local para monitoramento em background."""
    chave = _chave_local(lat, lon)
    with _lock_locais:
        if chave not in _locais_monitorados:
            _locais_monitorados[chave] = {
                'lat':      lat,
                'lon':      lon,
                'usuarios': [],
                'chave':    chave,
            }
        usuarios = _locais_monitorados[chave]['usuarios']
        if username not in usuarios:
            usuarios.append(username)
    logger.debug(
        f'Local registrado: {chave} — '
        f'usuários: {_locais_monitorados[chave]["usuarios"]}'
    )


def usuario_logado() -> str | None:
    return session.get('username')


def get_usuario() -> dict | None:
    """Retorna dados do usuário logado ou None se sessão inválida."""
    un = usuario_logado()
    if not un:
        return None
    try:
        return auth.obter_usuario(un)
    except Exception:
        return None


def requer_login(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not usuario_logado():
            return redirect(url_for('login'))
        # Verifica se usuário ainda existe e é válido
        u = get_usuario()
        if not u:
            session.clear()
            flash('Sessão expirada. Faça login novamente.', 'error')
            return redirect(url_for('login'))
        if u.get('status') not in ('ativo',):
            session.clear()
            flash('Conta suspensa ou inativa.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def requer_admin(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        u = get_usuario()
        if not u or u.get('perfil') != 'admin':
            flash('Acesso negado.', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


@app.context_processor
def contexto_global():
    try:
        u    = get_usuario()
        n_al = 0
        if u and u.get('localizacao', {}).get('lat'):
            al   = cache.get('alertas_usuario', username=u['username'])
            n_al = len(al or [])
        return dict(
            current_user=u,
            request_path=request.path,
            n_alertas=n_al,
            wmo_map=WMO_MAP,
        )
    except Exception:
        return dict(
            current_user=None,
            request_path=request.path,
            n_alertas=0,
            wmo_map=WMO_MAP,
        )


# ═══════════════════════════════════════════════════════════════════
# ROTAS PÚBLICAS
# ═══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    if usuario_logado() and get_usuario():
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    # Se já logado e usuário válido, vai para dashboard
    if usuario_logado() and get_usuario():
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        senha    = request.form.get('senha', '')
        u = auth.autenticar(username, senha)
        if u:
            session.clear()  # limpa sessão anterior
            session['username'] = u['username']
            session.permanent   = True
            app.permanent_session_lifetime = timedelta(days=30)
            logger.info(f'Login: {username}')
            # Registra local para monitoramento
            loc = u.get('localizacao', {})
            if loc.get('lat'):
                registrar_local_monitorado(
                    username, loc['lat'], loc['lon']
                )
            return redirect(url_for('dashboard'))
        flash('Usuário ou senha incorretos.', 'error')
    return render_template('login.html')


@app.route('/logout')
def logout():
    un = session.pop('username', None)
    session.clear()
    if un:
        logger.info(f'Logout: {un}')
    return redirect(url_for('login'))


@app.route('/cadastro', methods=['POST'])
def cadastro():
    nome   = request.form.get('nome',     '').strip()
    uname  = request.form.get('username', '').strip()
    email  = request.form.get('email',    '').strip()
    senha  = request.form.get('senha',    '')
    senha2 = request.form.get('senha2',   '')
    if senha != senha2:
        flash('As senhas não conferem.', 'error')
        return redirect(url_for('login'))
    ok, msg = auth.cadastrar_usuario(uname, nome, email, senha)
    flash(msg, 'success' if ok else 'error')
    return redirect(url_for('login'))


# ═══════════════════════════════════════════════════════════════════
# ROTAS AUTENTICADAS
# ═══════════════════════════════════════════════════════════════════

@app.route('/dashboard')
@requer_login
def dashboard():
    u = get_usuario()
    # Proteção extra — requer_login já garante, mas por segurança
    if not u:
        return redirect(url_for('login'))

    loc     = u.get('localizacao') or {}
    tem_loc = bool(loc.get('lat'))

    # Registra local para monitoramento em background
    if tem_loc:
        try:
            registrar_local_monitorado(
                u['username'], loc['lat'], loc['lon']
            )
        except Exception as e:
            logger.warning(f'Erro ao registrar local: {e}')

    return render_template('dashboard.html',
                           user=u,
                           tem_localizacao=tem_loc,
                           wmo_map=WMO_MAP)


@app.route('/radar')
@requer_login
def radar_page():
    u   = get_usuario()
    loc = (u.get('localizacao') or {}) if u else {}
    return render_template('radar.html',
                           user=u,
                           user_lat=loc.get('lat', -15.78),
                           user_lon=loc.get('lon', -47.93))


@app.route('/previsao')
@requer_login
def previsao():
    u   = get_usuario()
    loc = (u.get('localizacao') or {}) if u else {}
    return render_template('previsao.html',
                           user=u,
                           user_lat=loc.get('lat', 0),
                           user_lon=loc.get('lon', 0),
                           wmo_map=WMO_MAP)


@app.route('/alertas')
@requer_login
def alertas_page():
    u   = get_usuario()
    loc = (u.get('localizacao') or {}) if u else {}
    return render_template('alertas.html',
                           user=u,
                           user_lat=loc.get('lat', 0),
                           user_lon=loc.get('lon', 0))


@app.route('/configuracoes')
@requer_login
def configuracoes():
    u = get_usuario()
    return render_template('configuracoes.html', user=u)


@app.route('/configuracoes/localizacao', methods=['POST'])
@requer_login
def salvar_localizacao():
    u = get_usuario()
    if not u:
        return redirect(url_for('login'))
    try:
        lat = float(request.form.get('lat', 0))
        lon = float(request.form.get('lon', 0))
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError('Coordenadas fora do intervalo válido')
        loc = {
            'lat':          lat,
            'lon':          lon,
            'cidade':       request.form.get('cidade', ''),
            'estado':       request.form.get('estado', ''),
            'pais':         request.form.get('pais', ''),
            'display_name': request.form.get('display_name', ''),
        }
        auth.atualizar_localizacao(u['username'], loc)
        cache.limpar_namespace('alertas_usuario')
        registrar_local_monitorado(u['username'], lat, lon)
        flash('Localização salva com sucesso!', 'success')
    except Exception as e:
        flash(f'Erro ao salvar localização: {e}', 'error')
    return redirect(url_for('configuracoes'))


@app.route('/configuracoes/alertas', methods=['POST'])
@requer_login
def salvar_alertas():
    u = get_usuario()
    if not u:
        return redirect(url_for('login'))
    try:
        aj        = request.form.get('alertas_json', '{}')
        config_al = json.loads(aj)
        auth.atualizar_config_alertas(u['username'], config_al)
        cache.limpar_namespace('alertas_usuario')
        flash('Configurações de alerta salvas!', 'success')
    except Exception as e:
        flash(f'Erro: {e}', 'error')
    return redirect(url_for('configuracoes'))


@app.route('/configuracoes/senha', methods=['POST'])
@requer_login
def alterar_senha():
    u   = get_usuario()
    if not u:
        return redirect(url_for('login'))
    sa  = request.form.get('senha_atual',  '')
    ns  = request.form.get('nova_senha',   '')
    ns2 = request.form.get('nova_senha2',  '')
    if ns != ns2:
        flash('As senhas não conferem.', 'error')
    else:
        ok, msg = auth.alterar_senha(u['username'], sa, ns)
        flash(msg, 'success' if ok else 'error')
    return redirect(url_for('configuracoes'))


# ═══════════════════════════════════════════════════════════════════
# ADMIN
# ═══════════════════════════════════════════════════════════════════

@app.route('/admin')
@requer_login
@requer_admin
def admin():
    usuarios = auth.listar_usuarios()
    stats = {
        'total':     len(usuarios),
        'ativos':    sum(1 for u in usuarios if u['status'] == 'ativo'),
        'pendentes': sum(1 for u in usuarios if u['status'] == 'pendente'),
        'suspensos': sum(1 for u in usuarios if u['status'] == 'suspenso'),
    }
    return render_template('admin.html', usuarios=usuarios, stats=stats)


@app.route('/admin/usuario/<username>/<acao>', methods=['POST'])
@requer_login
@requer_admin
def admin_acao(username, acao):
    acoes = {
        'aprovar':   auth.aprovar_usuario,
        'rejeitar':  auth.rejeitar_usuario,
        'suspender': auth.suspender_usuario,
        'excluir':   auth.excluir_usuario,
        'reativar':  auth.reativar_usuario,
    }
    fn = acoes.get(acao)
    if fn:
        ok = fn(username)
        flash(
            f'{"Ação realizada" if ok else "Erro"}: {acao} — {username}',
            'success' if ok else 'error'
        )
    else:
        flash('Ação desconhecida.', 'error')
    return redirect(url_for('admin'))


@app.route('/admin/cache/limpar', methods=['POST'])
@requer_login
@requer_admin
def admin_limpar_cache():
    with cache._lock:
        cache._cache.clear()
    flash('Cache limpo com sucesso.', 'success')
    return redirect(url_for('admin'))


# ═══════════════════════════════════════════════════════════════════
# API — STATUS
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/status')
@requer_login
def api_status():
    """
    Verifica se há dados em cache para o usuário.
    Retorna rapidamente sem fazer chamadas externas.
    """
    try:
        u   = get_usuario()
        if not u:
            return jsonify({'tem_dados': False, 'tem_localizacao': False})

        loc = u.get('localizacao') or {}
        lat = loc.get('lat')
        lon = loc.get('lon')

        if not lat or not lon:
            return jsonify({
                'tem_dados':       False,
                'tem_cache':       False,
                'tem_localizacao': False,
            })

        chave_loc   = _chave_local(lat, lon)
        tem_meteo   = cache.get('tempo_atual',
                                lat=round(lat, 3),
                                lon=round(lon, 3)) is not None
        tem_alertas = cache.get('alertas_usuario',
                                username=u['username']) is not None

        return jsonify({
            'tem_dados':       tem_meteo,
            'tem_cache':       tem_meteo,
            'tem_localizacao': True,
            'tem_alertas':     tem_alertas,
            'chave_local':     chave_loc,
            'lat':             lat,
            'lon':             lon,
        })
    except Exception as e:
        logger.error(f'API status: {e}')
        return jsonify({'tem_dados': False, 'erro': str(e)}), 500


@app.route('/api/aguardar_dados')
@requer_login
def api_aguardar_dados():
    """
    Busca dados meteorológicos com timeout generoso.
    Se já tiver em cache retorna imediatamente.
    Resolve o dashboard ficando em 'analisando'.
    """
    try:
        u   = get_usuario()
        if not u:
            return jsonify({'erro': 'Sessão inválida', 'pronto': False}), 401

        loc = u.get('localizacao') or {}
        lat = float(request.args.get('lat', loc.get('lat', 0) or 0))
        lon = float(request.args.get('lon', loc.get('lon', 0) or 0))

        if not lat or not lon:
            return jsonify({'erro': 'Sem localização', 'pronto': False}), 400

        # Verifica cache primeiro
        dados = cache.get('tempo_atual',
                          lat=round(lat, 3), lon=round(lon, 3))
        if dados:
            return jsonify({'pronto': True, 'fonte': 'cache'})

        # Não tem cache — busca em background
        _resultado = {'pronto': False}

        def _buscar():
            try:
                d = meteorologia.buscar_tempo_completo(lat, lon)
                if d:
                    _resultado['pronto'] = True
                meteorologia.buscar_ensemble(lat, lon)
            except Exception as e:
                logger.warning(f'Busca background aguardar: {e}')

        t = threading.Thread(target=_buscar, daemon=True)
        t.start()

        # Polling até 10 segundos
        for _ in range(20):
            time.sleep(0.5)
            dados = cache.get('tempo_atual',
                              lat=round(lat, 3), lon=round(lon, 3))
            if dados:
                return jsonify({'pronto': True, 'fonte': 'api'})

        return jsonify({'pronto': False, 'motivo': 'timeout_api'})

    except Exception as e:
        logger.error(f'API aguardar dados: {e}')
        return jsonify({'pronto': False, 'erro': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
# API — METEOROLOGIA
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/meteo')
@requer_login
def api_meteo():
    try:
        u   = get_usuario()
        loc = (u.get('localizacao') or {}) if u else {}
        lat = float(request.args.get('lat', loc.get('lat', 0) or 0))
        lon = float(request.args.get('lon', loc.get('lon', 0) or 0))

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'erro': 'Coordenadas inválidas'}), 400

        dados = meteorologia.buscar_tempo_completo(lat, lon)
        if not dados:
            return jsonify({'erro': 'Sem dados do servidor meteorológico'}), 503

        threading.Thread(
            target=sistema_precisao.confirmar_com_dados_reais,
            args=(dados, lat, lon),
            daemon=True,
        ).start()

        return jsonify(dados)
    except Exception as e:
        logger.error(f'API meteo: {e}')
        return jsonify({'erro': str(e)}), 500


@app.route('/api/alertas')
@requer_login
def api_alertas():
    try:
        u   = get_usuario()
        if not u:
            return jsonify({'alertas': []}), 401

        loc = u.get('localizacao') or {}
        lat = float(request.args.get('lat', loc.get('lat', 0) or 0))
        lon = float(request.args.get('lon', loc.get('lon', 0) or 0))

        cached = cache.get('alertas_usuario', username=u['username'])
        if cached is not None:
            return jsonify({'alertas': cached})

        dados_m = meteorologia.buscar_tempo_completo(lat, lon)
        dados_e = meteorologia.buscar_ensemble(lat, lon)
        conf_al = u.get('config_alertas') or {}

        lista = alertas.verificar_alertas_usuario(
            dados_m, dados_e, lat, lon, conf_al
        )

        _registrar_previsoes_alertas(lista, dados_m, lat, lon)

        cache.set('alertas_usuario', lista,
                  ttl=config.CACHE_TTL['alertas'],
                  username=u['username'])
        return jsonify({'alertas': lista})
    except Exception as e:
        logger.error(f'API alertas: {e}')
        return jsonify({'alertas': [], 'erro': str(e)}), 500


def _registrar_previsoes_alertas(lista_alertas: list,
                                  dados_meteo: dict,
                                  lat: float, lon: float):
    try:
        for al in lista_alertas:
            tipo      = al.get('tipo', '')
            hora      = al.get('hora', '')
            confianca = al.get('concordancia', 1.0)
            if not hora:
                continue
            cat_map = {
                'tempestade':        'tempestade',
                'raio':              'raio',
                'chuva_forte':       'chuva_forte',
                'chuva_extrema':     'chuva_forte',
                'vento_forte':       'vento_forte',
                'furacao':           'vento_forte',
                'tornado':           'tornado',
                'microburst':        'microburst',
                'temperatura_alta':  'temperatura',
                'temperatura_baixa': 'temperatura',
            }
            cat = cat_map.get(tipo)
            if not cat:
                continue
            sistema_precisao.registrar_previsao(
                categoria=cat,
                previsao=al.get('dados', {}),
                hora_prevista=hora,
                lat=lat, lon=lon,
                confianca=confianca,
            )
    except Exception as e:
        logger.warning(f'Erro ao registrar previsões: {e}')


@app.route('/api/radar')
@requer_login
def api_radar():
    try:
        u   = get_usuario()
        loc = (u.get('localizacao') or {}) if u else {}
        lat = float(request.args.get('lat', loc.get('lat', -15.78) or -15.78))
        lon = float(request.args.get('lon', loc.get('lon', -47.93) or -47.93))
        dados = radar.obter_url_radar(lat, lon)
        return jsonify(dados)
    except Exception as e:
        logger.error(f'API radar: {e}')
        return jsonify({'erro': str(e)}), 500


@app.route('/api/geocoding')
@requer_login
def api_geocoding():
    q = request.args.get('q', '').strip()
    if len(q) < 2:
        return jsonify([])
    sugestoes = geocoding.buscar_sugestoes(q)
    return jsonify(sugestoes)


@app.route('/api/reverse')
@requer_login
def api_reverse():
    try:
        lat = float(request.args.get('lat'))
        lon = float(request.args.get('lon'))
        dados = geocoding.coords_para_cidade(lat, lon)
        return jsonify(dados or {})
    except Exception as e:
        return jsonify({'erro': str(e)}), 400


@app.route('/api/calibracao')
@requer_login
def api_calibracao():
    try:
        u   = get_usuario()
        loc = (u.get('localizacao') or {}) if u else {}
        lat = float(request.args.get('lat', loc.get('lat', 0) or 0))
        lon = float(request.args.get('lon', loc.get('lon', 0) or 0))
        cal = calibracao.CalibracaoEngine()
        return jsonify(cal.obter_estatisticas(lat, lon))
    except Exception as e:
        return jsonify({'erro': str(e)}), 500


@app.route('/api/ensemble/detalhado')
@requer_login
def api_ensemble_detalhado():
    try:
        u   = get_usuario()
        loc = (u.get('localizacao') or {}) if u else {}
        lat = float(request.args.get('lat', loc.get('lat', 0) or 0))
        lon = float(request.args.get('lon', loc.get('lon', 0) or 0))

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'erro': 'Coordenadas inválidas'}), 400

        dados_m  = meteorologia.buscar_tempo_completo(lat, lon)
        ensemble = meteorologia.buscar_ensemble(lat, lon)

        if not ensemble:
            return jsonify({
                'disponivel': False,
                'motivo': 'Ensemble não disponível',
            })

        idx    = meteorologia.hora_atual_indice(dados_m) if dados_m else 0
        resumo = meteorologia.resumo_ensemble_atual(ensemble, idx)

        if dados_m:
            hourly = dados_m.get('hourly', {})
            cape_arr = hourly.get('cape') or [None] * (idx + 1)
            resumo['referencia_principal'] = {
                'temperature_2m': dados_m.get('current', {}).get('temperature_2m'),
                'precipitation':  dados_m.get('current', {}).get('precipitation'),
                'wind_speed_10m': dados_m.get('current', {}).get('wind_speed_10m'),
                'wind_gusts_10m': dados_m.get('current', {}).get('wind_gusts_10m'),
                'cape':           cape_arr[idx] if idx < len(cape_arr) else None,
            }

        return jsonify(resumo)
    except Exception as e:
        logger.error(f'API ensemble detalhado: {e}')
        return jsonify({'erro': str(e), 'disponivel': False}), 500


@app.route('/api/ensemble/alerta')
@requer_login
def api_ensemble_alerta():
    try:
        u   = get_usuario()
        loc = (u.get('localizacao') or {}) if u else {}
        lat = float(request.args.get('lat', loc.get('lat', 0) or 0))
        lon = float(request.args.get('lon', loc.get('lon', 0) or 0))
        hora_iso = request.args.get('hora', '')

        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'erro': 'Coordenadas inválidas'}), 400

        ensemble = meteorologia.buscar_ensemble(lat, lon)
        if not ensemble:
            return jsonify({'disponivel': False,
                            'motivo': 'Ensemble não disponível'})

        dados_m = meteorologia.buscar_tempo_completo(lat, lon)
        idx     = 0

        if hora_iso and dados_m:
            try:
                times      = dados_m.get('hourly', {}).get('time', [])
                hora_trunc = hora_iso[:13]
                for i, t in enumerate(times):
                    if t[:13] == hora_trunc:
                        idx = i
                        break
            except Exception:
                idx = meteorologia.hora_atual_indice(dados_m)
        elif dados_m:
            idx = meteorologia.hora_atual_indice(dados_m)

        resumo = meteorologia.resumo_ensemble_por_hora(ensemble, idx)
        resumo['hora_solicitada'] = hora_iso
        resumo['idx_usado']       = idx

        return jsonify(resumo)
    except Exception as e:
        logger.error(f'API ensemble alerta: {e}')
        return jsonify({'erro': str(e), 'disponivel': False}), 500


# ═══════════════════════════════════════════════════════════════════
# API — PRECISÃO
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/precisao')
@requer_login
def api_precisao():
    try:
        return jsonify(sistema_precisao.obter_estatisticas())
    except Exception as e:
        logger.error(f'API precisão: {e}')
        return jsonify({'erro': str(e)}), 500


@app.route('/api/precisao/registrar', methods=['POST'])
@requer_login
def api_precisao_registrar():
    try:
        u   = get_usuario()
        if not u:
            return jsonify({'erro': 'Sessão inválida'}), 401
        loc = u.get('localizacao') or {}
        d   = request.get_json()
        if not d:
            return jsonify({'erro': 'Dados inválidos'}), 400

        pid = sistema_precisao.registrar_previsao_direcao(
            lat=float(loc.get('lat', 0) or 0),
            lon=float(loc.get('lon', 0) or 0),
            direcao=float(d.get('direcao', 0)),
            velocidade=float(d.get('velocidade', 0)),
            tipo_sistema=d.get('tipo', 'chuva'),
            hora_chegada=d.get('hora_chegada', ''),
            confianca=float(d.get('confianca', 0.75)),
        )
        return jsonify({'ok': True, 'id': pid})
    except Exception as e:
        return jsonify({'erro': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
# API — EXPLICAÇÕES
# ═══════════════════════════════════════════════════════════════════

def _lat_lon_usuario() -> tuple[float, float]:
    u   = get_usuario()
    loc = (u.get('localizacao') or {}) if u else {}
    lat = float(loc.get('lat') or request.args.get('lat', 0) or 0)
    lon = float(loc.get('lon') or request.args.get('lon', 0) or 0)
    return lat, lon


@app.route('/api/explicar/condicao-atual')
@requer_login
def api_explicar_condicao():
    try:
        lat, lon = _lat_lon_usuario()
        dados    = meteorologia.buscar_tempo_completo(lat, lon)
        if not dados:
            return jsonify({'erro': 'Sem dados'}), 503
        exp = explicador.explicar_condicao_atual(dados)
        return jsonify(exp)
    except Exception as e:
        logger.error(f'API explicar condição: {e}')
        return jsonify({'erro': str(e)}), 500


@app.route('/api/explicar/hora/<int:indice>')
@requer_login
def api_explicar_hora(indice):
    try:
        lat, lon = _lat_lon_usuario()
        dados    = meteorologia.buscar_tempo_completo(lat, lon)
        if not dados:
            return jsonify({'erro': 'Sem dados'}), 503
        exp = explicador.explicar_hora(dados, indice)
        return jsonify(exp)
    except Exception as e:
        logger.error(f'API explicar hora: {e}')
        return jsonify({'erro': str(e)}), 500


@app.route('/api/explicar/alerta', methods=['POST'])
@requer_login
def api_explicar_alerta():
    try:
        lat, lon = _lat_lon_usuario()
        alerta   = request.get_json()
        if not alerta:
            return jsonify({'erro': 'Dados inválidos'}), 400
        dados_m = meteorologia.buscar_tempo_completo(lat, lon)
        dados_e = meteorologia.buscar_ensemble(lat, lon)
        exp     = explicador.explicar_alerta(alerta, dados_m, dados_e)
        return jsonify(exp)
    except Exception as e:
        logger.error(f'API explicar alerta: {e}')
        return jsonify({'erro': str(e)}), 500


@app.route('/api/explicar/dia/<int:indice>')
@requer_login
def api_explicar_dia(indice):
    try:
        lat, lon = _lat_lon_usuario()
        dados    = meteorologia.buscar_tempo_completo(lat, lon)
        if not dados:
            return jsonify({'erro': 'Sem dados'}), 503
        exp = explicador.explicar_dia(dados, indice)
        return jsonify(exp)
    except Exception as e:
        logger.error(f'API explicar dia: {e}')
        return jsonify({'erro': str(e)}), 500


@app.route('/api/explicar/resumo')
@requer_login
def api_explicar_resumo():
    try:
        lat, lon = _lat_lon_usuario()
        dados_m  = meteorologia.buscar_tempo_completo(lat, lon)
        dados_e  = meteorologia.buscar_ensemble(lat, lon)
        u        = get_usuario()
        conf_al  = (u.get('config_alertas') or {}) if u else {}

        lista_al = alertas.verificar_alertas_usuario(
            dados_m, dados_e, lat, lon, conf_al
        )
        resumo = explicador.resumo_situacao(dados_m, lista_al, dados_e)
        return jsonify(resumo)
    except Exception as e:
        logger.error(f'API explicar resumo: {e}')
        return jsonify({'erro': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
# API — ADMIN
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/admin/sistema')
@requer_login
@requer_admin
def api_sistema():
    proc     = psutil.Process(os.getpid())
    uptime_s = int(time.time() - _inicio_srv)
    h, rem   = divmod(uptime_s, 3600)
    m, s     = divmod(rem, 60)
    with _lock_locais:
        n_locais = len(_locais_monitorados)
    return jsonify({
        'cpu':       psutil.cpu_percent(interval=0.5),
        'mem_mb':    round(proc.memory_info().rss / 1024 / 1024, 1),
        'uptime':    f'{h:02d}:{m:02d}:{s:02d}',
        'threads':   threading.active_count(),
        'pid':       os.getpid(),
        'locais_bg': n_locais,
    })


@app.route('/api/admin/cache')
@requer_login
@requer_admin
def api_cache_stats():
    return jsonify(cache.stats)


# ═══════════════════════════════════════════════════════════════════
# JOBS BACKGROUND
# ═══════════════════════════════════════════════════════════════════

def _carregar_todos_locais() -> list:
    """
    Retorna lista unificada de todos os locais a monitorar.
    Combina usuários ativos + locais registrados em runtime.
    Remove duplicatas por chave de localização.
    """
    locais = {}

    # 1. Usuários cadastrados no banco
    try:
        for u in auth.listar_usuarios():
            if u.get('status') == 'ativo':
                loc = u.get('localizacao') or {}
                lat = loc.get('lat')
                lon = loc.get('lon')
                if lat and lon:
                    chave = _chave_local(lat, lon)
                    if chave not in locais:
                        locais[chave] = {
                            'lat':      lat,
                            'lon':      lon,
                            'chave':    chave,
                            'usuarios': [],
                        }
                    if u['username'] not in locais[chave]['usuarios']:
                        locais[chave]['usuarios'].append(u['username'])
    except Exception as e:
        logger.warning(f'Erro ao carregar usuários para bg: {e}')

    # 2. Locais registrados em runtime
    with _lock_locais:
        for chave, info in _locais_monitorados.items():
            if chave not in locais:
                locais[chave] = {
                    'lat':      info['lat'],
                    'lon':      info['lon'],
                    'chave':    chave,
                    'usuarios': info['usuarios'][:],
                }

    return list(locais.values())


def job_atualizar_background():
    """
    Atualiza dados de todos os locais monitorados a cada 5 minutos.
    Roda sem necessidade de usuário abrir a página.
    Cache compartilhado por localização — vários usuários na mesma
    cidade usam o mesmo cache sem requisição duplicada.
    """
    time.sleep(30)  # aguarda servidor estabilizar
    logger.info('Job background iniciado — monitoramento contínuo ativo')

    while True:
        inicio_ciclo = time.time()
        try:
            locais = _carregar_todos_locais()
            logger.info(f'Job BG: atualizando {len(locais)} local(is)')

            for local in locais:
                lat   = local['lat']
                lon   = local['lon']
                chave = local['chave']
                try:
                    # Dados meteorológicos — cache compartilhado por local
                    dados = meteorologia.buscar_tempo_completo(lat, lon)
                    if dados:
                        logger.debug(f'BG meteo OK: {chave}')
                        sistema_precisao.confirmar_com_dados_reais(
                            dados, lat, lon
                        )

                    # Ensemble — cache compartilhado por local
                    ensemble = meteorologia.buscar_ensemble(lat, lon)

                    # Alertas — individuais por usuário
                    for username in local['usuarios']:
                        try:
                            u_data = auth.obter_usuario(username)
                            if not u_data or u_data.get('status') != 'ativo':
                                continue
                            conf_al = u_data.get('config_alertas') or {}
                            cache.delete('alertas_usuario', username=username)
                            lista_al = alertas.verificar_alertas_usuario(
                                dados, ensemble, lat, lon, conf_al
                            )
                            cache.set(
                                'alertas_usuario', lista_al,
                                ttl=config.CACHE_TTL['alertas'],
                                username=username,
                            )
                            _registrar_previsoes_alertas(
                                lista_al, dados, lat, lon
                            )
                            logger.debug(
                                f'BG alertas {username}: '
                                f'{len(lista_al)} alertas'
                            )
                        except Exception as e:
                            logger.warning(f'BG alertas {username}: {e}')

                    time.sleep(2)  # pausa entre locais

                except Exception as e:
                    logger.error(f'BG erro no local {chave}: {e}')

        except Exception as e:
            logger.error(f'Job BG erro geral: {e}')

        elapsed    = time.time() - inicio_ciclo
        sleep_time = max(10, 300 - elapsed)
        logger.debug(
            f'Job BG: ciclo em {elapsed:.0f}s, '
            f'próximo em {sleep_time:.0f}s'
        )
        time.sleep(sleep_time)


def job_monitorar_memoria():
    """Monitora memória a cada 2 min."""
    while True:
        time.sleep(120)
        try:
            proc   = psutil.Process(os.getpid())
            mem_mb = proc.memory_info().rss / 1024 / 1024
            if mem_mb > config.MEMORY_LIMIT_MB * 0.85:
                logger.warning(
                    f'Memória alta: {mem_mb:.0f} MB — limpando cache'
                )
                cache._limpar_expirados()
            if mem_mb > config.MEMORY_LIMIT_MB:
                with cache._lock:
                    remover = len(cache._cache) // 4
                    for _ in range(remover):
                        if cache._cache:
                            cache._cache.popitem(last=False)
                logger.warning(f'Cache reduzido ({remover} itens)')
        except Exception as e:
            logger.error(f'Job memória: {e}')


def job_sincronizar_locais():
    """
    Sincroniza locais monitorados com usuários cadastrados.
    Roda a cada 10 minutos para pegar usuários novos ou
    que mudaram de localização.
    """
    time.sleep(60)
    while True:
        try:
            for u in auth.listar_usuarios():
                if u.get('status') == 'ativo':
                    loc = u.get('localizacao') or {}
                    lat = loc.get('lat')
                    lon = loc.get('lon')
                    if lat and lon:
                        registrar_local_monitorado(
                            u['username'], lat, lon
                        )
            logger.debug(
                f'Sincronização: {len(_locais_monitorados)} locais ativos'
            )
        except Exception as e:
            logger.error(f'Job sincronizar locais: {e}')
        time.sleep(600)


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('  RRFS Meteorologia — iniciando')
    logger.info(f'  Acesse: http://localhost:{config.PORT}')
    logger.info(f'  Admin padrão: admin / admin')
    logger.info('=' * 60)

    # Carrega locais de todos os usuários ativos no início
    try:
        for u in auth.listar_usuarios():
            if u.get('status') == 'ativo':
                loc = u.get('localizacao') or {}
                if loc.get('lat'):
                    registrar_local_monitorado(
                        u['username'], loc['lat'], loc['lon']
                    )
        logger.info(
            f'Locais carregados: {len(_locais_monitorados)} local(is)'
        )
    except Exception as e:
        logger.warning(f'Erro ao carregar locais iniciais: {e}')

    # Inicia jobs em background
    jobs = [
        (job_atualizar_background, 'Job-Background'),
        (job_monitorar_memoria,    'Job-Memoria'),
        (job_sincronizar_locais,   'Job-SincLocais'),
    ]
    for job_fn, nome in jobs:
        t = threading.Thread(target=job_fn, daemon=True, name=nome)
        t.start()
        logger.info(f'Thread {nome} iniciada')

    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True,
        use_reloader=False,
    )

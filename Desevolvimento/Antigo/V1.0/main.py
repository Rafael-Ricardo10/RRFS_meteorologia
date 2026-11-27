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
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4 MB
Compress(app)

# ── Inicialização ─────────────────────────────────────────────────────────────
auth.inicializar_admin()
analisador  = alertas.AnalisadorAlertas()
_inicio_srv = time.time()

# WMO map para templates (serialização JSON)
WMO_MAP = {k: list(v) for k, v in meteorologia.WMO_CODES.items()}


# ═══════════════════════════════════════════════════════════════════
# HELPERS DE SESSÃO / DECORADORES
# ═══════════════════════════════════════════════════════════════════

def usuario_logado() -> str | None:
    return session.get('username')


def get_usuario() -> dict | None:
    un = usuario_logado()
    if not un:
        return None
    return auth.obter_usuario(un)


def requer_login(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not usuario_logado():
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


# ── Context processor global ──────────────────────────────────────────────────
@app.context_processor
def contexto_global():
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


# ═══════════════════════════════════════════════════════════════════
# ROTAS PÚBLICAS
# ═══════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    if usuario_logado():
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if usuario_logado():
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        senha    = request.form.get('senha', '')
        u = auth.autenticar(username, senha)
        if u:
            session['username'] = u['username']
            session.permanent   = True
            app.permanent_session_lifetime = timedelta(days=30)
            logger.info(f'Login: {username}')
            return redirect(url_for('dashboard'))
        flash('Usuário ou senha incorretos.', 'error')
    return render_template('login.html')


@app.route('/logout')
def logout():
    un = session.pop('username', None)
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
# ROTAS AUTENTICADAS — PÁGINAS
# ═══════════════════════════════════════════════════════════════════

@app.route('/dashboard')
@requer_login
def dashboard():
    u       = get_usuario()
    tem_loc = bool(u.get('localizacao', {}).get('lat'))
    return render_template('dashboard.html',
                           user=u,
                           tem_localizacao=tem_loc,
                           wmo_map=WMO_MAP)


@app.route('/radar')
@requer_login
def radar_page():
    u   = get_usuario()
    loc = u.get('localizacao', {})
    return render_template('radar.html',
                           user=u,
                           user_lat=loc.get('lat', -15.78),
                           user_lon=loc.get('lon', -47.93))


@app.route('/previsao')
@requer_login
def previsao():
    u   = get_usuario()
    loc = u.get('localizacao', {})
    return render_template('previsao.html',
                           user=u,
                           user_lat=loc.get('lat', 0),
                           user_lon=loc.get('lon', 0),
                           wmo_map=WMO_MAP)


@app.route('/alertas')
@requer_login
def alertas_page():
    u   = get_usuario()
    loc = u.get('localizacao', {})
    return render_template('alertas.html',
                           user=u,
                           user_lat=loc.get('lat', 0),
                           user_lon=loc.get('lon', 0))


@app.route('/configuracoes')
@requer_login
def configuracoes():
    u = get_usuario()
    return render_template('configuracoes.html', user=u)


# ── Salvar localização ────────────────────────────────────────────────────────
@app.route('/configuracoes/localizacao', methods=['POST'])
@requer_login
def salvar_localizacao():
    u = get_usuario()
    try:
        lat = float(request.form.get('lat', 0))
        lon = float(request.form.get('lon', 0))
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError('Coordenadas fora do intervalo válido')
        loc = {
            'lat':          lat,
            'lon':          lon,
            'cidade':       request.form.get('cidade',       ''),
            'estado':       request.form.get('estado',       ''),
            'pais':         request.form.get('pais',         ''),
            'display_name': request.form.get('display_name', ''),
        }
        auth.atualizar_localizacao(u['username'], loc)
        cache.limpar_namespace('alertas_usuario')
        flash('Localização salva com sucesso!', 'success')
    except Exception as e:
        flash(f'Erro ao salvar localização: {e}', 'error')
    return redirect(url_for('configuracoes'))


# ── Salvar alertas personalizados ─────────────────────────────────────────────
@app.route('/configuracoes/alertas', methods=['POST'])
@requer_login
def salvar_alertas():
    u = get_usuario()
    try:
        aj       = request.form.get('alertas_json', '{}')
        config_al = json.loads(aj)
        auth.atualizar_config_alertas(u['username'], config_al)
        cache.limpar_namespace('alertas_usuario')
        flash('Configurações de alerta salvas!', 'success')
    except Exception as e:
        flash(f'Erro: {e}', 'error')
    return redirect(url_for('configuracoes'))


# ── Alterar senha ─────────────────────────────────────────────────────────────
@app.route('/configuracoes/senha', methods=['POST'])
@requer_login
def alterar_senha():
    u   = get_usuario()
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
# ROTAS ADMIN
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
# API — DADOS METEOROLÓGICOS
# ═══════════════════════════════════════════════════════════════════

@app.route('/api/meteo')
@requer_login
def api_meteo():
    try:
        lat = float(request.args.get('lat', 0))
        lon = float(request.args.get('lon', 0))
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'erro': 'Coordenadas inválidas'}), 400
        dados = meteorologia.buscar_tempo_completo(lat, lon)
        if not dados:
            return jsonify({'erro': 'Sem dados do servidor meteorológico'}), 503
        return jsonify(dados)
    except Exception as e:
        logger.error(f'API meteo: {e}')
        return jsonify({'erro': str(e)}), 500


@app.route('/api/alertas')
@requer_login
def api_alertas():
    try:
        u   = get_usuario()
        lat = float(request.args.get('lat', 0))
        lon = float(request.args.get('lon', 0))

        # Cache por usuário
        cached = cache.get('alertas_usuario', username=u['username'])
        if cached is not None:
            return jsonify({'alertas': cached})

        dados_m = meteorologia.buscar_tempo_completo(lat, lon)
        dados_e = meteorologia.buscar_ensemble(lat, lon)
        conf_al = u.get('config_alertas', {})

        lista = alertas.verificar_alertas_usuario(
            dados_m, dados_e, lat, lon, conf_al
        )
        cache.set('alertas_usuario', lista,
                  ttl=config.CACHE_TTL['alertas'],
                  username=u['username'])
        return jsonify({'alertas': lista})
    except Exception as e:
        logger.error(f'API alertas: {e}')
        return jsonify({'alertas': [], 'erro': str(e)}), 500


@app.route('/api/radar')
@requer_login
def api_radar():
    try:
        lat = float(request.args.get('lat', -15.78))
        lon = float(request.args.get('lon', -47.93))
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
        lat = float(request.args.get('lat', 0))
        lon = float(request.args.get('lon', 0))
        cal = calibracao.CalibracaoEngine()
        return jsonify(cal.obter_estatisticas(lat, lon))
    except Exception as e:
        return jsonify({'erro': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
# API — EXPLICAÇÕES
# ═══════════════════════════════════════════════════════════════════

def _lat_lon_usuario() -> tuple[float, float]:
    """Retorna lat/lon do usuário logado."""
    u   = get_usuario()
    loc = u.get('localizacao', {}) if u else {}
    lat = float(loc.get('lat', request.args.get('lat', 0)))
    lon = float(loc.get('lon', request.args.get('lon', 0)))
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
        dados_m  = meteorologia.buscar_tempo_completo(lat, lon)
        dados_e  = meteorologia.buscar_ensemble(lat, lon)
        exp      = explicador.explicar_alerta(alerta, dados_m, dados_e)
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
        conf_al  = u.get('config_alertas', {}) if u else {}

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
    return jsonify({
        'cpu':     psutil.cpu_percent(interval=0.5),
        'mem_mb':  round(proc.memory_info().rss / 1024 / 1024, 1),
        'uptime':  f'{h:02d}:{m:02d}:{s:02d}',
        'threads': threading.active_count(),
        'pid':     os.getpid(),
    })


@app.route('/api/admin/cache')
@requer_login
@requer_admin
def api_cache_stats():
    return jsonify(cache.stats)


# ═══════════════════════════════════════════════════════════════════
# JOBS EM BACKGROUND
# ═══════════════════════════════════════════════════════════════════

def job_atualizar_alertas():
    """Invalida cache de alertas a cada 5 min para todos os usuários ativos."""
    while True:
        time.sleep(300)
        try:
            for u in auth.listar_usuarios():
                if u['status'] == 'ativo' and u.get('localizacao', {}).get('lat'):
                    cache.delete('alertas_usuario', username=u['username'])
            logger.debug('Cache de alertas invalidado')
        except Exception as e:
            logger.error(f'Job alertas: {e}')


def job_monitorar_memoria():
    """Monitora e limpa memória se necessário (roda a cada 2 min)."""
    while True:
        time.sleep(120)
        try:
            proc   = psutil.Process(os.getpid())
            mem_mb = proc.memory_info().rss / 1024 / 1024
            if mem_mb > config.MEMORY_LIMIT_MB * 0.85:
                logger.warning(f'Memória alta: {mem_mb:.0f} MB — limpando cache')
                cache._limpar_expirados()
            if mem_mb > config.MEMORY_LIMIT_MB:
                with cache._lock:
                    remover = len(cache._cache) // 4
                    for _ in range(remover):
                        if cache._cache:
                            cache._cache.popitem(last=False)
                logger.warning(f'Cache reduzido ({remover} itens removidos)')
        except Exception as e:
            logger.error(f'Job memória: {e}')


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info('=' * 60)
    logger.info('  RRFS Meteorologia — iniciando')
    logger.info(f'  Acesse: http://localhost:{config.PORT}')
    logger.info(f'  Admin padrão: admin / admin')
    logger.info('=' * 60)

    # Threads de background
    for job, nome in [
        (job_atualizar_alertas, 'Job-Alertas'),
        (job_monitorar_memoria, 'Job-Memoria'),
    ]:
        t = threading.Thread(target=job, daemon=True, name=nome)
        t.start()
        logger.info(f'Thread {nome} iniciada')

    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        threaded=True,
        use_reloader=False,
    )

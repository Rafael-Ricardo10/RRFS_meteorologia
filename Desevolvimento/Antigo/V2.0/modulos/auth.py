"""
Autenticação e gerenciamento de usuários.
Armazenamento em JSON (sem banco de dados externo).
"""
import json
import hashlib
import secrets
import logging
from datetime import datetime
from pathlib import Path
import config

logger = logging.getLogger('RRFS.Auth')

USUARIOS_DIR = config.PASTAS['usuarios']
DB_FILE      = USUARIOS_DIR / 'usuarios.json'


def _hash_senha(senha: str, salt: str = None) -> tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac('sha256', senha.encode(), salt.encode(), 260000)
    return h.hex(), salt


def _carregar_db() -> dict:
    if not DB_FILE.exists():
        return {'usuarios': {}}
    try:
        return json.loads(DB_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {'usuarios': {}}


def _salvar_db(db: dict):
    DB_FILE.write_text(
        json.dumps(db, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


def inicializar_admin():
    """Cria conta admin se não existir."""
    db = _carregar_db()
    if 'admin' not in db['usuarios']:
        h, s = _hash_senha(config.ADMIN_DEFAULT['password'])
        db['usuarios']['admin'] = {
            'username':   'admin',
            'nome':       'Administrador',
            'email':      'admin@rrfs.local',
            'senha_hash': h,
            'salt':       s,
            'perfil':     'admin',
            'status':     'ativo',
            'criado_em':  datetime.now().isoformat(),
            'ultimo_login': None,
            'localizacao': {},
            'config_alertas': {'condicoes_alerta': []},
            'senha_padrao': True,
        }
        _salvar_db(db)
        logger.info('Admin padrão criado (admin/admin)')


def autenticar(username: str, senha: str) -> dict | None:
    """Retorna dados do usuário se autenticado."""
    db = _carregar_db()
    user = db['usuarios'].get(username.lower())
    if not user:
        return None
    if user.get('status') not in ('ativo',):
        return None
    h, _ = _hash_senha(senha, user['salt'])
    if h != user['senha_hash']:
        return None
    # Atualiza último login
    user['ultimo_login'] = datetime.now().isoformat()
    _salvar_db(db)
    return {k: v for k, v in user.items()
            if k not in ('senha_hash', 'salt')}


def cadastrar_usuario(username: str, nome: str, email: str,
                      senha: str) -> tuple[bool, str]:
    """Cadastra novo usuário (status: pendente até admin aprovar)."""
    db = _carregar_db()
    username = username.lower().strip()

    if len(username) < 3:
        return False, 'Username muito curto (mínimo 3 caracteres)'
    if not username.isalnum():
        return False, 'Username deve conter apenas letras e números'
    if username in db['usuarios']:
        return False, 'Username já existe'
    if len(senha) < 6:
        return False, 'Senha muito curta (mínimo 6 caracteres)'
    if any(u['email'] == email for u in db['usuarios'].values()):
        return False, 'E-mail já cadastrado'

    h, s = _hash_senha(senha)
    db['usuarios'][username] = {
        'username':    username,
        'nome':        nome,
        'email':       email,
        'senha_hash':  h,
        'salt':        s,
        'perfil':      'usuario',
        'status':      'pendente',
        'criado_em':   datetime.now().isoformat(),
        'ultimo_login': None,
        'localizacao': {},
        'config_alertas': {'condicoes_alerta': []},
        'senha_padrao': False,
    }
    _salvar_db(db)
    return True, 'Cadastro realizado! Aguardando aprovação do administrador.'


def alterar_senha(username: str, senha_atual: str,
                  nova_senha: str) -> tuple[bool, str]:
    db = _carregar_db()
    user = db['usuarios'].get(username)
    if not user:
        return False, 'Usuário não encontrado'
    h, _ = _hash_senha(senha_atual, user['salt'])
    if h != user['senha_hash']:
        return False, 'Senha atual incorreta'
    if len(nova_senha) < 6:
        return False, 'Nova senha muito curta'
    h2, s2 = _hash_senha(nova_senha)
    user['senha_hash']  = h2
    user['salt']        = s2
    user['senha_padrao'] = False
    _salvar_db(db)
    return True, 'Senha alterada com sucesso'


def obter_usuario(username: str) -> dict | None:
    db = _carregar_db()
    user = db['usuarios'].get(username)
    if not user:
        return None
    return {k: v for k, v in user.items()
            if k not in ('senha_hash', 'salt')}


def atualizar_localizacao(username: str, localizacao: dict):
    db = _carregar_db()
    if username in db['usuarios']:
        db['usuarios'][username]['localizacao'] = localizacao
        _salvar_db(db)


def atualizar_config_alertas(username: str, config_alertas: dict):
    db = _carregar_db()
    if username in db['usuarios']:
        db['usuarios'][username]['config_alertas'] = config_alertas
        _salvar_db(db)


# ── Admin ─────────────────────────────────────────────────────────────────

def listar_usuarios() -> list:
    db = _carregar_db()
    return [
        {k: v for k, v in u.items() if k not in ('senha_hash', 'salt')}
        for u in db['usuarios'].values()
    ]


def aprovar_usuario(username: str) -> bool:
    db = _carregar_db()
    if username in db['usuarios']:
        db['usuarios'][username]['status'] = 'ativo'
        _salvar_db(db)
        return True
    return False


def rejeitar_usuario(username: str) -> bool:
    db = _carregar_db()
    if username in db['usuarios']:
        db['usuarios'][username]['status'] = 'rejeitado'
        _salvar_db(db)
        return True
    return False


def suspender_usuario(username: str) -> bool:
    db = _carregar_db()
    u = db['usuarios'].get(username)
    if u and u.get('perfil') != 'admin':
        db['usuarios'][username]['status'] = 'suspenso'
        _salvar_db(db)
        return True
    return False


def excluir_usuario(username: str) -> bool:
    db = _carregar_db()
    if username in db['usuarios'] and username != 'admin':
        del db['usuarios'][username]
        _salvar_db(db)
        return True
    return False


def reativar_usuario(username: str) -> bool:
    db = _carregar_db()
    if username in db['usuarios']:
        db['usuarios'][username]['status'] = 'ativo'
        _salvar_db(db)
        return True
    return False

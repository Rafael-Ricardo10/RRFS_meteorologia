#!/usr/bin/env bash
# =============================================================================
#  Instalador RRFS Meteorologia (Debian)
#  Uso: sudo ./install.sh            (menu interativo)
#       sudo ./install.sh install    |  sudo ./install.sh reinstall
#       sudo ./install.sh service    (gerenciar serviço)
#  Rode a partir da pasta que contém app.py, config.py, modulos/, templates/ e requirements.txt
# =============================================================================
set -Eeuo pipefail

APP_NAME="rrfs-meteorologia"
APP_DIR="/DATA/AppData/RRFS_meteorologia"
APP_USER="rrfs"
SERVICE_FILE="/etc/systemd/system/${APP_NAME}.service"
PORT="${PORT:-5000}"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# itens preservados na reinstalação (dados, banco SQLite do Flask, venv, segredos)
KEEP_ITEMS=(dados instance venv .env)

C_G="\033[1;32m"; C_Y="\033[1;33m"; C_R="\033[1;31m"; C_B="\033[1;34m"; C_0="\033[0m"
info()  { echo -e "${C_B}[INFO]${C_0} $*"; }
ok()    { echo -e "${C_G}[ OK ]${C_0} $*"; }
warn()  { echo -e "${C_Y}[AVISO]${C_0} $*"; }
die()   { echo -e "${C_R}[ERRO]${C_0} $*" >&2; exit 1; }

SERVICE_WAS_STOPPED=0
on_error() {
    echo -e "${C_R}[ERRO]${C_0} Falha na linha $1." >&2
    if [[ $SERVICE_WAS_STOPPED -eq 1 ]]; then
        warn "Tentando religar o serviço..."
        systemctl start "$APP_NAME" 2>/dev/null || true
    fi
}
trap 'on_error $LINENO' ERR

# ------------------------------------------------------------------ checagens
[[ $EUID -eq 0 ]] || die "Execute como root: sudo ./install.sh"
command -v apt-get >/dev/null || die "Este instalador é para Debian/Ubuntu (apt)."
[[ -f "$SRC_DIR/app.py" && -f "$SRC_DIR/requirements.txt" ]] \
    || die "app.py/requirements.txt não encontrados em $SRC_DIR. Rode o instalador na pasta do sistema."
[[ "$SRC_DIR" != "$APP_DIR" ]] || die "Rode o instalador a partir da pasta de origem, não de $APP_DIR."

is_installed() { [[ -f "$SERVICE_FILE" || -f "$APP_DIR/app.py" ]]; }

# ------------------------------------------------------------------- funções
install_packages() {
    info "Instalando dependências do sistema (apt)..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq python3 python3-venv python3-pip rsync ca-certificates >/dev/null
    ok "Pacotes do sistema prontos."
}

ensure_user() {
    if ! id "$APP_USER" &>/dev/null; then
        useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
        ok "Usuário de serviço '$APP_USER' criado."
    fi
}

copy_system_files() {
    info "Copiando arquivos do sistema para $APP_DIR ..."
    mkdir -p "$APP_DIR"
    rsync -a \
        --exclude 'install.sh' --exclude '.git' --exclude '.gitignore' \
        --exclude '__pycache__' --exclude '*.pyc' --exclude 'venv' --exclude '.venv' \
        --exclude 'dados' --exclude 'instance' --exclude '.env' \
        "$SRC_DIR"/ "$APP_DIR"/
    # primeira instalação: se a origem trouxe uma pasta dados/, copia só se o destino não tiver
    if [[ -d "$SRC_DIR/dados" && ! -d "$APP_DIR/dados" ]]; then
        cp -a "$SRC_DIR/dados" "$APP_DIR/dados"
    fi
    mkdir -p "$APP_DIR"/dados/{cache,models,raw,processed} "$APP_DIR/instance"
    ok "Arquivos copiados."
}

setup_venv() {
    if [[ -x "$APP_DIR/venv/bin/python" ]] && "$APP_DIR/venv/bin/python" -c "import sys" 2>/dev/null; then
        info "Ambiente virtual existente mantido."
    else
        [[ -d "$APP_DIR/venv" ]] && { warn "venv quebrado, recriando..."; rm -rf "$APP_DIR/venv"; }
        info "Criando ambiente virtual..."
        python3 -m venv "$APP_DIR/venv"
    fi

    info "Instalando/atualizando bibliotecas conforme requirements.txt..."
    "$APP_DIR/venv/bin/pip" install --quiet --upgrade pip

    # CORREÇÃO: Instala gunicorn separadamente (necessário para produção)
    "$APP_DIR/venv/bin/pip" install --quiet gunicorn

    "$APP_DIR/venv/bin/pip" install --upgrade -r "$APP_DIR/requirements.txt"
    "$APP_DIR/venv/bin/pip" check || warn "pip check reportou conflitos (veja acima)."

    # VALIDAÇÃO: Verifica se gunicorn foi instalado com sucesso
    if ! "$APP_DIR/venv/bin/gunicorn" --version >/dev/null 2>&1; then
        die "Falha ao instalar gunicorn. Verifique sua conexão de internet."
    fi

    ok "Bibliotecas verificadas e atualizadas."
}

ensure_env_file() {
    if [[ ! -f "$APP_DIR/.env" ]]; then
        local key
        key="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
        cat > "$APP_DIR/.env" << ENVEOF
SECRET_KEY=$key
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=mistral
ENVEOF
        ok "Arquivo .env criado com SECRET_KEY aleatória."
    fi
}

apply_permissions() {
    info "Aplicando permissões..."
    chown -R "$APP_USER:$APP_USER" "$APP_DIR"
    find "$APP_DIR" -type d -exec chmod 750 {} +
    find "$APP_DIR" -type f -exec chmod 640 {} +

    # CORREÇÃO: Mais cuidadoso com permissões do venv
    if [[ -d "$APP_DIR/venv/bin" ]]; then
        chmod 755 "$APP_DIR/venv/bin"
        chmod 755 "$APP_DIR/venv/bin/python" "$APP_DIR/venv/bin/gunicorn" 2>/dev/null || \
            warn "Alguns executáveis do venv podem não estar acessíveis"
    fi

    chmod 600 "$APP_DIR/.env"
    # o pai (/DATA/AppData) precisa permitir travessia
    chmod o+x "$(dirname "$APP_DIR")" 2>/dev/null || true
    ok "Permissões aplicadas."
}

create_service() {
    info "Criando serviço systemd..."
    cat > "$SERVICE_FILE" << 'SVCEOF'
[Unit]
Description=RRFS Meteorologia
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=rrfs
Group=rrfs
WorkingDirectory=/DATA/AppData/RRFS_meteorologia
EnvironmentFile=/DATA/AppData/RRFS_meteorologia/.env
ExecStart=/DATA/AppData/RRFS_meteorologia/venv/bin/gunicorn --workers 1 --threads 8 --timeout 180 --bind 0.0.0.0:5000 app:app
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVCEOF

    systemctl daemon-reload
    systemctl enable "$APP_NAME" >/dev/null 2>&1
    ok "Serviço criado e habilitado no boot."
}

start_service() {
    info "Iniciando serviço..."

    # CORREÇÃO: Aguarda um pouco antes de checar
    systemctl restart "$APP_NAME"
    SERVICE_WAS_STOPPED=0
    sleep 5

    # MELHOR DIAGNÓSTICO: Tenta 3 vezes com pequenas esperas
    local attempt=0
    local max_attempts=3
    while [[ $attempt -lt $max_attempts ]]; do
        if systemctl is-active --quiet "$APP_NAME"; then
            local ip; ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
            ok "Serviço ativo!  Acesse: http://${ip:-IP_DO_SERVIDOR}:$PORT"
            warn "Login padrão: admin / admin  -> TROQUE a senha logo após entrar."
            return 0
        fi
        attempt=$((attempt + 1))
        if [[ $attempt -lt $max_attempts ]]; then
            warn "Tentativa $attempt falhou, aguardando..."
            sleep 3
        fi
    done

    # Se chegou aqui, serviço não subiu
    echo ""
    warn "Serviço não subiu na primeira tentativa. Verificando logs..."
    echo ""
    journalctl -u "$APP_NAME" -n 50 --no-pager || true
    echo ""
    die "O serviço não subiu. Verifique: journalctl -u $APP_NAME -f"
}

do_install() {
    info "=== INSTALAÇÃO ==="
    install_packages
    ensure_user
    copy_system_files
    setup_venv
    ensure_env_file
    apply_permissions
    create_service
    start_service
}

do_reinstall() {
    info "=== REINSTALAÇÃO ==="
    install_packages
    ensure_user

    if systemctl is-active --quiet "$APP_NAME" 2>/dev/null; then
        info "Parando serviço..."
        systemctl stop "$APP_NAME"
        SERVICE_WAS_STOPPED=1
        sleep 2
    fi

    [[ -n "$APP_DIR" && "$APP_DIR" != "/" && -d "$APP_DIR" ]] || die "Pasta de instalação inválida."
    info "Removendo arquivos antigos (mantendo: ${KEEP_ITEMS[*]})..."
    local find_args=() item
    for item in "${KEEP_ITEMS[@]}"; do find_args+=( ! -name "$item" ); done
    find "$APP_DIR" -mindepth 1 -maxdepth 1 "${find_args[@]}" -exec rm -rf {} + 2>/dev/null || true

    copy_system_files
    setup_venv
    ensure_env_file
    apply_permissions
    create_service
    start_service
}

# ============================================================ GERENCIAMENTO DE SERVIÇO
service_status() {
    info "Status do Serviço $APP_NAME"
    echo "=============================================="
    if systemctl is-active --quiet "$APP_NAME"; then
        ok "Serviço está ATIVO"
        systemctl status "$APP_NAME" --no-pager || true
    else
        warn "Serviço está INATIVO"
        systemctl status "$APP_NAME" --no-pager || true
    fi
    echo "=============================================="
}

service_start() {
    info "Iniciando o serviço..."
    if systemctl is-active --quiet "$APP_NAME"; then
        warn "Serviço já está ativo."
        return 0
    fi
    systemctl start "$APP_NAME"
    sleep 3
    if systemctl is-active --quiet "$APP_NAME"; then
        ok "Serviço iniciado com sucesso!"
        local ip; ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
        info "Acesse: http://${ip:-IP_DO_SERVIDOR}:$PORT"
    else
        die "Falha ao iniciar o serviço. Verifique os logs."
    fi
}

service_stop() {
    info "Parando o serviço..."
    if ! systemctl is-active --quiet "$APP_NAME"; then
        warn "Serviço já está parado."
        return 0
    fi
    systemctl stop "$APP_NAME"
    sleep 2
    if ! systemctl is-active --quiet "$APP_NAME"; then
        ok "Serviço parado com sucesso!"
    else
        die "Falha ao parar o serviço."
    fi
}

service_restart() {
    info "Reiniciando o serviço..."
    systemctl restart "$APP_NAME"
    sleep 5
    if systemctl is-active --quiet "$APP_NAME"; then
        ok "Serviço reiniciado com sucesso!"
        local ip; ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
        info "Acesse: http://${ip:-IP_DO_SERVIDOR}:$PORT"
    else
        die "Falha ao reiniciar o serviço. Verifique os logs."
    fi
}

service_logs() {
    info "Exibindo últimos 100 linhas de logs (CTRL+C para sair)..."
    echo "=============================================="
    journalctl -u "$APP_NAME" -n 100 --no-pager
    echo "=============================================="
    echo ""
    read -rp "Deseja ver logs em tempo real? [S/n] " r
    if [[ ! "$r" =~ ^[nN]$ ]]; then
        info "Logs em tempo real (CTRL+C para sair)..."
        journalctl -u "$APP_NAME" -f || true
    fi
}

service_logs_full() {
    info "Exibindo TODOS os logs (últimos 500 linhas)..."
    echo "=============================================="
    journalctl -u "$APP_NAME" -n 500 --no-pager || true
    echo "=============================================="
}

service_enable() {
    info "Habilitando serviço no boot..."
    systemctl enable "$APP_NAME" >/dev/null 2>&1
    ok "Serviço habilitado para iniciar no boot."
}

service_disable() {
    info "Desabilitando serviço no boot..."
    systemctl disable "$APP_NAME" >/dev/null 2>&1
    ok "Serviço desabilitado (não iniciará mais no boot)."
}

service_reload() {
    info "Recarregando configurações do daemon..."
    systemctl daemon-reload
    ok "Daemon recarregado."
}

# ============================================================ MENU DE SERVIÇO
show_service_menu() {
    echo ""
    echo "=============================================="
    echo "   RRFS Meteorologia - Gerenciador de Serviço"
    echo "=============================================="

    if systemctl is-active --quiet "$APP_NAME" 2>/dev/null; then
        ok "Serviço STATUS: ATIVO ✓"
    else
        warn "Serviço STATUS: INATIVO ✗"
    fi

    echo ""
    echo "  1) Ver Status do Serviço"
    echo "  2) Iniciar Serviço"
    echo "  3) Parar Serviço"
    echo "  4) Reiniciar Serviço"
    echo "  5) Ver Logs (últimas 100 linhas)"
    echo "  6) Ver Todos os Logs (últimas 500 linhas)"
    echo "  7) Logs em Tempo Real"
    echo "  8) Habilitar no Boot"
    echo "  9) Desabilitar no Boot"
    echo " 10) Recarregar Daemon"
    echo " 11) Voltar ao Menu Principal"
    read -rp "Escolha [1-11]: " opt

    case "$opt" in
        1) service_status; service_menu_loop ;;
        2) service_start; service_menu_loop ;;
        3) service_stop; service_menu_loop ;;
        4) service_restart; service_menu_loop ;;
        5) service_logs; service_menu_loop ;;
        6) service_logs_full; service_menu_loop ;;
        7) journalctl -u "$APP_NAME" -f || true; service_menu_loop ;;
        8) service_enable; service_menu_loop ;;
        9) service_disable; service_menu_loop ;;
        10) service_reload; service_menu_loop ;;
        11) return ;;
        *) warn "Opção inválida!"; service_menu_loop ;;
    esac
}

service_menu_loop() {
    echo ""
    read -rp "Pressione ENTER para continuar..." dummy
    show_service_menu
}

# ----------------------------------------------------------------------- menu
ACTION="${1:-}"
if [[ -z "$ACTION" ]]; then
    echo "=============================================="
    echo "   RRFS Meteorologia - Instalador"
    echo "   Destino: $APP_DIR"
    is_installed && echo "   Status : JÁ INSTALADO" || echo "   Status : não instalado"
    echo "=============================================="
    echo "  1) Instalar"
    echo "  2) Reinstalar (mantém dados, instance, venv e .env)"
    echo "  3) Gerenciar Serviço"
    echo "  4) Sair"
    read -rp "Escolha [1-4]: " opt
    case "$opt" in
        1) ACTION="install" ;;
        2) ACTION="reinstall" ;;
        3) ACTION="service" ;;
        *) echo "Saindo."; exit 0 ;;
    esac
fi

case "$ACTION" in
    install)
        if is_installed; then
            warn "O sistema já está instalado em $APP_DIR."
            read -rp "Deseja fazer uma REINSTALAÇÃO (mantendo os dados)? [s/N] " r
            [[ "$r" =~ ^[sSyY]$ ]] && do_reinstall || echo "Nada feito."
        else
            do_install
        fi
        ;;
    reinstall)
        if ! is_installed; then
            warn "O sistema ainda não está instalado."
            read -rp "Deseja fazer a INSTALAÇÃO agora? [S/n] " r
            [[ "$r" =~ ^[nN]$ ]] && echo "Nada feito." || do_install
        else
            read -rp "Isso apaga o código antigo e mantém dados/instance/venv. Continuar? [s/N] " r
            [[ "$r" =~ ^[sSyY]$ ]] && do_reinstall || echo "Cancelado."
        fi
        ;;
    service)
        if ! is_installed; then
            die "O sistema não está instalado. Execute primeiro: sudo ./install.sh install"
        fi
        show_service_menu
        ;;
    *) die "Opção inválida: $ACTION (use install, reinstall ou service)" ;;
esac

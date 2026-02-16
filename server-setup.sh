#!/bin/bash
###############################################################################
# Скрипт первоначальной настройки сервера для Фармбазис ИИ-Техподдержки
# Запускать НА СЕРВЕРЕ от root:
#   curl -sL https://raw.githubusercontent.com/rosavskiy/FBTPbot/main/server-setup.sh | bash
#   или: bash server-setup.sh
###############################################################################

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

PROJECT_DIR="/opt/fbtpai"
REPO_URL="https://github.com/rosavskiy/FBTPbot.git"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Фармбазис ИИ-Техподдержка — Установка"
echo "═══════════════════════════════════════════════"
echo ""

# --- 1. Обновление системы ---
log "Обновление системы..."
apt-get update -qq
apt-get upgrade -y -qq

# --- 2. Установка Docker ---
if ! command -v docker &> /dev/null; then
    log "Установка Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
    log "Docker установлен"
else
    log "Docker уже установлен: $(docker --version)"
fi

# --- 3. Docker Compose Plugin ---
if ! docker compose version &> /dev/null 2>&1; then
    log "Установка Docker Compose Plugin..."
    apt-get install -y -qq docker-compose-plugin
    log "Docker Compose установлен"
else
    log "Docker Compose уже установлен: $(docker compose version)"
fi

# --- 4. Git ---
if ! command -v git &> /dev/null; then
    apt-get install -y -qq git
fi
log "Git: $(git --version)"

# --- 5. Создание swap (1 ГБ, на всякий случай) ---
if [ ! -f /swapfile ]; then
    log "Создание swap-файла (1 ГБ)..."
    fallocate -l 1G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    log "Swap создан и активирован"
else
    log "Swap уже существует"
fi

# --- 6. Клонирование проекта ---
if [ -d "${PROJECT_DIR}" ]; then
    warn "Директория ${PROJECT_DIR} уже существует. Обновляем..."
    cd ${PROJECT_DIR}
    git pull origin main
else
    log "Клонирование проекта..."
    git clone ${REPO_URL} ${PROJECT_DIR}
fi
cd ${PROJECT_DIR}
log "Проект загружен в ${PROJECT_DIR}"

# --- 7. Настройка .env ---
if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    warn "Создан файл backend/.env — НЕОБХОДИМО отредактировать!"
else
    log "backend/.env уже существует"
fi

# --- 8. Создание директорий ---
mkdir -p backend/data/chroma_db
mkdir -p backend/data/images

echo ""
echo "═══════════════════════════════════════════════"
echo "  Установка завершена!"
echo "═══════════════════════════════════════════════"
echo ""
echo "  Следующие шаги:"
echo ""
echo "  1. Отредактируйте .env файл:"
echo "     nano ${PROJECT_DIR}/backend/.env"
echo "     → Укажите OPENAI_API_KEY"
echo "     → Укажите TELEGRAM_BOT_TOKEN (опционально)"
echo ""
echo "  2. Загрузите инструкции с локальной машины:"
echo "     (на вашем компьютере выполните):"
echo "     scp -r ./instructions/* root@192.144.59.97:${PROJECT_DIR}/instructions/"
echo ""
echo "  3. Соберите и запустите Docker:"
echo "     cd ${PROJECT_DIR}"
echo "     docker compose build"
echo "     docker compose up -d"
echo ""
echo "  4. Проиндексируйте базу знаний:"
echo "     docker compose exec backend python -m app.indexer"
echo ""
echo "  5. Откройте в браузере:"
echo "     http://192.144.59.97"
echo ""
echo "═══════════════════════════════════════════════"

#!/bin/bash
###############################################################################
# Скрипт деплоя Фармбазис ИИ-Техподдержки на сервер
#
# Использование:
#   1. Настройте переменные ниже
#   2. chmod +x deploy.sh
#   3. ./deploy.sh
###############################################################################

set -e

# === НАСТРОЙКИ ===
SERVER_USER="root"
SERVER_HOST="41.216.182.31"
SERVER_DIR="/opt/fbtpai"
REPO_URL="https://github.com/rosavskiy/FBTPbot.git"
INSTRUCTIONS_LOCAL="./instructions"

# === ЦВЕТА ===
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[DEPLOY]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# === 1. Подготовка сервера (первый запуск) ===
setup_server() {
    log "Настройка сервера..."

    ssh ${SERVER_USER}@${SERVER_HOST} << 'ENDSSH'
        set -e

        # Установка Docker (если нет)
        if ! command -v docker &> /dev/null; then
            echo "Установка Docker..."
            curl -fsSL https://get.docker.com | sh
            systemctl enable docker
            systemctl start docker
        fi

        # Установка Docker Compose (если нет)
        if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
            echo "Установка Docker Compose Plugin..."
            apt-get update && apt-get install -y docker-compose-plugin
        fi

        # Установка git (если нет)
        if ! command -v git &> /dev/null; then
            apt-get update && apt-get install -y git
        fi

        echo "✅ Сервер готов"
ENDSSH
}

# === 2. Клонирование / обновление кода ===
deploy_code() {
    log "Деплой кода на сервер..."

    ssh ${SERVER_USER}@${SERVER_HOST} << ENDSSH
        set -e

        if [ -d "${SERVER_DIR}" ]; then
            cd ${SERVER_DIR}
            git pull origin main
            echo "✅ Код обновлён"
        else
            git clone ${REPO_URL} ${SERVER_DIR}
            cd ${SERVER_DIR}
            echo "✅ Код склонирован"
        fi
ENDSSH
}

# === 3. Загрузка инструкций (520MB) ===
upload_instructions() {
    log "Загрузка инструкций на сервер (может занять несколько минут)..."

    if [ ! -d "${INSTRUCTIONS_LOCAL}" ]; then
        err "Папка ${INSTRUCTIONS_LOCAL} не найдена!"
    fi

    # Создаём директорию на сервере
    ssh ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${SERVER_DIR}/instructions"

    # rsync для эффективной синхронизации
    rsync -avz --progress \
        ${INSTRUCTIONS_LOCAL}/ \
        ${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/instructions/

    log "✅ Инструкции загружены"
}

# === 4. Настройка .env ===
setup_env() {
    log "Настройка переменных окружения..."

    ssh ${SERVER_USER}@${SERVER_HOST} << ENDSSH
        cd ${SERVER_DIR}/backend

        if [ ! -f .env ]; then
            cp .env.example .env
            echo ""
            echo "⚠️  ВАЖНО: Отредактируйте файл ${SERVER_DIR}/backend/.env"
            echo "   Укажите OPENAI_API_KEY и TELEGRAM_BOT_TOKEN"
            echo "   Команда: nano ${SERVER_DIR}/backend/.env"
            echo ""
        else
            echo "✅ .env уже существует"
        fi
ENDSSH
}

# === 5. Запуск через Docker Compose ===
start_services() {
    log "Запуск сервисов..."

    ssh ${SERVER_USER}@${SERVER_HOST} << ENDSSH
        cd ${SERVER_DIR}

        # Сборка и запуск
        docker compose build
        docker compose up -d

        echo ""
        echo "✅ Сервисы запущены!"
        echo "   Backend:  http://${SERVER_HOST}:8000"
        echo "   Frontend: http://${SERVER_HOST}:3000"
        echo "   API docs: http://${SERVER_HOST}:8000/docs"
        echo ""
        docker compose ps
ENDSSH
}

# === 6. Индексация базы знаний ===
index_knowledge_base() {
    log "Индексация базы знаний (парсинг 542 статей)..."

    ssh ${SERVER_USER}@${SERVER_HOST} << ENDSSH
        cd ${SERVER_DIR}
        docker compose exec backend python -m app.indexer
ENDSSH

    log "✅ База знаний проиндексирована"
}

# === MAIN ===
case "${1:-full}" in
    setup)
        setup_server
        ;;
    code)
        deploy_code
        ;;
    instructions)
        upload_instructions
        ;;
    env)
        setup_env
        ;;
    start)
        start_services
        ;;
    index)
        index_knowledge_base
        ;;
    full)
        log "=== Полный деплой ==="
        setup_server
        deploy_code
        upload_instructions
        setup_env
        start_services
        index_knowledge_base
        log "=== Деплой завершён ==="
        ;;
    *)
        echo "Использование: $0 {setup|code|instructions|env|start|index|full}"
        exit 1
        ;;
esac

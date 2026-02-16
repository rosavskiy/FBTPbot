# Фармбазис ИИ-Техподдержка

Модуль интеллектуальной техподдержки для ООО "Фармбазис" (www.farmbazis.ru).

## Архитектура

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Веб-виджет /   │────▶│  FastAPI      │────▶│  RAG Engine     │
│  Страница чата  │◀────│  Backend     │◀────│  (LangChain)    │
└─────────────────┘     └──────┬───────┘     └────────┬────────┘
                               │                      │
                        ┌──────▼───────┐     ┌────────▼────────┐
                        │  SQLite DB   │     │  ChromaDB       │
                        │  (сессии,    │     │  (векторная БД) │
                        │   эскалации) │     └─────────────────┘
                        └──────┬───────┘
                               │
                     ┌─────────▼──────────┐
                     │ Telegram Bot       │
                     │ (уведомления ТП)   │
                     └────────────────────┘
```

## Стек технологий

- **Backend**: Python 3.11+, FastAPI, Uvicorn
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Векторная БД**: ChromaDB
- **RAG**: LangChain
- **Парсинг**: BeautifulSoup4
- **БД**: SQLite (dev) → PostgreSQL (prod)
- **Frontend**: React + TypeScript (виджет + страница)
- **Эскалация**: Telegram Bot API + Веб-панель оператора

## Быстрый старт

```bash
# 1. Установка зависимостей
cd backend
pip install -r requirements.txt

# 2. Настройка
cp .env.example .env
# Заполните OPENAI_API_KEY и TELEGRAM_BOT_TOKEN

# 3. Индексация базы знаний
python -m app.indexer

# 4. Запуск сервера
uvicorn app.main:app --reload

# 5. Фронтенд
cd ../frontend
npm install
npm run dev
```

## Структура проекта

```
backend/
  app/
    main.py            # FastAPI приложение
    config.py          # Конфигурация
    parser/            # Парсинг HTML инструкций
    indexer/           # Индексация в ChromaDB
    rag/               # RAG-движок
    api/               # API endpoints
    models/            # Pydantic модели
    database/          # SQLite/PostgreSQL
    telegram/          # Telegram-бот эскалации
    operator_panel/    # Панель оператора ТП
frontend/
  src/
    widget/            # Встраиваемый виджет
    chat/              # Полная страница чата
    operator/          # Панель оператора
instructions/          # HTML файлы руководства (542 шт.)
```

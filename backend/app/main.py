"""
Фармбазис ИИ-Техподдержка — главный модуль FastAPI.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.chat import router as chat_router
from app.api.escalation import router as escalation_router
from app.api.operator import router as operator_router
from app.config import settings
from app.database.models import init_db
from app.models.schemas import HealthResponse

logging.basicConfig(
    level=logging.DEBUG if settings.app_debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle: инициализация при старте, очистка при остановке."""
    logger.info("🚀 Инициализация Фармбазис ИИ-Техподдержки...")

    # Создаём директории
    Path("./data").mkdir(exist_ok=True)
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    # Инициализируем БД
    await init_db()
    logger.info("✅ База данных инициализирована")

    # Проверяем наличие индекса
    stats_path = Path(settings.chroma_persist_dir).parent / "indexing_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        logger.info(
            f"✅ База знаний загружена: "
            f"{stats.get('total_instructions', '?')} статей, "
            f"{stats.get('total_chunks', '?')} чанков"
        )
    else:
        logger.warning(
            "⚠️ База знаний не проиндексирована! "
            "Запустите: python -m app.indexer"
        )

    logger.info(f"✅ Сервер готов к работе на {settings.app_host}:{settings.app_port}")
    yield

    logger.info("Завершение работы...")


# Создание приложения
app = FastAPI(
    title="Фармбазис ИИ-Техподдержка",
    description=(
        "Модуль интеллектуальной техподдержки для ООО «Фармбазис». "
        "RAG-система на основе руководства пользователя."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Роутеры
app.include_router(chat_router)
app.include_router(escalation_router)
app.include_router(operator_router)

# Статические файлы (изображения из инструкций)
images_dir = Path(settings.chroma_persist_dir).parent / "images"
images_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=str(images_dir)), name="images")


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Проверка состояния системы."""
    stats_path = Path(settings.chroma_persist_dir).parent / "indexing_stats.json"
    kb_ready = stats_path.exists()
    stats = {}

    if kb_ready:
        stats = json.loads(stats_path.read_text())

    # Статистика заявок ТП
    support_stats_path = Path(settings.chroma_persist_dir).parent / "support_indexing_stats.json"
    support_count = 0
    if support_stats_path.exists():
        support_stats = json.loads(support_stats_path.read_text())
        support_count = support_stats.get("total_documents", 0)

    return HealthResponse(
        status="ok",
        version="1.0.0",
        knowledge_base_ready=kb_ready,
        total_articles=stats.get("total_instructions", 0),
        total_chunks=stats.get("total_chunks", 0),
        support_tickets_count=support_count,
    )

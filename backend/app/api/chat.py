"""
API эндпоинты чата — основной интерфейс пользователя.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import get_db
from app.database.service import DatabaseService
from app.models.schemas import ChatRequest, ChatResponse, SuggestedTopicSchema
from app.rag.engine import get_rag_engine
from app.rag.session_store import (
    clear_clarification_context,
    get_clarification_context,
    resolve_topic_choice,
    save_clarification_context,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Отправка сообщения в чат техподдержки.

    Если session_id не указан — создаётся новая сессия.
    Поддерживает режим уточнения: если запрос размытый,
    бот предложит выбрать тему или описать проблему подробнее.
    """
    db_service = DatabaseService(db)
    rag_engine = get_rag_engine()

    # Получаем или создаём сессию
    session = None
    if request.session_id:
        session = await db_service.get_session(request.session_id)

    if session is None:
        session = await db_service.create_session(
            user_ip=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("user-agent"),
        )

    # Сохраняем сообщение пользователя
    await db_service.add_message(
        session_id=session.id,
        role="user",
        content=request.message,
    )

    # Получаем историю чата для контекста
    history_messages = await db_service.get_chat_history(session.id, limit=10)
    chat_history = [
        {"role": msg.role, "content": msg.content}
        for msg in history_messages[:-1]  # Исключаем только что добавленное
    ]

    # ── Проверяем, не является ли сообщение выбором темы ──
    # Получаем контекст ДО resolve_topic_choice (он может очистить контекст)
    clarify_ctx = get_clarification_context(session.id)
    original_query_saved = clarify_ctx.get("original_query", request.message) if clarify_ctx else request.message

    topic_choice = resolve_topic_choice(session.id, request.message)
    if topic_choice is not None:
        # Пользователь выбрал тему из предложенных
        rag_response = await rag_engine.ask_by_topic(
            original_query=original_query_saved,
            article_id=topic_choice["article_id"],
            topic_title=topic_choice["title"],
            chat_history=chat_history,
        )

        # Сохраняем ответ бота
        await db_service.add_message(
            session_id=session.id,
            role="assistant",
            content=rag_response.answer,
            confidence=rag_response.confidence,
            source_articles=rag_response.source_articles,
        )

        return ChatResponse(
            answer=rag_response.answer,
            session_id=session.id,
            confidence=rag_response.confidence,
            needs_escalation=rag_response.needs_escalation,
            source_articles=rag_response.source_articles,
            youtube_links=rag_response.youtube_links,
            has_images=bool(rag_response.images),
            response_type="answer",
        )

    # ── Стандартный путь: ask с поддержкой уточнения ──
    rag_response, classification = await rag_engine.ask_with_clarification(
        question=request.message,
        chat_history=chat_history,
    )

    if classification is not None and not classification.is_complete:
        # Запрос размытый — предлагаем темы
        answer_text = classification.clarification_message or "Уточните ваш вопрос."

        topics_dicts = [
            {
                "title": t.title,
                "article_id": t.article_id,
                "score": t.score,
                "snippet": t.snippet,
            }
            for t in classification.suggested_topics
        ]

        # Сохраняем контекст уточнения для обработки ответа
        await save_clarification_context(
            session_id=session.id,
            original_query=request.message,
            topics=topics_dicts,
        )

        # Сохраняем уточняющее сообщение в историю
        await db_service.add_message(
            session_id=session.id,
            role="assistant",
            content=answer_text,
            confidence=0.5,
        )

        return ChatResponse(
            answer=answer_text,
            session_id=session.id,
            confidence=0.5,
            needs_escalation=False,
            response_type="clarification",
            suggested_topics=[
                SuggestedTopicSchema(**t) for t in topics_dicts
            ],
        )

    # Обычный ответ
    if rag_response is None:
        # Shouldn't happen, but safety fallback
        rag_response = await rag_engine.ask(
            question=request.message,
            chat_history=chat_history,
        )

    # Сохраняем ответ бота
    await db_service.add_message(
        session_id=session.id,
        role="assistant",
        content=rag_response.answer,
        confidence=rag_response.confidence,
        source_articles=rag_response.source_articles,
    )

    return ChatResponse(
        answer=rag_response.answer,
        session_id=session.id,
        confidence=rag_response.confidence,
        needs_escalation=rag_response.needs_escalation,
        source_articles=rag_response.source_articles,
        youtube_links=rag_response.youtube_links,
        has_images=bool(rag_response.images),
        response_type="answer",
    )

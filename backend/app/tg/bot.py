# -*- coding: utf-8 -*-
"""
Telegram-бот техподдержки Фармбазис.

Работает как отдельный процесс внутри Docker-контейнера backend.
Напрямую использует RAG engine для ответов на вопросы.
Поддерживает:
  - Ответы из базы знаний + заявок ТП
  - Уточняющие вопросы (inline-кнопки)
  - Эскалацию на оператора
  - YouTube-ссылки и ссылки на статьи
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import sys
from typing import Dict, Optional

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.config import settings
from app.rag.engine import get_rag_engine
from app.rag.session_store import (
    clear_clarification_context,
    get_clarification_context,
    resolve_topic_choice,
    save_clarification_context,
)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tg_bot")

# ── Константы ──
MAX_MESSAGE_LENGTH = 4096
WELCOME_TEXT = (
    "👋 Здравствуйте! Я бот техподдержки <b>Фармбазис</b>.\n\n"
    "Задайте вопрос по работе с программой, и я постараюсь помочь.\n\n"
    "Примеры вопросов:\n"
    "• Как сделать возврат?\n"
    "• Ошибка при проведении накладной\n"
    "• Не работает сканер маркировки\n\n"
    "Если я не смогу ответить — переведу на оператора."
)
HELP_TEXT = (
    "📖 <b>Как пользоваться ботом:</b>\n\n"
    "1. Просто напишите свой вопрос текстом\n"
    "2. Если вопрос широкий — я предложу уточнить тему кнопками\n"
    "3. Нажмите кнопку или напишите уточнение текстом\n\n"
    "<b>Команды:</b>\n"
    "/start — начать заново\n"
    "/help — эта справка\n"
    "/reset — сбросить контекст диалога"
)

# Хранилище chat_history для Telegram (по user_id)
# В production — заменить на Redis
_chat_histories: Dict[int, list] = {}
MAX_HISTORY = 10


def _get_history(user_id: int) -> list:
    """Получить историю чата для пользователя."""
    return _chat_histories.get(user_id, [])


def _add_to_history(user_id: int, role: str, content: str):
    """Добавить сообщение в историю."""
    if user_id not in _chat_histories:
        _chat_histories[user_id] = []
    _chat_histories[user_id].append({"role": role, "content": content})
    # Ограничиваем длину истории
    if len(_chat_histories[user_id]) > MAX_HISTORY * 2:
        _chat_histories[user_id] = _chat_histories[user_id][-MAX_HISTORY * 2:]


def _clear_history(user_id: int):
    """Очистить историю чата."""
    _chat_histories.pop(user_id, None)


def _session_id(user_id: int) -> str:
    """Формируем session_id для session_store из Telegram user_id."""
    return f"tg_{user_id}"


def _escape(text: str) -> str:
    """Экранирование HTML для Telegram."""
    return html.escape(text)


def _format_answer(
    answer: str,
    confidence: float = 0.0,
    source_articles: list | None = None,
    youtube_links: list | None = None,
    needs_escalation: bool = False,
) -> str:
    """Форматирование ответа для Telegram."""
    parts = [answer]

    # YouTube-ссылки
    if youtube_links:
        parts.append("")
        parts.append("🎥 <b>Видео-инструкции:</b>")
        for link in youtube_links:
            parts.append(f"▸ {link}")

    # Источники (если есть статьи из БЗ)
    if source_articles:
        kb_articles = [a for a in source_articles if a.isdigit()]
        if kb_articles:
            parts.append("")
            links = ", ".join(
                f'<a href="http://41.216.182.31/article/{a}">#{a}</a>'
                for a in kb_articles[:3]
            )
            parts.append(f"📚 Источники: {links}")

    # Эскалация
    if needs_escalation:
        parts.append("")
        parts.append("⚠️ <i>Рекомендую обратиться к оператору для детальной помощи.</i>")

    result = "\n".join(parts)

    # Telegram ограничение
    if len(result) > MAX_MESSAGE_LENGTH:
        result = result[: MAX_MESSAGE_LENGTH - 20] + "\n\n<i>…(обрезано)</i>"

    return result


def _build_topic_keyboard(topics: list[dict]) -> InlineKeyboardMarkup:
    """Создаёт inline-клавиатуру из предложенных тем."""
    buttons = []
    for i, topic in enumerate(topics):
        title = topic.get("title", f"Тема {i + 1}")
        # Telegram ограничивает callback_data до 64 байт
        callback = f"topic:{i}"
        # Обрезаем длинные заголовки для кнопок
        label = title if len(title) <= 60 else title[:57] + "..."
        buttons.append([InlineKeyboardButton(text=label, callback_data=callback)])

    return InlineKeyboardMarkup(buttons)


# ═══════════════════════════════════════════════════
#  Обработчики команд
# ═══════════════════════════════════════════════════


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start."""
    user_id = update.effective_user.id
    _clear_history(user_id)
    clear_clarification_context(_session_id(user_id))
    await update.message.reply_text(WELCOME_TEXT, parse_mode=ParseMode.HTML)
    logger.info(f"User {user_id} started bot")


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /help."""
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.HTML)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /reset — сброс контекста."""
    user_id = update.effective_user.id
    _clear_history(user_id)
    clear_clarification_context(_session_id(user_id))
    await update.message.reply_text(
        "🔄 Контекст диалога сброшен. Задайте новый вопрос.",
        parse_mode=ParseMode.HTML,
    )


# ═══════════════════════════════════════════════════
#  Обработка текстовых сообщений
# ═══════════════════════════════════════════════════


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстового сообщения пользователя."""
    user_id = update.effective_user.id
    sid = _session_id(user_id)
    text = update.message.text.strip()

    if not text:
        return

    # Информация о пользователе для логов
    user = update.effective_user
    username = user.username or user.first_name or str(user_id)
    logger.info(f"[DEMO] REQUEST|question={text[:120]}|source=telegram|user={username}")

    # Показываем «печатает...»
    await update.message.chat.send_action(ChatAction.TYPING)

    # Добавляем в историю
    _add_to_history(user_id, "user", text)
    chat_history = _get_history(user_id)[:-1]  # без текущего сообщения

    rag = get_rag_engine()

    # ── Проверяем, не выбирает ли пользователь тему текстом ──
    topic_choice = resolve_topic_choice(sid, text)
    if topic_choice is not None:
        # Пользователь написал номер темы текстом
        try:
            clarify_ctx = get_clarification_context(sid)
            original_query = clarify_ctx.get("original_query", text) if clarify_ctx else text
        except Exception:
            original_query = text

        rag_response = await rag.ask_by_topic(
            original_query=original_query,
            article_id=topic_choice["article_id"],
            topic_title=topic_choice["title"],
            chat_history=chat_history,
        )

        reply = _format_answer(
            answer=rag_response.answer,
            confidence=rag_response.confidence,
            source_articles=rag_response.source_articles,
            youtube_links=rag_response.youtube_links,
            needs_escalation=rag_response.needs_escalation,
        )
        _add_to_history(user_id, "assistant", rag_response.answer)

        await update.message.reply_text(reply, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        return

    # ── Стандартный путь: ask с поддержкой уточнения ──
    try:
        rag_response, classification = await rag.ask_with_clarification(
            question=text,
            chat_history=chat_history,
        )
    except Exception as e:
        logger.error(f"[TG] RAG error: {e}", exc_info=True)
        await update.message.reply_text(
            "😔 Произошла ошибка при обработке запроса. Попробуйте позже.",
            parse_mode=ParseMode.HTML,
        )
        return

    # ── Режим уточнения ──
    if classification is not None and not classification.is_complete:
        answer_text = classification.clarification_message or "Уточните ваш вопрос:"

        topics_dicts = [
            {
                "title": t.title,
                "article_id": t.article_id,
                "score": t.score,
                "snippet": t.snippet,
            }
            for t in classification.suggested_topics
        ]

        logger.info(f"[DEMO] CLARIFICATION_NEEDED|topics={len(topics_dicts)}|source=telegram|user={username}")

        # Сохраняем контекст для обработки выбора
        await save_clarification_context(
            session_id=sid,
            original_query=text,
            topics=topics_dicts,
        )

        _add_to_history(user_id, "assistant", answer_text)

        keyboard = _build_topic_keyboard(topics_dicts)
        await update.message.reply_text(
            f"🔍 {_escape(answer_text)}",
            parse_mode=ParseMode.HTML,
            reply_markup=keyboard,
        )
        return

    # ── Обычный ответ ──
    if rag_response is None:
        rag_response = await rag.ask(
            question=text,
            chat_history=chat_history,
        )

    reply = _format_answer(
        answer=rag_response.answer,
        confidence=rag_response.confidence,
        source_articles=rag_response.source_articles,
        youtube_links=rag_response.youtube_links,
        needs_escalation=rag_response.needs_escalation,
    )
    _add_to_history(user_id, "assistant", rag_response.answer)

    logger.info(
        f"[DEMO] COMPLETE|total_time=n/a|answer_len={len(rag_response.answer)}"
        f"|source=telegram|user={username}"
    )

    await update.message.reply_text(reply, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


# ═══════════════════════════════════════════════════
#  Обработка нажатий на inline-кнопки
# ═══════════════════════════════════════════════════


async def handle_topic_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатия кнопки выбора темы."""
    query = update.callback_query
    await query.answer()  # Убираем «часики» на кнопке

    user_id = query.from_user.id
    sid = _session_id(user_id)
    data = query.data  # "topic:0", "topic:1", ...

    if not data.startswith("topic:"):
        return

    try:
        idx = int(data.split(":")[1])
    except (ValueError, IndexError):
        return

    # Достаём контекст уточнения
    ctx = get_clarification_context(sid)
    if ctx is None:
        await query.edit_message_text(
            "⏰ Время для выбора темы истекло. Задайте вопрос заново.",
            parse_mode=ParseMode.HTML,
        )
        return

    topics = ctx.get("topics", [])
    if idx < 0 or idx >= len(topics):
        return

    topic = topics[idx]
    original_query = ctx.get("original_query", "")
    chat_history = _get_history(user_id)

    # Очищаем контекст
    clear_clarification_context(sid)

    # Обновляем сообщение — убираем кнопки, показываем выбор
    try:
        await query.edit_message_text(
            f"🔍 Выбрана тема: <b>{_escape(topic['title'])}</b>\n\n⏳ Формирую ответ...",
            parse_mode=ParseMode.HTML,
        )
    except Exception:
        pass

    logger.info(f"[DEMO] ASK_BY_TOPIC|article={topic.get('article_id', '?')}|title={topic.get('title', '?')}|source=telegram|user_id={user_id}")

    # Запрос к RAG
    rag = get_rag_engine()
    try:
        rag_response = await rag.ask_by_topic(
            original_query=original_query,
            article_id=topic["article_id"],
            topic_title=topic["title"],
            chat_history=chat_history,
        )
    except Exception as e:
        logger.error(f"[TG] RAG error on topic: {e}", exc_info=True)
        await query.edit_message_text(
            "😔 Произошла ошибка. Попробуйте задать вопрос заново.",
            parse_mode=ParseMode.HTML,
        )
        return

    reply = _format_answer(
        answer=rag_response.answer,
        confidence=rag_response.confidence,
        source_articles=rag_response.source_articles,
        youtube_links=rag_response.youtube_links,
        needs_escalation=rag_response.needs_escalation,
    )
    _add_to_history(user_id, "assistant", rag_response.answer)

    try:
        await query.edit_message_text(reply, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception:
        # Если сообщение слишком длинное для edit, отправляем новым
        await query.message.reply_text(reply, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


# ═══════════════════════════════════════════════════
#  Запуск бота
# ═══════════════════════════════════════════════════


def main():
    """Точка входа для Telegram-бота."""
    token = settings.telegram_bot_token
    if not token:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        logger.error("❌ TELEGRAM_BOT_TOKEN не задан!")
        sys.exit(1)

    logger.info("🤖 Запуск Telegram-бота Фармбазис ТП...")

    app = Application.builder().token(token).build()

    # Регистрация обработчиков
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("reset", cmd_reset))
    app.add_handler(CallbackQueryHandler(handle_topic_callback, pattern=r"^topic:\d+$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("✅ Бот запущен, ожидаю сообщения...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

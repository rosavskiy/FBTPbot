"""
Тесты для модуля уточняющих вопросов (query_classifier + session_store).

Запуск:
    cd backend
    python -m pytest tests/test_clarification.py -v
"""

import asyncio
import sys
import os

# Добавляем backend в PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from app.rag.query_classifier import (
    classify_query,
    ClassificationResult,
    SuggestedTopic,
    VAGUE_PATTERNS,
    BROAD_OBJECTS,
)
from app.rag.session_store import (
    save_clarification_context,
    get_clarification_context,
    clear_clarification_context,
    resolve_topic_choice,
    _store,
)


# ═══════════════════════════════════════════════════════════
# Утилиты для тестов
# ═══════════════════════════════════════════════════════════

def make_doc(article_id: str, title: str, text: str = "Текст чанка") -> Document:
    """Создаёт mock Document с метаданными."""
    return Document(
        page_content=text,
        metadata={
            "article_id": article_id,
            "title": title,
        },
    )


def make_scored_results(specs: list[tuple[str, str, float]]) -> list[tuple[Document, float]]:
    """
    Создаёт список (Document, score) из спецификаций.
    specs: [(article_id, title, score), ...]
    """
    results = []
    for article_id, title, score in specs:
        doc = make_doc(article_id, title)
        results.append((doc, score))
    return results


# ═══════════════════════════════════════════════════════════
# Тесты classify_query
# ═══════════════════════════════════════════════════════════

class TestClassifyQuery:
    """Тесты классификатора полноты запроса."""

    def test_empty_results_returns_complete(self):
        """Если нет результатов — считаем запрос complete (RAG сам скажет 'не найдено')."""
        result = classify_query("проблема в накладной", [])
        assert result.is_complete is True

    def test_single_topic_always_complete(self):
        """Один найденный документ — всегда отвечаем."""
        results = make_scored_results([
            ("1001", "Настройка накладных", 0.85),
        ])
        result = classify_query("проблема в накладной", results)
        assert result.is_complete is True

    def test_specific_query_always_complete(self):
        """Конкретный длинный запрос без размытых паттернов — отвечаем."""
        results = make_scored_results([
            ("1001", "Печать накладной", 0.80),
            ("1002", "Настройка принтера", 0.78),
            ("1003", "Формат бланка", 0.75),
        ])
        result = classify_query(
            "Как настроить формат печати расходной накладной в формате А4",
            results,
        )
        assert result.is_complete is True

    def test_vague_query_with_multiple_topics_needs_clarification(self):
        """Размытый запрос + несколько равновесных тем → уточнение."""
        results = make_scored_results([
            ("1001", "Не проводится накладная", 0.82),
            ("1002", "Не печатается накладная", 0.80),
            ("1003", "Ошибка цен в накладной", 0.79),
        ])
        result = classify_query("проблема в накладной", results)
        assert result.is_complete is False
        assert len(result.suggested_topics) >= 2
        assert result.clarification_message is not None

    def test_vague_query_with_clear_leader_is_complete(self):
        """Размытый запрос, но один результат явно лидирует → отвечаем."""
        results = make_scored_results([
            ("1001", "Не проводится накладная", 0.95),
            ("1002", "Не печатается накладная", 0.60),
        ])
        result = classify_query("проблема в накладной", results)
        assert result.is_complete is True

    def test_short_query_with_vague_word(self):
        """Короткий запрос с размытым словом."""
        results = make_scored_results([
            ("1001", "Ошибка обновления", 0.75),
            ("1002", "Ошибка авторизации", 0.73),
            ("1003", "Ошибка приёма данных", 0.70),
        ])
        result = classify_query("ошибка", results)
        assert result.is_complete is False

    def test_not_vague_without_patterns(self):
        """Запрос без размытых паттернов — complete."""
        results = make_scored_results([
            ("1001", "Настройка шрифтов", 0.70),
            ("1002", "Настройка языка", 0.68),
        ])
        result = classify_query("настроить шрифт в интерфейсе", results)
        assert result.is_complete is True

    def test_clarification_message_has_topic_titles(self):
        """Сообщение уточнения содержит названия тем."""
        results = make_scored_results([
            ("1001", "Не проводится накладная", 0.82),
            ("1002", "Не печатается накладная", 0.80),
        ])
        result = classify_query("проблема в накладной", results)
        if not result.is_complete:
            assert "Не проводится накладная" in result.clarification_message
            assert "Не печатается накладная" in result.clarification_message

    def test_max_topics_limit(self):
        """Не больше MAX_TOPICS_TO_SUGGEST тем."""
        results = make_scored_results([
            (f"100{i}", f"Тема {i}", 0.80 - i * 0.01) for i in range(10)
        ])
        result = classify_query("не работает", results)
        if not result.is_complete:
            assert len(result.suggested_topics) <= 5

    def test_dedup_by_article_id(self):
        """Чанки одной статьи не дублируют темы."""
        results = [
            (make_doc("1001", "Накладная", "Чанк 1"), 0.85),
            (make_doc("1001", "Накладная", "Чанк 2"), 0.83),
            (make_doc("1002", "Отчёт", "Чанк 3"), 0.80),
        ]
        result = classify_query("не работает отчёт", results)
        if not result.is_complete:
            article_ids = [t.article_id for t in result.suggested_topics]
            assert len(article_ids) == len(set(article_ids))


# ═══════════════════════════════════════════════════════════
# Тесты session_store
# ═══════════════════════════════════════════════════════════

class TestSessionStore:
    """Тесты хранилища контекста уточнения."""

    def setup_method(self):
        """Очищаем хранилище перед каждым тестом."""
        _store.clear()

    @pytest.mark.asyncio
    async def test_save_and_get(self):
        topics = [{"title": "Тема 1", "article_id": "1001", "score": 0.8, "snippet": "..."}]
        await save_clarification_context("sess-1", "проблема", topics)

        ctx = get_clarification_context("sess-1")
        assert ctx is not None
        assert ctx["state"] == "awaiting_clarification"
        assert ctx["original_query"] == "проблема"
        assert len(ctx["topics"]) == 1

    def test_get_nonexistent(self):
        ctx = get_clarification_context("nonexistent")
        assert ctx is None

    @pytest.mark.asyncio
    async def test_clear(self):
        await save_clarification_context("sess-1", "q", [])
        clear_clarification_context("sess-1")
        assert get_clarification_context("sess-1") is None

    @pytest.mark.asyncio
    async def test_resolve_topic_choice_by_number(self):
        topics = [
            {"title": "Тема A", "article_id": "1001", "score": 0.8, "snippet": "..."},
            {"title": "Тема B", "article_id": "1002", "score": 0.7, "snippet": "..."},
        ]
        await save_clarification_context("sess-2", "проблема", topics)

        chosen = resolve_topic_choice("sess-2", "1")
        assert chosen is not None
        assert chosen["article_id"] == "1001"
        assert chosen["title"] == "Тема A"

    @pytest.mark.asyncio
    async def test_resolve_topic_choice_number_2(self):
        topics = [
            {"title": "Тема A", "article_id": "1001", "score": 0.8, "snippet": "..."},
            {"title": "Тема B", "article_id": "1002", "score": 0.7, "snippet": "..."},
        ]
        await save_clarification_context("sess-3", "проблема", topics)

        chosen = resolve_topic_choice("sess-3", "2")
        assert chosen is not None
        assert chosen["article_id"] == "1002"

    @pytest.mark.asyncio
    async def test_resolve_topic_choice_free_text(self):
        topics = [
            {"title": "Тема A", "article_id": "1001", "score": 0.8, "snippet": "..."},
        ]
        await save_clarification_context("sess-4", "проблема", topics)

        chosen = resolve_topic_choice("sess-4", "не печатается при нажатии на кнопку")
        assert chosen is None
        # Контекст должен быть очищен
        assert get_clarification_context("sess-4") is None

    @pytest.mark.asyncio
    async def test_resolve_topic_out_of_range(self):
        topics = [
            {"title": "Тема A", "article_id": "1001", "score": 0.8, "snippet": "..."},
        ]
        await save_clarification_context("sess-5", "проблема", topics)

        chosen = resolve_topic_choice("sess-5", "5")
        # Номер вне диапазона — трактуется как текстовый ввод
        assert chosen is None

    def test_resolve_no_context(self):
        chosen = resolve_topic_choice("no-session", "1")
        assert chosen is None


# ═══════════════════════════════════════════════════════════
# Тесты интеграции classifier + session
# ═══════════════════════════════════════════════════════════

class TestClarificationFlow:
    """Интеграционные тесты полного потока уточнения."""

    def setup_method(self):
        _store.clear()

    @pytest.mark.asyncio
    async def test_full_clarification_flow(self):
        """Полный сценарий: размытый запрос → уточнение → выбор → ответ."""
        # 1. Размытый запрос
        results = make_scored_results([
            ("1001", "Накладная не проводится", 0.82),
            ("1002", "Не печатается накладная", 0.80),
            ("1003", "Ошибка цен в накладной", 0.79),
        ])
        classification = classify_query("проблема в накладной", results)
        assert classification.is_complete is False

        # 2. Сохраняем контекст
        topics_dicts = [
            {"title": t.title, "article_id": t.article_id, "score": t.score, "snippet": t.snippet}
            for t in classification.suggested_topics
        ]
        await save_clarification_context("sess-flow", "проблема в накладной", topics_dicts)

        # 3. Пользователь выбирает "2"
        chosen = resolve_topic_choice("sess-flow", "2")
        assert chosen is not None
        assert chosen["article_id"] == "1002"
        assert chosen["title"] == "Не печатается накладная"

    @pytest.mark.asyncio
    async def test_flow_with_text_refinement(self):
        """Пользователь уточняет текстом вместо номера."""
        results = make_scored_results([
            ("1001", "Накладная не проводится", 0.82),
            ("1002", "Не печатается накладная", 0.80),
        ])
        classification = classify_query("проблема в накладной", results)

        topics_dicts = [
            {"title": t.title, "article_id": t.article_id, "score": t.score, "snippet": t.snippet}
            for t in classification.suggested_topics
        ]
        await save_clarification_context("sess-text", "проблема в накладной", topics_dicts)

        # Пользователь пишет подробнее
        chosen = resolve_topic_choice("sess-text", "при проведении накладной ошибка остатков")
        assert chosen is None  # Не номер → обрабатывается как новый запрос


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

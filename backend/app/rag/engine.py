"""
RAG-движок для ИИ-техподдержки Фармбазис.

Реализует Retrieval-Augmented Generation:
1. Поиск релевантных чанков в ChromaDB
2. Формирование промпта с контекстом
3. Генерация ответа через OpenAI
4. Оценка уверенности (для эскалации)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import OpenAI

from app.config import settings
from app.indexer.knowledge_base import get_indexer, SUPPORT_COLLECTION_NAME

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Ты — ИИ-ассистент техподдержки ООО «Фармбазис» (ПО для аптек).

ГЛАВНОЕ ОГРАНИЧЕНИЕ: Ответ ДОЛЖЕН быть НЕ БОЛЕЕ 450 символов (кириллица). Пиши КРАТКО, только суть и действия. Без вступлений, без "рад помочь", без повторения вопроса.

ПРАВИЛА:
1. Отвечай ТОЛЬКО по контексту из базы знаний.
2. Нет информации — скажи кратко и предложи оператора.
3. Давай пошаговые инструкции, кратко.
4. Есть видео на YouTube — дай ссылку.
5. Русский язык, профессиональный тон.
6. Не раскрывай механику бота.

В конце ОБЯЗАТЕЛЬНО добавь (пользователь не увидит):
```confidence
{"confidence": <0.0-1.0>, "reason": "<кратко>"}
```
confidence: 0.0-0.3 — нет ответа, 0.3-0.6 — частичный, 0.6-1.0 — уверенный.
"""

# Лимит ответа для интеграции с Фармбазис (1024 байта UTF-8 ≈ 450 символов кириллицы)
MAX_ANSWER_BYTES = 1024

CONTEXT_TEMPLATE = """
КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:
{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}
"""


@dataclass
class RAGResponse:
    """Ответ RAG-системы."""
    answer: str                                    # Текст ответа для пользователя
    confidence: float = 0.0                        # Уровень уверенности (0-1)
    confidence_reason: str = ""                    # Пояснение к уверенности
    needs_escalation: bool = False                 # Нужна ли эскалация
    source_articles: List[str] = field(default_factory=list)  # ID статей-источников
    youtube_links: List[str] = field(default_factory=list)    # Найденные YouTube-ссылки
    images: List[dict] = field(default_factory=list)          # Найденные изображения


class RAGEngine:
    """RAG-движок для поиска и генерации ответов."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self._vector_store: Optional[Chroma] = None

    @property
    def vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = get_indexer().get_vector_store()
        return self._vector_store

    @property
    def support_vector_store(self):
        """Векторное хранилище заявок ТП (может быть None)."""
        if not hasattr(self, '_support_vector_store'):
            self._support_vector_store = get_indexer().get_support_vector_store()
        return self._support_vector_store

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """Поиск релевантных документов в обеих коллекциях."""
        top_k = top_k or settings.rag_top_k

        logger.info(f"[DEMO] SEARCH_START|query={query}|top_k={top_k}")

        # Поиск в основной коллекции (инструкции)
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=top_k
        )

        for doc, score in results:
            article_id = doc.metadata.get('article_id', '?')
            title = doc.metadata.get('title', 'Без названия')[:60]
            logger.info(f"[DEMO] CHUNK|score={score:.3f}|article={article_id}|title={title}|col=instructions")

        # Поиск во второй коллекции (заявки ТП)
        support_results = []
        if self.support_vector_store is not None:
            try:
                support_results = self.support_vector_store.similarity_search_with_relevance_scores(
                    query, k=top_k
                )
                for doc, score in support_results:
                    article_id = doc.metadata.get('article_id', '?')
                    title = doc.metadata.get('title', 'Заявка ТП')[:60]
                    logger.info(f"[DEMO] CHUNK|score={score:.3f}|article={article_id}|title={title}|col=support")
            except Exception as e:
                logger.warning(f"Ошибка поиска в коллекции support_tickets: {e}")

        # Объединяем и сортируем по score (лучшие сверху)
        all_results = results + support_results
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Берём top_k лучших из объединённых результатов
        top_results = all_results[:top_k]

        # Фильтруем по минимальному порогу релевантности
        filtered = [
            (doc, score) for doc, score in top_results
            if score >= settings.rag_confidence_threshold
        ]

        total_found = len(results) + len(support_results)
        logger.info(
            f"[DEMO] SEARCH_DONE|found={total_found}|merged_top={len(top_results)}|passed_filter={len(filtered)}|threshold={settings.rag_confidence_threshold}"
        )

        return [doc for doc, _ in filtered]

    def retrieve_with_scores(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """
        Поиск релевантных документов с возвратом score.

        Аналогичен retrieve(), но возвращает пары (Document, score)
        для использования в классификаторе запросов.
        """
        top_k = top_k or settings.rag_top_k

        logger.info(f"[DEMO] SEARCH_WITH_SCORES|query={query}|top_k={top_k}")

        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=top_k
        )

        support_results = []
        if self.support_vector_store is not None:
            try:
                support_results = self.support_vector_store.similarity_search_with_relevance_scores(
                    query, k=top_k
                )
            except Exception as e:
                logger.warning(f"Ошибка поиска в коллекции support_tickets: {e}")

        all_results = results + support_results
        all_results.sort(key=lambda x: x[1], reverse=True)
        top_results = all_results[:top_k]

        filtered = [
            (doc, score) for doc, score in top_results
            if score >= settings.rag_confidence_threshold
        ]

        return filtered

    def retrieve_by_article_id(self, article_id: str, query: str, top_k: int = 3) -> List[Document]:
        """
        Поиск чанков конкретной статьи, наиболее релевантных запросу.

        Используется когда пользователь выбрал тему из уточняющего списка.
        """
        # Ищем больше чанков, потом фильтруем по article_id
        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=20
        )

        matched = [
            (doc, score) for doc, score in results
            if str(doc.metadata.get("article_id", "")) == str(article_id)
        ]

        if not matched:
            # Fallback: поиск по всем коллекциям
            if self.support_vector_store is not None:
                try:
                    support_results = self.support_vector_store.similarity_search_with_relevance_scores(
                        query, k=20
                    )
                    matched = [
                        (doc, score) for doc, score in support_results
                        if str(doc.metadata.get("article_id", "")) == str(article_id)
                    ]
                except Exception:
                    pass

        matched.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in matched[:top_k]]

    def _build_context(self, documents: List[Document]) -> Tuple[str, List[str], List[str], List[dict]]:
        """
        Формирование контекста из документов.

        Returns:
            (context_text, article_ids, youtube_links, images)
        """
        context_parts = []
        article_ids = set()
        youtube_links = []
        images = []

        for doc in documents:
            meta = doc.metadata
            article_id = meta.get("article_id", "unknown")
            title = meta.get("title", "Без названия")

            article_ids.add(article_id)

            # Собираем YouTube ссылки
            yt_raw = meta.get("youtube_links")
            if yt_raw:
                try:
                    yt_list = json.loads(yt_raw)
                    for link in yt_list:
                        if link not in youtube_links:
                            youtube_links.append(link)
                except json.JSONDecodeError:
                    pass

            # Собираем изображения
            img_raw = meta.get("images_info")
            if img_raw:
                try:
                    img_list = json.loads(img_raw)
                    images.extend(img_list)
                except json.JSONDecodeError:
                    pass

            context_parts.append(
                f"--- Статья: {title} (ID: {article_id}) ---\n{doc.page_content}\n"
            )

        context_text = "\n".join(context_parts)
        return context_text, list(article_ids), youtube_links, images

    def _parse_confidence(self, answer: str) -> Tuple[str, float, str]:
        """
        Извлечение блока confidence из ответа LLM.

        Returns:
            (clean_answer, confidence, reason)
        """
        import re

        # Ищем блок confidence
        pattern = r'```confidence\s*\n?\{.*?"confidence"\s*:\s*([\d.]+).*?"reason"\s*:\s*"([^"]*)".*?\}\s*\n?```'
        match = re.search(pattern, answer, re.DOTALL)

        if match:
            confidence = float(match.group(1))
            reason = match.group(2)
            # Убираем блок confidence из ответа
            clean_answer = answer[:match.start()].strip()
            return clean_answer, confidence, reason

        # Если блок не найден, пробуем простой JSON
        json_pattern = r'\{[^{}]*"confidence"\s*:\s*([\d.]+)[^{}]*"reason"\s*:\s*"([^"]*)"[^{}]*\}'
        match = re.search(json_pattern, answer)
        if match:
            confidence = float(match.group(1))
            reason = match.group(2)
            clean_answer = answer[:match.start()].strip()
            return clean_answer, confidence, reason

        return answer, 0.5, "Не удалось извлечь оценку уверенности"

    async def ask(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None,
    ) -> RAGResponse:
        """
        Основной метод: задать вопрос RAG-системе.

        Args:
            question: Вопрос пользователя.
            chat_history: История чата [{"role": "user"/"assistant", "content": "..."}]

        Returns:
            RAGResponse с ответом и метаданными.
        """
        import time as _time
        _start = _time.time()

        logger.info(f"[DEMO] REQUEST|question={question}")

        # 1. Retrieval — поиск релевантных документов
        documents = self.retrieve(question)

        if not documents:
            logger.info(f"[DEMO] DECISION|confidence=0.0|escalation=True|reason=No documents found")
            return RAGResponse(
                answer="Ответ не найден в базе знаний. Передаю вопрос оператору.",
                confidence=0.0,
                confidence_reason="Нет релевантных документов в базе знаний",
                needs_escalation=True,
            )

        # 2. Формируем контекст
        context_text, article_ids, youtube_links, images = self._build_context(
            documents
        )
        logger.info(f"[DEMO] CONTEXT|articles={article_ids}|youtube={len(youtube_links)}|images={len(images)}")

        # 3. Формируем сообщения для LLM
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Добавляем историю чата (последние 6 сообщений)
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Добавляем текущий вопрос с контекстом
        user_message = CONTEXT_TEMPLATE.format(
            context=context_text,
            question=question,
        )
        messages.append({"role": "user", "content": user_message})

        # 4. Генерация ответа
        logger.info(f"[DEMO] GPT_CALL|model={settings.openai_model}|messages={len(messages)}|temperature=0.1")
        _gpt_start = _time.time()
        try:
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0.1,  # Низкая температура для точности
                max_tokens=800,
            )
            raw_answer = response.choices[0].message.content or ""
            _gpt_elapsed = _time.time() - _gpt_start
            _tokens_in = response.usage.prompt_tokens if response.usage else 0
            _tokens_out = response.usage.completion_tokens if response.usage else 0
            logger.info(f"[DEMO] GPT_DONE|time={_gpt_elapsed:.1f}s|tokens_in={_tokens_in}|tokens_out={_tokens_out}")
        except Exception as e:
            logger.error(f"Ошибка OpenAI API: {e}")
            return RAGResponse(
                answer="Техническая ошибка. Попробуйте позже или обратитесь к оператору.",
                confidence=0.0,
                confidence_reason=f"Ошибка API: {str(e)}",
                needs_escalation=True,
            )

        # 5. Извлекаем confidence
        clean_answer, confidence, reason = self._parse_confidence(raw_answer)

        # 5.1. Обрезаем ответ до лимита 1024 байт для интеграции с Фармбазис
        clean_answer = _truncate_to_bytes(clean_answer, MAX_ANSWER_BYTES)

        # 6. Определяем необходимость эскалации
        needs_escalation = confidence < settings.rag_confidence_threshold
        _total = _time.time() - _start

        logger.info(f"[DEMO] DECISION|confidence={confidence}|escalation={needs_escalation}|reason={reason}")
        logger.info(f"[DEMO] COMPLETE|total_time={_total:.1f}s|answer_len={len(clean_answer)}")

        return RAGResponse(
            answer=clean_answer,
            confidence=confidence,
            confidence_reason=reason,
            needs_escalation=needs_escalation,
            source_articles=article_ids,
            youtube_links=youtube_links,
            images=images,
        )

    async def ask_with_clarification(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None,
    ) -> Tuple[Optional[RAGResponse], Optional["ClassificationResult"]]:
        """
        Расширенный метод ask с поддержкой уточняющих вопросов.

        Сначала проверяет, достаточно ли конкретен запрос.
        Если нет — возвращает (None, ClassificationResult) с темами.
        Если да — возвращает (RAGResponse, None) как обычно.
        """
        from app.rag.query_classifier import classify_query

        # 1. Поиск с оценками
        scored_results = self.retrieve_with_scores(question)

        # 2. Классификация
        classification = classify_query(question, scored_results)

        if not classification.is_complete:
            logger.info(f"[DEMO] CLARIFICATION_NEEDED|topics={len(classification.suggested_topics)}")
            return None, classification

        # 3. Запрос достаточно конкретный — отвечаем как обычно
        documents = [doc for doc, _ in scored_results]
        if not documents:
            return RAGResponse(
                answer="Ответ не найден в базе знаний. Передаю вопрос оператору.",
                confidence=0.0,
                confidence_reason="Нет релевантных документов в базе знаний",
                needs_escalation=True,
            ), None

        response = await self._generate_response(question, documents, chat_history)
        return response, None

    async def ask_by_topic(
        self,
        original_query: str,
        article_id: str,
        topic_title: str,
        chat_history: Optional[List[dict]] = None,
    ) -> RAGResponse:
        """
        Генерация ответа по конкретной выбранной теме.

        Используется когда пользователь выбрал тему из уточняющего списка.
        """
        logger.info(f"[DEMO] ASK_BY_TOPIC|article={article_id}|title={topic_title}")

        documents = self.retrieve_by_article_id(article_id, original_query)

        if not documents:
            # Fallback: обычный поиск
            documents = self.retrieve(original_query)

        if not documents:
            return RAGResponse(
                answer="По этой теме информация не найдена. Передаю вопрос оператору.",
                confidence=0.0,
                confidence_reason="Нет документов по выбранной теме",
                needs_escalation=True,
            )

        return await self._generate_response(original_query, documents, chat_history)

    async def _generate_response(
        self,
        question: str,
        documents: List[Document],
        chat_history: Optional[List[dict]] = None,
    ) -> RAGResponse:
        """
        Внутренний метод генерации ответа через LLM.

        Вынесен из ask() для переиспользования в ask_with_clarification и ask_by_topic.
        """
        import time as _time
        _start = _time.time()

        context_text, article_ids, youtube_links, images = self._build_context(documents)
        logger.info(f"[DEMO] CONTEXT|articles={article_ids}|youtube={len(youtube_links)}|images={len(images)}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        user_message = CONTEXT_TEMPLATE.format(
            context=context_text,
            question=question,
        )
        messages.append({"role": "user", "content": user_message})

        logger.info(f"[DEMO] GPT_CALL|model={settings.openai_model}|messages={len(messages)}|temperature=0.1")
        _gpt_start = _time.time()
        try:
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0.1,
                max_tokens=800,
            )
            raw_answer = response.choices[0].message.content or ""
            _gpt_elapsed = _time.time() - _gpt_start
            _tokens_in = response.usage.prompt_tokens if response.usage else 0
            _tokens_out = response.usage.completion_tokens if response.usage else 0
            logger.info(f"[DEMO] GPT_DONE|time={_gpt_elapsed:.1f}s|tokens_in={_tokens_in}|tokens_out={_tokens_out}")
        except Exception as e:
            logger.error(f"Ошибка OpenAI API: {e}")
            return RAGResponse(
                answer="Техническая ошибка. Попробуйте позже или обратитесь к оператору.",
                confidence=0.0,
                confidence_reason=f"Ошибка API: {str(e)}",
                needs_escalation=True,
            )

        clean_answer, confidence, reason = self._parse_confidence(raw_answer)
        clean_answer = _truncate_to_bytes(clean_answer, MAX_ANSWER_BYTES)
        needs_escalation = confidence < settings.rag_confidence_threshold
        _total = _time.time() - _start

        logger.info(f"[DEMO] DECISION|confidence={confidence}|escalation={needs_escalation}|reason={reason}")
        logger.info(f"[DEMO] COMPLETE|total_time={_total:.1f}s|answer_len={len(clean_answer)}")

        return RAGResponse(
            answer=clean_answer,
            confidence=confidence,
            confidence_reason=reason,
            needs_escalation=needs_escalation,
            source_articles=article_ids,
            youtube_links=youtube_links,
            images=images,
        )


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    """Обрезает текст до max_bytes байт UTF-8, не ломая символы."""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    # Декодируем с игнорированием неполного последнего символа
    text = truncated.decode('utf-8', errors='ignore').rstrip()
    # Обрезаем по последнему предложению
    for sep in ('.\n', '.', '\n'):
        idx = text.rfind(sep)
        if idx > len(text) // 2:
            return text[:idx + 1]
    return text


# Singleton
_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine

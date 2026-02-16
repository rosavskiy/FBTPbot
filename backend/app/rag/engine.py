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
from app.indexer.knowledge_base import get_indexer

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Ты — ИИ-ассистент техподдержки компании ООО «Фармбазис» (www.farmbazis.ru).
Компания разрабатывает программное обеспечение для аптек.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста из базы знаний.
2. Если в контексте нет информации для ответа на вопрос — НЕ ВЫДУМЫВАЙ. Скажи, что не нашёл ответ, и предложи связаться с оператором.
3. Давай пошаговые, подробные инструкции.
4. Если к инструкции есть скриншоты — упомяни, что пользователь может посмотреть скриншоты в статье.
5. Если есть видео-инструкция на YouTube — обязательно дай ссылку.
6. Используй вежливый, профессиональный тон.
7. Отвечай на русском языке.
8. Не раскрывай внутреннюю механику работы бота.

В конце ответа ОБЯЗАТЕЛЬНО добавь JSON-блок оценки (пользователь его не увидит):
```confidence
{"confidence": <число от 0.0 до 1.0>, "reason": "<краткое пояснение>"}
```

Где confidence:
- 0.0-0.3 — ответ не найден, нужна эскалация
- 0.3-0.6 — частичный ответ, может потребоваться помощь оператора
- 0.6-1.0 — уверенный ответ на основе базы знаний
"""

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

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """Поиск релевантных документов."""
        top_k = top_k or settings.rag_top_k

        results = self.vector_store.similarity_search_with_relevance_scores(
            query, k=top_k
        )

        # Фильтруем по минимальному порогу релевантности
        filtered = [
            (doc, score) for doc, score in results
            if score >= settings.rag_confidence_threshold
        ]

        logger.info(
            f"Поиск по '{query[:50]}...': найдено {len(results)} результатов, "
            f"после фильтрации: {len(filtered)}"
        )

        return [doc for doc, _ in filtered]

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
        # 1. Retrieval — поиск релевантных документов
        documents = self.retrieve(question)

        if not documents:
            return RAGResponse(
                answer=(
                    "К сожалению, я не нашёл подходящей информации в базе знаний "
                    "по вашему вопросу. Давайте я передам ваш вопрос оператору "
                    "техподдержки — он сможет помочь более детально."
                ),
                confidence=0.0,
                confidence_reason="Нет релевантных документов в базе знаний",
                needs_escalation=True,
            )

        # 2. Формируем контекст
        context_text, article_ids, youtube_links, images = self._build_context(
            documents
        )

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
        try:
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                messages=messages,
                temperature=0.1,  # Низкая температура для точности
                max_tokens=2000,
            )
            raw_answer = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Ошибка OpenAI API: {e}")
            return RAGResponse(
                answer=(
                    "Произошла техническая ошибка. Пожалуйста, попробуйте ещё раз "
                    "или обратитесь к оператору техподдержки."
                ),
                confidence=0.0,
                confidence_reason=f"Ошибка API: {str(e)}",
                needs_escalation=True,
            )

        # 5. Извлекаем confidence
        clean_answer, confidence, reason = self._parse_confidence(raw_answer)

        # 6. Определяем необходимость эскалации
        needs_escalation = confidence < settings.rag_confidence_threshold

        return RAGResponse(
            answer=clean_answer,
            confidence=confidence,
            confidence_reason=reason,
            needs_escalation=needs_escalation,
            source_articles=article_ids,
            youtube_links=youtube_links,
            images=images,
        )


# Singleton
_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine

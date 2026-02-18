"""Pydantic-модели для API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# === Чат ===

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Сообщение пользователя")
    session_id: Optional[str] = Field(None, description="ID сессии (для продолжения диалога)")


class SuggestedTopicSchema(BaseModel):
    title: str = Field(..., description="Название темы")
    article_id: str = Field(..., description="ID статьи")
    score: float = Field(0.0, description="Релевантность (0-1)")
    snippet: str = Field("", description="Краткий фрагмент текста")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Ответ бота")
    session_id: str = Field(..., description="ID сессии")
    confidence: float = Field(..., description="Уровень уверенности (0-1)")
    needs_escalation: bool = Field(False, description="Требуется ли помощь оператора")
    source_articles: List[str] = Field(default_factory=list, description="ID статей-источников")
    youtube_links: List[str] = Field(default_factory=list, description="YouTube ссылки")
    has_images: bool = Field(False, description="Есть ли скриншоты в источниках")
    response_type: str = Field("answer", description="Тип ответа: answer | clarification")
    suggested_topics: Optional[List[SuggestedTopicSchema]] = Field(
        None, description="Предложенные темы для уточнения (при response_type=clarification)"
    )


# === Эскалация ===

class EscalationStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    resolved = "resolved"
    closed = "closed"


class EscalationRequest(BaseModel):
    session_id: str = Field(..., description="ID сессии чата")
    reason: Optional[str] = Field(None, description="Причина эскалации от пользователя")
    contact_info: Optional[str] = Field(None, description="Контактные данные (email/телефон)")


class EscalationResponse(BaseModel):
    escalation_id: str
    status: str = "pending"
    message: str = "Ваш запрос передан оператору техподдержки. Ожидайте ответа."
    position_in_queue: int = 0


class EscalationDetail(BaseModel):
    escalation_id: str
    session_id: str
    status: EscalationStatus
    reason: Optional[str] = None
    contact_info: Optional[str] = None
    chat_history: List[ChatMessage] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    operator_notes: Optional[str] = None


# === Панель оператора ===

class OperatorLoginRequest(BaseModel):
    username: str
    password: str


class OperatorLoginResponse(BaseModel):
    token: str
    username: str


class OperatorReplyRequest(BaseModel):
    escalation_id: str
    message: str
    close_ticket: bool = False


class EscalationListResponse(BaseModel):
    escalations: List[EscalationDetail]
    total: int
    pending_count: int


# === Обратная связь ===

class FeedbackRequest(BaseModel):
    session_id: str
    message_index: int = Field(0, description="Индекс сообщения")
    rating: int = Field(..., ge=1, le=5, description="Оценка 1-5")
    comment: Optional[str] = Field(None, max_length=500)


class FeedbackResponse(BaseModel):
    success: bool = True
    message: str = "Спасибо за обратную связь!"


# === Система ===

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    knowledge_base_ready: bool = False
    total_articles: int = 0
    total_chunks: int = 0
    support_tickets_count: int = 0

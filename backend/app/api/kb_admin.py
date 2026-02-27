"""
API-эндпоинты для управления базой знаний Q&A (квиз-режим + импорт).

Позволяет:
- Просматривать Q&A пары из JSON-файла
- Редактировать вопросы/ответы/метаданные
- Одобрять пары (approved)
- Удалять некачественные записи
- Импортировать новые данные
- Синхронизировать с ChromaDB (инкрементально + полный реиндекс)
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

from app.config import settings
from app.indexer.knowledge_base import (
    SUPPORT_COLLECTION_NAME,
    get_indexer,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/kb", tags=["kb-admin"])

# ─── Путь к JSON-файлу базы знаний ───────────────────────────────────
# На сервере: /app/data/support_kb.json (persistent volume)
# Локально: real_support/processed/support_qa_documents_merged_final.json
_KB_PATHS = [
    Path("/app/data/support_kb.json"),                    # Docker production
    Path("/app/real_support/processed/support_qa_documents_merged_final.json"),  # Docker alt
    Path(__file__).resolve().parents[3] / "real_support" / "processed" / "support_qa_documents_merged_final.json",  # Local dev
]
KB_JSON_PATH = next((p for p in _KB_PATHS if p.exists()), _KB_PATHS[0])

KB_BACKUP_DIR = KB_JSON_PATH.parent / "backups"


# ─── Pydantic-модели ─────────────────────────────────────────────────

class KBItemMetadata(BaseModel):
    source: str = "real_support_tickets"
    category: str = "Прочее"
    category_en: str = "general"
    tags: List[str] = []
    quality_score: int = 3
    question: str = ""
    answer: str = ""
    type: str = "qa_pair"


class KBItem(BaseModel):
    id: str
    text: str
    metadata: KBItemMetadata
    reviewed: bool = False
    review_date: Optional[str] = None


class KBItemUpdate(BaseModel):
    question: Optional[str] = None
    answer: Optional[str] = None
    category: Optional[str] = None
    category_en: Optional[str] = None
    tags: Optional[List[str]] = None
    quality_score: Optional[int] = None


class KBStats(BaseModel):
    total: int = 0
    reviewed: int = 0
    unreviewed: int = 0
    by_category: Dict[str, int] = {}
    by_quality: Dict[str, int] = {}
    avg_quality: float = 0.0


class KBImportResult(BaseModel):
    added: int = 0
    duplicates_skipped: int = 0
    errors: int = 0
    message: str = ""


class KBReindexResult(BaseModel):
    total_documents: int = 0
    duration_seconds: float = 0.0
    message: str = ""


# ─── Утилиты для работы с JSON-файлом ────────────────────────────────

def _load_kb() -> List[Dict[str, Any]]:
    """Загрузить базу знаний из JSON."""
    if not KB_JSON_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Файл БЗ не найден: {KB_JSON_PATH}")
    try:
        with open(KB_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка парсинга JSON: {e}")


def _save_kb(data: List[Dict[str, Any]], backup: bool = True):
    """Сохранить базу знаний в JSON с опциональным бэкапом."""
    if backup:
        KB_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = KB_BACKUP_DIR / f"kb_backup_{ts}.json"
        if KB_JSON_PATH.exists():
            shutil.copy2(KB_JSON_PATH, backup_path)
            logger.info(f"Бэкап создан: {backup_path}")
            # Оставляем только последние 20 бэкапов
            backups = sorted(KB_BACKUP_DIR.glob("kb_backup_*.json"))
            for old in backups[:-20]:
                old.unlink()

    with open(KB_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"БЗ сохранена: {len(data)} записей -> {KB_JSON_PATH}")


def _update_chromadb_document(item: Dict[str, Any]):
    """Инкрементально обновить один документ в ChromaDB."""
    try:
        indexer = get_indexer()
        store = indexer.get_support_vector_store()
        if store is None:
            logger.warning("support_vector_store не инициализирован — пропуск обновления ChromaDB")
            return

        collection = store._collection
        doc_id = item["id"]
        metadata = item.get("metadata", {})

        # Подготовка метаданных (ChromaDB не поддерживает списки)
        clean_meta = {
            "source": metadata.get("source", "real_support_tickets"),
            "category": metadata.get("category", "Прочее"),
            "category_en": metadata.get("category_en", "general"),
            "quality_score": metadata.get("quality_score", 0),
            "question": metadata.get("question", "")[:500],
            "doc_type": metadata.get("type", "qa_pair"),
            "article_id": f"tp_{doc_id}",
            "title": metadata.get("question", "Заявка ТП")[:200],
        }
        if metadata.get("tags"):
            clean_meta["tags"] = ", ".join(metadata["tags"])
        if item.get("reviewed"):
            clean_meta["reviewed"] = "true"

        text = item["text"]

        # Пробуем обновить, если не существует — добавляем
        existing = collection.get(ids=[doc_id])
        if existing and existing["ids"]:
            collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[clean_meta],
            )
            logger.info(f"ChromaDB: обновлён документ {doc_id}")
        else:
            collection.add(
                ids=[doc_id],
                documents=[text],
                metadatas=[clean_meta],
            )
            logger.info(f"ChromaDB: добавлен документ {doc_id}")

    except Exception as e:
        logger.error(f"Ошибка обновления ChromaDB для {item.get('id')}: {e}")


def _delete_chromadb_document(doc_id: str):
    """Удалить документ из ChromaDB."""
    try:
        indexer = get_indexer()
        store = indexer.get_support_vector_store()
        if store is None:
            return
        collection = store._collection
        collection.delete(ids=[doc_id])
        logger.info(f"ChromaDB: удалён документ {doc_id}")
    except Exception as e:
        logger.error(f"Ошибка удаления из ChromaDB {doc_id}: {e}")


# ─── Эндпоинты ───────────────────────────────────────────────────────

@router.get("/stats", response_model=KBStats)
async def get_kb_stats():
    """Статистика базы знаний."""
    data = _load_kb()
    by_category: Dict[str, int] = {}
    by_quality: Dict[str, int] = {}
    reviewed = 0
    total_quality = 0.0

    for item in data:
        meta = item.get("metadata", {})
        cat = meta.get("category", "Прочее")
        by_category[cat] = by_category.get(cat, 0) + 1

        qs = str(meta.get("quality_score", 0))
        by_quality[qs] = by_quality.get(qs, 0) + 1
        total_quality += meta.get("quality_score", 0)

        if item.get("reviewed"):
            reviewed += 1

    return KBStats(
        total=len(data),
        reviewed=reviewed,
        unreviewed=len(data) - reviewed,
        by_category=by_category,
        by_quality=by_quality,
        avg_quality=round(total_quality / max(len(data), 1), 2),
    )


@router.get("/items")
async def list_kb_items(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    reviewed: Optional[bool] = None,
    quality_min: Optional[int] = None,
    quality_max: Optional[int] = None,
    search: Optional[str] = None,
):
    """Список Q&A пар с пагинацией и фильтрацией."""
    data = _load_kb()

    # Фильтрация
    if category:
        data = [d for d in data if d.get("metadata", {}).get("category") == category]
    if reviewed is not None:
        data = [d for d in data if d.get("reviewed", False) == reviewed]
    if quality_min is not None:
        data = [d for d in data if d.get("metadata", {}).get("quality_score", 0) >= quality_min]
    if quality_max is not None:
        data = [d for d in data if d.get("metadata", {}).get("quality_score", 0) <= quality_max]
    if search:
        search_lower = search.lower()
        data = [
            d for d in data
            if search_lower in d.get("metadata", {}).get("question", "").lower()
            or search_lower in d.get("metadata", {}).get("answer", "").lower()
            or search_lower in d.get("text", "").lower()
        ]

    total = len(data)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "items": data[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.get("/items/{item_id}")
async def get_kb_item(item_id: str):
    """Получить одну Q&A пару."""
    data = _load_kb()
    for item in data:
        if item.get("id") == item_id:
            return item
    raise HTTPException(status_code=404, detail=f"Элемент {item_id} не найден")


@router.get("/quiz/next")
async def get_next_quiz_item(
    category: Optional[str] = None,
    skip_reviewed: bool = True,
):
    """
    Получить следующую неотревьюенную Q&A пару для квиза.
    Возвращает элемент + общий прогресс.
    """
    data = _load_kb()

    candidates = data
    if category:
        candidates = [d for d in candidates if d.get("metadata", {}).get("category") == category]
    if skip_reviewed:
        candidates = [d for d in candidates if not d.get("reviewed", False)]

    total = len(data)
    reviewed_count = sum(1 for d in data if d.get("reviewed", False))

    if not candidates:
        return {
            "item": None,
            "progress": {
                "total": total,
                "reviewed": reviewed_count,
                "remaining": 0,
                "percent": 100.0 if total > 0 else 0.0,
            },
            "message": "Все записи проверены! 🎉",
        }

    item = candidates[0]
    idx = data.index(item)

    return {
        "item": item,
        "index": idx,
        "progress": {
            "total": total,
            "reviewed": reviewed_count,
            "remaining": len(candidates),
            "percent": round(reviewed_count / max(total, 1) * 100, 1),
        },
    }


@router.put("/items/{item_id}")
async def update_kb_item(item_id: str, update: KBItemUpdate):
    """Обновить Q&A пару (вопрос, ответ, категорию, теги и т.д.)."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Элемент {item_id} не найден")

    item = data[found_idx]
    meta = item.get("metadata", {})

    # Применяем обновления
    if update.question is not None:
        meta["question"] = update.question
    if update.answer is not None:
        meta["answer"] = update.answer
    if update.category is not None:
        meta["category"] = update.category
    if update.category_en is not None:
        meta["category_en"] = update.category_en
    if update.tags is not None:
        meta["tags"] = update.tags
    if update.quality_score is not None:
        meta["quality_score"] = update.quality_score

    # Перестраиваем текст
    q = meta.get("question", "")
    a = meta.get("answer", "")
    item["text"] = f"Вопрос: {q}\n\nОтвет: {a}"
    item["metadata"] = meta

    data[found_idx] = item
    _save_kb(data)

    # Инкрементально обновляем ChromaDB
    _update_chromadb_document(item)

    return {"status": "ok", "item": item}


@router.post("/items/{item_id}/approve")
async def approve_kb_item(item_id: str):
    """Одобрить Q&A пару (пометить как проверенную, quality_score=5)."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Элемент {item_id} не найден")

    item = data[found_idx]
    item["reviewed"] = True
    item["review_date"] = datetime.now().isoformat()
    meta = item.get("metadata", {})
    meta["quality_score"] = 5
    item["metadata"] = meta

    data[found_idx] = item
    _save_kb(data)

    # Инкрементально обновляем ChromaDB
    _update_chromadb_document(item)

    return {"status": "ok", "item": item}


@router.post("/items/{item_id}/save-and-approve")
async def save_and_approve_kb_item(item_id: str, update: KBItemUpdate):
    """Обновить и сразу одобрить Q&A пару."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Элемент {item_id} не найден")

    item = data[found_idx]
    meta = item.get("metadata", {})

    # Применяем обновления
    if update.question is not None:
        meta["question"] = update.question
    if update.answer is not None:
        meta["answer"] = update.answer
    if update.category is not None:
        meta["category"] = update.category
    if update.category_en is not None:
        meta["category_en"] = update.category_en
    if update.tags is not None:
        meta["tags"] = update.tags
    if update.quality_score is not None:
        meta["quality_score"] = update.quality_score
    else:
        meta["quality_score"] = 5

    # Перестраиваем текст
    q = meta.get("question", "")
    a = meta.get("answer", "")
    item["text"] = f"Вопрос: {q}\n\nОтвет: {a}"
    item["metadata"] = meta
    item["reviewed"] = True
    item["review_date"] = datetime.now().isoformat()

    data[found_idx] = item
    _save_kb(data)

    _update_chromadb_document(item)

    return {"status": "ok", "item": item}


@router.delete("/items/{item_id}")
async def delete_kb_item(item_id: str):
    """Удалить Q&A пару из базы знаний."""
    data = _load_kb()

    found_idx = None
    for idx, item in enumerate(data):
        if item.get("id") == item_id:
            found_idx = idx
            break

    if found_idx is None:
        raise HTTPException(status_code=404, detail=f"Элемент {item_id} не найден")

    removed = data.pop(found_idx)
    _save_kb(data)

    _delete_chromadb_document(item_id)

    return {"status": "ok", "deleted_id": item_id, "remaining": len(data)}


@router.post("/import", response_model=KBImportResult)
async def import_kb_data(file: UploadFile = File(...)):
    """
    Импорт новых Q&A данных из JSON-файла.
    Дубликаты (по id) пропускаются, новые записи добавляются.
    """
    try:
        content = await file.read()
        new_items = json.loads(content.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {e}")

    if not isinstance(new_items, list):
        raise HTTPException(status_code=400, detail="Ожидается JSON-массив")

    data = _load_kb()
    existing_ids = {item["id"] for item in data}

    added = 0
    duplicates = 0
    errors = 0

    for new_item in new_items:
        try:
            if not isinstance(new_item, dict) or "id" not in new_item:
                errors += 1
                continue

            if new_item["id"] in existing_ids:
                duplicates += 1
                continue

            # Убеждаемся, что есть все нужные поля
            if "text" not in new_item:
                meta = new_item.get("metadata", {})
                q = meta.get("question", "")
                a = meta.get("answer", "")
                new_item["text"] = f"Вопрос: {q}\n\nОтвет: {a}"

            if "metadata" not in new_item:
                new_item["metadata"] = {
                    "source": "real_support_tickets",
                    "category": "Прочее",
                    "category_en": "general",
                    "tags": [],
                    "quality_score": 3,
                    "question": "",
                    "answer": "",
                    "type": "qa_pair",
                }

            new_item["reviewed"] = False
            data.append(new_item)
            existing_ids.add(new_item["id"])
            added += 1

            # Добавляем в ChromaDB
            _update_chromadb_document(new_item)

        except Exception as e:
            logger.error(f"Ошибка импорта элемента: {e}")
            errors += 1

    _save_kb(data)

    return KBImportResult(
        added=added,
        duplicates_skipped=duplicates,
        errors=errors,
        message=f"Импорт завершён: +{added} новых, {duplicates} дубликатов пропущено, {errors} ошибок",
    )


@router.post("/reindex", response_model=KBReindexResult)
async def reindex_kb():
    """
    Полная переиндексация support_tickets в ChromaDB из JSON-файла.
    Используйте после массовых правок.
    """
    import time

    start = time.time()

    try:
        indexer = get_indexer()
        count = indexer.index_support_tickets(KB_JSON_PATH)

        # Сбрасываем кеш vector_store в RAG-движке
        from app.rag.engine import _engine as _rag_engine
        if _rag_engine is not None and hasattr(_rag_engine, '_support_vector_store'):
            delattr(_rag_engine, '_support_vector_store')
            logger.info("Кеш support_vector_store в RAG-движке сброшен")

        # Также сбросим кеш в индексаторе
        indexer.support_vector_store = None

        duration = round(time.time() - start, 2)
        return KBReindexResult(
            total_documents=count,
            duration_seconds=duration,
            message=f"Переиндексация завершена: {count} документов за {duration}с",
        )
    except Exception as e:
        logger.error(f"Ошибка переиндексации: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка переиндексации: {e}")


@router.get("/categories")
async def get_categories():
    """Получить список всех категорий."""
    data = _load_kb()
    categories = {}
    for item in data:
        cat = item.get("metadata", {}).get("category", "Прочее")
        cat_en = item.get("metadata", {}).get("category_en", "general")
        categories[cat] = cat_en
    return {"categories": categories}

"""Тесты лимита длины ответа для интеграции с Фармбазис."""

import os
import sys


# Добавляем backend в PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rag.engine import MAX_ANSWER_BYTES, SYSTEM_PROMPT, _truncate_to_bytes


def test_answer_limit_is_doubled():
    assert MAX_ANSWER_BYTES == 2048
    assert "НЕ БОЛЕЕ 900 символов" in SYSTEM_PROMPT


def test_truncate_to_bytes_respects_new_utf8_limit():
    source_text = "я" * 1100

    truncated = _truncate_to_bytes(source_text, MAX_ANSWER_BYTES)

    assert len(truncated.encode("utf-8")) <= MAX_ANSWER_BYTES
    assert truncated == "я" * 1024
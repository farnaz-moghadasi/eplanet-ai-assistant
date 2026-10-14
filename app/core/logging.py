from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

_conversation_id: ContextVar[str | None] = ContextVar("conversation_id", default=None)


class JsonFormatter(logging.Formatter):
    _reserved = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        request_id = _request_id.get()

        if request_id:
            payload["request_id"] = request_id

        conversation_id = _conversation_id.get()

        if conversation_id:
            payload["conversation_id"] = conversation_id

        for key, value in record.__dict__.items():
            if key in self._reserved or key.startswith("_"):
                continue

            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)

        if record.exc_info:
            payload["exception_type"] = (
                record.exc_info[0].__name__ if record.exc_info[0] else None
            )

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, level: str = "INFO") -> None:
    root_logger = logging.getLogger()

    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    handler.setFormatter(JsonFormatter())

    root_logger.addHandler(handler)

    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    third_party_levels = {
        "httpx": logging.WARNING,
        "httpcore": logging.WARNING,
        "openai": logging.WARNING,
        "qdrant_client": logging.WARNING,
        "urllib3": logging.WARNING,
        "asyncio": logging.WARNING,
    }

    for logger_name, logger_level in third_party_levels.items():
        logger = logging.getLogger(logger_name)

        logger.setLevel(logger_level)

        logger.propagate = True


def bind_log_context(
    *, request_id: str | None = None, conversation_id: str | None = None
) -> None:
    if request_id is not None:
        _request_id.set(request_id)

    if conversation_id is not None:
        _conversation_id.set(conversation_id)


def clear_log_context() -> None:
    _request_id.set(None)

    _conversation_id.set(None)

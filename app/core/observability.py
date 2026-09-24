from __future__ import annotations

import logging
from typing import Any

from app.core.performance import (
    record_timing,
    record_value,
)


logger = logging.getLogger(
    "eplanet.observability"
)


_TIMING_EVENT_MAP = {
    "conversation.understanding.completed":
        "understanding_ms",

    "conversation.context_requirement.completed":
        "context_requirement_ms",

    "retrieval.bm25.completed":
        "bm25_ms",

    "retrieval.vector.completed":
        "vector_ms",

    "retrieval.completed":
        "retrieval_ms",

    "conversation.retrieval.completed":
        "conversation_retrieval_ms",

    "conversation.generation.completed":
        "generation_ms",

    "conversation.citation_validation.completed":
        "citation_validation_ms",

    "conversation.grounding.completed":
        "grounding_ms",

    "chat.request.completed":
        "total_ms",

    "chat.request.failed":
        "total_ms",
}


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    duration_ms = fields.get(
        "duration_ms"
    )

    timing_name = (
        _TIMING_EVENT_MAP.get(
            event
        )
    )

    if (
        timing_name is not None
        and duration_ms is not None
    ):
        record_timing(
            timing_name,
            duration_ms,
        )

    if event == "retrieval.completed":
        record_value(
            "reranker_mode",
            fields.get(
                "reranker_mode"
            ),
        )

        record_value(
            "retrieval_top_knowledge_key",
            fields.get(
                "top_knowledge_key"
            ),
        )

    logger.log(
        level,
        event,
        extra=fields,
    )
from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    Field,
)


class ChatRequest(BaseModel):
    conversation_id: str = Field(
        min_length=1,
        max_length=200,
    )

    message: str = Field(
        min_length=1,
        max_length=10000,
    )

    language: (
        str | None
    ) = None


class ChatCitationResponse(BaseModel):
    candidate_id: str
    knowledge_key: str


class ClarificationResponse(BaseModel):
    required: bool = False

    field: (
        str | None
    ) = None

    reason: (
        str | None
    ) = None

class PerformanceTraceResponse(
    BaseModel
):
    understanding_ms: (
        float | None
    ) = None

    context_requirement_ms: (
        float | None
    ) = None

    bm25_ms: (
        float | None
    ) = None

    vector_ms: (
        float | None
    ) = None

    hybrid_ms: (
        float | None
    ) = None

    eligibility_ms: (
        float | None
    ) = None

    reranker_ms: (
        float | None
    ) = None

    retrieval_ms: (
        float | None
    ) = None

    conversation_retrieval_ms: (
        float | None
    ) = None

    generation_ms: (
        float | None
    ) = None

    citation_validation_ms: (
        float | None
    ) = None

    grounding_ms: (
        float | None
    ) = None

    total_ms: (
        float | None
    ) = None
    
class ChatDebugResponse(BaseModel):
    language: (
        str | None
    ) = None

    domain: (
        str | None
    ) = None

    task: (
        str | None
    ) = None

    resolved_query: (
        str | None
    ) = None

    reranker_mode: (
        str | None
    ) = None

    understanding_mode: (
        str | None
    ) = None

    generation_mode: (
        str | None
    ) = None

    grounding_mode: (
        str | None
    ) = None

    top_knowledge_key: (
        str | None
    ) = None

    top_rerank_score: (
        float | None
    ) = None

    final_candidate_count: int = 0

    citation_valid: (
        bool | None
    ) = None

    grounding_valid: (
        bool | None
    ) = None

    abstained: (
        bool | None
    ) = None

    safe_context: dict[
        str,
        str,
    ] = Field(
        default_factory=dict
    )

    candidates: list[
        dict[str, Any]
    ] = Field(
        default_factory=list
    )

    eligibility_blocked: list[
        dict[str, Any]
    ] = Field(
        default_factory=list
    )

    performance: (
        PerformanceTraceResponse
        | None
    ) = None


class ChatResponse(BaseModel):
    conversation_id: str

    answer: str

    language: (
        str | None
    ) = None

    clarification: (
        ClarificationResponse
    )

    citations: list[
        ChatCitationResponse
    ] = Field(
        default_factory=list
    )

    abstained: bool = False

    abstain_reason: (
        str | None
    ) = None

    needs_handoff: bool = False

    debug: (
        ChatDebugResponse
        | None
    ) = None


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str

    runtime_initialized: bool

    retrieval_initialized: bool

    redis_ready: bool
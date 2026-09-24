from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    Field,
)


class ExpectedTurnResult(BaseModel):
    top_knowledge_key: (
        str | None
    ) = None

    acceptable_top_knowledge_keys: list[
        str
    ] = Field(
        default_factory=list
    )

    required_in_top_k: list[
        str
    ] = Field(
        default_factory=list
    )

    forbidden_knowledge_keys: list[
        str
    ] = Field(
        default_factory=list
    )

    forbidden_knowledge_prefixes: list[
        str
    ] = Field(
        default_factory=list
    )

    clarification_required: (
        bool | None
    ) = None

    clarification_field: (
        str | None
    ) = None

    abstained: (
        bool | None
    ) = None

    citation_valid: (
        bool | None
    ) = None

    grounding_valid: (
        bool | None
    ) = None

    expected_language: (
        str | None
    ) = None

    expected_domain: (
        str | None
    ) = None

    expected_task: (
        str | None
    ) = None

    expected_context: dict[
        str,
        str,
    ] = Field(
        default_factory=dict
    )


class EvaluationTurn(BaseModel):
    message: str

    language: (
        str | None
    ) = None

    expected: ExpectedTurnResult = (
        Field(
            default_factory=(
                ExpectedTurnResult
            )
        )
    )


class EvaluationTrustedSession(
    BaseModel
):
    is_authenticated: bool = False

    client_id: (
        str | None
    ) = None


class EvaluationCase(BaseModel):
    id: str
    description: str

    enabled: bool = True

    tags: list[str] = Field(
        default_factory=list
    )

    initial_context: dict[
        str,
        str,
    ] = Field(
        default_factory=dict
    )

    trusted_session: (
        EvaluationTrustedSession
        | None
    ) = None

    turns: list[
        EvaluationTurn
    ]


class CandidateDiagnostic(BaseModel):
    rank: int
    knowledge_key: str
    candidate_id: str

    rerank_score: (
        float | None
    ) = None

    hybrid_score: (
        float | None
    ) = None

    bm25_score: (
        float | None
    ) = None

    vector_score: (
        float | None
    ) = None


class GroundingClaimDiagnostic(
    BaseModel
):
    claim: str | None = None

    supported: bool

    supporting_candidate_ids: list[
        str
    ] = Field(
        default_factory=list
    )


class TurnDiagnostics(BaseModel):
    language: (
        str | None
    ) = None

    domain: (
        str | None
    ) = None

    task: (
        str | None
    ) = None

    client_specific: (
        bool | None
    ) = None

    understanding_confidence: (
        float | None
    ) = None

    resolved_query: (
        str | None
    ) = None

    missing_context_fields: list[
        str
    ] = Field(
        default_factory=list
    )

    awaiting_context_field: (
        str | None
    ) = None

    context_snapshot: dict[
        str,
        str,
    ] = Field(
        default_factory=dict
    )

    candidates: list[
        CandidateDiagnostic
    ] = Field(
        default_factory=list
    )

    eligibility_blocked: list[
        dict[str, Any]
    ] = Field(
        default_factory=list
    )

    reranker_mode: (
        str | None
    ) = None

    reranker_fallback_reason: (
        str | None
    ) = None

    generation_abstained: (
        bool | None
    ) = None

    generation_abstain_reason: (
        str | None
    ) = None

    generation_needs_handoff: (
        bool | None
    ) = None

    generation_citation_count: (
        int | None
    ) = None

    citation_valid: (
        bool | None
    ) = None

    citation_issues: list[
        str
    ] = Field(
        default_factory=list
    )

    grounding_valid: (
        bool | None
    ) = None

    grounding_skipped: (
        bool | None
    ) = None

    grounding_reason: (
        str | None
    ) = None

    grounding_issues: list[
        str
    ] = Field(
        default_factory=list
    )

    grounding_claims: list[
        GroundingClaimDiagnostic
    ] = Field(
        default_factory=list
    )


class TurnEvaluationResult(BaseModel):
    case_id: str
    turn_index: int

    passed: bool

    failures: list[str] = Field(
        default_factory=list
    )

    top_knowledge_key: (
        str | None
    ) = None

    ranked_knowledge_keys: list[
        str
    ] = Field(
        default_factory=list
    )

    reciprocal_rank: (
        float | None
    ) = None

    clarification_required: bool = False

    clarification_field: (
        str | None
    ) = None

    abstained: (
        bool | None
    ) = None

    citation_valid: (
        bool | None
    ) = None

    grounding_valid: (
        bool | None
    ) = None

    reranker_mode: (
        str | None
    ) = None

    latency_ms: float = 0.0

    diagnostics: TurnDiagnostics = (
        Field(
            default_factory=(
                TurnDiagnostics
            )
        )
    )


class EvaluationMetrics(BaseModel):
    total_cases: int
    enabled_cases: int
    skipped_cases: int

    total_turns: int
    passed_turns: int
    failed_turns: int

    turn_pass_rate: (
        float | None
    ) = None

    top1_accuracy: (
        float | None
    ) = None

    recall_at_5: (
        float | None
    ) = None

    mrr: (
        float | None
    ) = None

    clarification_accuracy: (
        float | None
    ) = None

    context_accuracy: (
        float | None
    ) = None

    citation_validity_rate: (
        float | None
    ) = None

    grounding_pass_rate: (
        float | None
    ) = None

    abstention_accuracy: (
        float | None
    ) = None

    leakage_rate: (
        float | None
    ) = None

    average_latency_ms: (
        float | None
    ) = None


class EvaluationReport(BaseModel):
    metrics: EvaluationMetrics

    results: list[
        TurnEvaluationResult
    ]

    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )
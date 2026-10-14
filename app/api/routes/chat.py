from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.api.rate_limit import enforce_chat_rate_limit
from app.api.schemas import (
    ChatCitationResponse,
    ChatDebugResponse,
    ChatRequest,
    ChatResponse,
    ClarificationResponse,
    PerformanceTraceResponse,
)
from app.core.config import settings
from app.core.observability import log_event
from app.core.performance import (
    clear_performance_trace,
    get_performance_trace,
    record_timing,
    start_performance_trace,
)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    runtime = getattr(request.app.state, "runtime", None)

    if runtime is None:
        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail=("Application runtime " "is not ready."),
        )

    await enforce_chat_rate_limit(redis=(runtime.state_store.redis), request=request)

    conversation_id = payload.conversation_id.strip()

    message = payload.message.strip()

    if not message:
        raise HTTPException(
            status_code=(status.HTTP_422_UNPROCESSABLE_ENTITY),
            detail=("message cannot be empty."),
        )

    start_performance_trace()

    request_started = time.perf_counter()

    try:
        async with runtime.conversation_lock(conversation_id=conversation_id):
            state = await runtime.get_conversation_state(
                conversation_id=(conversation_id)
            )

            result = await runtime.conversation_service.handle_message(
                message=message,
                state=state,
                language=(payload.language),
                trusted_session=None,
            )

            await runtime.save_conversation_state(state=state)

        total_ms = (time.perf_counter() - request_started) * 1000

        record_timing("api_total_ms", total_ms)

        performance = get_performance_trace()

        response = _build_response(
            conversation_id=(conversation_id),
            result=result,
            state=state,
            performance=(performance),
        )

        return response

    except HTTPException:
        raise

    except Exception as exc:
        import traceback

        traceback.print_exc()

        log_event(
            "api.chat.unhandled_exception",
            level=40,
            error_type=(exc.__class__.__name__),
            conversation_id=(conversation_id),
        )

        # Never expose raw exception text to the caller - full details
        # go to the server-side log above; the client only needs to
        # know the request failed.
        raise HTTPException(
            status_code=500, detail=("The assistant could not " "process the request.")
        ) from exc

    finally:
        clear_performance_trace()


def _build_response(
    *, conversation_id: str, result, state, performance: dict[str, Any]
) -> ChatResponse:
    clarification = _build_clarification(result)

    generation = result.generation_result

    answer = ""
    abstained = False
    abstain_reason = None
    needs_handoff = False

    citations: list[ChatCitationResponse] = []

    if generation is not None:
        answer = generation.payload.answer

        abstained = generation.payload.abstained

        abstain_reason = generation.payload.abstain_reason

        needs_handoff = generation.payload.needs_handoff

        citations = _build_citations(generation.payload.citations)

    elif result.clarification_question:
        answer = result.clarification_question

    debug = None

    if settings.DEMO_DEBUG_TRACE_ENABLED:
        debug = _build_debug_response(
            result=result, state=state, performance=(performance)
        )

    return ChatResponse(
        conversation_id=(conversation_id),
        answer=answer,
        language=(result.understanding.language),
        clarification=(clarification),
        citations=(citations),
        abstained=(abstained),
        abstain_reason=(abstain_reason),
        needs_handoff=(needs_handoff),
        debug=debug,
    )


def _build_clarification(result) -> ClarificationResponse:
    if not (result.resolved_turn.needs_clarification):
        return ClarificationResponse(required=False)

    field = None

    reason = result.resolved_turn.clarification_reason

    if result.resolved_turn.missing_context_fields:
        field = result.resolved_turn.missing_context_fields[0]

    return ClarificationResponse(required=True, field=field, reason=reason)


def _build_citations(citations) -> list[ChatCitationResponse]:
    result: list[ChatCitationResponse] = []

    for citation in citations:
        candidate_id = getattr(citation, "candidate_id", None)

        knowledge_key = getattr(citation, "knowledge_key", None)

        if not (candidate_id and knowledge_key):
            continue

        result.append(
            ChatCitationResponse(
                candidate_id=str(candidate_id), knowledge_key=str(knowledge_key)
            )
        )

    return result


def _build_debug_response(
    *, result, state, performance: dict[str, Any]
) -> ChatDebugResponse:
    retrieval = result.retrieval_result

    candidates: list[dict[str, Any]] = []

    top_knowledge_key = None
    top_rerank_score = None
    reranker_mode = None

    if retrieval is not None:
        reranker_mode = retrieval.reranker_mode

        for rank, item in enumerate(retrieval.final_candidates, start=1):
            candidate = item.candidate.candidate

            knowledge_key = candidate.metadata.knowledge_key

            rerank_score = getattr(item, "rerank_score", None)

            candidates.append(
                {
                    "rank": rank,
                    "knowledge_key": (knowledge_key),
                    "rerank_score": (rerank_score),
                }
            )

        if retrieval.final_candidates:
            first = retrieval.final_candidates[0]

            top_knowledge_key = first.candidate.candidate.metadata.knowledge_key

            top_rerank_score = getattr(first, "rerank_score", None)

    eligibility_blocked: list[dict[str, Any]] = []

    if retrieval is not None:
        for item in retrieval.eligibility_trace:
            if item.eligible:
                continue

            eligibility_blocked.append(
                {"knowledge_key": (item.knowledge_key), "reasons": list(item.reasons)}
            )

    citation_valid = None

    if result.citation_validation_result is not None:
        citation_valid = result.citation_validation_result.valid

    grounding_valid = None

    if result.grounding_validation_result is not None:
        grounding_valid = result.grounding_validation_result.valid

    abstained = None

    if result.generation_result is not None:
        abstained = result.generation_result.payload.abstained

    safe_context: dict[str, str] = {}

    for field_name in (
        "nationality",
        "country_of_residence",
        "kyc_document_country",
        "referenced_country",
    ):
        value = state.get_context_field(field_name)

        if value:
            safe_context[field_name] = str(value)

    return ChatDebugResponse(
        language=(result.understanding.language),
        domain=(result.understanding.domain.value),
        task=(result.understanding.task.value),
        resolved_query=(result.resolved_turn.resolved_query),
        reranker_mode=(reranker_mode),
        understanding_mode=(performance.get("understanding_mode")),
        generation_mode=(performance.get("generation_mode")),
        grounding_mode=(performance.get("grounding_mode")),
        top_knowledge_key=(top_knowledge_key),
        top_rerank_score=(top_rerank_score),
        final_candidate_count=len(candidates),
        citation_valid=(citation_valid),
        grounding_valid=(grounding_valid),
        abstained=(abstained),
        safe_context=(safe_context),
        candidates=(candidates),
        eligibility_blocked=(eligibility_blocked),
        performance=(
            PerformanceTraceResponse(
                understanding_ms=(performance.get("understanding_ms")),
                context_requirement_ms=(performance.get("context_requirement_ms")),
                bm25_ms=(performance.get("bm25_ms")),
                vector_ms=(performance.get("vector_ms")),
                retrieval_ms=(performance.get("retrieval_ms")),
                conversation_retrieval_ms=(
                    performance.get("conversation_retrieval_ms")
                ),
                generation_ms=(performance.get("generation_ms")),
                hybrid_ms=(performance.get("hybrid_ms")),
                eligibility_ms=(performance.get("eligibility_ms")),
                reranker_ms=(performance.get("reranker_ms")),
                citation_validation_ms=(performance.get("citation_validation_ms")),
                grounding_ms=(performance.get("grounding_ms")),
                total_ms=(
                    performance.get("api_total_ms") or performance.get("total_ms")
                ),
            )
        ),
    )

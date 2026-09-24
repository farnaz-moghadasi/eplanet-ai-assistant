from __future__ import annotations

import asyncio

from app.ai.openai import (
    OpenAIStructuredLLM,
)
from app.core.config import settings
from app.core.logging import configure_logging
from app.citations.validator import (
    CitationValidator,
)
from app.grounding.service import (
    GroundingValidationService,
)
from app.conversation.resolver import (
    ConversationResolver,
)
from app.conversation.service import (
    ConversationService,
    TrustedSessionContext,
)
from app.conversation.state import (
    ConversationState,
)
from app.conversation.understanding import (
    ConversationUnderstandingService,
)
from app.core.config import settings
from app.generation.service import (
    AnswerGenerationService,
)
from app.runtime.context_policy import (
    ContextPolicy,
)
from app.runtime.entity_registry import (
    EntityRegistry,
)
from scripts.test_pipeline_real_kb import (
    build_pipeline,
)


def print_candidates(
    result,
) -> None:
    retrieval = (
        result.retrieval_result
    )

    if retrieval is None:
        print(
            "Retrieval skipped."
        )
        return

    print(
        "Reranker mode: "
        f"{retrieval.reranker_mode}"
    )

    if (
        retrieval
        .reranker_fallback_reason
    ):
        print(
            "Reranker fallback: "
            f"{retrieval.reranker_fallback_reason}"
        )

    if not (
        retrieval.final_candidates
    ):
        print(
            "Final candidates: NONE"
        )
        return

    print(
        "Final candidates:"
    )

    for (
        rank,
        item,
    ) in enumerate(
        retrieval.final_candidates,
        start=1,
    ):
        hybrid = (
            item.candidate
        )

        candidate = (
            hybrid.candidate
        )

        print(
            f"  #{rank} "
            f"{candidate.metadata.knowledge_key}"
        )

        print(
            f"     title="
            f"{candidate.title}"
        )

        print(
            f"     page="
            f"{candidate.source_page_id}"
        )

        print(
            f"     rerank="
            f"{item.rerank_score:.6f}"
        )

        print(
            f"     reasons="
            f"{item.reasons}"
        )


def print_blocked(
    result,
) -> None:
    retrieval = (
        result.retrieval_result
    )

    if retrieval is None:
        return

    blocked = [
        trace
        for trace
        in retrieval.eligibility_trace
        if not trace.eligible
    ]

    if not blocked:
        return

    print(
        "Blocked by eligibility:"
    )

    for trace in blocked:
        print(
            f"  {trace.knowledge_key}"
            f" -> {trace.reasons}"
        )
        

def print_citation_validation(
    result,
) -> None:
    print()
    print(
        "CITATION VALIDATION"
    )
    print("-" * 100)

    validation = (
        result
        .citation_validation_result
    )

    if validation is None:
        print(
            "Citation validation skipped."
        )
        return

    print(
        "Valid: "
        f"{validation.valid}"
    )

    print(
        "Validated citations:"
    )

    if not validation.citations:
        print(
            "  NONE"
        )

    for citation in (
        validation.citations
    ):
        print(
            "  "
            f"{citation.knowledge_key}"
            " -> "
            f"{citation.candidate_id}"
        )

    print(
        "Validation issues:"
    )

    if not validation.issues:
        print(
            "  NONE"
        )

    for issue in (
        validation.issues
    ):
        print(
            "  "
            f"{issue.code}"
            " | candidate="
            f"{issue.candidate_id}"
            " | knowledge="
            f"{issue.knowledge_key}"
        )


def print_grounding_validation(
    result,
) -> None:
    print()
    print(
        "GROUNDING VALIDATION"
    )
    print("-" * 100)

    validation = (
        result
        .grounding_validation_result
    )

    if validation is None:
        print(
            "Grounding validation skipped."
        )
        return

    print(
        "Valid: "
        f"{validation.valid}"
    )

    print(
        "Skipped: "
        f"{validation.skipped}"
    )

    print(
        "Reason: "
        f"{validation.reason}"
    )

    print(
        "Claims:"
    )

    if not validation.claims:
        print(
            "  NONE"
        )

    for index, claim in enumerate(
        validation.claims,
        start=1,
    ):
        print(
            f"  #{index}"
        )

        print(
            "     claim="
            f"{claim.claim}"
        )

        print(
            "     supported="
            f"{claim.supported}"
        )

        print(
            "     evidence="
            f"{claim.supporting_candidate_ids}"
        )

        print(
            "     reason="
            f"{claim.reason}"
        )

    print(
        "Issues:"
    )

    if not validation.issues:
        print(
            "  NONE"
        )

    for issue in (
        validation.issues
    ):
        print(
            f"  {issue}"
        )

def print_generation(
    result,
) -> None:
    print()
    print(
        "GENERATION"
    )
    print("-" * 100)

    generation = (
        result.generation_result
    )

    if generation is None:
        print(
            "Generation skipped."
        )
        return

    payload = (
        generation.payload
    )

    print(
        "Answer:"
    )

    print(
        payload.answer
    )

    print(
        "Abstained: "
        f"{payload.abstained}"
    )

    print(
        "Abstain reason: "
        f"{payload.abstain_reason}"
    )

    print(
        "Needs handoff: "
        f"{payload.needs_handoff}"
    )

    print(
        "Evidence candidate IDs:"
    )

    for candidate_id in (
        generation
        .evidence_candidate_ids
    ):
        print(
            f"  {candidate_id}"
        )

    print(
        "Evidence knowledge keys:"
    )

    for knowledge_key in (
        generation
        .evidence_knowledge_keys
    ):
        print(
            f"  {knowledge_key}"
        )

    print(
        "Evidence pairs:"
    )

    if not generation.evidence_pairs:
        print(
            "  NONE"
        )

    for (
        candidate_id,
        knowledge_key,
    ) in sorted(
        generation.evidence_pairs
    ):
        print(
            "  "
            f"{knowledge_key}"
            " -> "
            f"{candidate_id}"
        )

    print(
        "Generated citations "
        "(sanitized):"
    )

    if not payload.citations:
        print(
            "  NONE"
        )

    for citation in (
        payload.citations
    ):
        print(
            "  "
            f"{citation.knowledge_key}"
            " -> "
            f"{citation.candidate_id}"
        )
    
    print_citation_validation(
        result
    )
    print_grounding_validation(
        result
    )


def print_result(
    *,
    label: str,
    result,
    state: ConversationState,
) -> None:
    print()
    print("=" * 100)
    print(label)
    print("=" * 100)

    turn = (
        result.resolved_turn
    )

    understanding = (
        result.understanding
    )

    print(
        "Original: "
        f"{turn.original_message}"
    )

    print(
        "Resolved: "
        f"{turn.resolved_query}"
    )

    print(
        "Language: "
        f"{turn.language}"
    )

    print()
    print(
        "UNDERSTANDING"
    )
    print("-" * 100)

    print(
        "Domain: "
        f"{understanding.domain.value}"
    )

    print(
        "Task: "
        f"{understanding.task.value}"
    )

    print(
        "Client specific: "
        f"{understanding.client_specific}"
    )

    print(
        "Topic relation: "
        f"{understanding.topic_relation.value}"
    )

    print(
        "Standalone query: "
        f"{understanding.standalone_query}"
    )

    print(
        "Retrieval concepts: "
        f"{understanding.retrieval_concepts}"
    )

    print(
        "Referenced countries: "
        f"{understanding.referenced_countries}"
    )

    print(
        "Referenced entities: "
        f"{understanding.referenced_entities}"
    )

    print(
        "Requires previous context: "
        f"{understanding.requires_previous_context}"
    )

    print(
        "Confidence: "
        f"{understanding.confidence}"
    )

    print(
        "Classification summary: "
        f"{understanding.classification_summary}"
    )

    print(
        "Knowledge hints: "
        f"{result.knowledge_hints}"
    )

    print()
    print(
        "CONTEXT"
    )
    print("-" * 100)

    print(
        "Platform: "
        f"{turn.platform}"
    )

    print(
        "Needs clarification: "
        f"{turn.needs_clarification}"
    )

    print(
        "Missing context: "
        f"{turn.missing_context_fields}"
    )

    print(
        "Clarification reason: "
        f"{turn.clarification_reason}"
    )

    print(
        "Awaiting context: "
        f"{state.awaiting_context_field}"
    )

    print(
        "Nationality: "
        f"{state.nationality}"
    )

    print(
        "Country of residence: "
        f"{state.country_of_residence}"
    )

    print(
        "KYC document country: "
        f"{state.kyc_document_country}"
    )

    print(
        "Referenced country: "
        f"{state.referenced_country}"
    )

    print(
        "Authenticated: "
        f"{state.is_authenticated}"
    )

    print(
        "Client ID present: "
        f"{bool(state.client_id)}"
    )

    if (
        result.context_requirement
    ):
        print(
            "Requirement matched keys: "
            f"{result.context_requirement.matched_knowledge_keys}"
        )

        print(
            "Requirement fields: "
            f"{result.context_requirement.required_fields}"
        )

        print(
            "Requirement missing fields: "
            f"{result.context_requirement.missing_fields}"
        )

        print(
            "Requirement reason: "
            f"{result.context_requirement.reason}"
        )

    print()

    print_candidates(
        result
    )

    print_blocked(
        result
    )

    print_generation(
        result
    )
    


async def run_platform_followup(
    service: ConversationService,
) -> None:
    state = ConversationState()

    first = (
        await service.handle_message(
            message=(
                "cTrader چجوری "
                "نصب کنم؟"
            ),
            state=state,
            language="FA",
        )
    )

    print_result(
        label=(
            "SCENARIO 1A - "
            "cTrader installation"
        ),
        result=first,
        state=state,
    )

    second = (
        await service.handle_message(
            message="MT5 چی؟",
            state=state,
            language="FA",
        )
    )

    print_result(
        label=(
            "SCENARIO 1B - "
            "MT5 follow-up"
        ),
        result=second,
        state=state,
    )


async def run_kyc_followup(
    service: ConversationService,
) -> None:
    state = ConversationState()

    first = (
        await service.handle_message(
            message=(
                "برای احراز هویت "
                "چه مدارکی لازمه؟"
            ),
            state=state,
            language="FA",
        )
    )

    print_result(
        label=(
            "SCENARIO 2A - "
            "KYC clarification"
        ),
        result=first,
        state=state,
    )

    if not (
        first.resolved_turn
        .needs_clarification
    ):
        print()
        print(
            "WARNING: KYC query did "
            "not request clarification."
        )
        return

    second = (
        await service.handle_message(
            message="افغانستان",
            state=state,
            language="FA",
        )
    )

    print_result(
        label=(
            "SCENARIO 2B - "
            "KYC Afghanistan"
        ),
        result=second,
        state=state,
    )


async def run_authenticated_case(
    service: ConversationService,
) -> None:
    state = ConversationState()

    result = (
        await service.handle_message(
            message=(
                "رمز حساب معاملاتی "
                "رو چجوری عوض کنم؟"
            ),
            state=state,
            language="FA",
            trusted_session=(
                TrustedSessionContext(
                    is_authenticated=True,
                    client_id=(
                        "TEST-CLIENT-001"
                    ),
                )
            ),
        )
    )

    print_result(
        label=(
            "SCENARIO 3 - "
            "authenticated client"
        ),
        result=result,
        state=state,
    )

async def main() -> None:
    configure_logging(
        level=settings.LOG_LEVEL
    )

    entity_registry = (
        EntityRegistry.from_file(
            settings.ENTITY_ALIASES_PATH
        )
    )

    resolver = (
        ConversationResolver(
            entity_registry=(
                entity_registry
            )
        )
    )

    structured_llm = (
        OpenAIStructuredLLM(
            api_key=(
                settings.OPENAI_API_KEY
            ),
            model=(
                settings.OPENAI_MODEL
            ),
        )
    )

    understanding_service = (
        ConversationUnderstandingService(
            llm=structured_llm
        )
    )

    generation_service = (
        AnswerGenerationService(
            llm=structured_llm,
            max_candidates=5,
            max_content_chars=4000,
        )
    )

    citation_validator = (
        CitationValidator()
    )

    grounding_service = (
        GroundingValidationService(
            llm=structured_llm,
            max_content_chars=4000,
        )
    )

    retrieval_runtime = (
        await build_pipeline()
    )

    service = (
        ConversationService(
            understanding_service=(
                understanding_service
            ),
            resolver=resolver,
            retrieval_pipeline=(
                retrieval_runtime.pipeline
            ),
            context_policy=(
                ContextPolicy()
            ),
            context_requirement_service=(
                retrieval_runtime
                .context_requirement_service
            ),
            entity_registry=(
                entity_registry
            ),
            generation_service=(
                generation_service
            ),
            citation_validator=(
                citation_validator
            ),
            grounding_service=(
                grounding_service
            ),
        )
    )

    try:
        await run_platform_followup(
            service
        )

        await run_kyc_followup(
            service
        )

        await run_authenticated_case(
            service
        )

    finally:
        await retrieval_runtime.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )
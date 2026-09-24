import json

import pytest

from app.conversation.resolver import (ConversationResolver,)
from app.conversation.service import (ConversationService,TrustedSessionContext,)
from app.conversation.state import (ConversationState,)
from app.knowledge.enums import (Audience,BotMode,FallbackPolicy,KnowledgeStatus,Language,MarketScope,SourceAuthority,)
from app.knowledge.schemas import (KnowledgeMetadata,)
from app.retrieval.hybrid import (HybridCandidate,)
from app.retrieval.pipeline import (EligibilityTrace,RetrievalPipelineResult,)
from app.retrieval.schemas import (KnowledgeCandidate,)
from app.runtime.context_policy import (ContextPolicy,)
from app.runtime.entity_registry import (EntityRegistry,)
from app.conversation.schemas import (ConversationDomain,ConversationTask,ConversationUnderstanding,TopicRelation,)

class FakeRetrievalPipeline:
    def __init__(
        self,
        result_factory,
    ):
        self.result_factory = (
            result_factory
        )
        self.calls = []

    async def retrieve(
        self,
        *,
        query,
        context,
        language=None,
        knowledge_hints=None,
    ):
        self.calls.append(
            {
                "query": query,
                "context": context,
                "language": language,
            }
        )

        return self.result_factory(
            query,
            context,
        )



class FakeUnderstandingService:
    async def understand(
        self,
        *,
        message,
        state,
    ):
        return ConversationUnderstanding(
            language=(
                state.language
                or "FA"
            ),
            domain=(
                ConversationDomain.GENERAL
            ),
            task=(
                ConversationTask.HOW_TO
            ),
            client_specific=False,
            topic_relation=(
                TopicRelation.NEW_TOPIC
            ),
            referenced_countries=[],
            referenced_entities=[],
            requires_previous_context=False,
            confidence=0.95,
        )

def empty_result(
    query: str,
):
    return RetrievalPipelineResult(
        query=query,
        bm25_candidates=[],
        vector_candidates=[],
        hybrid_candidates=[],
        eligible_candidates=[],
        final_candidates=[],
        eligibility_trace=[],
    )


def make_hybrid_candidate(
    knowledge_key: str,
) -> HybridCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key=knowledge_key,
        canonical_knowledge_key=(
            knowledge_key
        ),
        knowledge_type="HOW_TO",
        category="TEST",
        status=(
            KnowledgeStatus.APPROVED
        ),
        audience=Audience.BOTH,
        owner="Test",
        primary_source="Test",
        source_ids=["TEST"],
        source_authority=(
            SourceAuthority.APPROVED_DOCUMENT
        ),
        language=Language.FA,
        market_scope=(
            MarketScope.COUNTRY
        ),
        countries=["AF"],
        regions=[],
        required_context_fields=[
            "kyc_document_country"
        ],
        bot_mode=(
            BotMode.KNOWLEDGE
        ),
        fallback_policy=(
            FallbackPolicy.NONE
        ),
    )

    candidate = KnowledgeCandidate(
        candidate_id=(
            f"{knowledge_key}:chunk-1"
        ),
        title="Test knowledge",
        content="Test content",
        metadata=metadata,
        score=1.0,
        source_page_id="1",
    )

    return HybridCandidate(
        candidate=candidate,
        hybrid_score=0.04,
        bm25_rank=1,
        bm25_score=10.0,
        vector_rank=1,
        vector_score=0.8,
    )


def build_registry(
    tmp_path,
):
    config = {
        "entity_types": {
            "platform": {
                "fuzzy_enabled": True,
                "fuzzy_threshold": 0.70,
                "aliases": {
                    "ctrader": "cTrader",
                    "mt5": "MT5",
                    "metatrader": "MT5",
                },
            },
            "country": {
                "fuzzy_enabled": False,
                "fuzzy_threshold": 0.90,
                "aliases": {
                    "afghanistan": "AF",
                    "افغانستان": "AF",
                    "iran": "IR",
                    "ایران": "IR",
                },
            },
        },
        "context_field_entity_types": {
            "kyc_document_country":
                "country",
            "country_of_residence":
                "country",
            "nationality":
                "country",
            "referenced_country":
                "country",
        },
        "conversation": {
            "short_followup_max_tokens": 4,
            "correction_markers": [
                "نه منظورم",
                "i mean",
            ],
        },
    }

    path = (
        tmp_path
        / "entities.json"
    )

    path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return (
        EntityRegistry.from_file(
            path
        )
    )


def build_service(
    tmp_path,
    result_factory,
):
    registry = build_registry(
        tmp_path
    )

    resolver = (
        ConversationResolver(
            entity_registry=registry
        )
    )

    pipeline = (
        FakeRetrievalPipeline(
            result_factory
        )
    )

    service = ConversationService(
        understanding_service=(
            FakeUnderstandingService()
        ),
        resolver=resolver,
        retrieval_pipeline=pipeline,
        context_policy=ContextPolicy(),
    )

    return service, pipeline


@pytest.mark.asyncio
async def test_service_sends_resolved_query_to_pipeline(
    tmp_path,
):
    service, pipeline = (
        build_service(
            tmp_path,
            lambda query, context:
                empty_result(query),
        )
    )

    state = ConversationState()

    await service.handle_message(
        message="cTrader",
        state=state,
        language="FA",
    )

    assert len(
        pipeline.calls
    ) == 1

    assert (
        pipeline.calls[0]["query"]
        == "ctrader"
    )


@pytest.mark.asyncio
async def test_followup_changes_platform_using_conversation_context(
    tmp_path,
):
    service, pipeline = (
        build_service(
            tmp_path,
            lambda query, context:
                empty_result(query),
        )
    )

    state = ConversationState()

    await service.handle_message(
        message=(
            "cTrader چجوری نصب میشه؟"
        ),
        state=state,
        language="FA",
    )

    await service.handle_message(
        message="MT5 چی؟",
        state=state,
        language="FA",
    )

    assert len(
        pipeline.calls
    ) == 2

    second_query = (
        pipeline.calls[1]["query"]
    )

    assert "MT5" in second_query

    assert (
        "cTrader"
        not in second_query
    )


@pytest.mark.asyncio
async def test_missing_context_sets_pending_field(
    tmp_path,
):
    blocked_candidate = (
        make_hybrid_candidate(
            "KB-COUNTRY-001"
        )
    )

    def factory(
        query,
        context,
    ):
        return RetrievalPipelineResult(
            query=query,
            bm25_candidates=[],
            vector_candidates=[],
            hybrid_candidates=[
                blocked_candidate
            ],
            eligible_candidates=[],
            final_candidates=[],
            eligibility_trace=[
                EligibilityTrace(
                    knowledge_key=(
                        "KB-COUNTRY-001"
                    ),
                    eligible=False,
                    reasons=[
                        (
                            "MISSING_CONTEXT:"
                            "kyc_document_country"
                        ),
                        (
                            "COUNTRY_CONTEXT_"
                            "MISSING"
                        ),
                    ],
                )
            ],
        )

    service, _ = (
        build_service(
            tmp_path,
            factory,
        )
    )

    state = ConversationState()

    result = (
        await service.handle_message(
            message=(
                "مدارک احراز هویت"
            ),
            state=state,
            language="FA",
        )
    )

    assert (
        result.should_generate_answer
        is False
    )

    assert (
        result.resolved_turn
        .needs_clarification
        is True
    )

    assert (
        result.resolved_turn
        .missing_context_fields
        == [
            "kyc_document_country"
        ]
    )

    assert (
        state.awaiting_context_field
        == "kyc_document_country"
    )

    assert (
        result.clarification
        is not None
    )

    assert (
        result.clarification
        .field_name
        == "kyc_document_country"
    )


@pytest.mark.asyncio
async def test_pending_context_answer_is_stored(
    tmp_path,
):
    def factory(
        query,
        context,
    ):
        return empty_result(
            query
        )

    service, pipeline = (
        build_service(
            tmp_path,
            factory,
        )
    )

    state = ConversationState(
        language="FA",
        last_resolved_query=(
            "KYC document requirements"
        ),
        awaiting_context_field=(
            "kyc_document_country"
        ),
    )

    await service.handle_message(
        message="افغانستان",
        state=state,
        language="FA",
    )

    assert (
        state.kyc_document_country
        == "AF"
    )

    assert (
        state.awaiting_context_field
        is None
    )

    assert (
        pipeline.calls[-1]
        ["context"]
        .kyc_document_country
        == "AF"
    )


@pytest.mark.asyncio
async def test_trusted_session_sets_client_id(
    tmp_path,
):
    service, pipeline = (
        build_service(
            tmp_path,
            lambda query, context:
                empty_result(query),
        )
    )

    state = ConversationState()

    await service.handle_message(
        message="what is my balance",
        state=state,
        language="EN",
        trusted_session=(
            TrustedSessionContext(
                is_authenticated=True,
                client_id="CLIENT-123",
            )
        ),
    )

    assert (
        state.is_authenticated
        is True
    )

    assert (
        state.client_id
        == "CLIENT-123"
    )

    context = (
        pipeline.calls[-1][
            "context"
        ]
    )

    assert (
        context.is_authenticated
        is True
    )

    assert (
        context.client_id
        == "CLIENT-123"
    )

@pytest.mark.asyncio
async def test_authenticated_session_without_client_id_is_allowed(
    tmp_path,
):
    service, pipeline = (
        build_service(
            tmp_path,
            lambda query, context:
                empty_result(query),
        )
    )

    state = ConversationState()

    result = await service.handle_message(
        message="what is my balance",
        state=state,
        language="EN",
        trusted_session=(
            TrustedSessionContext(
                is_authenticated=True,
                client_id=None,
            )
        ),
    )

    assert (
        state.is_authenticated
        is True
    )

    assert (
        state.client_id
        is None
    )

    context = (
        pipeline.calls[-1][
            "context"
        ]
    )

    assert (
        context.is_authenticated
        is True
    )

    assert (
        context.client_id
        is None
    )

    assert (
        result.should_generate_answer
        is False
    )
@pytest.mark.asyncio
async def test_user_text_cannot_set_client_id(
    tmp_path,
):
    service, _ = (
        build_service(
            tmp_path,
            lambda query, context:
                empty_result(query),
        )
    )

    state = ConversationState()

    await service.handle_message(
        message=(
            "my client id is 12345"
        ),
        state=state,
        language="EN",
    )

    assert (
        state.is_authenticated
        is False
    )

    assert (
        state.client_id
        is None
    )

@pytest.mark.asyncio
async def test_fuzzy_greeting_slm_skips_retrieval(
    tmp_path,
):
    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
    )

    state = ConversationState()

    result = await service.handle_message(
        message="slm",
        state=state,
        language="EN",
    )

    assert pipeline.calls == []
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is False
    assert result.generation_result.payload.answer


@pytest.mark.asyncio
async def test_no_candidates_returns_safe_answer_instead_of_blank(
    tmp_path,
):
    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
    )

    state = ConversationState()

    result = await service.handle_message(
        message="unknown unsupported topic",
        state=state,
        language="EN",
    )

    assert len(pipeline.calls) == 1
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is True
    assert (
        result.generation_result.payload.abstain_reason
        == "NO_ELIGIBLE_KNOWLEDGE"
    )
    assert result.generation_result.payload.answer

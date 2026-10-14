import pytest

from app.conversation.readiness_service import (
    ConversationReadinessService,
    ReadinessCheckResponse,
)
from app.conversation.schemas import (
    ConversationDomain,
    ConversationTask,
    ConversationUnderstanding,
    TopicRelation,
)
from app.knowledge.enums import (
    Audience,
    BotMode,
    FallbackPolicy,
    KnowledgeStatus,
    Language,
    MarketScope,
    SourceAuthority,
)
from app.knowledge.schemas import KnowledgeMetadata
from app.retrieval.hybrid import HybridCandidate
from app.retrieval.reranker import RerankedCandidate
from app.retrieval.schemas import KnowledgeCandidate


class FakeStructuredLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def generate_structured(self, *, system_prompt, user_prompt, response_model):
        assert response_model is ReadinessCheckResponse

        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})

        return self.response


def make_candidate(
    title: str = "Chart delay troubleshooting",
    content: str = ("Chart delay is usually caused by " "using a VPN while connected."),
) -> RerankedCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key="KB-GEN-001-FA",
        canonical_knowledge_key="KB-GEN-001",
        knowledge_type="TROUBLESHOOTING",
        category="GENERAL",
        status=KnowledgeStatus.APPROVED,
        audience=Audience.BOTH,
        owner="Test",
        primary_source="Test",
        source_ids=["TEST"],
        source_authority=(SourceAuthority.APPROVED_DOCUMENT),
        language=Language.FA,
        market_scope=MarketScope.GLOBAL,
        countries=[],
        regions=[],
        required_context_fields=[],
        bot_mode=BotMode.KNOWLEDGE,
        fallback_policy=FallbackPolicy.NONE,
    )

    candidate = KnowledgeCandidate(
        candidate_id="KB-GEN-001-FA:chunk-1",
        title=title,
        content=content,
        metadata=metadata,
        score=1.0,
        source_page_id="1",
    )

    hybrid = HybridCandidate(
        candidate=candidate,
        hybrid_score=0.05,
        bm25_rank=1,
        bm25_score=10.0,
        vector_rank=1,
        vector_score=0.8,
    )

    return RerankedCandidate(candidate=hybrid, rerank_score=0.9, reasons=[])


def make_understanding() -> ConversationUnderstanding:
    return ConversationUnderstanding(
        language="fa",
        domain=ConversationDomain.GENERAL,
        task=ConversationTask.TROUBLESHOOTING,
        client_specific=False,
        topic_relation=TopicRelation.NEW_TOPIC,
        referenced_countries=[],
        referenced_entities=[],
        requires_previous_context=False,
        confidence=0.9,
    )


@pytest.mark.asyncio
async def test_returns_no_clarification_without_candidates():
    llm = FakeStructuredLLM(
        ReadinessCheckResponse(
            needs_clarification=True, question="should not be reached"
        )
    )

    service = ConversationReadinessService(llm=llm)

    result = await service.check(
        query="چرا نمودار دیر آپدیت میشه؟",
        language="fa",
        understanding=make_understanding(),
        candidates=[],
        conversation_context="",
    )

    assert result.needs_clarification is False
    assert llm.calls == []


@pytest.mark.asyncio
async def test_passes_through_a_real_clarifying_question():
    llm = FakeStructuredLLM(
        ReadinessCheckResponse(
            needs_clarification=True, question=(" آیا از فیلترشکن استفاده " "می‌کنید؟ ")
        )
    )

    service = ConversationReadinessService(llm=llm)

    result = await service.check(
        query="چرا نمودار دیر آپدیت میشه؟",
        language="fa",
        understanding=make_understanding(),
        candidates=[make_candidate()],
        conversation_context="",
    )

    assert result.needs_clarification is True
    assert result.question == "آیا از فیلترشکن استفاده می‌کنید؟"
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_empty_question_is_treated_as_no_clarification_needed():

    llm = FakeStructuredLLM(
        ReadinessCheckResponse(needs_clarification=True, question="   ")
    )

    service = ConversationReadinessService(llm=llm)

    result = await service.check(
        query="چرا نمودار دیر آپدیت میشه؟",
        language="fa",
        understanding=make_understanding(),
        candidates=[make_candidate()],
        conversation_context="",
    )

    assert result.needs_clarification is False


@pytest.mark.asyncio
async def test_triage_returns_no_clarification_without_near_candidates():
    llm = FakeStructuredLLM(
        ReadinessCheckResponse(
            needs_clarification=True, question="should not be reached"
        )
    )

    service = ConversationReadinessService(llm=llm)

    result = await service.check_before_abstaining(
        query="چرا نمودار دیر آپدیت میشه؟",
        language="fa",
        understanding=make_understanding(),
        near_candidates=[],
        conversation_context="",
    )

    assert result.needs_clarification is False
    assert llm.calls == []


@pytest.mark.asyncio
async def test_triage_passes_through_a_real_question():
    llm = FakeStructuredLLM(
        ReadinessCheckResponse(
            needs_clarification=True, question=" آیا از فیلترشکن استفاده می‌کنید؟ "
        )
    )

    service = ConversationReadinessService(llm=llm)

    hybrid = HybridCandidate(
        candidate=make_candidate().candidate.candidate,
        hybrid_score=0.02,
        bm25_rank=3,
        bm25_score=1.0,
        vector_rank=5,
        vector_score=0.3,
    )

    result = await service.check_before_abstaining(
        query="چرا نمودار دیر آپدیت میشه؟",
        language="fa",
        understanding=make_understanding(),
        near_candidates=[hybrid],
        conversation_context="",
    )

    assert result.needs_clarification is True
    assert result.question == "آیا از فیلترشکن استفاده می‌کنید؟"
    assert len(llm.calls) == 1
    # Must be clearly framed as unconfirmed/nearby content, not as a
    # source the assistant is about to cite.
    assert "NEARBY" in llm.calls[0]["user_prompt"]

import pytest

from app.conversation.schemas import (
    ConversationDomain,
    ConversationTask,
    ConversationUnderstanding,
)
from app.conversation.state import ConversationState
from app.knowledge.enums import (
    Audience,
    BotMode,
    FallbackPolicy,
    KnowledgeStatus,
    Language,
    MarketScope,
)
from app.knowledge.schemas import KnowledgeMetadata
from app.retrieval.schemas import KnowledgeCandidate
from app.runtime.context import (
    ContextTrust,
    ContextValue,
    IdentityState,
    RuntimeContext,
)
from app.runtime.pipeline import PipelineOutcomeState, RequestPipeline


class FakeUnderstandingService:
    def __init__(self, understanding: ConversationUnderstanding):
        self.understanding = understanding

    async def understand(
        self, *, message: str, state: ConversationState
    ) -> ConversationUnderstanding:
        return self.understanding


class FakeRetriever:
    def __init__(self, candidates: list[KnowledgeCandidate]):
        self.candidates = candidates

    async def retrieve(
        self, *, query: str, knowledge_hints: list[str], limit: int = 10
    ) -> list[KnowledgeCandidate]:
        return self.candidates[:limit]


def make_candidate(
    *, countries: list[str], required_context_fields: list[str]
) -> KnowledgeCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key="TEST-KYC",
        canonical_knowledge_key="TEST-KYC",
        knowledge_type="REQUIREMENTS",
        category="KYC",
        status=KnowledgeStatus.APPROVED,
        audience=Audience.BOTH,
        language=Language.FA,
        market_scope=MarketScope.COUNTRY,
        countries=countries,
        regions=[],
        required_context_fields=required_context_fields,
        bot_mode=BotMode.KNOWLEDGE,
        fallback_policy=FallbackPolicy.NONE,
    )

    return KnowledgeCandidate(
        candidate_id="TEST-KYC#01",
        title="Test KYC",
        content="Test content",
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_pipeline_uses_nationality_for_afghan_kyc():
    understanding = ConversationUnderstanding(
        language="fa",
        domain=ConversationDomain.KYC,
        task=ConversationTask.REQUIREMENTS,
        confidence=1.0,
    )

    candidate = make_candidate(
        countries=["AF"], required_context_fields=["nationality"]
    )

    pipeline = RequestPipeline(
        understanding_service=FakeUnderstandingService(understanding),
        retriever=FakeRetriever([candidate]),
    )

    context = RuntimeContext(
        identity_state=IdentityState.ANONYMOUS,
        nationality=ContextValue(
            value="AF", source="USER_DECLARED", trust=ContextTrust.DECLARED
        ),
        country_of_residence=ContextValue(
            value="IR", source="USER_DECLARED", trust=ContextTrust.DECLARED
        ),
    )

    result = await pipeline.process(
        message="برای KYC چه مدارکی لازم دارم؟",
        conversation_state=ConversationState(),
        runtime_context=context,
    )

    assert result.outcome.state == PipelineOutcomeState.READY_FOR_GENERATION

    assert result.outcome.allowed is True

    assert len(result.eligible_knowledge) == 1

    assert result.eligible_knowledge[0].metadata.knowledge_key == "TEST-KYC"


@pytest.mark.asyncio
async def test_pipeline_asks_for_nationality_when_missing():
    understanding = ConversationUnderstanding(
        language="fa",
        domain=ConversationDomain.KYC,
        task=ConversationTask.REQUIREMENTS,
        confidence=1.0,
    )

    candidate = make_candidate(
        countries=["AF"], required_context_fields=["nationality"]
    )

    pipeline = RequestPipeline(
        understanding_service=FakeUnderstandingService(understanding),
        retriever=FakeRetriever([candidate]),
    )

    context = RuntimeContext(
        identity_state=IdentityState.ANONYMOUS,
        country_of_residence=ContextValue(
            value="IR", source="USER_DECLARED", trust=ContextTrust.DECLARED
        ),
    )

    result = await pipeline.process(
        message="برای KYC چه مدارکی لازم دارم؟",
        conversation_state=ConversationState(),
        runtime_context=context,
    )

    assert result.outcome.state == PipelineOutcomeState.CONTEXT_REQUIRED

    assert result.outcome.allowed is False

    assert result.outcome.missing_context == ["nationality"]


@pytest.mark.asyncio
async def test_pipeline_blocks_market_mismatch():
    understanding = ConversationUnderstanding(
        language="fa",
        domain=ConversationDomain.KYC,
        task=ConversationTask.REQUIREMENTS,
        confidence=1.0,
    )

    candidate = make_candidate(
        countries=["IR"], required_context_fields=["nationality"]
    )

    pipeline = RequestPipeline(
        understanding_service=FakeUnderstandingService(understanding),
        retriever=FakeRetriever([candidate]),
    )

    context = RuntimeContext(
        identity_state=IdentityState.ANONYMOUS,
        nationality=ContextValue(
            value="AF", source="USER_DECLARED", trust=ContextTrust.DECLARED
        ),
    )

    result = await pipeline.process(
        message="برای KYC چه مدارکی لازم دارم؟",
        conversation_state=ConversationState(),
        runtime_context=context,
    )

    assert result.outcome.state == PipelineOutcomeState.NO_ELIGIBLE_KNOWLEDGE

    assert result.outcome.allowed is False

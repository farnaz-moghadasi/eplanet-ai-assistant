import pytest

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
from app.retrieval.schemas import KnowledgeCandidate
from app.retrieval.semantic_reranker import (
    SemanticKnowledgeReranker,
    SemanticRerankResponse,
)


class FakeStructuredLLM:
    async def generate_structured(self, *, system_prompt, user_prompt, response_model):
        assert response_model is SemanticRerankResponse

        return SemanticRerankResponse(
            items=[
                {
                    "candidate_id": "afghanistan",
                    "relevance_score": 0.97,
                    "rationale": ("Direct Afghanistan " "KYC requirements."),
                },
                {
                    "candidate_id": "general",
                    "relevance_score": 0.61,
                    "rationale": ("General KYC guidance."),
                },
            ]
        )


def make_candidate(
    *,
    candidate_id: str,
    knowledge_key: str,
    title: str,
    market_scope: MarketScope,
    countries: list[str],
) -> HybridCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key=knowledge_key,
        canonical_knowledge_key=(knowledge_key),
        knowledge_type="REQUIREMENTS",
        category="KYC",
        status=KnowledgeStatus.APPROVED,
        audience=Audience.BOTH,
        owner="Test",
        primary_source="Test",
        source_ids=["TEST"],
        source_authority=(SourceAuthority.APPROVED_DOCUMENT),
        language=Language.FA,
        market_scope=market_scope,
        countries=countries,
        regions=[],
        required_context_fields=[],
        bot_mode=BotMode.KNOWLEDGE,
        fallback_policy=(FallbackPolicy.NONE),
    )

    candidate = KnowledgeCandidate(
        candidate_id=candidate_id,
        title=title,
        content=title,
        metadata=metadata,
        score=1.0,
        source_page_id="1",
    )

    return HybridCandidate(
        candidate=candidate,
        hybrid_score=0.03,
        bm25_rank=1,
        bm25_score=1.0,
        vector_rank=1,
        vector_score=0.8,
    )


@pytest.mark.asyncio
async def test_country_specific_candidate_can_rank_above_global():
    general = make_candidate(
        candidate_id="general",
        knowledge_key=("KB-KYC-DOCS-GENERAL-FA"),
        title=("نکات عمومی مدارک احراز هویت"),
        market_scope=(MarketScope.GLOBAL),
        countries=[],
    )

    afghanistan = make_candidate(
        candidate_id="afghanistan",
        knowledge_key=("KB-KYC-DOCS-AF-FA"),
        title=("مدارک احراز هویت برای افغانستان"),
        market_scope=(MarketScope.COUNTRY),
        countries=["AF"],
    )

    reranker = SemanticKnowledgeReranker(llm=FakeStructuredLLM())

    result = await reranker.rerank(
        query=("چه مدارکی برای احراز هویت " "در افغانستان لازم است؟"),
        candidates=[general, afghanistan],
        language="FA",
        context={"kyc_document_country": "AF"},
        top_k=2,
    )

    assert len(result) == 2

    assert result[0].candidate.candidate.metadata.knowledge_key == "KB-KYC-DOCS-AF-FA"

    assert result[0].rerank_score == 0.97


@pytest.mark.asyncio
async def test_hybrid_score_breaks_semantic_tie():
    class TieLLM:
        async def generate_structured(
            self, *, system_prompt, user_prompt, response_model
        ):
            return SemanticRerankResponse(
                items=[
                    {
                        "candidate_id": "a",
                        "relevance_score": 0.8,
                        "rationale": "Relevant.",
                    },
                    {
                        "candidate_id": "b",
                        "relevance_score": 0.8,
                        "rationale": "Relevant.",
                    },
                ]
            )

    first = make_candidate(
        candidate_id="a",
        knowledge_key="KB-A",
        title="A",
        market_scope=(MarketScope.GLOBAL),
        countries=[],
    )

    second = make_candidate(
        candidate_id="b",
        knowledge_key="KB-B",
        title="B",
        market_scope=(MarketScope.GLOBAL),
        countries=[],
    )

    first.hybrid_score = 0.02
    second.hybrid_score = 0.04

    reranker = SemanticKnowledgeReranker(llm=TieLLM())

    result = await reranker.rerank(
        query="test", candidates=[first, second], language="EN", top_k=2
    )

    assert result[0].candidate.candidate.candidate_id == "b"

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
from app.retrieval.reranker import KnowledgeReranker
from app.retrieval.schemas import KnowledgeCandidate


def make_candidate(
    *,
    knowledge_key: str,
    title: str,
    content: str = "Test content",
    hybrid_score: float = 0.03,
) -> HybridCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key=knowledge_key,
        canonical_knowledge_key=knowledge_key.rsplit(
            "-",
            1,
        )[0],
        knowledge_type="HOW_TO",
        category="TEST",
        status=KnowledgeStatus.APPROVED,
        audience=Audience.BOTH,
        owner="Test",
        primary_source="Test Source",
        source_ids=["TEST-1"],
        source_authority=SourceAuthority.APPROVED_DOCUMENT,
        language=Language.EN,
        market_scope=MarketScope.GLOBAL,
        countries=[],
        regions=[],
        required_context_fields=[],
        bot_mode=BotMode.KNOWLEDGE,
        fallback_policy=FallbackPolicy.NONE,
    )

    candidate = KnowledgeCandidate(
        candidate_id=f"{knowledge_key}:chunk-1",
        title=title,
        content=content,
        metadata=metadata,
        score=1.0,
        source_page_id="1",
    )

    return HybridCandidate(
        candidate=candidate,
        hybrid_score=hybrid_score,
        bm25_rank=1,
        bm25_score=10.0,
        vector_rank=1,
        vector_score=0.7,
    )


def test_ctrader_query_boosts_ctrader_article():
    reranker = KnowledgeReranker()

    generic_platform = make_candidate(
        knowledge_key="KB-PLATFORM-001-EN",
        title="Choose a Trading Platform",
        hybrid_score=0.032,
    )

    ctrader_login = make_candidate(
        knowledge_key="KB-CTR-001-EN",
        title="Download and Log In to cTrader",
        hybrid_score=0.031,
    )

    results = reranker.rerank(
        query="How do I log in to cTrader?",
        candidates=[
            generic_platform,
            ctrader_login,
        ],
        limit=2,
    )

    assert (
        results[0]
        .candidate
        .candidate
        .metadata
        .knowledge_key
        == "KB-CTR-001-EN"
    )

    assert (
        "CTRADER_KEY_MATCH"
        in results[0].reasons
    )


def test_margin_query_prefers_margin_over_margin_call():
    reranker = KnowledgeReranker()

    margin_call = make_candidate(
        knowledge_key="KB-GEN-MARGIN-CALL-EN",
        title="Understand Margin Calls",
        hybrid_score=0.0325,
    )

    margin = make_candidate(
        knowledge_key="KB-GEN-MARGIN-EN",
        title="Understand Margin",
        hybrid_score=0.0325,
    )

    results = reranker.rerank(
        query="What is margin?",
        candidates=[
            margin_call,
            margin,
        ],
        limit=2,
    )

    assert (
        results[0]
        .candidate
        .candidate
        .metadata
        .knowledge_key
        == "KB-GEN-MARGIN-EN"
    )

    margin_call_result = next(
        item
        for item in results
        if (
            item.candidate
            .candidate
            .metadata
            .knowledge_key
            == "KB-GEN-MARGIN-CALL-EN"
        )
    )

    assert (
        "MARGIN_CALL_PENALTY"
        in margin_call_result.reasons
    )


def test_mt5_query_boosts_mt5_article():
    reranker = KnowledgeReranker()

    generic_article = make_candidate(
        knowledge_key="KB-PLATFORM-001-EN",
        title="Choose a Trading Platform",
        hybrid_score=0.032,
    )

    mt5_article = make_candidate(
        knowledge_key="KB-MT5-001-EN",
        title="Connect a Trading Account to MetaTrader 5",
        hybrid_score=0.031,
    )

    results = reranker.rerank(
        query="How do I connect my account to MT5?",
        candidates=[
            generic_article,
            mt5_article,
        ],
        limit=2,
    )

    assert (
        results[0]
        .candidate
        .candidate
        .metadata
        .knowledge_key
        == "KB-MT5-001-EN"
    )

    assert (
        "MT5_KEY_MATCH"
        in results[0].reasons
    )


def test_query_in_title_gets_boost():
    reranker = KnowledgeReranker()

    exact_title = make_candidate(
        knowledge_key="KB-TEST-001-EN",
        title="Reset Trading Account Password",
        hybrid_score=0.03,
    )

    other = make_candidate(
        knowledge_key="KB-TEST-002-EN",
        title="Trading Account Help",
        hybrid_score=0.03,
    )

    results = reranker.rerank(
        query="Reset Trading Account Password",
        candidates=[
            other,
            exact_title,
        ],
        limit=2,
    )

    assert (
        results[0]
        .candidate
        .candidate
        .metadata
        .knowledge_key
        == "KB-TEST-001-EN"
    )

    assert (
        "QUERY_IN_TITLE"
        in results[0].reasons
    )


def test_title_term_overlap_adds_boost():
    reranker = KnowledgeReranker()

    relevant = make_candidate(
        knowledge_key="KB-TEST-003-EN",
        title="Reset Client Portal Password",
        hybrid_score=0.03,
    )

    unrelated = make_candidate(
        knowledge_key="KB-TEST-004-EN",
        title="Deposit Methods",
        hybrid_score=0.03,
    )

    results = reranker.rerank(
        query="client portal password",
        candidates=[
            unrelated,
            relevant,
        ],
        limit=2,
    )

    assert (
        results[0]
        .candidate
        .candidate
        .metadata
        .knowledge_key
        == "KB-TEST-003-EN"
    )

    assert any(
        reason.startswith(
            "TITLE_TERM_OVERLAP:"
        )
        for reason in results[0].reasons
    )


def test_reranker_respects_limit():
    reranker = KnowledgeReranker()

    candidates = [
        make_candidate(
            knowledge_key=f"KB-TEST-{index:03d}-EN",
            title=f"Article {index}",
            hybrid_score=0.03,
        )
        for index in range(10)
    ]

    results = reranker.rerank(
        query="test",
        candidates=candidates,
        limit=3,
    )

    assert len(results) == 3
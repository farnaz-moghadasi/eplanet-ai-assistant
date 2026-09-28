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
from app.retrieval.eligibility import (
    EligibilityContext,
    KnowledgeEligibilityPolicy,
)
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import KnowledgeRetrievalPipeline
from app.retrieval.reranker import KnowledgeReranker
from app.retrieval.schemas import KnowledgeCandidate


class FakeRetriever:
    def __init__(
        self,
        candidates: list[KnowledgeCandidate],
    ):
        self.candidates = candidates

    async def retrieve(
        self,
        *,
        query: str,
        knowledge_hints: list[str],
        limit: int,
        **kwargs,
    ) -> list[KnowledgeCandidate]:
        return self.candidates[:limit]


def make_candidate(
    *,
    knowledge_key: str,
    title: str,
    score: float,
    language: Language = Language.EN,
    market_scope: MarketScope = MarketScope.GLOBAL,
    countries: list[str] | None = None,
    regions: list[str] | None = None,
    required_context_fields: list[str] | None = None,
    audience: Audience = Audience.BOTH,
    bot_mode: BotMode = BotMode.KNOWLEDGE,
) -> KnowledgeCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key=knowledge_key,
        canonical_knowledge_key=knowledge_key.rsplit(
            "-",
            1,
        )[0],
        knowledge_type="HOW_TO",
        category="TEST",
        status=KnowledgeStatus.APPROVED,
        audience=audience,
        owner="Test",
        primary_source="Test Source",
        source_ids=["TEST-1"],
        source_authority=SourceAuthority.APPROVED_DOCUMENT,
        language=language,
        market_scope=market_scope,
        countries=countries or [],
        regions=regions or [],
        required_context_fields=(
            required_context_fields or []
        ),
        bot_mode=bot_mode,
        fallback_policy=FallbackPolicy.NONE,
    )

    return KnowledgeCandidate(
        candidate_id=f"{knowledge_key}:chunk-1",
        title=title,
        content="Test content",
        metadata=metadata,
        score=score,
        source_page_id="1",
    )


class FakeSemanticReranker:
    def __init__(self, *, ranked, distinct_options):
        self.ranked = ranked
        self.distinct_options = distinct_options

    async def rerank(
        self,
        *,
        query,
        candidates,
        language=None,
        context=None,
        limit=5,
    ):
        from app.retrieval.reranker import RerankedCandidate

        by_key = {
            item.candidate.metadata.knowledge_key: item
            for item in candidates
        }

        results = [
            RerankedCandidate(
                candidate=by_key[key],
                rerank_score=score,
                reasons=[],
            )
            for key, score in self.ranked
            if key in by_key
        ]

        return results[:limit], self.distinct_options


def make_pipeline(
    *,
    bm25_candidates: list[KnowledgeCandidate],
    vector_candidates: list[KnowledgeCandidate],
    semantic_reranker=None,
    fast_path_enabled: bool = True,
) -> KnowledgeRetrievalPipeline:
    return KnowledgeRetrievalPipeline(
        bm25_retriever=FakeRetriever(
            bm25_candidates
        ),
        vector_retriever=FakeRetriever(
            vector_candidates
        ),
        hybrid_retriever=HybridRetriever(
            rrf_k=60
        ),
        eligibility_policy=KnowledgeEligibilityPolicy(),
        reranker=KnowledgeReranker(),
        semantic_reranker=semantic_reranker,
        bm25_top_k=20,
        vector_top_k=20,
        hybrid_top_k=20,
        final_top_k=5,
        fast_path_enabled=fast_path_enabled,
    )

@pytest.mark.asyncio
async def test_global_article_survives_pipeline():
    candidate = make_candidate(
        knowledge_key="KB-GLOBAL-001-EN",
        title="General Help",
        score=10.0,
    )

    pipeline = make_pipeline(
        bm25_candidates=[candidate],
        vector_candidates=[candidate],
    )

    result = await pipeline.retrieve(
        query="general help",
        context=EligibilityContext(),
        language="EN",
    )

    assert len(
        result.final_candidates
    ) == 1

    assert (
        result.final_candidates[0]
        .candidate
        .candidate
        .metadata
        .knowledge_key
        == "KB-GLOBAL-001-EN"
    )

@pytest.mark.asyncio
async def test_country_article_survives_when_context_matches():
    candidate = make_candidate(
        knowledge_key="KB-KYC-AF-001-EN",
        title="Afghan KYC Documents",
        score=10.0,
        market_scope=MarketScope.COUNTRY,
        countries=["AF"],
        required_context_fields=[
            "kyc_document_country"
        ],
    )

    pipeline = make_pipeline(
        bm25_candidates=[candidate],
        vector_candidates=[candidate],
    )

    result = await pipeline.retrieve(
        query="KYC documents",
        context=EligibilityContext(
            kyc_document_country="AF",
        ),
        language="EN",
    )

    assert len(
        result.final_candidates
    ) == 1

    assert (
        result.final_candidates[0]
        .candidate
        .candidate
        .metadata
        .knowledge_key
        == "KB-KYC-AF-001-EN"
    )

@pytest.mark.asyncio
async def test_country_article_is_removed_when_context_mismatches():
    candidate = make_candidate(
        knowledge_key="KB-KYC-AF-001-EN",
        title="Afghan KYC Documents",
        score=10.0,
        market_scope=MarketScope.COUNTRY,
        countries=["AF"],
        required_context_fields=[
            "kyc_document_country"
        ],
    )

    pipeline = make_pipeline(
        bm25_candidates=[candidate],
        vector_candidates=[candidate],
    )

    result = await pipeline.retrieve(
        query="KYC documents",
        context=EligibilityContext(
            kyc_document_country="IR",
        ),
        language="EN",
    )

    assert result.final_candidates == []
    assert result.eligible_candidates == []

    assert (
        result.eligibility_trace[0].eligible
        is False
    )

    assert (
        "COUNTRY_CONTEXT_MISMATCH"
        in result.eligibility_trace[0].reasons
    )

@pytest.mark.asyncio
async def test_country_article_is_removed_when_context_missing():
    candidate = make_candidate(
        knowledge_key="KB-KYC-AF-001-EN",
        title="Afghan KYC Documents",
        score=10.0,
        market_scope=MarketScope.COUNTRY,
        countries=["AF"],
        required_context_fields=[
            "kyc_document_country"
        ],
    )

    pipeline = make_pipeline(
        bm25_candidates=[candidate],
        vector_candidates=[candidate],
    )

    result = await pipeline.retrieve(
        query="KYC documents",
        context=EligibilityContext(),
        language="EN",
    )

    assert result.final_candidates == []

    reasons = (
        result
        .eligibility_trace[0]
        .reasons
    )

    assert (
        "MISSING_CONTEXT:kyc_document_country"
        in reasons
    )

    assert (
        "COUNTRY_CONTEXT_MISSING"
        in reasons
    )

@pytest.mark.asyncio
async def test_lookup_article_is_removed_for_anonymous_user():
    candidate = make_candidate(
        knowledge_key="KB-KYC-STATUS-EN",
        title="Check KYC Status",
        score=10.0,
        audience=Audience.CLIENT,
        bot_mode=BotMode.LOOKUP,
        required_context_fields=[
            "client_id"
        ],
    )

    pipeline = make_pipeline(
        bm25_candidates=[candidate],
        vector_candidates=[candidate],
    )

    result = await pipeline.retrieve(
        query="What is my KYC status?",
        context=EligibilityContext(
            is_authenticated=False,
            client_id=None,
        ),
        language="EN",
    )

    assert result.final_candidates == []

    reasons = (
        result
        .eligibility_trace[0]
        .reasons
    )

    assert (
        "CLIENT_AUTH_REQUIRED"
        in reasons
    )

    assert (
        "LOOKUP_AUTH_REQUIRED"
        in reasons
    )

@pytest.mark.asyncio
async def test_lookup_article_survives_for_authenticated_client():
    candidate = make_candidate(
        knowledge_key="KB-KYC-STATUS-EN",
        title="Check KYC Status",
        score=10.0,
        audience=Audience.CLIENT,
        bot_mode=BotMode.LOOKUP,
        required_context_fields=[
            "client_id"
        ],
    )

    pipeline = make_pipeline(
        bm25_candidates=[candidate],
        vector_candidates=[candidate],
    )

    result = await pipeline.retrieve(
        query="What is my KYC status?",
        context=EligibilityContext(
            is_authenticated=True,
            client_id="12345",
        ),
        language="EN",
    )

    assert len(
        result.final_candidates
    ) == 1

    assert (
        result.final_candidates[0]
        .candidate
        .candidate
        .metadata
        .knowledge_key
        == "KB-KYC-STATUS-EN"
    )

@pytest.mark.asyncio
async def test_pipeline_filters_wrong_language_from_bm25():
    fa_candidate = make_candidate(
        knowledge_key="KB-TEST-001-FA",
        title="راهنمای فارسی",
        score=10.0,
        language=Language.FA,
    )

    en_candidate = make_candidate(
        knowledge_key="KB-TEST-001-EN",
        title="English Guide",
        score=9.0,
        language=Language.EN,
    )

    pipeline = make_pipeline(
        bm25_candidates=[
            fa_candidate,
            en_candidate,
        ],
        vector_candidates=[
            en_candidate,
        ],
    )

    result = await pipeline.retrieve(
        query="English guide",
        context=EligibilityContext(),
        language="EN",
    )

    bm25_keys = [
        item.metadata.knowledge_key
        for item in result.bm25_candidates
    ]

    assert bm25_keys == [
        "KB-TEST-001-EN"
    ]

@pytest.mark.asyncio
async def test_pipeline_returns_eligibility_trace():
    allowed = make_candidate(
        knowledge_key="KB-GLOBAL-001-EN",
        title="Global Help",
        score=10.0,
    )

    blocked = make_candidate(
        knowledge_key="KB-KYC-IR-001-EN",
        title="Iran KYC",
        score=9.0,
        market_scope=MarketScope.COUNTRY,
        countries=["IR"],
        required_context_fields=[
            "kyc_document_country"
        ],
    )

    pipeline = make_pipeline(
        bm25_candidates=[
            allowed,
            blocked,
        ],
        vector_candidates=[
            allowed,
            blocked,
        ],
    )

    result = await pipeline.retrieve(
        query="help",
        context=EligibilityContext(
            kyc_document_country="AF",
        ),
        language="EN",
    )

    trace = {
        item.knowledge_key: item
        for item in result.eligibility_trace
    }

    assert (
        trace["KB-GLOBAL-001-EN"]
        .eligible
        is True
    )

    assert (
        trace["KB-KYC-IR-001-EN"]
        .eligible
        is False
    )

    assert (
        "COUNTRY_CONTEXT_MISMATCH"
        in trace[
            "KB-KYC-IR-001-EN"
        ].reasons
    )

@pytest.mark.asyncio
async def test_empty_query_returns_empty_result():
    pipeline = make_pipeline(
        bm25_candidates=[],
        vector_candidates=[],
    )

    result = await pipeline.retrieve(
        query="   ",
        context=EligibilityContext(),
        language="EN",
    )

    assert result.bm25_candidates == []
    assert result.vector_candidates == []
    assert result.hybrid_candidates == []
    assert result.eligible_candidates == []
    assert result.final_candidates == []
    assert result.eligibility_trace == []

@pytest.mark.asyncio
async def test_fast_path_used_when_vector_score_strong():
    candidate = make_candidate(
        knowledge_key="KB-GLOBAL-STRONG-EN",
        title="Strong Match",
        score=0.9,
    )

    pipeline = make_pipeline(
        bm25_candidates=[candidate],
        vector_candidates=[candidate],
    )

    result = await pipeline.retrieve(
        query="strong match",
        context=EligibilityContext(),
        language="EN",
    )

    assert (
        result.reranker_mode
        == "deterministic_fast_path"
    )

    assert len(
        result.final_candidates
    ) == 1

@pytest.mark.asyncio
async def test_fast_path_rejects_weak_vector_score():
    # Rank agreement alone (top-1 in both BM25 and vector search) is
    # not a relevance guarantee: an RRF-fused rank score lands in the
    # same narrow band whether the top candidate is genuinely relevant
    # or just the least-bad match for a poorly resolved query. A weak
    # raw vector score should stop the fast path from firing so the
    # answer does not get returned verbatim with no confidence check.
    candidate = make_candidate(
        knowledge_key="KB-GLOBAL-WEAK-EN",
        title="Weak Match",
        score=0.1,
    )

    pipeline = make_pipeline(
        bm25_candidates=[candidate],
        vector_candidates=[candidate],
    )

    result = await pipeline.retrieve(
        query="weak match",
        context=EligibilityContext(),
        language="EN",
    )

    assert (
        result.reranker_mode
        != "deterministic_fast_path"
    )

    assert result.final_candidates == []


@pytest.mark.asyncio
async def test_fast_path_keeps_the_approved_candidate_first():
    # The fast path approves the candidate BM25 and vector search both
    # rank first. The deterministic reranker's title-overlap bonus must
    # not be able to swap a different page into the top slot.
    creation = make_candidate(
        knowledge_key="KB-ACC-001-EN",
        title="Open a Trading Account",
        score=0.9,
    )

    conditions = make_candidate(
        knowledge_key="KB-ACC-003-EN",
        title="Demo Account and Trading Conditions",
        score=0.8,
    )

    pipeline = make_pipeline(
        bm25_candidates=[creation, conditions],
        vector_candidates=[creation, conditions],
    )

    result = await pipeline.retrieve(
        query="demo account",
        context=EligibilityContext(),
        language="EN",
    )

    assert (
        result.reranker_mode
        == "deterministic_fast_path"
    )

    top = result.final_candidates[0]

    assert (
        top.candidate.candidate.metadata.knowledge_key
        == "KB-ACC-001-EN"
    )


@pytest.mark.asyncio
async def test_distinct_options_propagate_through_retrieve():
    portal = make_candidate(
        knowledge_key="KB-ACCESS-001-FA",
        title="Recover client portal password",
        score=0.9,
    )

    mt5 = make_candidate(
        knowledge_key="KB-MT5-006-FA",
        title="Change trading password in MetaTrader 5",
        score=0.85,
    )

    options = [
        "client portal password",
        "MetaTrader 5 trading password",
    ]

    pipeline = make_pipeline(
        bm25_candidates=[portal, mt5],
        vector_candidates=[portal, mt5],
        semantic_reranker=FakeSemanticReranker(
            ranked=[
                ("KB-ACCESS-001-FA", 0.9),
                ("KB-MT5-006-FA", 0.85),
            ],
            distinct_options=options,
        ),
        fast_path_enabled=False,
    )

    result = await pipeline.retrieve(
        query="how do I change my account password",
        context=EligibilityContext(),
        language="EN",
    )

    assert result.reranker_mode == "semantic"
    assert result.distinct_options == options


@pytest.mark.asyncio
async def test_distinct_options_dropped_if_floor_removes_an_option():
    # A disambiguation is only meaningful while 2+ options are still
    # live candidates - if the relevance floor drops one, offering a
    # choice between "X" and a filtered-out ghost would be confusing.
    strong = make_candidate(
        knowledge_key="KB-ACCESS-001-FA",
        title="Recover client portal password",
        score=0.9,
    )

    weak = make_candidate(
        knowledge_key="KB-MT5-006-FA",
        title="Change trading password in MetaTrader 5",
        score=0.85,
    )

    pipeline = make_pipeline(
        bm25_candidates=[strong, weak],
        vector_candidates=[strong, weak],
        semantic_reranker=FakeSemanticReranker(
            ranked=[
                ("KB-ACCESS-001-FA", 0.9),
                # Below semantic_min_score (0.5 default): filtered out.
                ("KB-MT5-006-FA", 0.2),
            ],
            distinct_options=[
                "client portal password",
                "MetaTrader 5 trading password",
            ],
        ),
        fast_path_enabled=False,
    )

    result = await pipeline.retrieve(
        query="how do I change my account password",
        context=EligibilityContext(),
        language="EN",
    )

    assert result.distinct_options == []
    assert len(result.final_candidates) == 1

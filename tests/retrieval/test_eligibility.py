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
from app.retrieval.schemas import KnowledgeCandidate


def make_candidate(
    *,
    knowledge_key: str = "KB-TEST-001-EN",
    market_scope: MarketScope = MarketScope.GLOBAL,
    countries: list[str] | None = None,
    regions: list[str] | None = None,
    required_context_fields: list[str] | None = None,
    audience: Audience = Audience.BOTH,
    bot_mode: BotMode = BotMode.KNOWLEDGE,
) -> KnowledgeCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key=knowledge_key,
        canonical_knowledge_key="KB-TEST-001",
        knowledge_type="REQUIREMENTS",
        category="KYC",
        status=KnowledgeStatus.APPROVED,
        audience=audience,
        owner="Test",
        primary_source="Test Source",
        source_ids=["TEST-1"],
        source_authority=SourceAuthority.APPROVED_DOCUMENT,
        language=Language.EN,
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
        title="Test Article",
        content="Test content",
        metadata=metadata,
        score=1.0,
        source_page_id="1",
    )


def test_global_article_is_allowed():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        market_scope=MarketScope.GLOBAL,
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(),
    )

    assert result.eligible is True
    assert result.reasons == []


def test_country_article_is_allowed_when_context_matches():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        market_scope=MarketScope.COUNTRY,
        countries=["AF"],
        required_context_fields=[
            "kyc_document_country"
        ],
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(
            kyc_document_country="AF",
        ),
    )

    assert result.eligible is True
    assert result.reasons == []


def test_country_article_is_blocked_when_context_mismatches():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        market_scope=MarketScope.COUNTRY,
        countries=["AF"],
        required_context_fields=[
            "kyc_document_country"
        ],
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(
            kyc_document_country="IR",
        ),
    )

    assert result.eligible is False

    assert (
        "COUNTRY_CONTEXT_MISMATCH"
        in result.reasons
    )


def test_country_article_is_blocked_when_context_missing():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        market_scope=MarketScope.COUNTRY,
        countries=["AF"],
        required_context_fields=[
            "kyc_document_country"
        ],
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(),
    )

    assert result.eligible is False

    assert (
        "MISSING_CONTEXT:kyc_document_country"
        in result.reasons
    )

    assert (
        "COUNTRY_CONTEXT_MISSING"
        in result.reasons
    )


def test_country_article_without_context_basis_is_blocked():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        market_scope=MarketScope.COUNTRY,
        countries=["IR"],
        required_context_fields=[],
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(
            country_of_residence="IR",
        ),
    )

    assert result.eligible is False

    assert (
        "COUNTRY_CONTEXT_BASIS_MISSING"
        in result.reasons
    )


def test_lookup_article_is_allowed_for_authenticated_client():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        audience=Audience.CLIENT,
        bot_mode=BotMode.LOOKUP,
        required_context_fields=[
            "client_id"
        ],
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(
            is_authenticated=True,
            client_id="12345",
        ),
    )

    assert result.eligible is True
    assert result.reasons == []


def test_lookup_article_is_blocked_without_authentication():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        audience=Audience.CLIENT,
        bot_mode=BotMode.LOOKUP,
        required_context_fields=[
            "client_id"
        ],
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(
            is_authenticated=False,
            client_id=None,
        ),
    )

    assert result.eligible is False

    assert (
        "CLIENT_AUTH_REQUIRED"
        in result.reasons
    )

    assert (
        "MISSING_CONTEXT:client_id"
        in result.reasons
    )

    assert (
        "LOOKUP_AUTH_REQUIRED"
        in result.reasons
    )

    assert (
        "LOOKUP_CLIENT_ID_REQUIRED"
        in result.reasons
)


def test_internal_article_is_blocked():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        audience=Audience.INTERNAL,
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(
            is_authenticated=True,
            client_id="12345",
        ),
    )

    assert result.eligible is False

    assert (
        "AUDIENCE_INTERNAL"
        in result.reasons
    )


def test_region_article_is_allowed_when_region_matches():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        market_scope=MarketScope.REGION,
        regions=["MENA"],
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(
            resolved_regions={"MENA"},
        ),
    )

    assert result.eligible is True
    assert result.reasons == []


def test_region_article_is_blocked_when_region_mismatches():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        market_scope=MarketScope.REGION,
        regions=["MENA"],
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(
            resolved_regions={"EU"},
        ),
    )

    assert result.eligible is False

    assert (
        "REGION_CONTEXT_MISMATCH"
        in result.reasons
    )
def test_client_knowledge_requires_authentication():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        audience=Audience.CLIENT,
        bot_mode=BotMode.KNOWLEDGE,
        required_context_fields=[],
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(
            is_authenticated=False,
            client_id=None,
        ),
    )

    assert result.eligible is False

    assert (
        "CLIENT_AUTH_REQUIRED"
        in result.reasons
    )


def test_client_knowledge_does_not_require_client_id():
    policy = KnowledgeEligibilityPolicy()

    candidate = make_candidate(
        audience=Audience.CLIENT,
        bot_mode=BotMode.KNOWLEDGE,
        required_context_fields=[],
    )

    result = policy.evaluate(
        candidate=candidate,
        context=EligibilityContext(
            is_authenticated=True,
            client_id=None,
        ),
    )

    assert result.eligible is True
    assert result.reasons == []


def test_filter_returns_only_eligible_candidates():
    policy = KnowledgeEligibilityPolicy()

    global_candidate = make_candidate(
        knowledge_key="KB-GLOBAL-001-EN",
        market_scope=MarketScope.GLOBAL,
    )

    af_candidate = make_candidate(
        knowledge_key="KB-KYC-AF-001-EN",
        market_scope=MarketScope.COUNTRY,
        countries=["AF"],
        required_context_fields=[
            "kyc_document_country"
        ],
    )

    ir_candidate = make_candidate(
        knowledge_key="KB-KYC-IR-001-EN",
        market_scope=MarketScope.COUNTRY,
        countries=["IR"],
        required_context_fields=[
            "kyc_document_country"
        ],
    )

    results = policy.filter(
        candidates=[
            global_candidate,
            af_candidate,
            ir_candidate,
        ],
        context=EligibilityContext(
            kyc_document_country="AF",
        ),
    )

    keys = [
        item.metadata.knowledge_key
        for item in results
    ]

    assert keys == [
        "KB-GLOBAL-001-EN",
        "KB-KYC-AF-001-EN",
    ]
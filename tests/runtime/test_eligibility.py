from app.knowledge.enums import (
    Audience,
    BotMode,
    FallbackPolicy,
    KnowledgeStatus,
    Language,
    MarketScope,
)
from app.knowledge.schemas import KnowledgeMetadata
from app.runtime.context import (
    ContextTrust,
    ContextValue,
    IdentityState,
    RuntimeContext,
)
from app.runtime.eligibility import EligibilityState, KnowledgeEligibilityService


def make_metadata(
    *,
    audience=Audience.BOTH,
    bot_mode=BotMode.KNOWLEDGE,
    market_scope=MarketScope.GLOBAL,
    countries=None,
    regions=None,
    required_context_fields=None,
):
    return KnowledgeMetadata(
        knowledge_key="TEST-001",
        canonical_knowledge_key="TEST-001",
        knowledge_type="REQUIREMENTS",
        category="KYC",
        status=KnowledgeStatus.APPROVED,
        audience=audience,
        language=Language.EN,
        market_scope=market_scope,
        countries=countries or [],
        regions=regions or [],
        required_context_fields=required_context_fields or [],
        bot_mode=bot_mode,
        fallback_policy=FallbackPolicy.NONE,
    )


def test_afghan_nationality_in_iran_matches_afghan_kyc_article():
    metadata = make_metadata(
        market_scope=MarketScope.COUNTRY,
        countries=["AF"],
        required_context_fields=["nationality"],
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

    result = KnowledgeEligibilityService().evaluate(metadata, context)

    assert result.state == EligibilityState.ELIGIBLE
    assert result.eligible is True
    assert result.reason == "COUNTRY_MARKET_MATCH"


def test_afghan_nationality_in_iran_matches_iran_residence_article():
    metadata = make_metadata(
        market_scope=MarketScope.COUNTRY,
        countries=["IR"],
        required_context_fields=["country_of_residence"],
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

    result = KnowledgeEligibilityService().evaluate(metadata, context)

    assert result.state == EligibilityState.ELIGIBLE
    assert result.eligible is True


def test_country_article_asks_for_exact_missing_context():
    metadata = make_metadata(
        market_scope=MarketScope.COUNTRY,
        countries=["IR"],
        required_context_fields=["country_of_residence"],
    )

    context = RuntimeContext(
        identity_state=IdentityState.ANONYMOUS,
        nationality=ContextValue(
            value="AF", source="USER_DECLARED", trust=ContextTrust.DECLARED
        ),
    )

    result = KnowledgeEligibilityService().evaluate(metadata, context)

    assert result.state == EligibilityState.CLARIFICATION_REQUIRED
    assert result.eligible is False
    assert result.missing_context == ["country_of_residence"]


def test_country_market_mismatch_is_blocked():
    metadata = make_metadata(
        market_scope=MarketScope.COUNTRY,
        countries=["IR"],
        required_context_fields=["nationality"],
    )

    context = RuntimeContext(
        identity_state=IdentityState.ANONYMOUS,
        nationality=ContextValue(
            value="AF", source="USER_DECLARED", trust=ContextTrust.DECLARED
        ),
    )

    result = KnowledgeEligibilityService().evaluate(metadata, context)

    assert result.state == EligibilityState.BLOCKED
    assert result.eligible is False
    assert result.reason == "MARKET_MISMATCH"


def test_anonymous_user_cannot_access_client_audience():
    metadata = make_metadata(audience=Audience.CLIENT)

    context = RuntimeContext(identity_state=IdentityState.ANONYMOUS)

    result = KnowledgeEligibilityService().evaluate(metadata, context)

    assert result.state == EligibilityState.AUTHENTICATION_REQUIRED
    assert result.eligible is False


def test_lookup_rejects_declared_client_id():
    metadata = make_metadata(
        audience=Audience.CLIENT,
        bot_mode=BotMode.LOOKUP,
        required_context_fields=["client_id"],
    )

    context = RuntimeContext(
        identity_state=IdentityState.AUTHENTICATED,
        client_id=ContextValue(
            value="12345", source="USER_INPUT", trust=ContextTrust.DECLARED
        ),
    )

    result = KnowledgeEligibilityService().evaluate(metadata, context)

    assert result.state == EligibilityState.AUTHENTICATION_REQUIRED
    assert result.eligible is False


def test_lookup_accepts_verified_client_identity():
    metadata = make_metadata(
        audience=Audience.CLIENT,
        bot_mode=BotMode.LOOKUP,
        required_context_fields=["client_id"],
    )

    context = RuntimeContext(
        identity_state=IdentityState.AUTHENTICATED,
        client_id=ContextValue(
            value="12345", source="CLIENT_PORTAL", trust=ContextTrust.VERIFIED
        ),
    )

    result = KnowledgeEligibilityService().evaluate(metadata, context)

    assert result.state == EligibilityState.LOOKUP_REQUIRED
    assert result.eligible is True

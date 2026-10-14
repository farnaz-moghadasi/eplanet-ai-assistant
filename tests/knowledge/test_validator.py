from datetime import date, timedelta

from app.knowledge.enums import (
    Audience,
    BotMode,
    FallbackPolicy,
    KnowledgeStatus,
    Language,
    MarketScope,
)
from app.knowledge.schemas import KnowledgeMetadata
from app.knowledge.validator import KnowledgeValidator


def build_metadata(**overrides) -> KnowledgeMetadata:
    data = {
        "knowledge_key": "KB-GEN-001-EN",
        "canonical_knowledge_key": "KB-GEN-001",
        "knowledge_type": "FAQ",
        "category": "GENERAL",
        "status": KnowledgeStatus.APPROVED,
        "audience": Audience.BOTH,
        "language": Language.EN,
        "market_scope": MarketScope.GLOBAL,
        "countries": [],
        "regions": [],
        "required_context_fields": [],
        "bot_mode": BotMode.KNOWLEDGE,
        "fallback_policy": FallbackPolicy.NONE,
    }

    data.update(overrides)

    return KnowledgeMetadata(**data)


def test_valid_global_knowledge_is_ai_ready():
    validator = KnowledgeValidator()

    result = validator.validate(build_metadata())

    assert result.ai_ready is True
    assert result.reasons == []


def test_non_approved_content_is_rejected():
    validator = KnowledgeValidator()

    result = validator.validate(build_metadata(status=KnowledgeStatus.DRAFT))

    assert result.ai_ready is False
    assert "STATUS_NOT_APPROVED" in result.reasons


def test_unspecified_market_is_rejected():
    validator = KnowledgeValidator()

    result = validator.validate(build_metadata(market_scope=MarketScope.UNSPECIFIED))

    assert result.ai_ready is False
    assert "MARKET_SCOPE_UNSPECIFIED" in result.reasons


def test_country_scope_requires_country():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(market_scope=MarketScope.COUNTRY, countries=[])
    )

    assert result.ai_ready is False
    assert "COUNTRIES_REQUIRED" in result.reasons


def test_country_scope_is_allowed_without_context_field_for_now():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(
            market_scope=MarketScope.COUNTRY,
            countries=["IR"],
            required_context_fields=[],
        )
    )

    assert result.ai_ready is True


def test_region_scope_requires_region():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(market_scope=MarketScope.REGION, regions=[])
    )

    assert result.ai_ready is False
    assert "REGIONS_REQUIRED" in result.reasons


def test_internal_content_is_not_client_facing_ai_ready():
    validator = KnowledgeValidator()

    result = validator.validate(build_metadata(audience=Audience.INTERNAL))

    assert result.ai_ready is False
    assert "AUDIENCE_INTERNAL" in result.reasons


def test_lookup_requires_client_audience():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(
            audience=Audience.BOTH,
            bot_mode=BotMode.LOOKUP,
            required_context_fields=["client_id"],
        )
    )

    assert result.ai_ready is False
    assert "LOOKUP_REQUIRES_CLIENT_AUDIENCE" in result.reasons


def test_lookup_requires_client_id():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(
            audience=Audience.CLIENT,
            bot_mode=BotMode.LOOKUP,
            required_context_fields=[],
        )
    )

    assert result.ai_ready is False
    assert "LOOKUP_REQUIRES_CLIENT_ID" in result.reasons


def test_valid_client_lookup_is_ai_ready():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(
            knowledge_key="KB-KYC-003-EN",
            canonical_knowledge_key="KB-KYC-003",
            knowledge_type="TROUBLESHOOTING",
            category="KYC",
            audience=Audience.CLIENT,
            bot_mode=BotMode.LOOKUP,
            required_context_fields=["client_id"],
        )
    )

    assert result.ai_ready is True
    assert result.reasons == []


def test_supported_market_context_field_is_allowed():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(
            market_scope=MarketScope.COUNTRY,
            countries=["IR"],
            required_context_fields=["country_of_residence"],
        )
    )

    assert result.ai_ready is True


def test_kyc_document_country_context_is_supported():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(
            market_scope=MarketScope.COUNTRY,
            countries=["AF"],
            required_context_fields=["kyc_document_country"],
        )
    )

    assert result.ai_ready is True


def test_unknown_context_field_is_rejected():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(required_context_fields=["platform and device"])
    )

    assert result.ai_ready is False

    assert "UNSUPPORTED_REQUIRED_CONTEXT_FIELD:platform and device" in result.reasons


def test_future_content_is_not_yet_ai_ready():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(effective_from=date.today() + timedelta(days=1))
    )

    assert result.ai_ready is False
    assert "KNOWLEDGE_NOT_EFFECTIVE_YET" in result.reasons


def test_expired_content_is_rejected():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(valid_until=date.today() - timedelta(days=1))
    )

    assert result.ai_ready is False
    assert "KNOWLEDGE_EXPIRED" in result.reasons


def test_invalid_validity_range_is_rejected():
    validator = KnowledgeValidator()

    result = validator.validate(
        build_metadata(effective_from=date(2026, 10, 1), valid_until=date(2026, 9, 1))
    )

    assert result.ai_ready is False
    assert "INVALID_VALIDITY_RANGE" in result.reasons


def test_not_applicable_bot_mode_is_rejected():
    validator = KnowledgeValidator()

    result = validator.validate(build_metadata(bot_mode=BotMode.NOT_APPLICABLE))

    assert result.ai_ready is False
    assert "BOT_MODE_NOT_AI_ACCESSIBLE" in result.reasons

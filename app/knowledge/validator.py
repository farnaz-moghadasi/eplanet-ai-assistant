from datetime import date

from app.knowledge.enums import (
    Audience,
    BotMode,
    KnowledgeStatus,
    MarketScope,
)
from app.knowledge.schemas import (
    KnowledgeMetadata,
    KnowledgeValidationResult,
)


ALLOWED_REQUIRED_CONTEXT_FIELDS = {
    "client_id",
    "nationality",
    "country_of_residence",
    "kyc_document_country",
    "referenced_country",
}


class KnowledgeValidator:
    def validate(
        self,
        metadata: KnowledgeMetadata,
    ) -> KnowledgeValidationResult:
        reasons: list[str] = []

        self._validate_required_metadata(metadata, reasons)
        self._validate_status(metadata, reasons)
        self._validate_audience(metadata, reasons)
        self._validate_market(metadata, reasons)
        self._validate_context_fields(metadata, reasons)
        self._validate_bot_mode(metadata, reasons)
        self._validate_validity(metadata, reasons)

        return KnowledgeValidationResult(
            ai_ready=len(reasons) == 0,
            reasons=reasons,
        )

    def _validate_required_metadata(
        self,
        metadata: KnowledgeMetadata,
        reasons: list[str],
    ) -> None:
        if not metadata.knowledge_key.strip():
            reasons.append("KNOWLEDGE_KEY_MISSING")

        if not metadata.canonical_knowledge_key.strip():
            reasons.append("CANONICAL_KNOWLEDGE_KEY_MISSING")

        if not metadata.knowledge_type.strip():
            reasons.append("KNOWLEDGE_TYPE_MISSING")

        if not metadata.category.strip():
            reasons.append("CATEGORY_MISSING")

    def _validate_status(
        self,
        metadata: KnowledgeMetadata,
        reasons: list[str],
    ) -> None:
        if metadata.status != KnowledgeStatus.APPROVED:
            reasons.append("STATUS_NOT_APPROVED")

    def _validate_audience(
        self,
        metadata: KnowledgeMetadata,
        reasons: list[str],
    ) -> None:
        if metadata.audience == Audience.INTERNAL:
            reasons.append("AUDIENCE_INTERNAL")

    def _validate_market(
        self,
        metadata: KnowledgeMetadata,
        reasons: list[str],
    ) -> None:
        if metadata.market_scope == MarketScope.UNSPECIFIED:
            reasons.append("MARKET_SCOPE_UNSPECIFIED")
            return

        if (
            metadata.market_scope == MarketScope.COUNTRY
            and not metadata.countries
        ):
            reasons.append("COUNTRIES_REQUIRED")

        if (
            metadata.market_scope == MarketScope.REGION
            and not metadata.regions
        ):
            reasons.append("REGIONS_REQUIRED")

    def _validate_context_fields(
        self,
        metadata: KnowledgeMetadata,
        reasons: list[str],
    ) -> None:
        normalized_fields = {
            field.strip().lower()
            for field in metadata.required_context_fields
            if field.strip()
        }

        unsupported_fields = sorted(
            normalized_fields - ALLOWED_REQUIRED_CONTEXT_FIELDS
        )

        for field in unsupported_fields:
            reasons.append(
                f"UNSUPPORTED_REQUIRED_CONTEXT_FIELD:{field}"
            )

        if metadata.bot_mode == BotMode.LOOKUP:
            if metadata.audience != Audience.CLIENT:
                reasons.append("LOOKUP_REQUIRES_CLIENT_AUDIENCE")

            if "client_id" not in normalized_fields:
                reasons.append("LOOKUP_REQUIRES_CLIENT_ID")

    def _validate_bot_mode(
        self,
        metadata: KnowledgeMetadata,
        reasons: list[str],
    ) -> None:
        if metadata.bot_mode == BotMode.NOT_APPLICABLE:
            reasons.append("BOT_MODE_NOT_AI_ACCESSIBLE")

    def _validate_validity(
        self,
        metadata: KnowledgeMetadata,
        reasons: list[str],
    ) -> None:
        today = date.today()

        if (
            metadata.effective_from
            and metadata.effective_from > today
        ):
            reasons.append("KNOWLEDGE_NOT_EFFECTIVE_YET")

        if (
            metadata.valid_until
            and metadata.valid_until < today
        ):
            reasons.append("KNOWLEDGE_EXPIRED")

        if (
            metadata.effective_from
            and metadata.valid_until
            and metadata.effective_from > metadata.valid_until
        ):
            reasons.append("INVALID_VALIDITY_RANGE")
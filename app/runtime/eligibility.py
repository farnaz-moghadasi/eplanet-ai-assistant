from enum import Enum

from pydantic import BaseModel, Field

from app.knowledge.enums import Audience, BotMode, MarketScope
from app.knowledge.schemas import KnowledgeMetadata
from app.runtime.context import (
    IdentityState,
    RuntimeContext,
    has_verified_client_identity,
)
from app.runtime.market_access import MarketAccessResolver


class EligibilityState(str, Enum):
    ELIGIBLE = "ELIGIBLE"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"
    LOOKUP_REQUIRED = "LOOKUP_REQUIRED"
    HANDOFF_REQUIRED = "HANDOFF_REQUIRED"
    BLOCKED = "BLOCKED"


class EligibilityResult(BaseModel):
    state: EligibilityState
    eligible: bool
    reason: str | None = None
    missing_context: list[str] = Field(default_factory=list)


class KnowledgeEligibilityService:
    def __init__(self):
        self.market_resolver = MarketAccessResolver()

    def evaluate(
        self, metadata: KnowledgeMetadata, context: RuntimeContext
    ) -> EligibilityResult:

        #  Internal content is never customer-facing.
        if metadata.audience == Audience.INTERNAL:
            return EligibilityResult(
                state=EligibilityState.BLOCKED,
                eligible=False,
                reason="INTERNAL_AUDIENCE",
            )

        #  Audience gate.
        if (
            context.identity_state == IdentityState.ANONYMOUS
            and metadata.audience == Audience.CLIENT
        ):
            return EligibilityResult(
                state=EligibilityState.AUTHENTICATION_REQUIRED,
                eligible=False,
                reason="CLIENT_AUDIENCE_REQUIRES_AUTHENTICATION",
            )

        #  Bot mode gate.
        if metadata.bot_mode == BotMode.NOT_APPLICABLE:
            return EligibilityResult(
                state=EligibilityState.BLOCKED,
                eligible=False,
                reason="BOT_MODE_NOT_APPLICABLE",
            )

        if metadata.bot_mode == BotMode.HUMAN:
            return EligibilityResult(
                state=EligibilityState.HANDOFF_REQUIRED,
                eligible=False,
                reason="HUMAN_HANDOFF_REQUIRED",
            )

        if metadata.bot_mode == BotMode.GUIDED:
            return EligibilityResult(
                state=EligibilityState.HANDOFF_REQUIRED,
                eligible=False,
                reason="GUIDED_FLOW_NOT_SUPPORTED",
            )

        #  Required context gate.
        missing_context: list[str] = []

        for field_name in metadata.required_context_fields:
            if field_name == "client_id":
                if not has_verified_client_identity(context):
                    missing_context.append("client_id")
                continue

            context_value = getattr(context, field_name, None)

            if context_value is None or not context_value.value:
                missing_context.append(field_name)

        if missing_context:
            if "client_id" in missing_context:
                return EligibilityResult(
                    state=EligibilityState.AUTHENTICATION_REQUIRED,
                    eligible=False,
                    reason="TRUSTED_CLIENT_ID_REQUIRED",
                    missing_context=missing_context,
                )

            return EligibilityResult(
                state=EligibilityState.CLARIFICATION_REQUIRED,
                eligible=False,
                reason="REQUIRED_CONTEXT_MISSING",
                missing_context=missing_context,
            )

        #  LOOKUP requires a verified portal identity.
        if metadata.bot_mode == BotMode.LOOKUP:
            if not has_verified_client_identity(context):
                return EligibilityResult(
                    state=EligibilityState.AUTHENTICATION_REQUIRED,
                    eligible=False,
                    reason="VERIFIED_CLIENT_IDENTITY_REQUIRED",
                    missing_context=["client_id"],
                )

            return EligibilityResult(
                state=EligibilityState.LOOKUP_REQUIRED,
                eligible=True,
                reason="TRUSTED_LOOKUP_REQUIRED",
            )

        # Global content needs no market matching.
        if metadata.market_scope == MarketScope.GLOBAL:
            return EligibilityResult(
                state=EligibilityState.ELIGIBLE, eligible=True, reason="GLOBAL_MARKET"
            )

        #  Country-scoped content.
        if metadata.market_scope == MarketScope.COUNTRY:
            market_fields = self.market_resolver.get_market_context_fields(
                metadata.required_context_fields
            )

            # Bad metadata should not be repaired by runtime guesses.
            if not market_fields:
                return EligibilityResult(
                    state=EligibilityState.BLOCKED,
                    eligible=False,
                    reason="COUNTRY_MARKET_CONTEXT_FIELD_MISSING",
                )

            # Current schema has one countries list, so it cannot safely
            # encode separate nationality and residence applicability.
            if len(market_fields) > 1:
                return EligibilityResult(
                    state=EligibilityState.BLOCKED,
                    eligible=False,
                    reason="AMBIGUOUS_COUNTRY_MARKET_CONTEXT",
                )

            field_name = market_fields[0]

            resolution = self.market_resolver.resolve_for_field(
                context=context, field_name=field_name
            )

            if not resolution.resolved_market:
                return EligibilityResult(
                    state=EligibilityState.CLARIFICATION_REQUIRED,
                    eligible=False,
                    reason="MARKET_CONTEXT_MISSING",
                    missing_context=[field_name],
                )

            applicable_countries = {
                country.strip().upper()
                for country in metadata.countries
                if country.strip()
            }

            if resolution.resolved_market not in applicable_countries:
                return EligibilityResult(
                    state=EligibilityState.BLOCKED,
                    eligible=False,
                    reason="MARKET_MISMATCH",
                )

            return EligibilityResult(
                state=EligibilityState.ELIGIBLE,
                eligible=True,
                reason="COUNTRY_MARKET_MATCH",
            )

        #  Region resolution is deliberately not implemented yet.
        if metadata.market_scope == MarketScope.REGION:
            return EligibilityResult(
                state=EligibilityState.BLOCKED,
                eligible=False,
                reason="REGION_RESOLUTION_NOT_IMPLEMENTED",
            )

        return EligibilityResult(
            state=EligibilityState.BLOCKED,
            eligible=False,
            reason="UNSUPPORTED_MARKET_SCOPE",
        )

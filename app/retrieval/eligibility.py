from dataclasses import dataclass, field

from app.retrieval.schemas import KnowledgeCandidate


MARKET_CONTEXT_FIELDS = {
    "nationality",
    "country_of_residence",
    "kyc_document_country",
    "referenced_country",
}


@dataclass
class EligibilityContext:
    language: str | None = None

    is_authenticated: bool = False
    client_id: str | None = None

    nationality: str | None = None
    country_of_residence: str | None = None
    kyc_document_country: str | None = None
    referenced_country: str | None = None

    resolved_regions: set[str] = field(
        default_factory=set
    )

    def get(
        self,
        field_name: str,
    ) -> str | None:
        value = getattr(
            self,
            field_name,
            None,
        )

        if value is None:
            return None

        value = str(value).strip()

        return value or None


@dataclass
class EligibilityResult:
    eligible: bool
    reasons: list[str]


class KnowledgeEligibilityPolicy:
    def evaluate(
        self,
        *,
        candidate: KnowledgeCandidate,
        context: EligibilityContext,
    ) -> EligibilityResult:
        metadata = candidate.metadata
        reasons: list[str] = []

        self._check_audience(
            metadata=metadata,
            context=context,
            reasons=reasons,
        )

        self._check_required_context(
            metadata=metadata,
            context=context,
            reasons=reasons,
        )

        self._check_bot_mode(
            metadata=metadata,
            context=context,
            reasons=reasons,
        )

        self._check_market(
            metadata=metadata,
            context=context,
            reasons=reasons,
        )

        return EligibilityResult(
            eligible=not reasons,
            reasons=reasons,
        )

    def filter(
        self,
        *,
        candidates: list[KnowledgeCandidate],
        context: EligibilityContext,
    ) -> list[KnowledgeCandidate]:
        eligible: list[KnowledgeCandidate] = []

        for candidate in candidates:
            result = self.evaluate(
                candidate=candidate,
                context=context,
            )

            if result.eligible:
                eligible.append(candidate)

        return eligible

    def _check_audience(
        self,
        *,
        metadata,
        context: EligibilityContext,
        reasons: list[str],
    ) -> None:
        audience = metadata.audience.value

        if audience == "INTERNAL":
            reasons.append(
                "AUDIENCE_INTERNAL"
            )
            return

        if audience == "CLIENT":
            if not context.is_authenticated:
                reasons.append(
                    "CLIENT_AUTH_REQUIRED")


    def _check_required_context(
        self,
        *,
        metadata,
        context: EligibilityContext,
        reasons: list[str],
    ) -> None:
        required_fields = (
            metadata.required_context_fields
            or []
        )

        for field_name in required_fields:
            field_name = str(
                field_name
            ).strip()

            if not field_name:
                continue

            value = context.get(
                field_name
            )

            if value is None:
                reasons.append(
                    f"MISSING_CONTEXT:{field_name}"
                )

    def _check_bot_mode(
        self,
        *,
        metadata,
        context: EligibilityContext,
        reasons: list[str],
    ) -> None:
        bot_mode = metadata.bot_mode.value

        if bot_mode == "NOT_APPLICABLE":
            reasons.append(
                "BOT_MODE_NOT_AI_ACCESSIBLE"
            )
            return

        if bot_mode == "LOOKUP":
            if not context.is_authenticated:
                reasons.append(
                    "LOOKUP_AUTH_REQUIRED"
                )

            if not context.client_id:
                reasons.append(
                    "LOOKUP_CLIENT_ID_REQUIRED"
                )

    def _check_market(
        self,
        *,
        metadata,
        context: EligibilityContext,
        reasons: list[str],
    ) -> None:
        market_scope = (
            metadata.market_scope.value
        )

        if market_scope == "GLOBAL":
            return

        if market_scope == "UNSPECIFIED":
            reasons.append(
                "MARKET_SCOPE_UNSPECIFIED"
            )
            return

        if market_scope == "COUNTRY":
            self._check_country_market(
                metadata=metadata,
                context=context,
                reasons=reasons,
            )
            return

        if market_scope == "REGION":
            self._check_region_market(
                metadata=metadata,
                context=context,
                reasons=reasons,
            )
            return

        reasons.append(
            f"UNSUPPORTED_MARKET_SCOPE:{market_scope}"
        )

    def _check_country_market(
        self,
        *,
        metadata,
        context: EligibilityContext,
        reasons: list[str],
    ) -> None:
        allowed_countries = {
            str(country).strip().upper()
            for country in (
                metadata.countries
                or []
            )
            if str(country).strip()
        }

        if not allowed_countries:
            reasons.append(
                "COUNTRY_SCOPE_WITHOUT_COUNTRIES"
            )
            return

        required_fields = {
            str(field_name).strip()
            for field_name in (
                metadata.required_context_fields
                or []
            )
        }

        market_fields = (
            required_fields
            & MARKET_CONTEXT_FIELDS
        )

        if not market_fields:
            reasons.append(
                "COUNTRY_CONTEXT_BASIS_MISSING"
            )
            return

        context_values: set[str] = set()

        for field_name in market_fields:
            value = context.get(
                field_name
            )

            if value:
                context_values.add(
                    value.upper()
                )

        if not context_values:
            reasons.append(
                "COUNTRY_CONTEXT_MISSING"
            )
            return

        if not (
            context_values
            & allowed_countries
        ):
            reasons.append(
                "COUNTRY_CONTEXT_MISMATCH"
            )

    def _check_region_market(
        self,
        *,
        metadata,
        context: EligibilityContext,
        reasons: list[str],
    ) -> None:
        allowed_regions = {
            str(region).strip().upper()
            for region in (
                metadata.regions
                or []
            )
            if str(region).strip()
        }

        if not allowed_regions:
            reasons.append(
                "REGION_SCOPE_WITHOUT_REGIONS"
            )
            return

        resolved_regions = {
            str(region).strip().upper()
            for region in (
                context.resolved_regions
                or set()
            )
            if str(region).strip()
        }

        if not resolved_regions:
            reasons.append(
                "REGION_CONTEXT_MISSING"
            )
            return

        if not (
            resolved_regions
            & allowed_regions
        ):
            reasons.append(
                "REGION_CONTEXT_MISMATCH"
            )
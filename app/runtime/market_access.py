from dataclasses import dataclass

from app.runtime.context import RuntimeContext

SUPPORTED_MARKET_CONTEXT_FIELDS = {"nationality", "country_of_residence"}


@dataclass(frozen=True)
class MarketResolution:
    resolved_market: str | None
    context_field: str | None
    source: str | None
    trust: str | None
    conflict: bool = False


class MarketAccessResolver:
    def get_market_context_fields(
        self, required_context_fields: list[str]
    ) -> list[str]:
        return [
            field_name
            for field_name in required_context_fields
            if field_name in SUPPORTED_MARKET_CONTEXT_FIELDS
        ]

    def resolve_for_field(
        self, *, context: RuntimeContext, field_name: str
    ) -> MarketResolution:
        if field_name not in SUPPORTED_MARKET_CONTEXT_FIELDS:
            return MarketResolution(
                resolved_market=None, context_field=field_name, source=None, trust=None
            )

        context_value = getattr(context, field_name, None)

        if context_value is None or not context_value.value:
            return MarketResolution(
                resolved_market=None, context_field=field_name, source=None, trust=None
            )

        return MarketResolution(
            resolved_market=str(context_value.value).strip().upper(),
            context_field=field_name,
            source=context_value.source,
            trust=(context_value.trust.value if context_value.trust else None),
        )

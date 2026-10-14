from app.runtime.context import (
    ContextTrust,
    ContextValue,
    IdentityState,
    RuntimeContext,
)
from app.runtime.market_access import MarketAccessResolver


def test_resolves_nationality_when_requested():
    context = RuntimeContext(
        identity_state=IdentityState.ANONYMOUS,
        nationality=ContextValue(
            value="AF", source="USER_DECLARED", trust=ContextTrust.DECLARED
        ),
        country_of_residence=ContextValue(
            value="IR", source="USER_DECLARED", trust=ContextTrust.DECLARED
        ),
    )

    resolver = MarketAccessResolver()

    result = resolver.resolve_for_field(context=context, field_name="nationality")

    assert result.resolved_market == "AF"
    assert result.context_field == "nationality"
    assert result.source == "USER_DECLARED"
    assert result.trust == ContextTrust.DECLARED.value


def test_resolves_country_of_residence_when_requested():
    context = RuntimeContext(
        identity_state=IdentityState.ANONYMOUS,
        nationality=ContextValue(
            value="AF", source="USER_DECLARED", trust=ContextTrust.DECLARED
        ),
        country_of_residence=ContextValue(
            value="IR", source="USER_DECLARED", trust=ContextTrust.DECLARED
        ),
    )

    resolver = MarketAccessResolver()

    result = resolver.resolve_for_field(
        context=context, field_name="country_of_residence"
    )

    assert result.resolved_market == "IR"
    assert result.context_field == "country_of_residence"


def test_referenced_country_is_not_a_market_identity_field():
    context = RuntimeContext(
        identity_state=IdentityState.ANONYMOUS,
        referenced_country=ContextValue(
            value="AF", source="CURRENT_QUERY", trust=ContextTrust.QUERY_CONTEXT
        ),
    )

    resolver = MarketAccessResolver()

    result = resolver.resolve_for_field(
        context=context, field_name="referenced_country"
    )

    assert result.resolved_market is None


def test_missing_requested_market_context_returns_none():
    context = RuntimeContext(identity_state=IdentityState.ANONYMOUS)

    resolver = MarketAccessResolver()

    result = resolver.resolve_for_field(context=context, field_name="nationality")

    assert result.resolved_market is None
    assert result.context_field == "nationality"

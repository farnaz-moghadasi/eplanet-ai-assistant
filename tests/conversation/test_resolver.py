import json

from app.conversation.resolver import (
    ConversationResolver,
)
from app.conversation.state import (
    ConversationState,
)
from app.runtime.entity_registry import (
    EntityRegistry,
)


def build_resolver(tmp_path):
    config = {
        "entity_types": {
            "platform": {
                "fuzzy_enabled": True,
                "fuzzy_threshold": 0.70,
                "aliases": {
                    "ctrader": "cTrader",
                    "c trader": "cTrader",
                    "mt5": "MT5",
                    "metatrader": "MT5",
                    "metatrader 5": "MT5",
                },
            },
            "country": {
                "fuzzy_enabled": False,
                "fuzzy_threshold": 0.90,
                "aliases": {
                    "afghanistan": "AF",
                    "afghan": "AF",
                    "افغانستان": "AF",
                    "افغان": "AF",
                    "iran": "IR",
                    "ایران": "IR",
                },
            },
        },
        "context_field_entity_types": {
            "nationality": "country",
            "country_of_residence": "country",
            "kyc_document_country": "country",
            "referenced_country": "country",
        },
        "conversation": {
            "short_followup_max_tokens": 4,
            "correction_markers": [
                "نه منظورم",
                "منظورم",
                "i mean",
                "no i mean",
            ],
        },
    }

    path = (
        tmp_path
        / "entity_aliases.json"
    )

    path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    registry = (
        EntityRegistry.from_file(
            path
        )
    )

    return ConversationResolver(
        entity_registry=registry
    )


def test_platform_is_extracted(
    tmp_path,
):
    resolver = build_resolver(
        tmp_path
    )

    state = ConversationState()

    result = resolver.resolve(
        message=(
            "cTrader رو چجوری نصب کنم؟"
        ),
        state=state,
        language="FA",
    )

    assert result.platform == "cTrader"

    assert (
        result.entities["platform"]
        == "cTrader"
    )

    assert (
        state.last_platform
        == "cTrader"
    )


def test_platform_typo_is_resolved(
    tmp_path,
):
    resolver = build_resolver(
        tmp_path
    )

    state = ConversationState()

    result = resolver.resolve(
        message="ctarder رو میخوام",
        state=state,
        language="FA",
    )

    assert (
        result.platform
        == "cTrader"
    )


def test_mt5_followup_replaces_previous_platform(
    tmp_path,
):
    resolver = build_resolver(
        tmp_path
    )

    state = ConversationState()

    first = resolver.resolve(
        message=(
            "cTrader چجوری نصب میشه؟"
        ),
        state=state,
        language="FA",
    )

    assert (
        first.platform
        == "cTrader"
    )

    second = resolver.resolve(
        message="MT5 چی؟",
        state=state,
        language="FA",
    )

    assert (
        second.platform
        == "MT5"
    )

    assert (
        "MT5"
        in second.resolved_query
    )

    assert (
        "cTrader"
        not in second.resolved_query
    )

    assert (
        "previous_query"
        in second.inherited_context
    )


def test_followup_without_new_entity_uses_previous_platform(
    tmp_path,
):
    resolver = build_resolver(
        tmp_path
    )

    state = ConversationState(
        last_platform="cTrader",
        last_resolved_query=(
            "How do I log in "
            "to cTrader?"
        ),
    )

    result = resolver.resolve(
        message="چجوری؟",
        state=state,
        language="FA",
    )

    assert (
        result.platform
        == "cTrader"
    )

    assert (
        result.inherited_context[
            "platform"
        ]
        == "cTrader"
    )


def test_pending_country_context_is_resolved(
    tmp_path,
):
    resolver = build_resolver(
        tmp_path
    )

    state = ConversationState(
        language="FA",
        last_resolved_query=(
            "مدارک احراز هویت"
        ),
        awaiting_context_field=(
            "kyc_document_country"
        ),
    )

    result = resolver.resolve(
        message="افغانستان",
        state=state,
        language="FA",
    )

    assert (
        state.get_context_field(
            "kyc_document_country"
        )
        == "AF"
    )

    assert (
        state.awaiting_context_field
        is None
    )

    assert (
        result.entities[
            "kyc_document_country"
        ]
        == "AF"
    )


def test_pending_context_does_not_guess_unknown_value(
    tmp_path,
):
    resolver = build_resolver(
        tmp_path
    )

    state = ConversationState(
        awaiting_context_field=(
            "kyc_document_country"
        ),
        last_resolved_query=(
            "KYC docs"
        ),
    )

    result = resolver.resolve(
        message="نمیدونم",
        state=state,
        language="FA",
    )

    assert (
        state.kyc_document_country
        is None
    )

    assert (
        state.awaiting_context_field
        == "kyc_document_country"
    )

    assert (
        result.original_message
        == "نمیدونم"
    )


def test_language_is_preserved_from_state(
    tmp_path,
):
    resolver = build_resolver(
        tmp_path
    )

    state = ConversationState(
        language="KU"
    )

    result = resolver.resolve(
        message="MT5 چی؟",
        state=state,
    )

    assert result.language == "KU"


def test_unknown_message_is_not_forced_to_entity(
    tmp_path,
):
    resolver = build_resolver(
        tmp_path
    )

    state = ConversationState()

    result = resolver.resolve(
        message="سلام خوبی؟",
        state=state,
        language="FA",
    )

    assert result.platform is None
    assert result.entities == {}


def test_resolver_adds_turn_to_history(
    tmp_path,
):
    resolver = build_resolver(
        tmp_path
    )

    state = ConversationState()

    resolver.resolve(
        message="cTrader",
        state=state,
        language="FA",
    )

    assert (
        state.turn_count
        == 1
    )

    assert (
        len(state.recent_turns)
        == 1
    )


def test_correction_preserves_previous_query_context(
    tmp_path,
):
    resolver = build_resolver(
        tmp_path
    )

    state = ConversationState(
        last_resolved_query=(
            "How do I reset my "
            "portal password?"
        )
    )

    result = resolver.resolve(
        message=(
            "نه منظورم رمز "
            "حساب معاملاتی بود"
        ),
        state=state,
        language="FA",
    )

    assert (
        "previous_query"
        in result.inherited_context
    )

    assert (
        "منظورم"
        in result.resolved_query
    )


def test_disambiguation_choice_matched_by_keyword(
    tmp_path,
):
    resolver = build_resolver(tmp_path)

    state = ConversationState(
        language="FA",
        last_resolved_query=(
            "how do i change my account password"
        ),
        awaiting_context_field="disambiguation",
        pending_disambiguation_options=[
            "client portal password",
            "MetaTrader 5 trading password",
        ],
        pending_disambiguation_query=(
            "how do i change my account password"
        ),
    )

    result = resolver.resolve(
        message="MetaTrader 5",
        state=state,
        language="EN",
    )

    assert state.awaiting_context_field is None
    assert state.pending_disambiguation_options == []
    assert (
        "MetaTrader 5 trading password"
        in result.resolved_query
    )
    assert (
        result.entities[
            "disambiguation_choice"
        ]
        == "MetaTrader 5 trading password"
    )


def test_disambiguation_choice_matched_by_ordinal(
    tmp_path,
):
    resolver = build_resolver(tmp_path)

    state = ConversationState(
        language="FA",
        last_resolved_query=(
            "رمز حسابم رو چجوری عوض کنم؟"
        ),
        awaiting_context_field="disambiguation",
        pending_disambiguation_options=[
            "رمز عبور پنل کاربری",
            "رمز معاملاتی MetaTrader 5",
        ],
        pending_disambiguation_query=(
            "رمز حسابم رو چجوری عوض کنم؟"
        ),
    )

    result = resolver.resolve(
        message="دومی",
        state=state,
        language="FA",
    )

    assert (
        "رمز معاملاتی MetaTrader 5"
        in result.resolved_query
    )


def test_disambiguation_no_match_keeps_pending_state(
    tmp_path,
):
    # An unrelated reply must not be forced into one of the offered
    # options - the pending state stays so the caller can decide what
    # to do (e.g. fall through to a fresh understanding pass).
    resolver = build_resolver(tmp_path)

    state = ConversationState(
        language="EN",
        last_resolved_query=(
            "how do i change my account password"
        ),
        awaiting_context_field="disambiguation",
        pending_disambiguation_options=[
            "client portal password",
            "MetaTrader 5 trading password",
        ],
        pending_disambiguation_query=(
            "how do i change my account password"
        ),
    )

    resolver.resolve(
        message="what is the weather today",
        state=state,
        language="EN",
    )

    assert (
        state.awaiting_context_field
        == "disambiguation"
    )
    assert (
        state.pending_disambiguation_options
        != []
    )

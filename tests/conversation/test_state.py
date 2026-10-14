import pytest

from app.conversation.state import ConversationState, ConversationTurn


def test_add_turn_updates_last_context():
    state = ConversationState()

    turn = ConversationTurn(
        user_message="cTrader چجوری نصب کنم؟",
        resolved_query="نصب cTrader",
        language="FA",
        intent="platform_installation",
        topic="trading_platform",
        platform="cTrader",
    )

    state.add_turn(turn)

    assert state.turn_count == 1
    assert state.language == "FA"
    assert state.last_intent == "platform_installation"
    assert state.last_topic == "trading_platform"
    assert state.last_platform == "cTrader"
    assert state.last_user_message == "cTrader چجوری نصب کنم؟"
    assert state.last_resolved_query == "نصب cTrader"


def test_state_preserves_previous_values_when_turn_omits_them():
    state = ConversationState(
        language="FA",
        last_intent="platform_installation",
        last_topic="trading_platform",
        last_platform="cTrader",
    )

    turn = ConversationTurn(
        user_message="MT5 چی؟",
        resolved_query="نصب MetaTrader 5",
        language="FA",
        platform="MT5",
    )

    state.add_turn(turn)

    assert state.last_intent == "platform_installation"
    assert state.last_topic == "trading_platform"
    assert state.last_platform == "MT5"


def test_recent_turns_are_capped():
    state = ConversationState()

    for index in range(10):
        state.add_turn(ConversationTurn(user_message=f"message {index}"), max_turns=3)

    assert len(state.recent_turns) == 3

    assert [turn.user_message for turn in state.recent_turns] == [
        "message 7",
        "message 8",
        "message 9",
    ]

    assert state.turn_count == 10


def test_update_context_normalizes_country_codes():
    state = ConversationState()

    state.update_context(
        nationality=" af ",
        country_of_residence=" ir ",
        kyc_document_country=" af ",
        referenced_country=" tr ",
    )

    assert state.nationality == "AF"
    assert state.country_of_residence == "IR"
    assert state.kyc_document_country == "AF"
    assert state.referenced_country == "TR"


def test_update_context_does_not_clear_existing_values_with_none():
    state = ConversationState(nationality="AF", country_of_residence="IR")

    state.update_context(nationality=None, country_of_residence=None)

    assert state.nationality == "AF"
    assert state.country_of_residence == "IR"


def test_pending_context_can_be_set_and_cleared():
    state = ConversationState()

    state.set_pending_context("kyc_document_country")

    assert state.awaiting_context_field == "kyc_document_country"

    state.clear_pending_context()

    assert state.awaiting_context_field is None


def test_authenticated_client_can_be_set():
    state = ConversationState()

    state.set_authenticated_client(client_id=" 12345 ")

    assert state.is_authenticated is True
    assert state.client_id == "12345"


def test_authenticated_client_rejects_empty_client_id():
    state = ConversationState()

    with pytest.raises(ValueError, match="client_id cannot be empty"):
        state.set_authenticated_client(client_id="   ")


def test_authenticated_client_can_be_cleared():
    state = ConversationState(is_authenticated=True, client_id="12345")

    state.clear_authenticated_client()

    assert state.is_authenticated is False
    assert state.client_id is None


def test_add_turn_keeps_entities():
    state = ConversationState()

    turn = ConversationTurn(
        user_message="MT5 چی؟",
        resolved_query="نصب MetaTrader 5",
        language="FA",
        intent="platform_installation",
        topic="trading_platform",
        platform="MT5",
        entities={"platform": "MT5"},
    )

    state.add_turn(turn)

    assert len(state.recent_turns) == 1

    assert state.recent_turns[0].entities["platform"] == "MT5"

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    user_message: str
    resolved_query: str | None = None

    language: str | None = None
    intent: str | None = None
    topic: str | None = None
    platform: str | None = None

    entities: dict[str, Any] = Field(default_factory=dict)


class ResolvedTurn(BaseModel):
    original_message: str
    resolved_query: str

    language: str | None = None

    intent: str | None = None
    topic: str | None = None
    platform: str | None = None

    entities: dict[str, Any] = Field(default_factory=dict)

    inherited_context: dict[str, Any] = Field(default_factory=dict)

    missing_context_fields: list[str] = Field(default_factory=list)

    needs_clarification: bool = False

    clarification_reason: str | None = None

    confidence: float | None = None


class ConversationState(BaseModel):
    conversation_id: str | None = None

    language: str | None = None

    last_intent: str | None = None
    last_topic: str | None = None
    last_platform: str | None = None

    last_user_message: str | None = None
    last_resolved_query: str | None = None

    nationality: str | None = None

    country_of_residence: str | None = None

    kyc_document_country: str | None = None

    referenced_country: str | None = None

    context_values: dict[str, str] = Field(default_factory=dict)

    is_authenticated: bool = False
    client_id: str | None = None

    awaiting_context_field: str | None = None


    pending_disambiguation_options: list[str] = Field(default_factory=list)

    pending_disambiguation_query: str | None = None


    pending_open_clarification_question: str | None = None

    pending_open_clarification_query: str | None = None

    diagnostic_clarification_rounds: int = 0

    turn_count: int = 0

    recent_turns: list[ConversationTurn] = Field(default_factory=list)

    def add_turn(self, turn: ConversationTurn, *, max_turns: int = 8) -> None:
        self.recent_turns.append(turn)

        if len(self.recent_turns) > max_turns:
            self.recent_turns = self.recent_turns[-max_turns:]

        self.turn_count += 1

        self.last_user_message = turn.user_message

        if turn.resolved_query:
            self.last_resolved_query = turn.resolved_query

        if turn.language:
            self.language = turn.language

        if turn.intent:
            self.last_intent = turn.intent

        if turn.topic:
            self.last_topic = turn.topic

        if turn.platform:
            self.last_platform = turn.platform

    def set_context_field(self, *, field_name: str, value: str) -> None:
        field_name = field_name.strip()

        value = value.strip()

        if not field_name:
            raise ValueError("field_name cannot be empty")

        if not value:
            raise ValueError("context value cannot be empty")

        normalized_value = value.upper()

        self.context_values[field_name] = normalized_value

        if hasattr(self, field_name):
            setattr(self, field_name, normalized_value)

    def get_context_field(self, field_name: str) -> str | None:
        direct = getattr(self, field_name, None)

        if direct:
            return str(direct)

        return self.context_values.get(field_name)

    def update_context(
        self,
        *,
        nationality: str | None = None,
        country_of_residence: str | None = None,
        kyc_document_country: str | None = None,
        referenced_country: str | None = None,
    ) -> None:
        values = {
            "nationality": nationality,
            "country_of_residence": country_of_residence,
            "kyc_document_country": kyc_document_country,
            "referenced_country": referenced_country,
        }

        for field_name, value in values.items():
            if value is None:
                continue

            self.set_context_field(field_name=field_name, value=value)

    def set_authenticated_client(self, *, client_id: str) -> None:
        client_id = client_id.strip()

        if not client_id:
            raise ValueError("client_id cannot be empty")

        self.is_authenticated = True
        self.client_id = client_id

    def clear_authenticated_client(self) -> None:
        self.is_authenticated = False
        self.client_id = None

    def set_pending_context(self, field_name: str) -> None:
        field_name = field_name.strip()

        if not field_name:
            raise ValueError("field_name cannot be empty")

        self.awaiting_context_field = field_name

    def clear_pending_context(self) -> None:
        self.awaiting_context_field = None
        self.pending_disambiguation_options = []
        self.pending_disambiguation_query = None
        self.pending_open_clarification_question = None
        self.pending_open_clarification_query = None


def state_to_prompt_text(state: ConversationState) -> str:
    parts: list[str] = []

    if state.language:
        parts.append(f"Language: {state.language}")

    if state.last_intent:
        parts.append(f"Last intent: {state.last_intent}")

    if state.last_topic:
        parts.append(f"Last topic: {state.last_topic}")

    if state.last_platform:
        parts.append(f"Last platform: {state.last_platform}")

    if state.last_user_message:
        parts.append("Last user message: " f"{state.last_user_message}")

    if state.last_resolved_query:
        parts.append("Last resolved query: " f"{state.last_resolved_query}")

    if state.nationality:
        parts.append(f"Nationality: {state.nationality}")

    if state.country_of_residence:
        parts.append("Country of residence: " f"{state.country_of_residence}")

    if state.kyc_document_country:
        parts.append("KYC document country: " f"{state.kyc_document_country}")

    if state.referenced_country:
        parts.append("Referenced country: " f"{state.referenced_country}")

    if state.awaiting_context_field:
        parts.append("Awaiting context field: " f"{state.awaiting_context_field}")

    parts.append(
        "Authenticated: yes" if state.is_authenticated else "Authenticated: no"
    )

    if state.recent_turns:
        parts.append("Recent conversation:")

        for turn in state.recent_turns[-5:]:
            parts.append(f"- User: {turn.user_message}")

            if turn.resolved_query:
                parts.append("  Resolved: " f"{turn.resolved_query}")

            if turn.intent:
                parts.append("  Intent: " f"{turn.intent}")

            if turn.topic:
                parts.append("  Topic: " f"{turn.topic}")

            if turn.platform:
                parts.append("  Platform: " f"{turn.platform}")

    return "\n".join(parts)

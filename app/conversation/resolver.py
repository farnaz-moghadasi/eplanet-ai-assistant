from __future__ import annotations

import re

from app.conversation.state import (
    ConversationState,
    ConversationTurn,
    ResolvedTurn,
)
from app.runtime.entity_registry import (
    EntityRegistry,
)


class ConversationResolver:
    def __init__(
        self,
        *,
        entity_registry: EntityRegistry,
    ):
        self.entity_registry = (
            entity_registry
        )

    def resolve(
        self,
        *,
        message: str,
        state: ConversationState,
        language: str | None = None,
    ) -> ResolvedTurn:
        original_message = message

        normalized = (
            self.entity_registry
            .normalize_text(message)
        )

        resolved_language = (
            language
            or state.language
        )

        pending = (
            self._resolve_pending_context(
                message=normalized,
                original_message=(
                    original_message
                ),
                language=(
                    resolved_language
                ),
                state=state,
            )
        )

        if pending is not None:
            self._commit_turn(
                state=state,
                resolved=pending,
            )

            return pending

        entities = (
            self.entity_registry
            .resolve_entities(
                normalized
            )
        )

        is_correction = (
            self._is_correction(
                normalized
            )
        )

        is_followup = (
            self._looks_like_followup(
                normalized
            )
        )

        inherited_context: dict[
            str,
            str,
        ] = {}

        resolved_query = normalized

        if (
            is_correction
            and state.last_resolved_query
        ):
            resolved_query = (
                self._resolve_correction(
                    previous_query=(
                        state.last_resolved_query
                    ),
                    current_message=(
                        normalized
                    ),
                )
            )

            inherited_context[
                "previous_query"
            ] = state.last_resolved_query

        elif (
            is_followup
            and state.last_resolved_query
        ):
            resolved_query = (
                self._resolve_followup(
                    previous_query=(
                        state.last_resolved_query
                    ),
                    current_message=(
                        normalized
                    ),
                    entities=entities,
                    state=state,
                )
            )

            inherited_context[
                "previous_query"
            ] = state.last_resolved_query

        platform = entities.get(
            "platform"
        )

        if (
            platform is None
            and is_followup
            and state.last_platform
        ):
            platform = (
                state.last_platform
            )

            inherited_context[
                "platform"
            ] = state.last_platform

        result = ResolvedTurn(
            original_message=(
                original_message
            ),
            resolved_query=(
                resolved_query
            ),
            language=(
                resolved_language
            ),
            intent=(
                state.last_intent
                if is_followup
                else None
            ),
            topic=(
                state.last_topic
                if is_followup
                else None
            ),
            platform=platform,
            entities=entities,
            inherited_context=(
                inherited_context
            ),
            missing_context_fields=[],
            needs_clarification=False,
            clarification_reason=None,
            confidence=(
                self._estimate_confidence(
                    entities=entities,
                    is_followup=is_followup,
                    is_correction=(
                        is_correction
                    ),
                )
            ),
        )

        self._commit_turn(
            state=state,
            resolved=result,
        )

        return result

    def _resolve_pending_context(
        self,
        *,
        message: str,
        original_message: str,
        language: str | None,
        state: ConversationState,
    ) -> ResolvedTurn | None:
        field_name = (
            state.awaiting_context_field
        )

        if not field_name:
            return None

        if field_name == "disambiguation":
            return (
                self
                ._resolve_disambiguation_choice(
                    message=message,
                    original_message=(
                        original_message
                    ),
                    language=language,
                    state=state,
                )
            )

        entity_type = (
            self.entity_registry
            .entity_type_for_context_field(
                field_name
            )
        )

        if entity_type is None:
            return None

        value = (
            self.entity_registry
            .resolve_pending_context_entity(
                entity_type=entity_type,
                text=message,
            )
        )

        if value is None:
            return None

        previous_query = (
            state.last_resolved_query
            or original_message
        )

        state.set_context_field(
            field_name=field_name,
            value=value,
        )

        state.clear_pending_context()

        return ResolvedTurn(
            original_message=(
                original_message
            ),
            resolved_query=(
                previous_query
            ),
            language=language,
            intent=state.last_intent,
            topic=state.last_topic,
            platform=(
                state.last_platform
            ),
            entities={
                field_name: value
            },
            inherited_context={
                "resolved_context_field":
                    field_name
            },
            missing_context_fields=[],
            needs_clarification=False,
            clarification_reason=None,
            confidence=0.98,
        )

    _DISAMBIGUATION_ORDINALS = (
        ("1", "۱", "اول", "اولی", "یک", "first", "one"),
        ("2", "۲", "دوم", "دومی", "دو", "second", "two"),
        ("3", "۳", "سوم", "سومی", "سه", "third", "three"),
        ("4", "۴", "چهارم", "چهارمی", "fourth", "four"),
    )

    def _resolve_disambiguation_choice(
        self,
        *,
        message: str,
        original_message: str,
        language: str | None,
        state: ConversationState,
    ) -> ResolvedTurn | None:
        options = (
            state
            .pending_disambiguation_options
        )

        if not options:
            state.clear_pending_context()
            return None

        chosen = (
            self._match_disambiguation_choice(
                message=message,
                options=options,
            )
        )

        if chosen is None:
            return None

        previous_query = (
            state.pending_disambiguation_query
            or state.last_resolved_query
            or original_message
        )

        combined_query = (
            f"{previous_query} ({chosen})"
        )

        state.clear_pending_context()

        return ResolvedTurn(
            original_message=(
                original_message
            ),
            resolved_query=combined_query,
            language=language,
            intent=state.last_intent,
            topic=state.last_topic,
            platform=state.last_platform,
            entities={
                "disambiguation_choice":
                    chosen
            },
            inherited_context={
                "resolved_context_field":
                    "disambiguation"
            },
            missing_context_fields=[],
            needs_clarification=False,
            clarification_reason=None,
            confidence=0.9,
        )

    def _match_disambiguation_choice(
        self,
        *,
        message: str,
        options: list[str],
    ) -> str | None:
        normalized_message = (
            self.entity_registry
            .normalize_text(message)
        )

        for (
            index,
            markers,
        ) in enumerate(
            self._DISAMBIGUATION_ORDINALS
        ):
            if index >= len(options):
                break

            if any(
                marker in normalized_message
                for marker in markers
            ):
                return options[index]

        message_tokens = set(
            normalized_message.split()
        )

        best_option: str | None = None
        best_overlap = 0

        for option in options:
            option_tokens = set(
                self.entity_registry
                .normalize_text(option)
                .split()
            )

            overlap = len(
                option_tokens
                & message_tokens
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_option = option

        return best_option

    def _resolve_followup(
        self,
        *,
        previous_query: str,
        current_message: str,
        entities: dict[str, str],
        state: ConversationState,
    ) -> str:
        result = previous_query

        replacement_applied = False

        for (
            entity_type,
            new_value,
        ) in entities.items():
            previous_value = (
                self._previous_entity_value(
                    entity_type=(
                        entity_type
                    ),
                    state=state,
                )
            )

            if not previous_value:
                continue

            updated = re.sub(
                re.escape(
                    previous_value
                ),
                new_value,
                result,
                flags=re.IGNORECASE,
            )

            if updated != result:
                replacement_applied = (
                    True
                )

                result = updated

        if replacement_applied:
            return result

        # Do not concatenate an ambiguous short follow-up onto the old
        # query. ConversationUnderstanding produces the semantic
        # standalone query. Keeping only the current text here avoids
        # polluting conversation history with strings such as
        # "ECN commission Standard?".
        return current_message.strip()

    def _resolve_correction(
        self,
        *,
        previous_query: str,
        current_message: str,
    ) -> str:
        # Corrections are also resolved semantically by the understanding
        # layer. The previous query remains available in inherited_context.
        return current_message.strip()

    def _previous_entity_value(
        self,
        *,
        entity_type: str,
        state: ConversationState,
    ) -> str | None:
        if entity_type == "platform":
            return (
                state.last_platform
            )

        if state.recent_turns:
            latest = (
                state.recent_turns[-1]
            )

            value = latest.entities.get(
                entity_type
            )

            if value:
                return str(value)

        return None

    def _looks_like_followup(
        self,
        text: str,
    ) -> bool:
        tokens = text.split()

        if not tokens:
            return False

        max_tokens = (
            self.entity_registry
            .short_followup_max_tokens()
        )

        return (
            len(tokens)
            <= max_tokens
        )

    def _is_correction(
        self,
        text: str,
    ) -> bool:
        normalized = (
            self.entity_registry
            .normalize_text(text)
        )

        return any(
            marker in normalized
            for marker
            in (
                self.entity_registry
                .correction_markers()
            )
        )

    @staticmethod
    def _estimate_confidence(
        *,
        entities: dict[str, str],
        is_followup: bool,
        is_correction: bool,
    ) -> float:
        score = 0.55

        if entities:
            score += 0.20

        if is_followup:
            score += 0.10

        if is_correction:
            score += 0.05

        return min(
            score,
            1.0,
        )

    @staticmethod
    def _commit_turn(
        *,
        state: ConversationState,
        resolved: ResolvedTurn,
    ) -> None:
        state.add_turn(
            ConversationTurn(
                user_message=(
                    resolved.original_message
                ),
                resolved_query=(
                    resolved.resolved_query
                ),
                language=(
                    resolved.language
                ),
                intent=(
                    resolved.intent
                ),
                topic=(
                    resolved.topic
                ),
                platform=(
                    resolved.platform
                ),
                entities=(
                    resolved.entities
                ),
            )
        )
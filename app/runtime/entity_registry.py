from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EntityTypeConfig:
    aliases: dict[str, str]
    fuzzy_enabled: bool = False
    fuzzy_threshold: float = 0.8


class EntityRegistry:
    def __init__(
        self,
        *,
        entity_types: dict[
            str,
            EntityTypeConfig,
        ],
        context_field_entity_types: dict[
            str,
            str,
        ] | None = None,
        conversation_config: dict[
            str,
            Any,
        ] | None = None,
    ):
        self.entity_types = entity_types

        self.context_field_entity_types = (
            context_field_entity_types
            or {}
        )

        self.conversation_config = (
            conversation_config
            or {}
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
    ) -> "EntityRegistry":
        path = Path(path)

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw = json.load(file)

        entity_types: dict[
            str,
            EntityTypeConfig,
        ] = {}

        for (
            entity_type,
            config,
        ) in raw.get(
            "entity_types",
            {},
        ).items():
            aliases = {
                cls.normalize_text(alias):
                canonical
                for (
                    alias,
                    canonical,
                ) in config.get(
                    "aliases",
                    {},
                ).items()
            }

            entity_types[
                entity_type
            ] = EntityTypeConfig(
                aliases=aliases,
                fuzzy_enabled=bool(
                    config.get(
                        "fuzzy_enabled",
                        False,
                    )
                ),
                fuzzy_threshold=float(
                    config.get(
                        "fuzzy_threshold",
                        0.8,
                    )
                ),
            )

        return cls(
            entity_types=entity_types,
            context_field_entity_types=(
                raw.get(
                    "context_field_entity_types",
                    {},
                )
            ),
            conversation_config=(
                raw.get(
                    "conversation",
                    {},
                )
            ),
        )

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        text = str(text or "").strip()

        text = text.replace(
            "\u200c",
            " ",
        )

        text = text.replace(
            "ي",
            "ی",
        )

        text = text.replace(
            "ك",
            "ک",
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.lower()

    def resolve_entities(
        self,
        text: str,
    ) -> dict[str, str]:
        results: dict[
            str,
            str,
        ] = {}

        for entity_type in self.entity_types:
            value = self.resolve_entity(
                entity_type=entity_type,
                text=text,
            )

            if value is not None:
                results[
                    entity_type
                ] = value

        return results
    def resolve_pending_context_entity(
        self,
        *,
        entity_type: str,
        text: str,
    ) -> str | None:
        exact = self.resolve_entity(
            entity_type=entity_type,
            text=text,
        )

        if exact is not None:
            return exact

        config = self.entity_types.get(
            entity_type
        )

        if config is None:
            return None

        normalized = self.normalize_text(
            text
        )

        tokens = re.findall(
            r"[\w-]+",
            normalized,
            flags=re.UNICODE,
        )

        if not tokens:
            return None

        threshold = float(
            self.conversation_config.get(
                "pending_context_fuzzy_threshold",
                0.78,
            )
        )

        minimum_margin = float(
            self.conversation_config.get(
                "pending_context_fuzzy_margin",
                0.08,
            )
        )

        scores_by_value: dict[
            str,
            float,
        ] = {}

        for token in tokens:
            if len(token) < 3:
                continue

            for alias, canonical in (
                config.aliases.items()
            ):
                if " " in alias:
                    continue

                if len(alias) < 3:
                    continue

                score = SequenceMatcher(
                    None,
                    token,
                    alias,
                ).ratio()

                previous = (
                    scores_by_value.get(
                        canonical,
                        0.0,
                    )
                )

                if score > previous:
                    scores_by_value[
                        canonical
                    ] = score

        if not scores_by_value:
            return None

        ranked = sorted(
            scores_by_value.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        best_value, best_score = (
            ranked[0]
        )

        if best_score < threshold:
            return None

        if len(ranked) > 1:
            second_score = ranked[1][1]

            if (
                best_score
                - second_score
                < minimum_margin
            ):
                return None

        return best_value
    def resolve_entity(
        self,
        *,
        entity_type: str,
        text: str,
    ) -> str | None:
        config = self.entity_types.get(
            entity_type
        )

        if config is None:
            return None

        normalized = self.normalize_text(
            text
        )

        exact = self._resolve_exact(
            text=normalized,
            aliases=config.aliases,
        )

        if exact is not None:
            return exact

        if not config.fuzzy_enabled:
            return None

        return self._resolve_fuzzy(
            text=normalized,
            config=config,
        )

    def entity_type_for_context_field(
        self,
        field_name: str,
    ) -> str | None:
        return (
            self.context_field_entity_types
            .get(field_name)
        )

    def short_followup_max_tokens(
        self,
    ) -> int:
        return int(
            self.conversation_config.get(
                "short_followup_max_tokens",
                4,
            )
        )

    def correction_markers(
        self,
    ) -> tuple[str, ...]:
        values = (
            self.conversation_config.get(
                "correction_markers",
                [],
            )
        )

        return tuple(
            self.normalize_text(value)
            for value in values
            if str(value).strip()
        )

    def _resolve_exact(
        self,
        *,
        text: str,
        aliases: dict[str, str],
    ) -> str | None:
        sorted_aliases = sorted(
            aliases.items(),
            key=lambda item: len(
                item[0]
            ),
            reverse=True,
        )

        for (
            alias,
            canonical,
        ) in sorted_aliases:
            if self._phrase_match(
                text=text,
                phrase=alias,
            ):
                return canonical

        return None

    def _resolve_fuzzy(
        self,
        *,
        text: str,
        config: EntityTypeConfig,
    ) -> str | None:
        words = re.findall(
            r"[^\W_]+",
            text,
            flags=re.UNICODE,
        )

        if not words:
            return None

        best_value: str | None = None
        best_score = 0.0

        for alias, canonical in config.aliases.items():
            if not self._is_fuzzy_candidate(alias):
                continue

            alias_words = re.findall(
                r"[^\W_]+",
                alias,
                flags=re.UNICODE,
            )

            if not alias_words:
                continue

            width = len(alias_words)

            if width > len(words):
                candidates = [" ".join(words)]
            else:
                candidates = [
                    " ".join(words[index:index + width])
                    for index in range(
                        0,
                        len(words) - width + 1,
                    )
                ]

            alias_text = " ".join(alias_words)

            for candidate_text in candidates:
                # Avoid aggressive fuzzy matching when lengths are wildly
                # different. This reduces accidental entity matches.
                if (
                    abs(len(candidate_text) - len(alias_text))
                    > max(3, int(len(alias_text) * 0.4))
                ):
                    continue

                score = SequenceMatcher(
                    None,
                    candidate_text.casefold(),
                    alias_text.casefold(),
                ).ratio()

                if score > best_score:
                    best_score = score
                    best_value = canonical

        if best_score >= config.fuzzy_threshold:
            return best_value

        return None

    @staticmethod
    def _is_fuzzy_candidate(
        alias: str,
    ) -> bool:
        compact = re.sub(
            r"[^\w]",
            "",
            alias,
            flags=re.UNICODE,
        )

        return len(compact) >= 3

    @staticmethod
    def _phrase_match(
        *,
        text: str,
        phrase: str,
    ) -> bool:
        if not phrase:
            return False

        if " " in phrase:
            return phrase in text

        pattern = (
            rf"(?<!\w)"
            rf"{re.escape(phrase)}"
            rf"(?!\w)"
        )

        return (
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            is not None
        )
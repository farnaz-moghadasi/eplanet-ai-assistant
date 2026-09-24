from __future__ import annotations

from dataclasses import dataclass, field

from app.conversation.schemas import (
    ConversationUnderstanding,
)
from app.conversation.state import (
    ConversationState,
)
from app.retrieval.document import (
    KnowledgeChunk,
)
from app.retrieval.schemas import (
    KnowledgeCandidate,
)


@dataclass(frozen=True)
class ContextRequirementEntry:
    knowledge_key: str
    canonical_knowledge_key: str

    category: str
    knowledge_type: str

    language: str | None

    audience: str
    bot_mode: str
    market_scope: str

    required_context_fields: tuple[
        str,
        ...,
    ]

    title: str | None = None


@dataclass
class ContextRequirementDecision:
    required: bool

    missing_fields: list[str] = field(
        default_factory=list
    )

    required_fields: list[str] = field(
        default_factory=list
    )

    matched_knowledge_keys: list[str] = field(
        default_factory=list
    )

    reason: str | None = None


class ContextRequirementIndex:
    def __init__(
        self,
    ) -> None:
        self._entries: dict[
            str,
            ContextRequirementEntry,
        ] = {}

    def add_candidate(
        self,
        candidate: KnowledgeCandidate,
    ) -> None:
        self._add_metadata(
            metadata=candidate.metadata,
            title=candidate.title,
        )

    def add_candidates(
        self,
        candidates: list[
            KnowledgeCandidate
        ],
    ) -> None:
        for candidate in candidates:
            self.add_candidate(
                candidate
            )

    def add_chunk(
        self,
        chunk: KnowledgeChunk,
    ) -> None:
        self._add_metadata(
            metadata=chunk.metadata,
            title=chunk.title,
        )

    def add_chunks(
        self,
        chunks: list[
            KnowledgeChunk
        ],
    ) -> None:
        for chunk in chunks:
            self.add_chunk(
                chunk
            )

    @classmethod
    def from_candidates(
        cls,
        candidates: list[
            KnowledgeCandidate
        ],
    ) -> "ContextRequirementIndex":
        index = cls()

        index.add_candidates(
            candidates
        )

        return index

    @classmethod
    def from_chunks(
        cls,
        chunks: list[
            KnowledgeChunk
        ],
    ) -> "ContextRequirementIndex":
        index = cls()

        index.add_chunks(
            chunks
        )

        return index

    def entries(
        self,
    ) -> list[
        ContextRequirementEntry
    ]:
        return list(
            self._entries.values()
        )

    def __len__(
        self,
    ) -> int:
        return len(
            self._entries
        )

    def _add_metadata(
        self,
        *,
        metadata,
        title: str | None,
    ) -> None:
        required_fields = tuple(
            field_name.strip()
            for field_name
            in metadata.required_context_fields
            if field_name
            and field_name.strip()
        )

        if not required_fields:
            return

        entry = ContextRequirementEntry(
            knowledge_key=(
                metadata.knowledge_key
            ),
            canonical_knowledge_key=(
                metadata.canonical_knowledge_key
            ),
            category=(
                metadata.category
            ),
            knowledge_type=(
                metadata.knowledge_type
            ),
            language=(
                metadata.language.value
                if metadata.language
                else None
            ),
            audience=(
                metadata.audience.value
            ),
            bot_mode=(
                metadata.bot_mode.value
            ),
            market_scope=(
                metadata.market_scope.value
            ),
            required_context_fields=(
                required_fields
            ),
            title=title,
        )

        self._entries[
            metadata.knowledge_key
        ] = entry


class ContextRequirementService:
    def __init__(
        self,
        index: ContextRequirementIndex,
    ):
        self.index = index

    def decide(
        self,
        *,
        understanding: (
            ConversationUnderstanding
        ),
        state: ConversationState,
        language: str | None = None,
    ) -> ContextRequirementDecision:
        matching = self._matching_entries(
            understanding=understanding,
            language=language,
        )

        if not matching:
            return (
                ContextRequirementDecision(
                    required=False
                )
            )

        required_fields = (
            self._collect_required_fields(
                matching
            )
        )

        if not required_fields:
            return (
                ContextRequirementDecision(
                    required=False
                )
            )

        missing_fields = [
            field_name
            for field_name
            in required_fields
            if not self._has_context_value(
                state=state,
                field_name=field_name,
            )
        ]

        matched_keys = [
            entry.knowledge_key
            for entry in matching
        ]

        if not missing_fields:
            return (
                ContextRequirementDecision(
                    required=False,
                    required_fields=(
                        required_fields
                    ),
                    matched_knowledge_keys=(
                        matched_keys
                    ),
                )
            )

        return ContextRequirementDecision(
            required=True,
            missing_fields=(
                missing_fields
            ),
            required_fields=(
                required_fields
            ),
            matched_knowledge_keys=(
                matched_keys
            ),
            reason=(
                "MISSING_REQUIRED_CONTEXT"
            ),
        )

    def _matching_entries(
        self,
        *,
        understanding: (
            ConversationUnderstanding
        ),
        language: str | None,
    ) -> list[
        ContextRequirementEntry
    ]:
        domain = self._normalize(
            understanding.domain.value
        )

        task = self._normalize(
            understanding.task.value
        )

        if domain in {
            "",
            "UNKNOWN",
            "GENERAL",
        }:
            return []

        normalized_language = (
            self._normalize_language(
                language
                or understanding.language
            )
        )

        category_matches: list[
            ContextRequirementEntry
        ] = []

        task_matches: list[
            ContextRequirementEntry
        ] = []

        for entry in self.index.entries():
            if (
                normalized_language
                and entry.language
                and self._normalize_language(
                    entry.language
                )
                != normalized_language
            ):
                continue

            if not self._category_matches(
                entry.category,
                domain,
            ):
                continue

            if (
                not understanding
                .client_specific
                and self._is_lookup_entry(
                    entry
                )
            ):
                continue

            category_matches.append(
                entry
            )

            if self._knowledge_type_matches(
                entry.knowledge_type,
                task,
            ):
                task_matches.append(
                    entry
                )

        if task_matches:
            return task_matches

        return category_matches

    @staticmethod
    def _collect_required_fields(
        entries: list[
            ContextRequirementEntry
        ],
    ) -> list[str]:
        result: list[str] = []

        seen: set[str] = set()

        for entry in entries:
            for field_name in (
                entry.required_context_fields
            ):
                normalized = (
                    field_name.strip()
                )

                if not normalized:
                    continue

                key = (
                    normalized.casefold()
                )

                if key in seen:
                    continue

                seen.add(key)

                result.append(
                    normalized
                )

        return result

    @staticmethod
    def _has_context_value(
        *,
        state: ConversationState,
        field_name: str,
    ) -> bool:
        if field_name == "client_id":
            return bool(
                state.client_id
            )

        value = (
            state.get_context_field(
                field_name
            )
        )

        return bool(
            value
            and value.strip()
        )

    @classmethod
    def _category_matches(
        cls,
        category: str,
        domain: str,
    ) -> bool:
        category = cls._normalize(
            category
        )

        if not category:
            return False

        if category == domain:
            return True

        category_tokens = set(
            category
            .replace("-", " ")
            .replace("_", " ")
            .split()
        )

        domain_tokens = set(
            domain
            .replace("-", " ")
            .replace("_", " ")
            .split()
        )

        return bool(
            category_tokens
            & domain_tokens
        )

    @classmethod
    def _knowledge_type_matches(
        cls,
        knowledge_type: str,
        task: str,
    ) -> bool:
        knowledge_type = cls._normalize(
            knowledge_type
        )

        if not knowledge_type:
            return False

        if (
            not task
            or task == "UNKNOWN"
        ):
            return False

        if knowledge_type == task:
            return True

        type_tokens = set(
            knowledge_type
            .replace("-", " ")
            .replace("_", " ")
            .split()
        )

        task_tokens = set(
            task
            .replace("-", " ")
            .replace("_", " ")
            .split()
        )

        return bool(
            type_tokens
            & task_tokens
        )

    @staticmethod
    def _is_lookup_entry(
        entry: ContextRequirementEntry,
    ) -> bool:
        return (
            entry.bot_mode
            .strip()
            .upper()
            == "LOOKUP"
        )

    @staticmethod
    def _normalize(
        value: str | None,
    ) -> str:
        return (
            str(value or "")
            .strip()
            .upper()
        )

    @staticmethod
    def _normalize_language(
        value: str | None,
    ) -> str:
        return (
            str(value or "")
            .strip()
            .upper()
        )
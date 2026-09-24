from __future__ import annotations

import re

from app.ai.base import StructuredLLM
from app.conversation.prompts import (
    CONVERSATION_UNDERSTANDING_SYSTEM_PROMPT,
    build_conversation_understanding_prompt,
)
from app.conversation.schemas import (
    ConversationDomain,
    ConversationTask,
    ConversationUnderstanding,
    TopicRelation,
)
from app.conversation.state import (
    ConversationState,
    state_to_prompt_text,
)
from app.core.performance import (
    record_value,
)
from app.runtime.entity_registry import (
    EntityRegistry,
)


class ConversationUnderstandingService:
    _FA_HOW_TO_MARKERS = (
        "چگونه",
        "چطور",
        "چجوری",
        "چگونه می",
        "چطور می",
    )

    _EN_HOW_TO_MARKERS = (
        "how do i",
        "how can i",
        "how to",
        "how do you",
    )

    _DOWNLOAD_MARKERS = (
        "download",
        "دانلود",
        "دریافت",
    )

    _INSTALL_MARKERS = (
        "install",
        "installation",
        "نصب",
    )

    _INDICATOR_MARKERS = (
        "indicator",
        "indicators",
        "اندیکاتور",
    )

    _CLIENT_SPECIFIC_MARKERS = (
        "my account",
        "my balance",
        "my kyc",
        "my deposit",
        "my withdrawal",
        "my profile",
        "my manager",
        "حساب من",
        "حسابم",
        "موجودی من",
        "موجودیم",
        "احراز هویت من",
        "kyc من",
        "واریز من",
        "واریزم",
        "برداشت من",
        "برداشتم",
    )

    _SENSITIVE_MARKERS = (
        "kyc",
        "احراز هویت",
        "verification",
        "deposit",
        "deposits",
        "واریز",
        "withdrawal",
        "withdrawals",
        "برداشت",
        "bonus",
        "bonuses",
        "بونوس",
        "promotion",
        "promotions",
    )

    def __init__(
        self,
        llm: StructuredLLM,
        *,
        entity_registry: (
            EntityRegistry
            | None
        ) = None,
    ):
        self.llm = llm
        self.entity_registry = (
            entity_registry
        )

    async def understand(
        self,
        *,
        message: str,
        state: ConversationState,
    ) -> ConversationUnderstanding:
        deterministic = (
            self
            ._try_deterministic_understanding(
                message=message,
                state=state,
            )
        )

        if deterministic is not None:
            record_value(
                "understanding_mode",
                "deterministic_low_risk",
            )

            return deterministic

        record_value(
            "understanding_mode",
            "llm",
        )

        state_text = state_to_prompt_text(
            state
        )

        user_prompt = (
            build_conversation_understanding_prompt(
                current_message=message,
                state_text=state_text,
            )
        )

        result = (
            await self.llm.generate_structured(
                system_prompt=(
                    CONVERSATION_UNDERSTANDING_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
                response_model=(
                    ConversationUnderstanding
                ),
            )
        )

        return result

    def _try_deterministic_understanding(
        self,
        *,
        message: str,
        state: ConversationState,
    ) -> ConversationUnderstanding | None:
        if self.entity_registry is None:
            return None

        original = str(
            message
            or ""
        ).strip()

        if not original:
            return None

        normalized = (
            self.entity_registry
            .normalize_text(
                original
            )
        )

        if not normalized:
            return None

        if state.awaiting_context_field:
            return None

        if (
            self._contains_any(
                normalized,
                self._SENSITIVE_MARKERS,
            )
        ):
            return None

        if (
            self._contains_any(
                normalized,
                self._CLIENT_SPECIFIC_MARKERS,
            )
        ):
            return None

        country = (
            self.entity_registry
            .resolve_entity(
                entity_type="country",
                text=normalized,
            )
        )

        if country is not None:
            return None

        platform = (
            self.entity_registry
            .resolve_entity(
                entity_type="platform",
                text=normalized,
            )
        )

        if platform is None:
            return None

        if self._looks_context_dependent(
            normalized=normalized,
            state=state,
        ):
            return None

        language = (
            self._detect_supported_language(
                normalized
            )
        )

        if language is None:
            return None

        task = (
            self._detect_platform_task(
                normalized
            )
        )

        if task is None:
            return None

        concepts = (
            self._build_retrieval_concepts(
                normalized=normalized,
                platform=platform,
                task=task,
            )
        )

        return ConversationUnderstanding(
            language=language,
            domain=(
                ConversationDomain.PLATFORM
            ),
            task=task,
            client_specific=False,
            topic_relation=(
                TopicRelation.NEW_TOPIC
            ),
            referenced_countries=[],
            referenced_entities=[
                platform
            ],
            requires_previous_context=False,
            standalone_query=original,
            retrieval_concepts=concepts,
            confidence=0.99,
            classification_summary=(
                "Deterministic low-risk "
                "platform request."
            ),
        )

    def _looks_context_dependent(
        self,
        *,
        normalized: str,
        state: ConversationState,
    ) -> bool:
        tokens = normalized.split()

        if not tokens:
            return True

        if (
            len(tokens)
            <= self.entity_registry
            .short_followup_max_tokens()
            and state.last_resolved_query
        ):
            return True

        correction_markers = (
            self.entity_registry
            .correction_markers()
        )

        if any(
            marker in normalized
            for marker in correction_markers
        ):
            return True

        followup_markers = (
            "what about",
            "and mt5",
            "and ctrader",
            "اون چطور",
            "این چطور",
            "برای mt5",
            "برای ctrader",
        )

        if any(
            marker in normalized
            for marker in followup_markers
        ):
            return True

        return False

    def _detect_supported_language(
        self,
        normalized: str,
    ) -> str | None:
        if (
            self._contains_any(
                normalized,
                self._FA_HOW_TO_MARKERS,
            )
            or re.search(
                r"[\u067e\u0686\u0698\u06af]",
                normalized,
            )
        ):
            return "fa"

        if self._contains_any(
            normalized,
            self._EN_HOW_TO_MARKERS,
        ):
            return "en"

        return None

    def _detect_platform_task(
        self,
        normalized: str,
    ) -> ConversationTask | None:
        is_how_to = (
            self._contains_any(
                normalized,
                self._FA_HOW_TO_MARKERS,
            )
            or self._contains_any(
                normalized,
                self._EN_HOW_TO_MARKERS,
            )
        )

        if not is_how_to:
            return None

        if (
            self._contains_any(
                normalized,
                self._DOWNLOAD_MARKERS,
            )
            or self._contains_any(
                normalized,
                self._INSTALL_MARKERS,
            )
            or self._contains_any(
                normalized,
                self._INDICATOR_MARKERS,
            )
        ):
            return (
                ConversationTask.HOW_TO
            )

        return None

    def _build_retrieval_concepts(
        self,
        *,
        normalized: str,
        platform: str,
        task: ConversationTask,
    ) -> list[str]:
        concepts: list[str] = [
            platform,
            task.value,
        ]

        if self._contains_any(
            normalized,
            self._DOWNLOAD_MARKERS,
        ):
            concepts.append(
                "download"
            )

        if self._contains_any(
            normalized,
            self._INSTALL_MARKERS,
        ):
            concepts.append(
                "installation"
            )

        if self._contains_any(
            normalized,
            self._INDICATOR_MARKERS,
        ):
            concepts.append(
                "indicator"
            )

        result: list[str] = []
        seen: set[str] = set()

        for value in concepts:
            normalized_value = (
                str(value).strip()
            )

            if not normalized_value:
                continue

            key = (
                normalized_value
                .casefold()
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            result.append(
                normalized_value
            )

        return result

    @staticmethod
    def _contains_any(
        text: str,
        markers: tuple[str, ...],
    ) -> bool:
        return any(
            marker in text
            for marker in markers
        )
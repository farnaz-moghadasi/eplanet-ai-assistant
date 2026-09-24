from __future__ import annotations

from dataclasses import dataclass

from app.ai.openai import (
    OpenAIStructuredLLM,
)
from app.citations.validator import (
    CitationValidator,
)
from app.conversation.clarification_service import (
    ClarificationGenerationService,
)
from app.conversation.redis_store import (
    RedisConversationStateStore,
)
from app.conversation.resolver import (
    ConversationResolver,
)
from app.conversation.service import (
    ConversationService,
)
from app.conversation.state import (
    ConversationState,
)
from app.conversation.understanding import (
    ConversationUnderstandingService,
)
from app.core.config import (
    settings,
)
from app.generation.service import (
    AnswerGenerationService,
)
from app.grounding.service import (
    GroundingValidationService,
)
from app.runtime.context_policy import (
    ContextPolicy,
)
from app.runtime.entity_registry import (
    EntityRegistry,
)

from scripts.test_conversation_real_kb import (
    build_pipeline,
)


@dataclass
class ApplicationRuntime:
    conversation_service: (
        ConversationService
    )

    retrieval_runtime: object

    state_store: (
        RedisConversationStateStore
    )

    async def get_conversation_state(
        self,
        *,
        conversation_id: str,
    ) -> ConversationState:
        normalized_id = (
            conversation_id.strip()
        )

        if not normalized_id:
            raise ValueError(
                "conversation_id cannot "
                "be empty."
            )

        state = (
            await self.state_store.get(
                normalized_id
            )
        )

        if state is not None:
            return state

        return ConversationState(
            conversation_id=(
                normalized_id
            )
        )

    async def save_conversation_state(
        self,
        *,
        state: ConversationState,
    ) -> None:
        await self.state_store.save(
            state
        )

    def conversation_lock(
        self,
        *,
        conversation_id: str,
    ):
        return self.state_store.lock(
            conversation_id
        )

    async def redis_ready(
        self,
    ) -> bool:
        return (
            await self
            .state_store
            .ping()
        )

    async def close(
        self,
    ) -> None:
        close_method = getattr(
            self.retrieval_runtime,
            "close",
            None,
        )

        if close_method is not None:
            result = close_method()

            if hasattr(
                result,
                "__await__",
            ):
                await result

        await self.state_store.close()


async def build_runtime() -> (
    ApplicationRuntime
):
    entity_registry = (
        EntityRegistry.from_file(
            settings
            .ENTITY_ALIASES_PATH
        )
    )

    resolver = (
        ConversationResolver(
            entity_registry=(
                entity_registry
            )
        )
    )

    structured_llm = (
        OpenAIStructuredLLM(
            api_key=(
                settings
                .OPENAI_API_KEY
            ),
            model=(
                settings
                .OPENAI_MODEL
            ),
        )
    )

    understanding_service = (
        ConversationUnderstandingService(
            llm=structured_llm,
            entity_registry=(
                entity_registry
            ),
        )
    )

    generation_service = (
        AnswerGenerationService(
            llm=structured_llm,
            max_candidates=5,
            max_content_chars=4000,
        )
    )

    clarification_service = (
        ClarificationGenerationService(
            llm=structured_llm
        )
    )

    citation_validator = (
        CitationValidator()
    )

    grounding_service = (
        GroundingValidationService(
            llm=structured_llm,
            max_content_chars=4000,
        )
    )

    retrieval_runtime = (
        await build_pipeline()
    )

    conversation_service = (
        ConversationService(
            understanding_service=(
                understanding_service
            ),
            resolver=resolver,
            retrieval_pipeline=(
                retrieval_runtime.pipeline
            ),
            context_policy=(
                ContextPolicy()
            ),
            context_requirement_service=(
                retrieval_runtime
                .context_requirement_service
            ),
            clarification_service=(
                clarification_service
            ),
            entity_registry=(
                entity_registry
            ),
            generation_service=(
                generation_service
            ),
            citation_validator=(
                citation_validator
            ),
            grounding_service=(
                grounding_service
            ),
        )
    )

    state_store = (
        RedisConversationStateStore(
            redis_url=(
                settings.REDIS_URL
            ),
            ttl_seconds=(
                settings
                .CONVERSATION_STATE_TTL_SECONDS
            ),
        )
    )

    redis_ready = (
        await state_store.ping()
    )

    if not redis_ready:
        await retrieval_runtime.close()

        raise RuntimeError(
            "Redis is unavailable."
        )

    return ApplicationRuntime(
        conversation_service=(
            conversation_service
        ),
        retrieval_runtime=(
            retrieval_runtime
        ),
        state_store=(
            state_store
        ),
    )
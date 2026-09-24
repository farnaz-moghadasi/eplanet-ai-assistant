from __future__ import annotations

import asyncio

from app.ai.openai import (
    OpenAIStructuredLLM,
)
from app.citations.validator import (
    CitationValidator,
)
from app.conversation.resolver import (
    ConversationResolver,
)
from app.conversation.service import (
    ConversationService,
    TrustedSessionContext,
)
from app.conversation.state import (
    ConversationState,
)
from app.conversation.understanding import (
    ConversationUnderstandingService,
)
from app.core.config import settings
from app.core.logging import (
    configure_logging,
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
from scripts.test_pipeline_real_kb import (
    build_pipeline,
)


def clarification_message(
    field_name: str | None,
    language: str | None,
) -> str:
    language = (
        language
        or "FA"
    ).upper()

    messages = {
        "FA": {
            "nationality": (
                "تابعیت شما مربوط به کدام کشور است؟"
            ),
            "country_of_residence": (
                "در حال حاضر در کدام کشور "
                "زندگی می‌کنید؟"
            ),
            "kyc_document_country": (
                "برای احراز هویت، مدارک مربوط "
                "به کدام کشور را دارید؟"
            ),
            "referenced_country": (
                "منظورتان کدام کشور است؟"
            ),
            "client_id": (
                "برای بررسی اطلاعات شخصی حساب، "
                "نیاز به ورود و احراز هویت دارید."
            ),
        },
        "EN": {
            "nationality": (
                "What is your nationality?"
            ),
            "country_of_residence": (
                "Which country do you currently "
                "live in?"
            ),
            "kyc_document_country": (
                "Which country's documents will "
                "you use for KYC?"
            ),
            "referenced_country": (
                "Which country are you referring to?"
            ),
            "client_id": (
                "You need to be authenticated "
                "before I can access account-specific "
                "information."
            ),
        },
        "AR": {
            "nationality": (
                "ما هي جنسيتك؟"
            ),
            "country_of_residence": (
                "في أي دولة تقيم حالياً؟"
            ),
            "kyc_document_country": (
                "وثائق أي دولة ستستخدم للتحقق "
                "من الهوية؟"
            ),
            "referenced_country": (
                "أي دولة تقصد؟"
            ),
            "client_id": (
                "يجب تسجيل الدخول للتحقق من "
                "المعلومات الخاصة بحسابك."
            ),
        },
    }

    language_messages = (
        messages.get(
            language,
            messages["FA"],
        )
    )

    return (
        language_messages.get(
            field_name,
            (
                "برای ادامه به اطلاعات بیشتری "
                "نیاز دارم."
            ),
        )
    )


def print_debug(
    result,
    state: ConversationState,
) -> None:
    retrieval = (
        result.retrieval_result
    )

    generation = (
        result.generation_result
    )

    top_key = None
    reranker_mode = None

    if retrieval is not None:
        reranker_mode = (
            retrieval.reranker_mode
        )

        if retrieval.final_candidates:
            top_key = (
                retrieval
                .final_candidates[0]
                .candidate
                .candidate
                .metadata
                .knowledge_key
            )

    print()
    print(
        "----- DEBUG -----"
    )

    print(
        "domain:",
        result.understanding.domain.value,
    )

    print(
        "task:",
        result.understanding.task.value,
    )

    print(
        "client_specific:",
        result.understanding.client_specific,
    )

    print(
        "resolved_query:",
        result.resolved_turn.resolved_query,
    )

    print(
        "reranker_mode:",
        reranker_mode,
    )

    print(
        "top_knowledge_key:",
        top_key,
    )

    print(
        "awaiting_context:",
        state.awaiting_context_field,
    )

    print(
        "nationality:",
        state.nationality,
    )

    print(
        "country_of_residence:",
        state.country_of_residence,
    )

    print(
        "kyc_document_country:",
        state.kyc_document_country,
    )

    print(
        "referenced_country:",
        state.referenced_country,
    )

    print(
        "authenticated:",
        state.is_authenticated,
    )

    print(
        "client_id:",
        (
            "***"
            if state.client_id
            else None
        ),
    )

    if generation is not None:
        print(
            "abstained:",
            generation.payload.abstained,
        )

        print(
            "citations:",
            [
                (
                    citation.knowledge_key
                )
                for citation
                in generation.payload.citations
            ],
        )

    print(
        "-----------------"
    )
    print()


def print_help() -> None:
    print()
    print(
        "Commands:"
    )

    print(
        "  /help"
        "              show commands"
    )

    print(
        "  /debug on"
        "          enable debug output"
    )

    print(
        "  /debug off"
        "         disable debug output"
    )

    print(
        "  /state"
        "             show conversation context"
    )

    print(
        "  /reset"
        "             reset conversation"
    )

    print(
        "  /auth CLIENT_ID"
        "    simulate authenticated user"
    )

    print(
        "  /logout"
        "            remove authentication"
    )

    print(
        "  /quit"
        "              exit"
    )

    print()


def print_state(
    state: ConversationState,
) -> None:
    print()
    print(
        "Conversation state:"
    )

    print(
        "  awaiting_context_field:",
        state.awaiting_context_field,
    )

    print(
        "  nationality:",
        state.nationality,
    )

    print(
        "  country_of_residence:",
        state.country_of_residence,
    )

    print(
        "  kyc_document_country:",
        state.kyc_document_country,
    )

    print(
        "  referenced_country:",
        state.referenced_country,
    )

    print(
        "  last_topic:",
        state.last_topic,
    )

    print(
        "  last_intent:",
        state.last_intent,
    )

    print(
        "  last_resolved_query:",
        state.last_resolved_query,
    )

    print(
        "  authenticated:",
        state.is_authenticated,
    )

    print(
        "  client_id_present:",
        bool(
            state.client_id
        ),
    )

    print()


async def build_service():
    entity_registry = (
        EntityRegistry.from_file(
            settings.ENTITY_ALIASES_PATH
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
                settings.OPENAI_API_KEY
            ),
            model=(
                settings.OPENAI_MODEL
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

    service = (
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

    return (
        service,
        retrieval_runtime,
    )


async def main() -> None:
    configure_logging(
        level=settings.LOG_LEVEL
    )

    (
        service,
        retrieval_runtime,
    ) = await build_service()

    state = ConversationState()

    authenticated_client_id: (
        str
        | None
    ) = None

    debug_enabled = False

    print()
    print(
        "=" * 70
    )
    print(
        "ePlanet AI Assistant - "
        "Interactive Knowledge Test"
    )
    print(
        "=" * 70
    )

    print(
        "Type naturally in FA / EN / AR."
    )

    print(
        "Use /help for commands."
    )

    print()

    try:
        while True:
            try:
                message = input(
                    "You > "
                ).strip()

            except (
                EOFError,
                KeyboardInterrupt,
            ):
                print()
                break

            if not message:
                continue

            if message.lower() in {
                "/quit",
                "/exit",
            }:
                break

            if message.lower() == "/help":
                print_help()
                continue

            if message.lower() == "/state":
                print_state(
                    state
                )
                continue

            if message.lower() == "/debug on":
                debug_enabled = True

                print(
                    "Debug enabled."
                )
                continue

            if message.lower() == "/debug off":
                debug_enabled = False

                print(
                    "Debug disabled."
                )
                continue

            if message.lower() == "/reset":
                state = (
                    ConversationState()
                )

                authenticated_client_id = (
                    None
                )

                print(
                    "Conversation reset."
                )

                print()
                continue

            if message.lower() == "/logout":
                authenticated_client_id = (
                    None
                )

                state.is_authenticated = (
                    False
                )

                state.client_id = None

                print(
                    "Logged out."
                )

                print()
                continue

            if (
                message
                .lower()
                .startswith(
                    "/auth "
                )
            ):
                client_id = (
                    message[
                        len(
                            "/auth "
                        )
                    :
                    ]
                    .strip()
                )

                if not client_id:
                    print(
                        "Usage: /auth CLIENT_ID"
                    )
                    continue

                authenticated_client_id = (
                    client_id
                )

                print(
                    "Authenticated test "
                    "session enabled."
                )

                print()
                continue

            trusted_session = None

            if authenticated_client_id:
                trusted_session = (
                    TrustedSessionContext(
                        is_authenticated=True,
                        client_id=(
                            authenticated_client_id
                        ),
                    )
                )

            try:
                result = (
                    await service
                    .handle_message(
                        message=message,
                        state=state,
                        trusted_session=(
                            trusted_session
                        ),
                    )
                )

            except Exception as exc:
                print()
                print(
                    "Bot > "
                    "Internal error while "
                    "processing the message."
                )

                print(
                    "Error:",
                    (
                        f"{exc.__class__.__name__}: "
                        f"{exc}"
                    ),
                )

                print()
                continue

            clarification_field = None

            if (
                result.context_requirement
                is not None
                and result
                .context_requirement
                .required
            ):
                missing_fields = (
                    result
                    .context_requirement
                    .missing_fields
                )

                if missing_fields:
                    clarification_field = (
                        missing_fields[0]
                    )

            if (
                clarification_field
                is None
                and result.clarification
                is not None
                and result
                .clarification
                .required
            ):
                clarification_field = (
                    result
                    .clarification
                    .field_name
                )

            if clarification_field:
                answer = (
                    clarification_message(
                        clarification_field,
                        result
                        .understanding
                        .language,
                    )
                )

                print()
                print(
                    f"Bot > {answer}"
                )
                print()

                if debug_enabled:
                    print_debug(
                        result,
                        state,
                    )

                continue

            generation = (
                result.generation_result
            )

            if generation is None:
                print()
                print(
                    "Bot > "
                    "پاسخی برای این سؤال "
                    "پیدا نشد."
                )
                print()

                if debug_enabled:
                    print_debug(
                        result,
                        state,
                    )

                continue

            payload = (
                generation.payload
            )

            if payload.abstained:
                print()
                print(
                    "Bot > "
                    "برای پاسخ مطمئن به این "
                    "سؤال اطلاعات کافی ندارم."
                )

                if payload.needs_handoff:
                    print(
                        "ممکن است نیاز باشد "
                        "موضوع توسط پشتیبانی "
                        "بررسی شود."
                    )

                print()

                if debug_enabled:
                    print_debug(
                        result,
                        state,
                    )

                continue

            answer = (
                payload.answer
                or ""
            ).strip()

            if not answer:
                print()
                print(
                    "Bot > "
                    "پاسخی تولید نشد."
                )
                print()

                if debug_enabled:
                    print_debug(
                        result,
                        state,
                    )

                continue

            print()
            print(
                "Bot >"
            )

            print(
                answer
            )

            if payload.citations:
                print()
                print(
                    "Sources:"
                )

                for citation in (
                    payload.citations
                ):
                    print(
                        "  - "
                        f"{citation.knowledge_key}"
                    )

            print()

            if debug_enabled:
                print_debug(
                    result,
                    state,
                )

    finally:
        await (
            retrieval_runtime
            .close()
        )

    print()
    print(
        "Bye."
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )
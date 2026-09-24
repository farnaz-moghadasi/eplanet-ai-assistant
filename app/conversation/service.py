from __future__ import annotations

import re
import time
import uuid
from dataclasses import (
    dataclass,
    field,
)

from app.core.performance import (
    record_timing,
    record_value,
)
from app.citations.validator import (
    CitationValidationResult,
    CitationValidator,
)
from app.conversation.clarification_service import (
    ClarificationGenerationService,
)
from app.conversation.resolver import (
    ConversationResolver,
)
from app.conversation.schemas import (
    ConversationDomain,
    ConversationTask,
    ConversationUnderstanding,
    TopicRelation,
)
from app.conversation.state import (
    ConversationState,
    ResolvedTurn,
)
from app.conversation.understanding import (
    ConversationUnderstandingService,
)
from app.core.logging import (
    bind_log_context,
    clear_log_context,
)
from app.core.observability import (
    log_event,
)
from app.generation.schemas import (
    GeneratedAnswerPayload,
)
from app.generation.service import (
    AnswerGenerationService,
    GenerationResult,
)
from app.grounding.schemas import (
    GroundingEvidence,
    GroundingValidationResult,
)
from app.grounding.service import (
    GroundingValidationService,
)
from app.retrieval.eligibility import (
    EligibilityContext,
)
from app.retrieval.pipeline import (
    KnowledgeRetrievalPipeline,
    RetrievalPipelineResult,
)
from app.runtime.context_policy import (
    ClarificationDecision,
    ContextPolicy,
)
from app.runtime.context_requirements import (
    ContextRequirementDecision,
    ContextRequirementService,
)
from app.runtime.entity_registry import (
    EntityRegistry,
)


@dataclass
class TrustedSessionContext:
    is_authenticated: bool = False
    client_id: str | None = None


@dataclass
class ConversationServiceResult:
    understanding: ConversationUnderstanding
    resolved_turn: ResolvedTurn

    retrieval_result: (
        RetrievalPipelineResult
        | None
    )

    clarification: (
        ClarificationDecision
        | None
    )

    context_requirement: (
        ContextRequirementDecision
        | None
    )

    generation_result: (
        GenerationResult
        | None
    )

    citation_validation_result: (
        CitationValidationResult
        | None
    )

    grounding_validation_result: (
        GroundingValidationResult
        | None
    )

    clarification_question: (
        str | None
    ) = None

    should_generate_answer: bool = False

    knowledge_hints: list[str] = field(
        default_factory=list
    )


class ConversationService:
    _DETERMINISTIC_GROUNDING_BLOCKED_CATEGORIES = {
        "KYC",
        "DEPOSIT",
        "DEPOSITS",
        "WITHDRAWAL",
        "WITHDRAWALS",
        "BONUS",
        "BONUSES",
        "PROMOTION",
        "PROMOTIONS",
    }

    _DETERMINISTIC_GROUNDING_BLOCKED_TYPES = {
        "POLICY_RULE",
    }

    _DETERMINISTIC_GROUNDING_MAX_ANSWER_CHARS = (
        2000
    )

    # Whole-message-only matches (never a substring match) for greetings
    # and conversational openers with no actual question yet. A message
    # like "سلام میخوام احراز هویت کنم" does NOT match these - only the
    # bare opener does - so any real question still goes through the full
    # pipeline untouched.
    _GREETING_PHRASES_FA = {
        "سلام",
        "سلام خوبی",
        "سلام خوبید",
        "سلام وقت بخیر",
        "درود",
        "درود بر شما",
        "صبح بخیر",
        "عصر بخیر",
        "شب بخیر",
        "سلام و عرض ادب",
        "سلام و احترام",
    }

    _GREETING_PHRASES_EN = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "good morning",
        "good afternoon",
        "good evening",
    }

    # Common Latin-script (Finglish) spellings of "سلام" - typed by
    # Persian speakers without a Persian keyboard, not English at all,
    # so these map to the FA response rather than the EN one.
    _GREETING_PHRASES_FA_LATIN = {
        "slm",
        "salam",
        "salaam",
    }

    _OPENER_PHRASES_FA = {
        "سوال دارم",
        "یه سوال دارم",
        "یک سوال دارم",
        "سوالی دارم",
        "میخوام یه سوال بپرسم",
        "میخوام سوال بپرسم",
        "میخواستم یه سوال بپرسم",
        "کمک میخوام",
        "به کمک نیاز دارم",
    }

    _OPENER_PHRASES_EN = {
        "i have a question",
        "i have a question for you",
        "can i ask a question",
        "can i ask you something",
        "i need help",
        "need help",
    }

    _GREETING_RESPONSE_FA = (
        "سلام! خوش اومدید 👋 چطور می‌تونم کمکتون کنم؟ "
        "می‌تونید مثلاً درباره‌ی احراز هویت، واریز و برداشت، یا "
        "پلتفرم‌های معاملاتی سؤال بپرسید."
    )

    _GREETING_RESPONSE_EN = (
        "Hello! Welcome. How can I help you today? You can ask me "
        "about KYC verification, deposits and withdrawals, or "
        "trading platforms, for example."
    )

    _OPENER_RESPONSE_FA = (
        "بله، حتماً! سؤالتون رو بپرسید، در خدمتتون هستم."
    )

    _OPENER_RESPONSE_EN = (
        "Sure, go ahead — I'm listening. What's your question?"
    )

    def __init__(
        self,
        *,
        understanding_service: (
            ConversationUnderstandingService
        ),
        resolver: ConversationResolver,
        retrieval_pipeline: (
            KnowledgeRetrievalPipeline
        ),
        context_policy: ContextPolicy,
        context_requirement_service: (
            ContextRequirementService
            | None
        ) = None,
        clarification_service: (
            ClarificationGenerationService
            | None
        ) = None,
        entity_registry: (
            EntityRegistry
            | None
        ) = None,
        generation_service: (
            AnswerGenerationService
            | None
        ) = None,
        citation_validator: (
            CitationValidator
            | None
        ) = None,
        grounding_service: (
            GroundingValidationService
            | None
        ) = None,
    ):
        self.understanding_service = (
            understanding_service
        )

        self.resolver = resolver

        self.retrieval_pipeline = (
            retrieval_pipeline
        )

        self.context_policy = (
            context_policy
        )

        self.context_requirement_service = (
            context_requirement_service
        )

        self.clarification_service = (
            clarification_service
        )

        self.entity_registry = (
            entity_registry
        )

        self.generation_service = (
            generation_service
        )

        self.citation_validator = (
            citation_validator
        )

        self.grounding_service = (
            grounding_service
        )

    async def handle_message(
        self,
        *,
        message: str,
        state: ConversationState,
        language: str | None = None,
        trusted_session: (
            TrustedSessionContext
            | None
        ) = None,
        knowledge_hints: (
            list[str]
            | None
        ) = None,
    ) -> ConversationServiceResult:
        request_id = uuid.uuid4().hex
        started_at = time.perf_counter()

        bind_log_context(
            request_id=request_id,
            conversation_id=(
                state.conversation_id
            ),
        )

        log_event(
            "chat.request.started",
            authenticated=(
                bool(
                    trusted_session
                    and trusted_session
                    .is_authenticated
                )
            ),
            message_length=len(
                message
            ),
        )

        try:
            self._apply_trusted_session(
                state=state,
                trusted_session=(
                    trusted_session
                ),
            )

            opener_match = (
                self
                ._match_conversational_opener(
                    message
                )
            )

            if opener_match is not None:
                return (
                    self
                    ._handle_conversational_opener(
                        message=message,
                        state=state,
                        language=language,
                        started_at=started_at,
                        answer=opener_match[0],
                        detected_language=(
                            opener_match[1]
                        ),
                    )
                )

            understanding_started = (
                time.perf_counter()
            )

            understanding = (
                await self
                .understanding_service
                .understand(
                    message=message,
                    state=state,
                )
            )

            understanding_duration_ms = (
                (
                    time.perf_counter()
                    - understanding_started
                )
                * 1000
            )

            record_timing(
                "understanding_ms",
                understanding_duration_ms,
            )

            log_event(
                "conversation.understanding.completed",
                language=(
                    understanding.language
                ),
                domain=(
                    understanding.domain.value
                ),
                task=(
                    understanding.task.value
                ),
                client_specific=(
                    understanding.client_specific
                ),
                confidence=(
                    understanding.confidence
                ),
                duration_ms=round(
                    understanding_duration_ms,
                    2,
                ),
            )

            effective_language = (
                understanding.language
                or language
                or state.language
            )

            self._apply_referenced_country(
                understanding=understanding,
                state=state,
            )

            deterministic_turn = (
                self.resolver.resolve(
                    message=message,
                    state=state,
                    language=(
                        effective_language
                    ),
                )
            )

            semantic_intent = (
                self._semantic_intent(
                    understanding
                )
            )

            semantic_topic = (
                understanding.domain.value
            )

            resolved_query = (
                self._resolve_retrieval_query(
                    understanding=(
                        understanding
                    ),
                    deterministic_turn=(
                        deterministic_turn
                    ),
                )
            )

            resolved_turn = (
                deterministic_turn.model_copy(
                    update={
                        "resolved_query":
                            resolved_query,
                        "intent":
                            semantic_intent,
                        "topic":
                            semantic_topic,
                    }
                )
            )

            state.last_intent = (
                semantic_intent
            )

            state.last_topic = (
                semantic_topic
            )

            state.last_resolved_query = (
                resolved_query
            )

            hints = (
                self._build_knowledge_hints(
                    understanding=(
                        understanding
                    ),
                    explicit_hints=(
                        knowledge_hints
                        or []
                    ),
                )
            )

            context_requirement_started = (
                time.perf_counter()
            )

            requirement_decision = (
                self
                ._evaluate_context_requirements(
                    understanding=(
                        understanding
                    ),
                    state=state,
                    language=(
                        effective_language
                    ),
                )
            )

            record_timing(
                "context_requirement_ms",
                (
                    time.perf_counter()
                    - context_requirement_started
                )
                * 1000,
            )

            if requirement_decision:
                log_event(
                    "context.requirement.evaluated",
                    required=(
                        requirement_decision
                        .required
                    ),
                    missing_fields=(
                        requirement_decision
                        .missing_fields
                    ),
                    reason=(
                        requirement_decision
                        .reason
                    ),
                )

            if (
                requirement_decision
                and requirement_decision.required
            ):
                first_missing = (
                    requirement_decision
                    .missing_fields[0]
                )

                state.set_pending_context(
                    first_missing
                )

                resolved_turn = (
                    resolved_turn.model_copy(
                        update={
                            "missing_context_fields":
                                (
                                    requirement_decision
                                    .missing_fields
                                ),
                            "needs_clarification":
                                True,
                            "clarification_reason":
                                (
                                    requirement_decision
                                    .reason
                                ),
                        }
                    )
                )

                clarification_question = (
                    await self
                    ._generate_clarification_question(
                        original_message=message,
                        missing_field=first_missing,
                        reason=(
                            requirement_decision
                            .reason
                        ),
                        understanding=understanding,
                        language=(
                            effective_language
                        ),
                    )
                )

                log_event(
                    "chat.request.clarification_required",
                    clarification_source=(
                        "context_requirement"
                    ),
                    field=first_missing,
                    total_duration_ms=round(
                        (
                            time.perf_counter()
                            - started_at
                        )
                        * 1000,
                        2,
                    ),
                )

                return ConversationServiceResult(
                    understanding=understanding,
                    resolved_turn=resolved_turn,
                    retrieval_result=None,
                    clarification=None,
                    context_requirement=(
                        requirement_decision
                    ),
                    generation_result=None,
                    citation_validation_result=None,
                    grounding_validation_result=None,
                    clarification_question=(
                        clarification_question
                    ),
                    should_generate_answer=False,
                    knowledge_hints=hints,
                )

            eligibility_context = (
                self
                ._build_eligibility_context(
                    state=state,
                    understanding=understanding,
                )
            )

            retrieval_started = (
                time.perf_counter()
            )

            retrieval_result = (
                await self
                .retrieval_pipeline
                .retrieve(
                    query=resolved_query,
                    context=(
                        eligibility_context
                    ),
                    language=(
                        effective_language
                    ),
                    knowledge_hints=hints,
                )
            )

            conversation_retrieval_duration_ms = (
                (
                    time.perf_counter()
                    - retrieval_started
                )
                * 1000
            )

            record_timing(
                "conversation_retrieval_ms",
                conversation_retrieval_duration_ms,
            )

            log_event(
                "conversation.retrieval.completed",
                bm25_candidates=len(
                    retrieval_result
                    .bm25_candidates
                ),
                vector_candidates=len(
                    retrieval_result
                    .vector_candidates
                ),
                hybrid_candidates=len(
                    retrieval_result
                    .hybrid_candidates
                ),
                eligible_candidates=len(
                    retrieval_result
                    .eligible_candidates
                ),
                final_candidates=len(
                    retrieval_result
                    .final_candidates
                ),
                reranker_mode=(
                    retrieval_result
                    .reranker_mode
                ),
                duration_ms=round(
                    conversation_retrieval_duration_ms,
                    2,
                ),
            )

            clarification = (
                self.context_policy.decide(
                    retrieval_result=(
                        retrieval_result
                    )
                )
            )

            if (
                clarification.required
                and clarification.field_name
            ):
                state.set_pending_context(
                    clarification.field_name
                )

                resolved_turn = (
                    resolved_turn.model_copy(
                        update={
                            "missing_context_fields":
                                [
                                    clarification
                                    .field_name
                                ],
                            "needs_clarification":
                                True,
                            "clarification_reason":
                                clarification
                                .reason,
                        }
                    )
                )

                clarification_question = (
                    await self
                    ._generate_clarification_question(
                        original_message=message,
                        missing_field=(
                            clarification
                            .field_name
                        ),
                        reason=(
                            clarification.reason
                        ),
                        understanding=understanding,
                        language=(
                            effective_language
                        ),
                    )
                )

                log_event(
                    "chat.request.clarification_required",
                    clarification_source=(
                        "retrieval_policy"
                    ),
                    field=(
                        clarification.field_name
                    ),
                    reason=(
                        clarification.reason
                    ),
                    total_duration_ms=round(
                        (
                            time.perf_counter()
                            - started_at
                        )
                        * 1000,
                        2,
                    ),
                )

                return ConversationServiceResult(
                    understanding=understanding,
                    resolved_turn=resolved_turn,
                    retrieval_result=(
                        retrieval_result
                    ),
                    clarification=clarification,
                    context_requirement=(
                        requirement_decision
                    ),
                    generation_result=None,
                    citation_validation_result=None,
                    grounding_validation_result=None,
                    clarification_question=(
                        clarification_question
                    ),
                    should_generate_answer=False,
                    knowledge_hints=hints,
                )

            should_generate = bool(
                retrieval_result
                .final_candidates
            )

            generation_result = None
            citation_validation_result = None
            grounding_validation_result = None

            if (
                should_generate
                and self.generation_service
                is not None
            ):
                use_deterministic_generation = (
                    self
                    ._can_use_deterministic_generation(
                        understanding=understanding,
                        retrieval_result=(
                            retrieval_result
                        ),
                    )
                )

                generation_mode = (
                    "deterministic_knowledge"
                    if use_deterministic_generation
                    else "llm"
                )

                record_value(
                    "generation_mode",
                    generation_mode,
                )

                generation_started = (
                    time.perf_counter()
                )

                if use_deterministic_generation:
                    generation_result = (
                        self
                        .generation_service
                        .generate_deterministic(
                            candidates=(
                                retrieval_result
                                .final_candidates
                            )
                        )
                    )
                else:
                    generation_result = (
                        await self
                        .generation_service
                        .generate(
                            query=resolved_query,
                            language=(
                                effective_language
                            ),
                            candidates=(
                                retrieval_result
                                .final_candidates
                            ),
                            context=(
                                self
                                ._build_generation_context(
                                    state=state,
                                    understanding=understanding,
                                )
                            ),
                        )
                    )

                generation_duration_ms = (
                    (
                        time.perf_counter()
                        - generation_started
                    )
                    * 1000
                )

                record_timing(
                    "generation_ms",
                    generation_duration_ms,
                )

                log_event(
                    "generation.completed",
                    abstained=(
                        generation_result
                        .payload
                        .abstained
                    ),
                    citation_count=len(
                        generation_result
                        .payload
                        .citations
                    ),
                    evidence_count=len(
                        generation_result
                        .evidence_candidate_ids
                    ),
                    answer_length=len(
                        generation_result
                        .payload
                        .answer
                    ),
                    duration_ms=round(
                        generation_duration_ms,
                        2,
                    ),
                )

                citation_validation_started = (
                    time.perf_counter()
                )

                (
                    generation_result,
                    citation_validation_result,
                ) = self._validate_citations(
                    generation_result
                )

                record_timing(
                    "citation_validation_ms",
                    (
                        time.perf_counter()
                        - citation_validation_started
                    )
                    * 1000,
                )

                if (
                    citation_validation_result
                    is not None
                ):
                    log_event(
                        "citation_validation.completed",
                        valid=(
                            citation_validation_result
                            .valid
                        ),
                        citation_count=len(
                            citation_validation_result
                            .citations
                        ),
                        issue_count=len(
                            citation_validation_result
                            .issues
                        ),
                    )

                if (
                    citation_validation_result
                    is not None
                    and not (
                        citation_validation_result
                        .valid
                    )
                ):
                    generation_result = (
                        self
                        ._fail_closed_generation(
                            generation_result,
                            reason=(
                                "CITATION_"
                                "VALIDATION_FAILED"
                            ),
                            language=(
                                effective_language
                            ),
                        )
                    )

                    log_event(
                        "generation.fail_closed",
                        reason=(
                            "CITATION_VALIDATION_FAILED"
                        ),
                    )

                elif (
                    self.grounding_service
                    is not None
                ):
                    grounding_started = (
                        time.perf_counter()
                    )

                    use_deterministic_grounding = (
                        self
                        ._can_use_deterministic_grounding(
                            understanding=(
                                understanding
                            ),
                            retrieval_result=(
                                retrieval_result
                            ),
                            generation_result=(
                                generation_result
                            ),
                            citation_validation_result=(
                                citation_validation_result
                            ),
                        )
                    )

                    if use_deterministic_grounding:
                        grounding_mode = (
                            "deterministic_low_risk"
                        )

                        grounding_validation_result = (
                            self
                            .grounding_service
                            .validate_deterministic(
                                answer=(
                                    generation_result
                                    .payload
                                    .answer
                                ),
                                citations=(
                                    generation_result
                                    .payload
                                    .citations
                                ),
                                evidence=(
                                    self
                                    ._build_grounding_evidence(
                                        retrieval_result
                                    )
                                ),
                                abstained=(
                                    generation_result
                                    .payload
                                    .abstained
                                ),
                            )
                        )

                    else:
                        grounding_mode = (
                            "llm"
                        )

                        grounding_validation_result = (
                            await self
                            .grounding_service
                            .validate(
                                answer=(
                                    generation_result
                                    .payload
                                    .answer
                                ),
                                citations=(
                                    generation_result
                                    .payload
                                    .citations
                                ),
                                evidence=(
                                    self
                                    ._build_grounding_evidence(
                                        retrieval_result
                                    )
                                ),
                                abstained=(
                                    generation_result
                                    .payload
                                    .abstained
                                ),
                            )
                        )

                    grounding_duration_ms = (
                        (
                            time.perf_counter()
                            - grounding_started
                        )
                        * 1000
                    )

                    record_timing(
                        "grounding_ms",
                        grounding_duration_ms,
                    )

                    record_value(
                        "grounding_mode",
                        grounding_mode,
                    )

                    unsupported_claims = sum(
                        1
                        for claim in (
                            grounding_validation_result
                            .claims
                        )
                        if not claim.supported
                    )

                    log_event(
                        "grounding.completed",
                        mode=grounding_mode,
                        valid=(
                            grounding_validation_result
                            .valid
                        ),
                        skipped=(
                            grounding_validation_result
                            .skipped
                        ),
                        reason=(
                            grounding_validation_result
                            .reason
                        ),
                        claim_count=len(
                            grounding_validation_result
                            .claims
                        ),
                        unsupported_claims=(
                            unsupported_claims
                        ),
                        issue_count=len(
                            grounding_validation_result
                            .issues
                        ),
                        duration_ms=round(
                            grounding_duration_ms,
                            2,
                        ),
                    )

                    if not (
                        grounding_validation_result
                        .valid
                    ):
                        generation_result = (
                            self
                            ._fail_closed_generation(
                                generation_result,
                                reason=(
                                    "GROUNDING_"
                                    "VALIDATION_FAILED"
                                ),
                                language=(
                                    effective_language
                                ),
                            )
                        )

                        log_event(
                            "generation.fail_closed",
                            reason=(
                                "GROUNDING_VALIDATION_FAILED"
                            ),
                        )

            elif not should_generate:
                # No eligible candidate survived retrieval/reranking
                # (including the semantic_min_score floor) - stay
                # honest instead of silently returning an empty
                # answer with no explanation.
                generation_result = (
                    self
                    ._build_no_candidates_result(
                        language=(
                            effective_language
                        )
                    )
                )

                record_value(
                    "generation_mode",
                    "no_eligible_knowledge",
                )

                log_event(
                    "generation.no_eligible_knowledge",
                    language=(
                        effective_language
                    ),
                )

            final_abstained = (
                generation_result
                is not None
                and generation_result
                .payload
                .abstained
            )

            top_knowledge_key = None
            top_score = None

            if (
                retrieval_result
                .final_candidates
            ):
                top_candidate = (
                    retrieval_result
                    .final_candidates[0]
                )

                top_knowledge_key = (
                    top_candidate
                    .candidate
                    .candidate
                    .metadata
                    .knowledge_key
                )

                top_score = (
                    top_candidate
                    .rerank_score
                )

            total_duration_ms = (
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000
            )

            record_timing(
                "total_ms",
                total_duration_ms,
            )

            log_event(
                "chat.request.completed",
                generated=(
                    generation_result
                    is not None
                ),
                abstained=(
                    final_abstained
                ),
                final_candidate_count=len(
                    retrieval_result
                    .final_candidates
                ),
                top_knowledge_key=(
                    top_knowledge_key
                ),
                top_rerank_score=(
                    top_score
                ),
                total_duration_ms=round(
                    total_duration_ms,
                    2,
                ),
            )

            return ConversationServiceResult(
                understanding=understanding,
                resolved_turn=resolved_turn,
                retrieval_result=(
                    retrieval_result
                ),
                clarification=None,
                context_requirement=(
                    requirement_decision
                ),
                generation_result=(
                    generation_result
                ),
                citation_validation_result=(
                    citation_validation_result
                ),
                grounding_validation_result=(
                    grounding_validation_result
                ),
                clarification_question=None,
                should_generate_answer=(
                    should_generate
                ),
                knowledge_hints=hints,
            )

        except Exception as exc:
            log_event(
                "chat.request.failed",
                level=40,
                error_type=(
                    exc.__class__.__name__
                ),
                total_duration_ms=round(
                    (
                        time.perf_counter()
                        - started_at
                    )
                    * 1000,
                    2,
                ),
            )

            raise

        finally:
            clear_log_context()

    @classmethod
    def _normalize_opener(
        cls,
        text: str,
    ) -> str:
        normalized = (
            str(text or "")
            .strip()
            .lower()
        )

        normalized = normalized.replace(
            "‌",
            "",
        )

        normalized = normalized.replace(
            "ي",
            "ی",
        )

        normalized = normalized.replace(
            "ك",
            "ک",
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()

        normalized = normalized.strip(
            "،!؟?.,؛;: "
        )

        return normalized.strip()

    @classmethod
    def _match_conversational_opener(
        cls,
        message: str,
    ) -> tuple[str, str] | None:
        normalized = cls._normalize_opener(
            message
        )

        if not normalized:
            return None

        if (
            normalized
            in cls._GREETING_PHRASES_FA
        ):
            return (
                cls._GREETING_RESPONSE_FA,
                "fa",
            )

        if (
            normalized
            in cls._GREETING_PHRASES_EN
        ):
            return (
                cls._GREETING_RESPONSE_EN,
                "en",
            )

        if (
            normalized
            in cls._GREETING_PHRASES_FA_LATIN
        ):
            return (
                cls._GREETING_RESPONSE_FA,
                "fa",
            )

        if (
            normalized
            in cls._OPENER_PHRASES_FA
        ):
            return (
                cls._OPENER_RESPONSE_FA,
                "fa",
            )

        if (
            normalized
            in cls._OPENER_PHRASES_EN
        ):
            return (
                cls._OPENER_RESPONSE_EN,
                "en",
            )

        return None

    def _handle_conversational_opener(
        self,
        *,
        message: str,
        state: ConversationState,
        language: str | None,
        started_at: float,
        answer: str,
        detected_language: str,
    ) -> ConversationServiceResult:
        resolved_turn = self.resolver.resolve(
            message=message,
            state=state,
            language=(
                detected_language
                or language
                or state.language
            ),
        )

        understanding = (
            ConversationUnderstanding(
                language=detected_language,
                domain=(
                    ConversationDomain.GENERAL
                ),
                task=(
                    ConversationTask.UNKNOWN
                ),
                client_specific=False,
                topic_relation=(
                    TopicRelation.NEW_TOPIC
                ),
                referenced_countries=[],
                referenced_entities=[],
                requires_previous_context=False,
                standalone_query=message,
                retrieval_concepts=[],
                confidence=0.99,
                classification_summary=(
                    "Deterministic greeting/"
                    "small-talk fast path."
                ),
            )
        )

        generation_result = GenerationResult(
            payload=GeneratedAnswerPayload(
                answer=answer,
                citations=[],
                abstained=False,
                abstain_reason=None,
                needs_handoff=False,
            ),
            evidence_candidate_ids=[],
            evidence_knowledge_keys=[],
            evidence_pairs=set(),
        )

        record_value(
            "understanding_mode",
            "deterministic_greeting",
        )

        record_value(
            "generation_mode",
            "deterministic_greeting",
        )

        total_duration_ms = (
            (
                time.perf_counter()
                - started_at
            )
            * 1000
        )

        record_timing(
            "total_ms",
            total_duration_ms,
        )

        log_event(
            "chat.request.completed",
            generated=True,
            abstained=False,
            final_candidate_count=0,
            top_knowledge_key=None,
            top_rerank_score=None,
            fast_path="greeting_opener",
            total_duration_ms=round(
                total_duration_ms,
                2,
            ),
        )

        return ConversationServiceResult(
            understanding=understanding,
            resolved_turn=resolved_turn,
            retrieval_result=None,
            clarification=None,
            context_requirement=None,
            generation_result=(
                generation_result
            ),
            citation_validation_result=None,
            grounding_validation_result=None,
            clarification_question=None,
            should_generate_answer=True,
            knowledge_hints=[],
        )

    async def _generate_clarification_question(
        self,
        *,
        original_message: str,
        missing_field: str,
        reason: str | None,
        understanding: (
            ConversationUnderstanding
        ),
        language: str | None,
    ) -> str | None:
        normalized_language = (
            language
            or understanding.language
            or "fa"
        ).lower()

        if missing_field == "kyc_document_country":
            record_value(
                "clarification_mode",
                "deterministic",
            )

            if normalized_language == "en":
                return (
                    "Which country is your "
                    "identity document from?"
                )

            if normalized_language == "ar":
                return (
                    "إلى أي دولة تنتمي وثيقة "
                    "الهوية الخاصة بك؟"
                )

            if normalized_language == "ku":
                return (
                    "بەڵگەنامەی ناسنامەکەت "
                    "هی کام وڵاتێکە؟"
                )

            return (
                "مدرک هویتی‌تان مربوط به "
                "کدام کشور است؟"
            )

        if self.clarification_service is None:
            return None

        started = time.perf_counter()

        result = (
            await self
            .clarification_service
            .generate(
                original_message=(
                    original_message
                ),
                missing_field=(
                    missing_field
                ),
                reason=reason,
                understanding=understanding,
                language=language,
            )
        )

        duration_ms = (
            (
                time.perf_counter()
                - started
            )
            * 1000
        )

        record_timing(
            "clarification_generation_ms",
            duration_ms,
        )

        record_value(
            "clarification_mode",
            "llm",
        )

        log_event(
            "clarification.generated",
            field=missing_field,
            duration_ms=round(
                duration_ms,
                2,
            ),
        )

        return (
            result.question
            .strip()
        )

    def _validate_citations(
        self,
        generation_result: GenerationResult,
    ) -> tuple[
        GenerationResult,
        CitationValidationResult | None,
    ]:
        if self.citation_validator is None:
            return (
                generation_result,
                None,
            )

        payload = (
            generation_result.payload
        )

        require_citation = (
            not payload.abstained
            and bool(
                generation_result
                .evidence_pairs
            )
            and bool(
                payload.answer.strip()
            )
        )

        validation = (
            self.citation_validator
            .validate(
                citations=(
                    payload.citations
                ),
                evidence_candidate_ids=(
                    generation_result
                    .evidence_candidate_ids
                ),
                evidence_knowledge_keys=(
                    generation_result
                    .evidence_knowledge_keys
                ),
                evidence_pairs=(
                    generation_result
                    .evidence_pairs
                ),
                require_citation=(
                    require_citation
                ),
            )
        )

        sanitized_payload = (
            payload.model_copy(
                update={
                    "citations":
                        validation.citations
                }
            )
        )

        sanitized_generation = (
            GenerationResult(
                payload=(
                    sanitized_payload
                ),
                evidence_candidate_ids=(
                    generation_result
                    .evidence_candidate_ids
                ),
                evidence_knowledge_keys=(
                    generation_result
                    .evidence_knowledge_keys
                ),
                evidence_pairs=(
                    generation_result
                    .evidence_pairs
                ),
            )
        )

        return (
            sanitized_generation,
            validation,
        )

    @classmethod
    def _fail_closed_generation(
        cls,
        generation_result: (
            GenerationResult
        ),
        *,
        reason: str,
        language: str | None = None,
    ) -> GenerationResult:
        payload = (
            generation_result
            .payload
            .model_copy(
                update={
                    "answer": (
                        cls
                        ._no_answer_fallback_text(
                            language
                        )
                    ),
                    "citations": [],
                    "abstained": True,
                    "abstain_reason": (
                        reason
                    ),
                    "needs_handoff": True,
                }
            )
        )

        return GenerationResult(
            payload=payload,
            evidence_candidate_ids=(
                generation_result
                .evidence_candidate_ids
            ),
            evidence_knowledge_keys=(
                generation_result
                .evidence_knowledge_keys
            ),
            evidence_pairs=(
                generation_result
                .evidence_pairs
            ),
        )

    _NO_ANSWER_FALLBACK_FA = (
        "متأسفم، برای این سؤال اطلاعات "
        "مطمئنی توی پایگاه دانش پیدا "
        "نکردم. می‌تونید سؤال رو "
        "واضح‌تر مطرح کنید یا با "
        "پشتیبانی تماس بگیرید."
    )

    _NO_ANSWER_FALLBACK_EN = (
        "I'm sorry, I couldn't find "
        "reliable information for this "
        "in our knowledge base. Please "
        "try rephrasing your question, "
        "or contact support for help."
    )

    _NO_ANSWER_FALLBACK_AR = (
        "عذرًا، لم أجد معلومات موثوقة "
        "حول هذا في قاعدة المعرفة. "
        "يمكنك إعادة صياغة سؤالك أو "
        "التواصل مع فريق الدعم."
    )

    _NO_ANSWER_FALLBACK_KU = (
        "ببورە، زانیاری دڵنیام لەم "
        "بارەیەوە لە بنکەی زانیاریدا "
        "نەدۆزیەوە. تکایە پرسیارەکەت "
        "بە شێوەیەکی تر بنووسە یان "
        "پەیوەندی بە پشتیبانی بکە."
    )

    @classmethod
    def _no_answer_fallback_text(
        cls,
        language: str | None,
    ) -> str:
        normalized = (
            str(language or "fa")
            .strip()
            .lower()
        )

        if normalized == "en":
            return cls._NO_ANSWER_FALLBACK_EN

        if normalized == "ar":
            return cls._NO_ANSWER_FALLBACK_AR

        if normalized == "ku":
            return cls._NO_ANSWER_FALLBACK_KU

        return cls._NO_ANSWER_FALLBACK_FA

    @classmethod
    def _build_no_candidates_result(
        cls,
        *,
        language: str | None,
    ) -> GenerationResult:
        payload = GeneratedAnswerPayload(
            answer=(
                cls
                ._no_answer_fallback_text(
                    language
                )
            ),
            citations=[],
            abstained=True,
            abstain_reason=(
                "NO_ELIGIBLE_KNOWLEDGE"
            ),
            needs_handoff=True,
        )

        return GenerationResult(
            payload=payload,
            evidence_candidate_ids=[],
            evidence_knowledge_keys=[],
            evidence_pairs=set(),
        )

    def _can_use_deterministic_grounding(
        self,
        *,
        understanding: (
            ConversationUnderstanding
        ),
        retrieval_result: (
            RetrievalPipelineResult
        ),
        generation_result: (
            GenerationResult
        ),
        citation_validation_result: (
            CitationValidationResult
            | None
        ),
    ) -> bool:
        if (
            citation_validation_result
            is None
            or not (
                citation_validation_result
                .valid
            )
        ):
            return False

        payload = (
            generation_result.payload
        )

        if payload.abstained:
            return False

        answer = (
            payload.answer
            or ""
        ).strip()

        if not answer:
            return False

        if (
            len(answer)
            > self
            ._DETERMINISTIC_GROUNDING_MAX_ANSWER_CHARS
        ):
            return False

        citations = (
            payload.citations
        )

        if len(citations) != 1:
            return False

        if (
            retrieval_result
            .reranker_mode
            != "deterministic_fast_path"
        ):
            return False

        if understanding.client_specific:
            return False

        if (
            understanding
            .referenced_countries
        ):
            return False

        cited_candidate_id = (
            citations[0]
            .candidate_id
        )

        cited_candidate = None

        for item in (
            retrieval_result
            .final_candidates
        ):
            candidate = (
                item
                .candidate
                .candidate
            )

            if (
                candidate.candidate_id
                == cited_candidate_id
            ):
                cited_candidate = (
                    candidate
                )
                break

        if cited_candidate is None:
            return False

        metadata = (
            cited_candidate.metadata
        )

        market_scope = (
            getattr(
                metadata.market_scope,
                "value",
                metadata.market_scope,
            )
        )

        if (
            str(market_scope)
            .strip()
            .upper()
            != "GLOBAL"
        ):
            return False

        if (
            metadata
            .required_context_fields
        ):
            return False

        audience = (
            getattr(
                metadata.audience,
                "value",
                metadata.audience,
            )
        )

        if (
            str(audience)
            .strip()
            .upper()
            not in {
                "PUBLIC",
                "BOTH",
            }
        ):
            return False

        bot_mode = (
            getattr(
                metadata.bot_mode,
                "value",
                metadata.bot_mode,
            )
        )

        if (
            str(bot_mode)
            .strip()
            .upper()
            != "KNOWLEDGE"
        ):
            return False

        category = (
            str(
                metadata.category
                or ""
            )
            .strip()
            .upper()
        )

        if (
            category
            in self
            ._DETERMINISTIC_GROUNDING_BLOCKED_CATEGORIES
        ):
            return False

        knowledge_type = (
            str(
                metadata.knowledge_type
                or ""
            )
            .strip()
            .upper()
        )

        if (
            knowledge_type
            in self
            ._DETERMINISTIC_GROUNDING_BLOCKED_TYPES
        ):
            return False

        return True

    @staticmethod
    def _build_grounding_evidence(
        retrieval_result: (
            RetrievalPipelineResult
        ),
    ) -> list[
        GroundingEvidence
    ]:
        evidence: list[
            GroundingEvidence
        ] = []

        for item in (
            retrieval_result
            .final_candidates
        ):
            candidate = (
                item
                .candidate
                .candidate
            )

            evidence.append(
                GroundingEvidence(
                    candidate_id=(
                        candidate
                        .candidate_id
                    ),
                    knowledge_key=(
                        candidate
                        .metadata
                        .knowledge_key
                    ),
                    title=(
                        candidate.title
                    ),
                    content=(
                        candidate.content
                        or ""
                    ),
                )
            )

        return evidence

    def _evaluate_context_requirements(
        self,
        *,
        understanding: (
            ConversationUnderstanding
        ),
        state: ConversationState,
        language: str | None,
    ) -> (
        ContextRequirementDecision
        | None
    ):
        if (
            self.context_requirement_service
            is None
        ):
            return None

        return (
            self.context_requirement_service
            .decide(
                understanding=(
                    understanding
                ),
                state=state,
                language=language,
            )
        )

    def _apply_referenced_country(
        self,
        *,
        understanding: (
            ConversationUnderstanding
        ),
        state: ConversationState,
    ) -> None:
        countries = [
            country.strip()
            for country
            in understanding
            .referenced_countries
            if country
            and country.strip()
        ]

        if len(countries) == 1:
            value = countries[0]

            if self.entity_registry:
                resolved = (
                    self.entity_registry
                    .resolve_entity(
                        entity_type=(
                            "country"
                        ),
                        text=value,
                    )
                )

                if resolved:
                    value = resolved

            state.set_context_field(
                field_name=(
                    "referenced_country"
                ),
                value=value,
            )

            return

        if (
            understanding.topic_relation
            == TopicRelation.NEW_TOPIC
        ):
            state.referenced_country = (
                None
            )

            state.context_values.pop(
                "referenced_country",
                None,
            )

    @staticmethod
    def _resolve_retrieval_query(
        *,
        understanding: (
            ConversationUnderstanding
        ),
        deterministic_turn: (
            ResolvedTurn
        ),
    ) -> str:
        resolved_context_field = (
            deterministic_turn
            .inherited_context
            .get(
                "resolved_context_field"
            )
        )

        if resolved_context_field:
            return (
                deterministic_turn
                .resolved_query
            )

        standalone_query = (
            understanding
            .standalone_query
        )

        if (
            standalone_query
            and standalone_query.strip()
        ):
            return (
                standalone_query.strip()
            )

        return (
            deterministic_turn
            .resolved_query
        )

    @staticmethod
    def _semantic_intent(
        understanding: (
            ConversationUnderstanding
        ),
    ) -> str:
        return (
            f"{understanding.domain.value}:"
            f"{understanding.task.value}"
        )

    @staticmethod
    def _build_knowledge_hints(
        *,
        understanding: (
            ConversationUnderstanding
        ),
        explicit_hints: list[str],
    ) -> list[str]:
        values: list[str] = []

        values.extend(
            explicit_hints
        )

        values.append(
            understanding.domain.value
        )

        values.append(
            understanding.task.value
        )

        values.extend(
            understanding
            .retrieval_concepts
        )

        values.extend(
            understanding
            .referenced_entities
        )

        values.extend(
            understanding
            .referenced_countries
        )

        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = (
                str(value).strip()
            )

            if not normalized:
                continue

            key = (
                normalized.casefold()
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            result.append(
                normalized
            )

        return result

    @staticmethod
    def _apply_trusted_session(
        *,
        state: ConversationState,
        trusted_session: (
            TrustedSessionContext
            | None
        ),
    ) -> None:
        if trusted_session is None:
            return

        state.is_authenticated = (
            trusted_session
            .is_authenticated
        )

        if not (
            trusted_session
            .is_authenticated
        ):
            state.client_id = None
            return

        client_id = (
            trusted_session.client_id
        )

        if client_id is None:
            state.client_id = None
            return

        client_id = (
            client_id.strip()
        )

        state.client_id = (
            client_id
            or None
        )

    @staticmethod
    def _build_eligibility_context(
        *,
        state: ConversationState,
        understanding: ConversationUnderstanding,
    ) -> EligibilityContext:
        # Deliberately NOT gated by understanding.domain: which context
        # field actually matters is already determined downstream, per
        # knowledge page, by that page's own required_context_fields
        # (app/retrieval/eligibility.py). Gating here too by a domain
        # classification (LLM output, or a test double) made this fragile
        # - a one-word pending-context answer like "افغانستان" can easily
        # get classified as GENERAL rather than KYC, silently dropping
        # kyc_document_country from the context even though it was just
        # correctly resolved onto state. Always pass through what state
        # actually holds; it's harmless for a field an unrelated domain's
        # pages never declare as required.
        return EligibilityContext(
            language=(
                understanding.language
                or state.language
            ),
            is_authenticated=(
                state.is_authenticated
            ),
            client_id=(
                state.client_id
            ),
            nationality=(
                state.get_context_field(
                    "nationality"
                )
            ),
            country_of_residence=(
                state.get_context_field(
                    "country_of_residence"
                )
            ),
            kyc_document_country=(
                state.get_context_field(
                    "kyc_document_country"
                )
            ),
            referenced_country=(
                state.get_context_field(
                    "referenced_country"
                )
            ),
        )

    @staticmethod
    def _build_generation_context(
        *,
        state: ConversationState,
        understanding: ConversationUnderstanding,
    ) -> dict[str, str]:
        domain = understanding.domain

        values: dict[str, str | None] = {}

        if domain == ConversationDomain.KYC:
            values = {
                "nationality":
                    state.get_context_field(
                        "nationality"
                    ),
                "kyc_document_country":
                    state.get_context_field(
                        "kyc_document_country"
                    ),
                "referenced_country":
                    state.get_context_field(
                        "referenced_country"
                    ),
            }

        elif domain in {
            ConversationDomain.DEPOSIT,
            ConversationDomain.WITHDRAWAL,
        }:
            values = {
                "country_of_residence":
                    state.get_context_field(
                        "country_of_residence"
                    ),
                "referenced_country":
                    state.get_context_field(
                        "referenced_country"
                    ),
            }

        return {
            key: str(value).strip()
            for key, value
            in values.items()
            if (
                value
                and str(value).strip()
            )
        }

    def _can_use_deterministic_generation(
        self,
        *,
        understanding: ConversationUnderstanding,
        retrieval_result: RetrievalPipelineResult,
    ) -> bool:
        if self.generation_service is None:
            return False

        candidates = (
            retrieval_result.final_candidates
        )

        if not candidates:
            return False

        if not (
            self.generation_service
            .can_generate_deterministically(
                candidates=candidates
            )
        ):
            return False

        # Troubleshooting questions often require matching the exact
        # problem, not simply returning a nearby approved article.
        # Route them through grounded LLM generation so adjacent tasks
        # such as password-change vs password-recovery are not conflated.
        if (
            understanding.task
            == ConversationTask.TROUBLESHOOTING
        ):
            return False

        top = (
            candidates[0]
            .candidate
            .candidate
        )

        content = (
            top.content
            or ""
        ).casefold()

        internal_markers = (
            "source-based procedure for review",
            "migration review notes",
            "internal review notes",
            "internal notes",
            "livechat source",
        )

        if any(
            marker in content
            for marker in internal_markers
        ):
            return False

        return True
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
from app.conversation.understanding import (
    CLIENT_SPECIFIC_MARKERS,
    HOW_TO_MARKERS,
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
        "Sure, go ahead  I'm listening. What's your question?"
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
                self._canonical_language(
                    understanding.language
                    or language
                    or state.language
                )
            )

            self._apply_referenced_country(
                understanding=understanding,
                state=state,
            )

            self._apply_self_country_correction(
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

            if (
                not state.is_authenticated
                and self._needs_client_data(
                    understanding=understanding,
                    message=message,
                )
            ):
                generation_result = (
                    self
                    ._build_client_data_unavailable_result(
                        language=(
                            effective_language
                        )
                    )
                )

                log_event(
                    "chat.request.client_data_unavailable",
                )

                return ConversationServiceResult(
                    understanding=understanding,
                    resolved_turn=(
                        deterministic_turn
                    ),
                    retrieval_result=None,
                    clarification=None,
                    context_requirement=None,
                    generation_result=(
                        generation_result
                    ),
                    citation_validation_result=None,
                    grounding_validation_result=None,
                    clarification_question=None,
                    should_generate_answer=False,
                    knowledge_hints=[],
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

            if retrieval_result.distinct_options:
                state.set_pending_context(
                    "disambiguation"
                )

                state.pending_disambiguation_options = (
                    retrieval_result
                    .distinct_options
                )

                state.pending_disambiguation_query = (
                    resolved_query
                )

                disambiguation_question = (
                    self
                    ._build_disambiguation_question(
                        options=(
                            retrieval_result
                            .distinct_options
                        ),
                        language=(
                            effective_language
                        ),
                    )
                )

                log_event(
                    "chat.request.clarification_required",
                    clarification_source=(
                        "disambiguation"
                    ),
                    options=(
                        retrieval_result
                        .distinct_options
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
                    clarification=None,
                    context_requirement=None,
                    generation_result=None,
                    citation_validation_result=None,
                    grounding_validation_result=None,
                    clarification_question=(
                        disambiguation_question
                    ),
                    should_generate_answer=False,
                    knowledge_hints=hints,
                )

            clarification = (
                self.context_policy.decide(
                    retrieval_result=(
                        retrieval_result
                    ),
                    understanding=(
                        understanding
                    ),
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
                advice = (
                    self._is_advice_request(
                        understanding=(
                            understanding
                        ),
                    )
                )

                out_of_scope = (
                    not advice
                    and self._is_out_of_scope(
                        understanding=(
                            understanding
                        ),
                        retrieval_result=(
                            retrieval_result
                        ),
                    )
                )

                generation_result = (
                    self
                    ._build_no_candidates_result(
                        language=(
                            effective_language
                        ),
                        out_of_scope=(
                            out_of_scope
                        ),
                        advice=advice,
                    )
                )

                record_value(
                    "generation_mode",
                    (
                        "investment_advice"
                        if advice
                        else "out_of_scope"
                        if out_of_scope
                        else "no_eligible_knowledge"
                    ),
                )

                log_event(
                    "generation.no_eligible_knowledge",
                    language=(
                        effective_language
                    ),
                    out_of_scope=out_of_scope,
                    advice_request=advice,
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

    _LANGUAGE_ALIASES = {
        "english": "en",
        "persian": "fa",
        "farsi": "fa",
        "\u0641\u0627\u0631\u0633\u06cc": "fa",
        "arabic": "ar",
        "\u0627\u0644\u0639\u0631\u0628\u064a\u0629": "ar",
        "kurdish": "ku",
        "sorani": "ku",
        "ckb": "ku",
        "\u06a9\u0648\u0631\u062f\u06cc": "ku",
    }

    @classmethod
    def _canonical_language(
        cls,
        language: str | None,
    ) -> str | None:
        # The understanding model returns the language inconsistently
        # ("fa", "FA", "English", "Persian (Farsi)", ...). Retrieval
        # filters compare it to the pages' language codes, so a name
        # like "English" silently matched no page at all and the user
        # got "nothing found". Map every spelling to the code.
        if not language or not str(language).strip():
            return None

        normalized = (
            str(language)
            .strip()
            .lower()
        )

        if normalized in cls._LANGUAGE_ALIASES:
            return cls._LANGUAGE_ALIASES[
                normalized
            ]

        first_word = re.split(
            r"[^\w]+",
            normalized,
        )[0]

        return cls._LANGUAGE_ALIASES.get(
            first_word,
            normalized,
        )

    @classmethod
    def _normalize_language_code(
        cls,
        language: str | None,
    ) -> str:
        # Fallback messages default to Persian when nothing is known.
        return (
            cls._canonical_language(language)
            or "fa"
        )

    @classmethod
    def _no_answer_fallback_text(
        cls,
        language: str | None,
    ) -> str:
        normalized = (
            cls._normalize_language_code(
                language
            )
        )

        if normalized == "en":
            return cls._NO_ANSWER_FALLBACK_EN

        if normalized == "ar":
            return cls._NO_ANSWER_FALLBACK_AR

        if normalized == "ku":
            return cls._NO_ANSWER_FALLBACK_KU

        return cls._NO_ANSWER_FALLBACK_FA

    # Shown instead of the generic "couldn't find reliable information"
    # message when the question is clearly outside what this assistant
    # covers (weather, jokes, capitals, investment advice, ...): telling
    # the user what it *can* help with is more useful than implying the
    # knowledge base simply lacks an answer they might rephrase into.
    _OUT_OF_SCOPE_FA = (
        "این سؤال خارج از حوزهٔ کاری من هست. من فقط دربارهٔ خدمات "
        "eplanet کمک می‌کنم؛ مثلاً احراز هویت (KYC)، واریز و برداشت، "
        "حساب‌های معاملاتی و پلتفرم‌های MT5 و cTrader. اگه سؤالی در "
        "این زمینه‌ها دارید، بپرسید."
    )

    _OUT_OF_SCOPE_EN = (
        "That's outside what I can help with. I can answer questions "
        "about eplanet services, such as KYC verification, deposits and "
        "withdrawals, trading accounts, and the MT5 and cTrader "
        "platforms. Feel free to ask me about any of those."
    )

    _OUT_OF_SCOPE_AR = (
        "هذا السؤال خارج نطاق ما أستطيع المساعدة فيه. يمكنني الإجابة "
        "عن الأسئلة المتعلقة بخدمات eplanet، مثل توثيق الهوية (KYC) "
        "والإيداع والسحب وحسابات التداول ومنصتي MT5 وcTrader. "
        "تفضّل بسؤالي عن أي منها."
    )

    _OUT_OF_SCOPE_KU = (
        "ئەم پرسیارە لە دەرەوەی بواری کارەکانی منە. من تەنها لەبارەی "
        "خزمەتگوزارییەکانی eplanet یارمەتی دەدەم، وەک پشتڕاستکردنەوەی "
        "ناسنامە (KYC)، خستنەسەر و دەرهێنانی پارە، هەژمارەکانی "
        "مامەڵەکردن و پلاتفۆرمەکانی MT5 و cTrader. ئەگەر پرسیارێکت "
        "لەم بوارانەدا هەیە، بپرسە."
    )

    # Shown when the user asks for a buy/sell/hold recommendation, a
    # signal or a price forecast. The assistant only explains the
    # service; it must not look like it simply failed to find the
    # "answer" in the knowledge base.
    _ADVICE_FA = (
        "من نمی‌تونم توصیهٔ سرمایه‌گذاری، سیگنال معاملاتی یا پیش‌بینی "
        "قیمت بدم. اما دربارهٔ حساب‌های معاملاتی، واریز و برداشت، "
        "احراز هویت (KYC) و پلتفرم‌های MT5 و cTrader خوشحال می‌شم "
        "کمکتون کنم."
    )

    _ADVICE_EN = (
        "I can't give investment advice, trading signals or price "
        "predictions. I'm happy to help with trading accounts, deposits "
        "and withdrawals, KYC verification, and the MT5 and cTrader "
        "platforms, though."
    )

    _ADVICE_AR = (
        "لا يمكنني تقديم نصائح استثمارية أو إشارات تداول أو توقعات "
        "للأسعار. لكن يسعدني مساعدتك في حسابات التداول والإيداع "
        "والسحب وتوثيق الهوية (KYC) ومنصتي MT5 وcTrader."
    )

    _ADVICE_KU = (
        "من ناتوانم ئامۆژگاری وەبەرهێنان، سیگناڵی مامەڵەکردن یان "
        "پێشبینی نرخ بدەم. بەڵام دڵخۆش دەبم یارمەتیت بدەم لەبارەی "
        "هەژمارەکانی مامەڵەکردن، خستنەسەر و دەرهێنانی پارە، "
        "پشتڕاستکردنەوەی ناسنامە (KYC) و پلاتفۆرمەکانی MT5 و cTrader."
    )

    @classmethod
    def _advice_text(
        cls,
        language: str | None,
    ) -> str:
        normalized = (
            cls._normalize_language_code(
                language
            )
        )

        if normalized == "en":
            return cls._ADVICE_EN

        if normalized == "ar":
            return cls._ADVICE_AR

        if normalized == "ku":
            return cls._ADVICE_KU

        return cls._ADVICE_FA

    @classmethod
    def _is_advice_request(
        cls,
        *,
        understanding: (
            ConversationUnderstanding
        ),
    ) -> bool:
        return (
            understanding.is_advice_request
            and understanding.domain.value
            in {"GENERAL", "UNKNOWN"}
        )

    # Top vector similarity seen for off-topic questions was 0.20-0.41;
    # real in-scope questions sit at or above ~0.42-0.50. Treated as out
    # of scope only when ALL of these hold: the understanding step
    # flagged it off-topic, its domain is GENERAL/UNKNOWN, and nothing in
    # the knowledge base is even vaguely similar. (Requiring the model's
    # flag avoids telling a customer that e.g. "what are your support
    # hours?" is outside my scope just because the KB has no page on it.)
    _OUT_OF_SCOPE_MAX_VECTOR_SCORE = 0.45

    @classmethod
    def _out_of_scope_text(
        cls,
        language: str | None,
    ) -> str:
        normalized = (
            cls._normalize_language_code(
                language
            )
        )

        if normalized == "en":
            return cls._OUT_OF_SCOPE_EN

        if normalized == "ar":
            return cls._OUT_OF_SCOPE_AR

        if normalized == "ku":
            return cls._OUT_OF_SCOPE_KU

        return cls._OUT_OF_SCOPE_FA

    @classmethod
    def _is_out_of_scope(
        cls,
        *,
        understanding: (
            ConversationUnderstanding
        ),
        retrieval_result: (
            RetrievalPipelineResult
        ),
    ) -> bool:
        if not understanding.is_off_topic:
            return False

        if understanding.domain.value not in {
            "GENERAL",
            "UNKNOWN",
        }:
            return False

        best_vector_score = max(
            (
                item.vector_score
                for item in (
                    retrieval_result
                    .hybrid_candidates
                )
                if item.vector_score
                is not None
            ),
            default=0.0,
        )

        return (
            best_vector_score
            < cls._OUT_OF_SCOPE_MAX_VECTOR_SCORE
        )

    # This project has no live integration to the client's actual
    # account data anywhere (balance, KYC status, transaction history,
    # account manager, ...) - is_authenticated/client_id only gate
    # which KNOWLEDGE PAGES are eligible, not a real data lookup. A
    # request that needs the user's own live data can never be
    # answered correctly by retrieval, no matter who is logged in: the
    # nearest KB match is at best a generic explanation of the concept
    # (e.g. "what balance means") that reads like an answer without
    # being one. Handled before retrieval runs at all, not as a
    # fallback after a misleading answer was already generated.
    _CLIENT_DATA_UNAVAILABLE_FA = (
        "من به اطلاعات شخصی و لحظه‌ای حساب شما (مثل موجودی، تراکنش‌ها، "
        "وضعیت احراز هویت یا مدیر حساب) دسترسی مستقیم ندارم. لطفاً از "
        "طریق پنل کاربری بررسی کنید یا با پشتیبانی تماس بگیرید."
    )

    _CLIENT_DATA_UNAVAILABLE_EN = (
        "I don't have direct access to your personal, real-time account "
        "data (such as balance, transactions, verification status, or "
        "your account manager). Please check the client portal or "
        "contact support for that."
    )

    _CLIENT_DATA_UNAVAILABLE_AR = (
        "ليس لدي وصول مباشر إلى بيانات حسابك الشخصية واللحظية (مثل "
        "الرصيد أو المعاملات أو حالة التحقق أو مدير الحساب). يرجى "
        "التحقق عبر بوابة العميل أو التواصل مع الدعم."
    )

    _CLIENT_DATA_UNAVAILABLE_KU = (
        "من ڕاستەوخۆ دەستم ناگات بە زانیاری کەسی و ڕاستەقینەی هەژمارەکەت "
        "(وەک باڵانس، مامەڵەکان، دۆخی پشتڕاستکردنەوە یان بەڕێوەبەری "
        "هەژمار). تکایە لە ڕێگەی پۆرتاڵی کڕیارەوە بزانە یان پەیوەندی بە "
        "پشتیبانی بکە."
    )

    @classmethod
    def _client_data_unavailable_text(
        cls,
        language: str | None,
    ) -> str:
        normalized = (
            cls._normalize_language_code(
                language
            )
        )

        if normalized == "en":
            return cls._CLIENT_DATA_UNAVAILABLE_EN

        if normalized == "ar":
            return cls._CLIENT_DATA_UNAVAILABLE_AR

        if normalized == "ku":
            return cls._CLIENT_DATA_UNAVAILABLE_KU

        return cls._CLIENT_DATA_UNAVAILABLE_FA

    @classmethod
    def _build_client_data_unavailable_result(
        cls,
        *,
        language: str | None,
    ) -> GenerationResult:
        payload = GeneratedAnswerPayload(
            answer=(
                cls
                ._client_data_unavailable_text(
                    language
                )
            ),
            citations=[],
            abstained=True,
            abstain_reason=(
                "CLIENT_DATA_NOT_AVAILABLE"
            ),
            needs_handoff=True,
        )

        return GenerationResult(
            payload=payload,
            evidence_candidate_ids=[],
            evidence_knowledge_keys=[],
            evidence_pairs=set(),
        )

    def _needs_client_data(
        self,
        *,
        understanding: (
            ConversationUnderstanding
        ),
        message: str,
    ) -> bool:
        if understanding.client_specific:
            return True

        # Deterministic backstop: the classification model does not
        # reliably set client_specific=True even for its own textbook
        # examples (e.g. "what is my account balance?"). Falling
        # through silently here means a generic KB page gets used to
        # answer a question about the user's own data, which reads
        # like a real answer without being one - worse than asking the
        # model again would be, so this is checked by keyword instead.
        #
        # But a personal-account keyword does not always mean "give me
        # my live data": "how do I verify my account?" or "how do I
        # change my account password?" both name "my account" while
        # asking for a procedure, which the knowledge base can answer
        # normally. Real classification confusion was observed on
        # client_specific specifically, not on HOW_TO - so a how-to
        # phrasing is trusted here to mean "this is procedural, not a
        # live-data request" even when a personal keyword also matches.
        if understanding.task.value == "HOW_TO":
            return False

        normalized = message

        if self.entity_registry:
            normalized = (
                self.entity_registry
                .normalize_text(message)
            )

        if any(
            marker in normalized
            for marker in HOW_TO_MARKERS
        ):
            return False

        return any(
            marker in normalized
            for marker in (
                CLIENT_SPECIFIC_MARKERS
            )
        )

    _DISAMBIGUATION_INTRO_FA = (
        "این سؤال می‌تونه به چند حالت مختلف مربوط باشه. منظورتون کدومه؟"
    )

    _DISAMBIGUATION_INTRO_EN = (
        "This could mean a couple of different things. Which one did "
        "you mean?"
    )

    _DISAMBIGUATION_INTRO_AR = (
        "قد يعني هذا أكثر من شيء. أيهما تقصد؟"
    )

    _DISAMBIGUATION_INTRO_KU = (
        "ئەمە دەکرێت چەند مانایەکی جیاواز هەبێت. کامیانت مەبەستە؟"
    )

    @classmethod
    def _disambiguation_intro_text(
        cls,
        language: str | None,
    ) -> str:
        normalized = (
            cls._normalize_language_code(
                language
            )
        )

        if normalized == "en":
            return cls._DISAMBIGUATION_INTRO_EN

        if normalized == "ar":
            return cls._DISAMBIGUATION_INTRO_AR

        if normalized == "ku":
            return cls._DISAMBIGUATION_INTRO_KU

        return cls._DISAMBIGUATION_INTRO_FA

    @classmethod
    def _build_disambiguation_question(
        cls,
        *,
        options: list[str],
        language: str | None,
    ) -> str:
        intro = (
            cls._disambiguation_intro_text(
                language
            )
        )

        numbered = "\n".join(
            f"{index}) {option}"
            for index, option in enumerate(
                options,
                start=1,
            )
        )

        return f"{intro}\n{numbered}"

    @classmethod
    def _build_no_candidates_result(
        cls,
        *,
        language: str | None,
        out_of_scope: bool = False,
        advice: bool = False,
    ) -> GenerationResult:
        if advice:
            payload = GeneratedAnswerPayload(
                answer=cls._advice_text(
                    language
                ),
                citations=[],
                abstained=True,
                abstain_reason=(
                    "INVESTMENT_ADVICE_NOT_SUPPORTED"
                ),
                needs_handoff=False,
            )

            return GenerationResult(
                payload=payload,
                evidence_candidate_ids=[],
                evidence_knowledge_keys=[],
                evidence_pairs=set(),
            )

        if out_of_scope:
            payload = GeneratedAnswerPayload(
                answer=(
                    cls._out_of_scope_text(
                        language
                    )
                ),
                citations=[],
                abstained=True,
                abstain_reason="OUT_OF_SCOPE",
                needs_handoff=False,
            )

            return GenerationResult(
                payload=payload,
                evidence_candidate_ids=[],
                evidence_knowledge_keys=[],
                evidence_pairs=set(),
            )

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

    @staticmethod
    def _answer_is_verbatim_cited_content(
        *,
        answer: str,
        cited_candidate_id: str,
        retrieval_result: (
            RetrievalPipelineResult
        ),
    ) -> bool:
        for item in (
            retrieval_result.final_candidates
        ):
            candidate = item.candidate.candidate

            if (
                candidate.candidate_id
                == cited_candidate_id
            ):
                content = (
                    candidate.content or ""
                ).strip()

                # The verbatim answer is the page as displayed (the Q&A
                # template's Question/Answer scaffolding removed), so
                # accept either form.
                return bool(content) and (
                    answer.strip()
                    in {
                        content,
                        AnswerGenerationService
                        .to_display_text(
                            content,
                            language=getattr(
                                candidate
                                .metadata
                                .language,
                                "value",
                                None,
                            ),
                        ),
                    }
                )

        return False

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

        # An answer that is literally the cited source text needs no LLM
        # grounding pass: every "claim" in it is the approved document
        # itself, so there is nothing the check could catch. Grounding
        # only verifies an answer against its cited evidence - it cannot
        # tell whether the *right* document was retrieved, so this holds
        # no matter which reranker path picked the document. (Before
        # this, a verbatim answer after the semantic reranker still
        # paid for a 4-19s LLM grounding call.)
        if (
            not understanding.client_specific
            and self._answer_is_verbatim_cited_content(
                answer=answer,
                cited_candidate_id=(
                    citations[0].candidate_id
                ),
                retrieval_result=(
                    retrieval_result
                ),
            )
        ):
            return True

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

    # Which stored "fact about the user" a corrected country should
    # overwrite, keyed by the classified domain of the correction
    # message itself - consistent with how every other context-field
    # decision in this file is domain-based (ContextRequirementService,
    # ContextPolicy). KYC content is scoped by the ID document's
    # country; deposit/withdrawal/account content by residence.
    _SELF_COUNTRY_FIELD_BY_DOMAIN = {
        "KYC": "kyc_document_country",
        "DEPOSIT": "country_of_residence",
        "WITHDRAWAL": "country_of_residence",
        "ACCOUNT": "country_of_residence",
    }

    def _apply_self_country_correction(
        self,
        *,
        understanding: (
            ConversationUnderstanding
        ),
        state: ConversationState,
    ) -> None:
        if not understanding.corrects_self_country:
            return

        countries = [
            country.strip()
            for country in (
                understanding
                .referenced_countries
            )
            if country and country.strip()
        ]

        if len(countries) != 1:
            return

        field_name = (
            self
            ._SELF_COUNTRY_FIELD_BY_DOMAIN
            .get(
                understanding.domain.value
            )
        )

        if field_name is None:
            return

        value = countries[0]

        if self.entity_registry:
            resolved = (
                self.entity_registry
                .resolve_entity(
                    entity_type="country",
                    text=value,
                )
            )

            if resolved:
                value = resolved

        state.set_context_field(
            field_name=field_name,
            value=value,
        )

        log_event(
            "context.self_country_corrected",
            field=field_name,
            value=value,
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
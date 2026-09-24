from enum import Enum

from pydantic import BaseModel, Field

from app.conversation.schemas import ConversationUnderstanding
from app.conversation.state import ConversationState
from app.conversation.understanding import (
    ConversationUnderstandingService,
)
from app.retrieval.base import KnowledgeRetriever
from app.retrieval.schemas import KnowledgeCandidate
from app.runtime.context import RuntimeContext
from app.runtime.decision import (
    RuntimeDecision,
    RuntimeDecisionState,
    evaluate_request_plan,
)
from app.runtime.eligibility import (
    EligibilityResult,
    EligibilityState,
    KnowledgeEligibilityService,
)
from app.runtime.planner import (
    ExecutionMode,
    RequestPlan,
    build_request_plan,
)


class PipelineOutcomeState(str, Enum):
    READY_FOR_GENERATION = "READY_FOR_GENERATION"

    READY_FOR_LOOKUP = "READY_FOR_LOOKUP"

    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"

    CONTEXT_REQUIRED = "CONTEXT_REQUIRED"

    HANDOFF_REQUIRED = "HANDOFF_REQUIRED"

    NO_ELIGIBLE_KNOWLEDGE = "NO_ELIGIBLE_KNOWLEDGE"

    RETRIEVAL_NOT_CONFIGURED = "RETRIEVAL_NOT_CONFIGURED"


class CandidateEvaluation(BaseModel):
    candidate: KnowledgeCandidate
    eligibility: EligibilityResult


class PipelineOutcome(BaseModel):
    state: PipelineOutcomeState

    allowed: bool

    reason: str

    missing_context: list[str] = Field(
        default_factory=list
    )


class RequestPipelineResult(BaseModel):
    understanding: ConversationUnderstanding

    plan: RequestPlan

    preflight_decision: RuntimeDecision

    candidate_evaluations: list[CandidateEvaluation] = Field(
        default_factory=list
    )

    eligible_knowledge: list[KnowledgeCandidate] = Field(
        default_factory=list
    )

    outcome: PipelineOutcome


class RequestPipeline:
    def __init__(
        self,
        understanding_service: ConversationUnderstandingService,
        retriever: KnowledgeRetriever | None = None,
    ):
        self.understanding_service = understanding_service
        self.retriever = retriever

        self.eligibility_service = (
            KnowledgeEligibilityService()
        )

    async def process(
        self,
        *,
        message: str,
        conversation_state: ConversationState,
        runtime_context: RuntimeContext,
    ) -> RequestPipelineResult:

        understanding = (
            await self.understanding_service.understand(
                message=message,
                state=conversation_state,
            )
        )

        plan = build_request_plan(
            understanding
        )

        preflight_decision = evaluate_request_plan(
            plan,
            runtime_context,
        )

        # ---------------------------------------------------------
        # 1. Preflight hard gate
        # ---------------------------------------------------------

        if (
            preflight_decision.state
            == RuntimeDecisionState.AUTHENTICATION_REQUIRED
        ):
            return RequestPipelineResult(
                understanding=understanding,
                plan=plan,
                preflight_decision=preflight_decision,
                outcome=PipelineOutcome(
                    state=(
                        PipelineOutcomeState
                        .AUTHENTICATION_REQUIRED
                    ),
                    allowed=False,
                    reason=preflight_decision.reason,
                    missing_context=(
                        preflight_decision.missing_context
                    ),
                ),
            )

        if (
            preflight_decision.state
            == RuntimeDecisionState.CONTEXT_REQUIRED
        ):
            return RequestPipelineResult(
                understanding=understanding,
                plan=plan,
                preflight_decision=preflight_decision,
                outcome=PipelineOutcome(
                    state=PipelineOutcomeState.CONTEXT_REQUIRED,
                    allowed=False,
                    reason=preflight_decision.reason,
                    missing_context=(
                        preflight_decision.missing_context
                    ),
                ),
            )

        # ---------------------------------------------------------
        # 2. Pure lookup request
        # ---------------------------------------------------------

        if (
            plan.mode == ExecutionMode.LOOKUP
            and not plan.requires_knowledge
        ):
            return RequestPipelineResult(
                understanding=understanding,
                plan=plan,
                preflight_decision=preflight_decision,
                outcome=PipelineOutcome(
                    state=PipelineOutcomeState.READY_FOR_LOOKUP,
                    allowed=True,
                    reason="LIVE_LOOKUP_REQUIRED",
                ),
            )

        # ---------------------------------------------------------
        # 3. Knowledge retrieval
        # ---------------------------------------------------------

        if plan.requires_knowledge:
            if self.retriever is None:
                return RequestPipelineResult(
                    understanding=understanding,
                    plan=plan,
                    preflight_decision=preflight_decision,
                    outcome=PipelineOutcome(
                        state=(
                            PipelineOutcomeState
                            .RETRIEVAL_NOT_CONFIGURED
                        ),
                        allowed=False,
                        reason="KNOWLEDGE_RETRIEVER_NOT_CONFIGURED",
                    ),
                )

            candidates = await self.retriever.retrieve(
                query=message,
                knowledge_hints=plan.knowledge_hints,
            )

            candidate_evaluations: list[
                CandidateEvaluation
            ] = []

            eligible_knowledge: list[
                KnowledgeCandidate
            ] = []

            for candidate in candidates:
                eligibility = (
                    self.eligibility_service.evaluate(
                        candidate.metadata,
                        runtime_context,
                    )
                )

                candidate_evaluations.append(
                    CandidateEvaluation(
                        candidate=candidate,
                        eligibility=eligibility,
                    )
                )

                if (
                    eligibility.state
                    == EligibilityState.ELIGIBLE
                    and eligibility.eligible
                ):
                    eligible_knowledge.append(candidate)

            # -----------------------------------------------------
            # 4. At least one valid knowledge candidate exists
            # -----------------------------------------------------

            if eligible_knowledge:
                if plan.requires_live_state:
                    final_state = (
                        PipelineOutcomeState.READY_FOR_LOOKUP
                    )
                    reason = (
                        "KNOWLEDGE_READY_LIVE_LOOKUP_REQUIRED"
                    )
                else:
                    final_state = (
                        PipelineOutcomeState
                        .READY_FOR_GENERATION
                    )
                    reason = "ELIGIBLE_KNOWLEDGE_FOUND"

                return RequestPipelineResult(
                    understanding=understanding,
                    plan=plan,
                    preflight_decision=preflight_decision,
                    candidate_evaluations=(
                        candidate_evaluations
                    ),
                    eligible_knowledge=eligible_knowledge,
                    outcome=PipelineOutcome(
                        state=final_state,
                        allowed=True,
                        reason=reason,
                    ),
                )

            # -----------------------------------------------------
            # 5. No eligible candidate:
            #    inspect why candidates were rejected
            # -----------------------------------------------------

            authentication_results = [
                item
                for item in candidate_evaluations
                if item.eligibility.state
                == EligibilityState.AUTHENTICATION_REQUIRED
            ]

            if authentication_results:
                missing = self._collect_missing_context(
                    authentication_results
                )

                return RequestPipelineResult(
                    understanding=understanding,
                    plan=plan,
                    preflight_decision=preflight_decision,
                    candidate_evaluations=(
                        candidate_evaluations
                    ),
                    outcome=PipelineOutcome(
                        state=(
                            PipelineOutcomeState
                            .AUTHENTICATION_REQUIRED
                        ),
                        allowed=False,
                        reason=(
                            "KNOWLEDGE_REQUIRES_AUTHENTICATION"
                        ),
                        missing_context=missing,
                    ),
                )

            clarification_results = [
                item
                for item in candidate_evaluations
                if item.eligibility.state
                == EligibilityState.CLARIFICATION_REQUIRED
            ]

            if clarification_results:
                missing = self._collect_missing_context(
                    clarification_results
                )

                return RequestPipelineResult(
                    understanding=understanding,
                    plan=plan,
                    preflight_decision=preflight_decision,
                    candidate_evaluations=(
                        candidate_evaluations
                    ),
                    outcome=PipelineOutcome(
                        state=(
                            PipelineOutcomeState.CONTEXT_REQUIRED
                        ),
                        allowed=False,
                        reason=(
                            "KNOWLEDGE_CONTEXT_REQUIRED"
                        ),
                        missing_context=missing,
                    ),
                )

            handoff_results = [
                item
                for item in candidate_evaluations
                if item.eligibility.state
                == EligibilityState.HANDOFF_REQUIRED
            ]

            if handoff_results:
                return RequestPipelineResult(
                    understanding=understanding,
                    plan=plan,
                    preflight_decision=preflight_decision,
                    candidate_evaluations=(
                        candidate_evaluations
                    ),
                    outcome=PipelineOutcome(
                        state=(
                            PipelineOutcomeState
                            .HANDOFF_REQUIRED
                        ),
                        allowed=False,
                        reason="KNOWLEDGE_REQUIRES_HANDOFF",
                    ),
                )

            return RequestPipelineResult(
                understanding=understanding,
                plan=plan,
                preflight_decision=preflight_decision,
                candidate_evaluations=(
                    candidate_evaluations
                ),
                outcome=PipelineOutcome(
                    state=(
                        PipelineOutcomeState
                        .NO_ELIGIBLE_KNOWLEDGE
                    ),
                    allowed=False,
                    reason="NO_ELIGIBLE_KNOWLEDGE_FOUND",
                ),
            )

        # ---------------------------------------------------------
        # 6. Defensive fallback
        # ---------------------------------------------------------

        return RequestPipelineResult(
            understanding=understanding,
            plan=plan,
            preflight_decision=preflight_decision,
            outcome=PipelineOutcome(
                state=PipelineOutcomeState.NO_ELIGIBLE_KNOWLEDGE,
                allowed=False,
                reason="NO_EXECUTION_PATH",
            ),
        )

    @staticmethod
    def _collect_missing_context(
        evaluations: list[CandidateEvaluation],
    ) -> list[str]:
        fields: list[str] = []

        for evaluation in evaluations:
            for field_name in (
                evaluation.eligibility.missing_context
            ):
                if field_name not in fields:
                    fields.append(field_name)

        return fields
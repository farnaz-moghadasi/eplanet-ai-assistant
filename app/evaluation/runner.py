from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

from app.conversation.service import (
    ConversationService,
    TrustedSessionContext,
)
from app.conversation.state import (
    ConversationState,
)
from app.evaluation.schemas import (
    CandidateDiagnostic,
    EvaluationCase,
    EvaluationMetrics,
    EvaluationReport,
    ExpectedTurnResult,
    GroundingClaimDiagnostic,
    TurnDiagnostics,
    TurnEvaluationResult,
)


@dataclass
class _MetricCounters:
    top1_total: int = 0
    top1_passed: int = 0

    recall_total: int = 0
    recall_passed: int = 0

    clarification_total: int = 0
    clarification_passed: int = 0

    context_total: int = 0
    context_passed: int = 0

    citation_total: int = 0
    citation_passed: int = 0

    grounding_total: int = 0
    grounding_passed: int = 0

    abstention_total: int = 0
    abstention_passed: int = 0

    leakage_checks: int = 0
    leakage_failures: int = 0


class EvaluationRunner:
    def __init__(
        self,
        *,
        service: ConversationService,
    ):
        self.service = service

    async def run(
        self,
        cases: list[
            EvaluationCase
        ],
    ) -> EvaluationReport:
        results: list[
            TurnEvaluationResult
        ] = []

        counters = (
            _MetricCounters()
        )

        reciprocal_ranks: list[
            float
        ] = []

        latencies: list[
            float
        ] = []

        enabled_cases = [
            case
            for case in cases
            if case.enabled
        ]

        skipped_cases = (
            len(cases)
            - len(enabled_cases)
        )

        for case in enabled_cases:
            case_results = (
                await self._run_case(
                    case=case,
                    counters=counters,
                    reciprocal_ranks=(
                        reciprocal_ranks
                    ),
                    latencies=latencies,
                )
            )

            results.extend(
                case_results
            )

        passed_turns = sum(
            1
            for result in results
            if result.passed
        )

        failed_turns = (
            len(results)
            - passed_turns
        )

        metrics = EvaluationMetrics(
            total_cases=len(
                cases
            ),
            enabled_cases=len(
                enabled_cases
            ),
            skipped_cases=(
                skipped_cases
            ),
            total_turns=len(
                results
            ),
            passed_turns=(
                passed_turns
            ),
            failed_turns=(
                failed_turns
            ),
            turn_pass_rate=(
                self._ratio(
                    passed_turns,
                    len(results),
                )
            ),
            top1_accuracy=(
                self._ratio(
                    counters
                    .top1_passed,
                    counters
                    .top1_total,
                )
            ),
            recall_at_5=(
                self._ratio(
                    counters
                    .recall_passed,
                    counters
                    .recall_total,
                )
            ),
            mrr=(
                statistics.mean(
                    reciprocal_ranks
                )
                if reciprocal_ranks
                else None
            ),
            clarification_accuracy=(
                self._ratio(
                    counters
                    .clarification_passed,
                    counters
                    .clarification_total,
                )
            ),
            context_accuracy=(
                self._ratio(
                    counters
                    .context_passed,
                    counters
                    .context_total,
                )
            ),
            citation_validity_rate=(
                self._ratio(
                    counters
                    .citation_passed,
                    counters
                    .citation_total,
                )
            ),
            grounding_pass_rate=(
                self._ratio(
                    counters
                    .grounding_passed,
                    counters
                    .grounding_total,
                )
            ),
            abstention_accuracy=(
                self._ratio(
                    counters
                    .abstention_passed,
                    counters
                    .abstention_total,
                )
            ),
            leakage_rate=(
                self._ratio(
                    counters
                    .leakage_failures,
                    counters
                    .leakage_checks,
                )
            ),
            average_latency_ms=(
                statistics.mean(
                    latencies
                )
                if latencies
                else None
            ),
        )

        return EvaluationReport(
            metrics=metrics,
            results=results,
            metadata={
                "metric_note": (
                    "Reranker scores are not "
                    "accuracy metrics."
                )
            },
        )

    async def _run_case(
        self,
        *,
        case: EvaluationCase,
        counters: _MetricCounters,
        reciprocal_ranks: list[
            float
        ],
        latencies: list[
            float
        ],
    ) -> list[
        TurnEvaluationResult
    ]:
        state = ConversationState(
            conversation_id=(
                f"eval-{case.id}"
            )
        )

        for field_name, value in (
            case.initial_context.items()
        ):
            state.set_context_field(
                field_name=field_name,
                value=value,
            )

        trusted_session = None

        if (
            case.trusted_session
            is not None
        ):
            trusted_session = (
                TrustedSessionContext(
                    is_authenticated=(
                        case
                        .trusted_session
                        .is_authenticated
                    ),
                    client_id=(
                        case
                        .trusted_session
                        .client_id
                    ),
                )
            )

        results: list[
            TurnEvaluationResult
        ] = []

        for index, turn in enumerate(
            case.turns,
            start=1,
        ):
            started_at = (
                time.perf_counter()
            )

            service_result = (
                await self.service
                .handle_message(
                    message=(
                        turn.message
                    ),
                    state=state,
                    language=(
                        turn.language
                    ),
                    trusted_session=(
                        trusted_session
                    ),
                )
            )

            latency_ms = round(
                (
                    time.perf_counter()
                    - started_at
                )
                * 1000,
                2,
            )

            latencies.append(
                latency_ms
            )

            result = (
                self._evaluate_turn(
                    case_id=case.id,
                    turn_index=index,
                    expected=(
                        turn.expected
                    ),
                    service_result=(
                        service_result
                    ),
                    state=state,
                    counters=counters,
                    reciprocal_ranks=(
                        reciprocal_ranks
                    ),
                    latency_ms=(
                        latency_ms
                    ),
                )
            )

            results.append(
                result
            )

        return results

    def _evaluate_turn(
        self,
        *,
        case_id: str,
        turn_index: int,
        expected: ExpectedTurnResult,
        service_result,
        state: ConversationState,
        counters: _MetricCounters,
        reciprocal_ranks: list[
            float
        ],
        latency_ms: float,
    ) -> TurnEvaluationResult:
        failures: list[str] = []

        retrieval = (
            service_result
            .retrieval_result
        )

        ranked_keys: list[
            str
        ] = []

        candidate_diagnostics: list[
            CandidateDiagnostic
        ] = []

        reranker_mode = None
        reranker_fallback_reason = None

        if retrieval is not None:
            reranker_mode = (
                retrieval.reranker_mode
            )

            reranker_fallback_reason = (
                retrieval
                .reranker_fallback_reason
            )

            for rank, item in enumerate(
                retrieval.final_candidates,
                start=1,
            ):
                hybrid = item.candidate
                candidate = hybrid.candidate

                knowledge_key = (
                    candidate
                    .metadata
                    .knowledge_key
                )

                ranked_keys.append(
                    knowledge_key
                )

                candidate_diagnostics.append(
                    CandidateDiagnostic(
                        rank=rank,
                        knowledge_key=(
                            knowledge_key
                        ),
                        candidate_id=(
                            candidate
                            .candidate_id
                        ),
                        rerank_score=(
                            self
                            ._safe_float(
                                getattr(
                                    item,
                                    "rerank_score",
                                    None,
                                )
                            )
                        ),
                        hybrid_score=(
                            self
                            ._safe_float(
                                getattr(
                                    hybrid,
                                    "hybrid_score",
                                    None,
                                )
                            )
                        ),
                        bm25_score=(
                            self
                            ._safe_float(
                                getattr(
                                    hybrid,
                                    "bm25_score",
                                    None,
                                )
                            )
                        ),
                        vector_score=(
                            self
                            ._safe_float(
                                getattr(
                                    hybrid,
                                    "vector_score",
                                    None,
                                )
                            )
                        ),
                    )
                )

        top_key = (
            ranked_keys[0]
            if ranked_keys
            else None
        )

        relevant_keys = (
            self._relevant_keys(
                expected
            )
        )

        reciprocal_rank = None

        if relevant_keys:
            counters.top1_total += 1

            if top_key in relevant_keys:
                counters.top1_passed += 1

            else:
                failures.append(
                    (
                        "TOP1_MISMATCH:"
                        f"expected="
                        f"{sorted(relevant_keys)}:"
                        f"actual={top_key}"
                    )
                )

            counters.recall_total += 1

            if any(
                key in ranked_keys[:5]
                for key
                in relevant_keys
            ):
                counters.recall_passed += 1

            else:
                failures.append(
                    "RECALL_AT_5_MISS"
                )

            reciprocal_rank = (
                self._reciprocal_rank(
                    ranked_keys=(
                        ranked_keys
                    ),
                    relevant_keys=(
                        relevant_keys
                    ),
                )
            )

            reciprocal_ranks.append(
                reciprocal_rank
            )

        for required_key in (
            expected.required_in_top_k
        ):
            if (
                required_key
                not in ranked_keys[:5]
            ):
                failures.append(
                    (
                        "REQUIRED_KEY_MISSING:"
                        f"{required_key}"
                    )
                )

        leakage_detected = False

        if (
            expected
            .forbidden_knowledge_keys
            or expected
            .forbidden_knowledge_prefixes
        ):
            counters.leakage_checks += 1

            for key in ranked_keys:
                if (
                    key
                    in expected
                    .forbidden_knowledge_keys
                ):
                    leakage_detected = True

                if any(
                    key.startswith(
                        prefix
                    )
                    for prefix in (
                        expected
                        .forbidden_knowledge_prefixes
                    )
                ):
                    leakage_detected = True

            if leakage_detected:
                counters.leakage_failures += 1

                failures.append(
                    "FORBIDDEN_KNOWLEDGE_LEAKAGE"
                )

        actual_clarification = (
            service_result
            .resolved_turn
            .needs_clarification
        )

        actual_clarification_field = (
            state.awaiting_context_field
        )

        if (
            expected
            .clarification_required
            is not None
        ):
            counters.clarification_total += 1

            clarification_ok = (
                actual_clarification
                == expected
                .clarification_required
            )

            if (
                clarification_ok
                and expected
                .clarification_field
                is not None
            ):
                clarification_ok = (
                    actual_clarification_field
                    == expected
                    .clarification_field
                )

            if clarification_ok:
                counters.clarification_passed += 1

            else:
                failures.append(
                    (
                        "CLARIFICATION_MISMATCH:"
                        f"expected_required="
                        f"{expected.clarification_required}:"
                        f"actual_required="
                        f"{actual_clarification}:"
                        f"expected_field="
                        f"{expected.clarification_field}:"
                        f"actual_field="
                        f"{actual_clarification_field}"
                    )
                )

        if expected.expected_context:
            counters.context_total += 1

            context_ok = True

            for field_name, value in (
                expected
                .expected_context
                .items()
            ):
                actual_value = (
                    state.get_context_field(
                        field_name
                    )
                )

                if (
                    str(actual_value)
                    != str(value)
                ):
                    context_ok = False

                    failures.append(
                        (
                            "CONTEXT_MISMATCH:"
                            f"{field_name}:"
                            f"expected={value}:"
                            f"actual={actual_value}"
                        )
                    )

            if context_ok:
                counters.context_passed += 1

        generation = (
            service_result
            .generation_result
        )

        actual_abstained = None
        abstain_reason = None
        generation_needs_handoff = None
        generation_citation_count = None

        if generation is not None:
            actual_abstained = (
                generation
                .payload
                .abstained
            )

            abstain_reason = (
                generation
                .payload
                .abstain_reason
            )

            generation_needs_handoff = (
                generation
                .payload
                .needs_handoff
            )

            generation_citation_count = len(
                generation
                .payload
                .citations
            )

        if (
            expected.abstained
            is not None
        ):
            counters.abstention_total += 1

            if (
                actual_abstained
                == expected.abstained
            ):
                counters.abstention_passed += 1

            else:
                failures.append(
                    (
                        "ABSTENTION_MISMATCH:"
                        f"expected="
                        f"{expected.abstained}:"
                        f"actual="
                        f"{actual_abstained}:"
                        f"reason="
                        f"{abstain_reason}"
                    )
                )

        citation_validation = (
            service_result
            .citation_validation_result
        )

        actual_citation_valid = None
        citation_issues: list[str] = []

        if (
            citation_validation
            is not None
        ):
            actual_citation_valid = (
                citation_validation.valid
            )

            citation_issues = [
                str(issue)
                for issue in (
                    getattr(
                        citation_validation,
                        "issues",
                        [],
                    )
                    or []
                )
            ]

        if (
            expected.citation_valid
            is not None
        ):
            counters.citation_total += 1

            if (
                actual_citation_valid
                == expected
                .citation_valid
            ):
                counters.citation_passed += 1

            else:
                failures.append(
                    (
                        "CITATION_VALIDITY_MISMATCH:"
                        f"expected="
                        f"{expected.citation_valid}:"
                        f"actual="
                        f"{actual_citation_valid}"
                    )
                )

        grounding_validation = (
            service_result
            .grounding_validation_result
        )

        actual_grounding_valid = None
        grounding_skipped = None
        grounding_reason = None
        grounding_issues: list[
            str
        ] = []
        grounding_claims: list[
            GroundingClaimDiagnostic
        ] = []

        if (
            grounding_validation
            is not None
        ):
            actual_grounding_valid = (
                grounding_validation.valid
            )

            grounding_skipped = (
                grounding_validation.skipped
            )

            grounding_reason = (
                grounding_validation.reason
            )

            grounding_issues = [
                str(issue)
                for issue in (
                    grounding_validation
                    .issues
                    or []
                )
            ]

            for claim in (
                grounding_validation
                .claims
            ):
                grounding_claims.append(
                    GroundingClaimDiagnostic(
                        claim=(
                            self
                            ._extract_claim_text(
                                claim
                            )
                        ),
                        supported=(
                            bool(
                                claim.supported
                            )
                        ),
                        supporting_candidate_ids=list(
                            claim
                            .supporting_candidate_ids
                        ),
                    )
                )

        if (
            expected.grounding_valid
            is not None
        ):
            counters.grounding_total += 1

            if (
                actual_grounding_valid
                == expected
                .grounding_valid
            ):
                counters.grounding_passed += 1

            else:
                failures.append(
                    (
                        "GROUNDING_VALIDITY_MISMATCH:"
                        f"expected="
                        f"{expected.grounding_valid}:"
                        f"actual="
                        f"{actual_grounding_valid}:"
                        f"reason="
                        f"{grounding_reason}"
                    )
                )

        understanding = (
            service_result
            .understanding
        )

        if (
            expected.expected_language
            is not None
            and (
                understanding.language.upper()
                != expected
                .expected_language.upper()
            )
        ):
            failures.append(
                (
                    "LANGUAGE_MISMATCH:"
                    f"expected="
                    f"{expected.expected_language}:"
                    f"actual="
                    f"{understanding.language}"
                )
            )

        if (
            expected.expected_domain
            is not None
            and (
                understanding.domain.value
                != expected
                .expected_domain
            )
        ):
            failures.append(
                (
                    "DOMAIN_MISMATCH:"
                    f"expected="
                    f"{expected.expected_domain}:"
                    f"actual="
                    f"{understanding.domain.value}"
                )
            )

        if (
            expected.expected_task
            is not None
            and (
                understanding.task.value
                != expected
                .expected_task
            )
        ):
            failures.append(
                (
                    "TASK_MISMATCH:"
                    f"expected="
                    f"{expected.expected_task}:"
                    f"actual="
                    f"{understanding.task.value}"
                )
            )

        context_snapshot = (
            self._safe_context_snapshot(
                state
            )
        )

        eligibility_blocked = (
            self._build_eligibility_diagnostics(
                retrieval
            )
        )

        diagnostics = TurnDiagnostics(
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
            understanding_confidence=(
                self._safe_float(
                    understanding.confidence
                )
            ),
            resolved_query=(
                service_result
                .resolved_turn
                .resolved_query
            ),
            missing_context_fields=list(
                service_result
                .resolved_turn
                .missing_context_fields
            ),
            awaiting_context_field=(
                state.awaiting_context_field
            ),
            context_snapshot=(
                context_snapshot
            ),
            candidates=(
                candidate_diagnostics
            ),
            eligibility_blocked=(
                eligibility_blocked
            ),
            reranker_mode=(
                reranker_mode
            ),
            reranker_fallback_reason=(
                reranker_fallback_reason
            ),
            generation_abstained=(
                actual_abstained
            ),
            generation_abstain_reason=(
                abstain_reason
            ),
            generation_needs_handoff=(
                generation_needs_handoff
            ),
            generation_citation_count=(
                generation_citation_count
            ),
            citation_valid=(
                actual_citation_valid
            ),
            citation_issues=(
                citation_issues
            ),
            grounding_valid=(
                actual_grounding_valid
            ),
            grounding_skipped=(
                grounding_skipped
            ),
            grounding_reason=(
                grounding_reason
            ),
            grounding_issues=(
                grounding_issues
            ),
            grounding_claims=(
                grounding_claims
            ),
        )

        return TurnEvaluationResult(
            case_id=case_id,
            turn_index=turn_index,
            passed=not failures,
            failures=failures,
            top_knowledge_key=(
                top_key
            ),
            ranked_knowledge_keys=(
                ranked_keys
            ),
            reciprocal_rank=(
                reciprocal_rank
            ),
            clarification_required=(
                actual_clarification
            ),
            clarification_field=(
                actual_clarification_field
            ),
            abstained=(
                actual_abstained
            ),
            citation_valid=(
                actual_citation_valid
            ),
            grounding_valid=(
                actual_grounding_valid
            ),
            reranker_mode=(
                reranker_mode
            ),
            latency_ms=(
                latency_ms
            ),
            diagnostics=(
                diagnostics
            ),
        )

    @staticmethod
    def _relevant_keys(
        expected: ExpectedTurnResult,
    ) -> set[str]:
        keys: set[str] = set(
            expected
            .acceptable_top_knowledge_keys
        )

        if expected.top_knowledge_key:
            keys.add(
                expected
                .top_knowledge_key
            )

        return keys

    @staticmethod
    def _reciprocal_rank(
        *,
        ranked_keys: list[str],
        relevant_keys: set[str],
    ) -> float:
        for index, key in enumerate(
            ranked_keys,
            start=1,
        ):
            if key in relevant_keys:
                return 1.0 / index

        return 0.0

    @staticmethod
    def _ratio(
        numerator: int,
        denominator: int,
    ) -> float | None:
        if denominator == 0:
            return None

        return round(
            numerator / denominator,
            4,
        )

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float | None:
        if value is None:
            return None

        try:
            return float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _safe_context_snapshot(
        state: ConversationState,
    ) -> dict[str, str]:
        safe_fields = [
            "nationality",
            "country_of_residence",
            "kyc_document_country",
            "referenced_country",
        ]

        result: dict[
            str,
            str,
        ] = {}

        for field_name in (
            safe_fields
        ):
            value = (
                state.get_context_field(
                    field_name
                )
            )

            if value is None:
                continue

            normalized = str(
                value
            ).strip()

            if normalized:
                result[
                    field_name
                ] = normalized

        return result

    @staticmethod
    def _build_eligibility_diagnostics(
        retrieval,
    ) -> list[
        dict[str, Any]
    ]:
        if retrieval is None:
            return []

        result: list[
            dict[str, Any]
        ] = []

        for item in (
            retrieval
            .eligibility_trace
        ):
            if item.eligible:
                continue

            result.append(
                {
                    "knowledge_key": (
                        item.knowledge_key
                    ),
                    "reasons": list(
                        item.reasons
                    ),
                }
            )

        return result

    @staticmethod
    def _extract_claim_text(
        claim,
    ) -> str | None:
        for attr in (
            "claim",
            "text",
            "claim_text",
            "statement",
        ):
            value = getattr(
                claim,
                attr,
                None,
            )

            if value:
                return str(
                    value
                )

        return None
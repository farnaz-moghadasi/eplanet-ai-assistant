from __future__ import annotations

from dataclasses import dataclass, field

from app.retrieval.pipeline import (
    RetrievalPipelineResult,
)


@dataclass
class ClarificationDecision:
    required: bool

    field_name: str | None = None

    reason: str | None = None

    candidate_keys: list[str] = field(
        default_factory=list
    )


class ContextPolicy:
    def decide(
        self,
        *,
        retrieval_result: (
            RetrievalPipelineResult
        ),
    ) -> ClarificationDecision:
        missing_by_candidate = (
            self._collect_missing_fields(
                retrieval_result
            )
        )

        if not missing_by_candidate:
            return ClarificationDecision(
                required=False
            )

        hybrid_rank = {
            (
                item.candidate
                .metadata
                .knowledge_key
            ): rank
            for (
                rank,
                item,
            ) in enumerate(
                retrieval_result
                .hybrid_candidates,
                start=1,
            )
        }

        top_eligible_rank = (
            self._top_eligible_rank(
                retrieval_result=(
                    retrieval_result
                ),
                hybrid_rank=(
                    hybrid_rank
                ),
            )
        )

        field_candidates: dict[
            str,
            list[str],
        ] = {}

        field_best_rank: dict[
            str,
            int,
        ] = {}

        for (
            knowledge_key,
            fields,
        ) in missing_by_candidate.items():
            rank = hybrid_rank.get(
                knowledge_key
            )

            if rank is None:
                continue

            for field_name in fields:
                field_candidates.setdefault(
                    field_name,
                    [],
                ).append(
                    knowledge_key
                )

                previous_rank = (
                    field_best_rank.get(
                        field_name
                    )
                )

                if (
                    previous_rank is None
                    or rank < previous_rank
                ):
                    field_best_rank[
                        field_name
                    ] = rank

        if not field_best_rank:
            return ClarificationDecision(
                required=False
            )

        selected_field = min(
            field_best_rank,
            key=lambda field: (
                field_best_rank[field],
                field,
            ),
        )

        blocked_rank = (
            field_best_rank[
                selected_field
            ]
        )

        should_clarify = (
            top_eligible_rank is None
            or blocked_rank
            <= top_eligible_rank
        )

        if not should_clarify:
            return ClarificationDecision(
                required=False
            )

        return ClarificationDecision(
            required=True,
            field_name=selected_field,
            reason=(
                "MISSING_REQUIRED_CONTEXT"
            ),
            candidate_keys=(
                field_candidates[
                    selected_field
                ]
            ),
        )

    def _collect_missing_fields(
        self,
        retrieval_result: (
            RetrievalPipelineResult
        ),
    ) -> dict[str, set[str]]:
        result: dict[
            str,
            set[str],
        ] = {}

        for trace in (
            retrieval_result
            .eligibility_trace
        ):
            fields: set[str] = set()

            for reason in trace.reasons:
                prefix = (
                    "MISSING_CONTEXT:"
                )

                if reason.startswith(
                    prefix
                ):
                    field_name = (
                        reason[
                            len(prefix):
                        ].strip()
                    )

                    if field_name:
                        fields.add(
                            field_name
                        )

            if fields:
                result[
                    trace.knowledge_key
                ] = fields

        return result

    @staticmethod
    def _top_eligible_rank(
        *,
        retrieval_result: (
            RetrievalPipelineResult
        ),
        hybrid_rank: dict[
            str,
            int,
        ],
    ) -> int | None:
        ranks: list[int] = []

        for item in (
            retrieval_result
            .eligible_candidates
        ):
            key = (
                item.candidate
                .metadata
                .knowledge_key
            )

            rank = hybrid_rank.get(
                key
            )

            if rank is not None:
                ranks.append(rank)

        if not ranks:
            return None

        return min(ranks)
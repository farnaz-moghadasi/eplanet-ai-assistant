from __future__ import annotations

from dataclasses import dataclass, field

from app.conversation.schemas import (
    ConversationUnderstanding,
)
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
    # Measured top-1 vector similarity: 0.20-0.41 for off-topic
    # questions, 0.49-0.69 for real in-scope ones.
    _GENERAL_DOMAIN_MIN_VECTOR_SCORE = 0.45

    def decide(
        self,
        *,
        retrieval_result: (
            RetrievalPipelineResult
        ),
        understanding: (
            ConversationUnderstanding
            | None
        ) = None,
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

        # A candidate that is merely a noisy lexical/embedding
        # neighbour (e.g. an Iran withdrawal-fee page surfacing for a
        # trading-commission question just because both mention fee
        # percentages) should not be able to block the conversation
        # on a clarification question that has nothing to do with
        # what the user actually asked. Only let a missing-context
        # candidate participate in blocking if its own knowledge
        # category is actually aligned with the classified domain of
        # this request.
        domain_value = (
            understanding.domain.value
            if understanding is not None
            else ""
        )

        # A domain of GENERAL/UNKNOWN means the request wasn't
        # classified into a product area. Two very different things land
        # here: genuinely off-topic questions (weather, jokes, "who are
        # you?", investment advice) and real questions the classifier
        # simply labelled loosely (e.g. "what is the commission for
        # gold?"). A country- or login-scoped page that merely
        # surfaced in raw retrieval must never turn the first kind into
        # a bogus clarification, but the second kind still needs the
        # clarification to reach its real answer. So:
        #   - the understanding step flagged it off-topic / advice: never
        #     ask anything;
        #   - otherwise only candidates that are genuinely similar to the
        #     query (vector similarity above the level off-topic
        #     questions reach) may block.
        # (When no understanding is supplied at all, keep the legacy
        # rank-only behaviour.)
        if (
            understanding is not None
            and domain_value.strip().upper()
            in {"", "GENERAL", "UNKNOWN"}
        ):
            if (
                understanding.is_off_topic
                or understanding.is_advice_request
            ):
                return ClarificationDecision(
                    required=False
                )

            missing_by_candidate = (
                self._filter_by_min_vector_score(
                    missing_by_candidate=(
                        missing_by_candidate
                    ),
                    retrieval_result=(
                        retrieval_result
                    ),
                    min_score=(
                        self
                        ._GENERAL_DOMAIN_MIN_VECTOR_SCORE
                    ),
                )
            )

            if not missing_by_candidate:
                return ClarificationDecision(
                    required=False
                )

        if domain_value.strip().upper() not in {
            "",
            "GENERAL",
            "UNKNOWN",
        }:
            missing_by_candidate = (
                self._filter_by_domain_alignment(
                    missing_by_candidate=(
                        missing_by_candidate
                    ),
                    retrieval_result=(
                        retrieval_result
                    ),
                    domain=domain_value,
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

    @staticmethod
    def _filter_by_min_vector_score(
        *,
        missing_by_candidate: dict[
            str, set[str]
        ],
        retrieval_result: (
            RetrievalPipelineResult
        ),
        min_score: float,
    ) -> dict[str, set[str]]:
        score_by_key = {
            item.candidate.metadata.knowledge_key: (
                item.vector_score
            )
            for item in (
                retrieval_result
                .hybrid_candidates
            )
        }

        return {
            knowledge_key: fields
            for (
                knowledge_key,
                fields,
            ) in missing_by_candidate.items()
            if (
                score_by_key.get(
                    knowledge_key
                )
                is not None
                and score_by_key[knowledge_key]
                >= min_score
            )
        }

    @staticmethod
    def _filter_by_domain_alignment(
        *,
        missing_by_candidate: dict[
            str, set[str]
        ],
        retrieval_result: (
            RetrievalPipelineResult
        ),
        domain: str,
    ) -> dict[str, set[str]]:
        category_by_key: dict[
            str, str
        ] = {
            item.candidate.metadata.knowledge_key: str(
                item.candidate.metadata.category
                or ""
            )
            for item in (
                retrieval_result
                .hybrid_candidates
            )
        }

        return {
            knowledge_key: fields
            for (
                knowledge_key,
                fields,
            ) in missing_by_candidate.items()
            if ContextPolicy._domain_aligns_with_category(
                domain=domain,
                category=category_by_key.get(
                    knowledge_key, ""
                ),
            )
        }

    @staticmethod
    def _domain_aligns_with_category(
        *,
        domain: str,
        category: str,
    ) -> bool:
        domain = domain.strip().upper()
        category = category.strip().upper()

        if not domain or not category:
            return False

        if domain == category:
            return True

        domain_tokens = set(
            domain.replace("-", " ")
            .replace("_", " ")
            .split()
        )

        category_tokens = set(
            category.replace("-", " ")
            .replace("_", " ")
            .split()
        )

        for domain_token in domain_tokens:
            for (
                category_token
            ) in category_tokens:
                if (
                    domain_token
                    == category_token
                    or category_token.startswith(
                        domain_token
                    )
                    or domain_token.startswith(
                        category_token
                    )
                ):
                    return True

        return False

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
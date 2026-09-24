from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import (
    dataclass,
    field,
)

from app.core.performance import (
    record_timing,
    record_value,
)
from app.retrieval.base import (
    KnowledgeRetriever,
)
from app.retrieval.eligibility import (
    EligibilityContext,
    EligibilityResult,
    KnowledgeEligibilityPolicy,
)
from app.retrieval.hybrid import (
    HybridCandidate,
    HybridRetriever,
)
from app.retrieval.reranker import (
    KnowledgeReranker,
    RerankedCandidate,
    SemanticKnowledgeReranker,
)
from app.retrieval.schemas import (
    KnowledgeCandidate,
)


logger = logging.getLogger(
    __name__
)


@dataclass
class EligibilityTrace:
    knowledge_key: str
    eligible: bool

    reasons: list[str] = field(
        default_factory=list
    )


@dataclass
class RetrievalPipelineResult:
    query: str

    bm25_candidates: list[
        KnowledgeCandidate
    ]

    vector_candidates: list[
        KnowledgeCandidate
    ]

    hybrid_candidates: list[
        HybridCandidate
    ]

    eligible_candidates: list[
        HybridCandidate
    ]

    final_candidates: list[
        RerankedCandidate
    ]

    eligibility_trace: list[
        EligibilityTrace
    ] = field(
        default_factory=list
    )

    reranker_mode: str = (
        "deterministic"
    )

    reranker_fallback_reason: (
        str | None
    ) = None


class KnowledgeRetrievalPipeline:
    _FAST_PATH_BLOCKED_CATEGORIES = {
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

    _FAST_PATH_BLOCKED_TYPES = {
        "POLICY_RULE",
    }

    def __init__(
        self,
        *,
        bm25_retriever: (
            KnowledgeRetriever
        ),
        vector_retriever: (
            KnowledgeRetriever
        ),
        hybrid_retriever: (
            HybridRetriever
        ),
        eligibility_policy: (
            KnowledgeEligibilityPolicy
        ),
        reranker: KnowledgeReranker,
        semantic_reranker: (
            SemanticKnowledgeReranker
            | None
        ) = None,
        bm25_top_k: int = 20,
        vector_top_k: int = 20,
        hybrid_top_k: int = 20,
        semantic_rerank_top_k: (
            int | None
        ) = 12,
        final_top_k: int = 5,
        fast_path_enabled: bool = True,
        semantic_min_score: float = 0.50,
    ):
        self.bm25_retriever = (
            bm25_retriever
        )

        self.vector_retriever = (
            vector_retriever
        )

        self.hybrid_retriever = (
            hybrid_retriever
        )

        self.eligibility_policy = (
            eligibility_policy
        )

        self.reranker = reranker

        self.semantic_reranker = (
            semantic_reranker
        )

        self.bm25_top_k = (
            bm25_top_k
        )

        self.vector_top_k = (
            vector_top_k
        )

        self.hybrid_top_k = (
            hybrid_top_k
        )

        self.semantic_rerank_top_k = (
            semantic_rerank_top_k
        )

        self.final_top_k = (
            final_top_k
        )

        self.fast_path_enabled = (
            fast_path_enabled
        )

        self.semantic_min_score = float(
            semantic_min_score
        )

    async def retrieve(
        self,
        *,
        query: str,
        context: EligibilityContext,
        language: str | None = None,
        knowledge_hints: (
            list[str]
            | None
        ) = None,
    ) -> RetrievalPipelineResult:
        retrieval_started = (
            time.perf_counter()
        )

        query = query.strip()

        if not query:
            record_timing(
                "retrieval_ms",
                0,
            )

            return RetrievalPipelineResult(
                query=query,
                bm25_candidates=[],
                vector_candidates=[],
                hybrid_candidates=[],
                eligible_candidates=[],
                final_candidates=[],
                eligibility_trace=[],
                reranker_mode="none",
            )

        hints = (
            knowledge_hints
            or []
        )

        (
            bm25_candidates,
            vector_candidates,
        ) = await asyncio.gather(
            self._timed_bm25(
                query=query,
                knowledge_hints=hints,
                language=language,
            ),
            self._timed_vector(
                query=query,
                knowledge_hints=hints,
                language=language,
            ),
        )

        hybrid_started = (
            time.perf_counter()
        )

        hybrid_candidates = (
            self.hybrid_retriever
            .merge(
                bm25_candidates=(
                    bm25_candidates
                ),
                vector_candidates=(
                    vector_candidates
                ),
                limit=(
                    self.hybrid_top_k
                ),
            )
        )

        record_timing(
            "hybrid_ms",
            (
                time.perf_counter()
                - hybrid_started
            )
            * 1000,
        )

        eligibility_started = (
            time.perf_counter()
        )

        (
            eligible_candidates,
            eligibility_trace,
        ) = self._apply_eligibility(
            candidates=(
                hybrid_candidates
            ),
            context=context,
        )

        record_timing(
            "eligibility_ms",
            (
                time.perf_counter()
                - eligibility_started
            )
            * 1000,
        )

        if eligible_candidates:
            top_hybrid = (
                eligible_candidates[0]
            )

            record_value(
                "top_bm25_rank",
                top_hybrid.bm25_rank,
            )

            record_value(
                "top_vector_rank",
                top_hybrid.vector_rank,
            )

            record_value(
                "top_hybrid_key",
                (
                    top_hybrid
                    .candidate
                    .metadata
                    .knowledge_key
                ),
            )

        reranker_started = (
            time.perf_counter()
        )

        (
            final_candidates,
            reranker_mode,
            fallback_reason,
        ) = await self._rerank(
            query=query,
            candidates=(
                eligible_candidates
            ),
            bm25_candidates=(
                bm25_candidates
            ),
            vector_candidates=(
                vector_candidates
            ),
            context=context,
            language=language,
        )

        record_timing(
            "reranker_ms",
            (
                time.perf_counter()
                - reranker_started
            )
            * 1000,
        )

        record_value(
            "reranker_mode",
            reranker_mode,
        )

        record_timing(
            "retrieval_ms",
            (
                time.perf_counter()
                - retrieval_started
            )
            * 1000,
        )

        return RetrievalPipelineResult(
            query=query,
            bm25_candidates=(
                bm25_candidates
            ),
            vector_candidates=(
                vector_candidates
            ),
            hybrid_candidates=(
                hybrid_candidates
            ),
            eligible_candidates=(
                eligible_candidates
            ),
            final_candidates=(
                final_candidates
            ),
            eligibility_trace=(
                eligibility_trace
            ),
            reranker_mode=(
                reranker_mode
            ),
            reranker_fallback_reason=(
                fallback_reason
            ),
        )

    async def _timed_bm25(
        self,
        *,
        query: str,
        knowledge_hints: list[str],
        language: str | None,
    ) -> list[
        KnowledgeCandidate
    ]:
        started = time.perf_counter()

        try:
            return await self._retrieve_bm25(
                query=query,
                knowledge_hints=(
                    knowledge_hints
                ),
                language=language,
            )

        finally:
            record_timing(
                "bm25_ms",
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
            )

    async def _timed_vector(
        self,
        *,
        query: str,
        knowledge_hints: list[str],
        language: str | None,
    ) -> list[
        KnowledgeCandidate
    ]:
        started = time.perf_counter()

        try:
            return await self._retrieve_vector(
                query=query,
                knowledge_hints=(
                    knowledge_hints
                ),
                language=language,
            )

        finally:
            record_timing(
                "vector_ms",
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
            )

    async def _rerank(
        self,
        *,
        query: str,
        candidates: list[
            HybridCandidate
        ],
        bm25_candidates: list[
            KnowledgeCandidate
        ],
        vector_candidates: list[
            KnowledgeCandidate
        ],
        context: EligibilityContext,
        language: str | None,
    ) -> tuple[
        list[RerankedCandidate],
        str,
        str | None,
    ]:
        if not candidates:
            return (
                [],
                "none",
                None,
            )

        if self._can_use_fast_path(
            candidates=candidates,
            bm25_candidates=(
                bm25_candidates
            ),
            vector_candidates=(
                vector_candidates
            ),
        ):
            results = (
                self.reranker.rerank(
                    query=query,
                    candidates=candidates,
                    limit=(
                        self.final_top_k
                    ),
                )
            )

            return (
                results,
                "deterministic_fast_path",
                None,
            )

        if (
            self.semantic_reranker
            is None
        ):
            return (
                self.reranker.rerank(
                    query=query,
                    candidates=candidates,
                    limit=(
                        self.final_top_k
                    ),
                ),
                "deterministic",
                None,
            )

        semantic_candidates = (
            self._semantic_candidate_pool(
                candidates
            )
        )

        semantic_context = (
            self._semantic_context(
                context
            )
        )

        try:
            results = (
                await self
                .semantic_reranker
                .rerank(
                    query=query,
                    candidates=(
                        semantic_candidates
                    ),
                    language=language,
                    context=(
                        semantic_context
                    ),
                    limit=(
                        self.final_top_k
                    ),
                )
            )

            results = [
                item
                for item in results
                if (
                    item.rerank_score
                    >= self.semantic_min_score
                )
            ]

            return (
                results,
                "semantic",
                None,
            )

        except Exception as exc:
            fallback_reason = (
                f"{exc.__class__.__name__}: "
                f"{str(exc)}"
            )[:500]

            logger.warning(
                (
                    "Semantic reranker "
                    "failed. Falling back "
                    "to deterministic "
                    "reranking. reason=%s"
                ),
                fallback_reason,
            )

            results = (
                self.reranker.rerank(
                    query=query,
                    candidates=candidates,
                    limit=(
                        self.final_top_k
                    ),
                )
            )

            return (
                results,
                "deterministic_fallback",
                fallback_reason,
            )
    def _can_use_fast_path(
        self,
        *,
        candidates: list[
            HybridCandidate
        ],
        bm25_candidates: list[
            KnowledgeCandidate
        ],
        vector_candidates: list[
            KnowledgeCandidate
        ],
    ) -> bool:
        if not self.fast_path_enabled:
            return False

        if not candidates:
            return False

        top_hybrid = candidates[0]

        top_candidate = (
            top_hybrid.candidate
        )

        metadata = (
            top_candidate.metadata
        )

        # The winning candidate must have support
        # from both lexical and semantic retrieval.
        if (
            top_hybrid.bm25_rank is None
            or top_hybrid.vector_rank is None
        ):
            return False

        # Strong retrieval consensus:
        # candidate must be near the top in both systems.
        if (
            top_hybrid.bm25_rank > 2
            or top_hybrid.vector_rank > 2
        ):
            return False

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
            ._FAST_PATH_BLOCKED_CATEGORIES
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
            ._FAST_PATH_BLOCKED_TYPES
        ):
            return False

        return True
    def _semantic_candidate_pool(
        self,
        candidates: list[
            HybridCandidate
        ],
    ) -> list[
        HybridCandidate
    ]:
        limit = (
            self.semantic_rerank_top_k
        )

        if (
            limit is None
            or limit <= 0
        ):
            return candidates

        return candidates[:limit]

    @staticmethod
    def _semantic_context(
        context: EligibilityContext,
    ) -> dict[str, str]:
        values: dict[
            str,
            str,
        ] = {}

        fields = {
            "nationality":
                context.nationality,
            "country_of_residence":
                context.country_of_residence,
            "kyc_document_country":
                context.kyc_document_country,
            "referenced_country":
                context.referenced_country,
        }

        for key, value in (
            fields.items()
        ):
            if not value:
                continue

            normalized = str(
                value
            ).strip()

            if normalized:
                values[key] = (
                    normalized
                )

        return values

    async def _retrieve_bm25(
        self,
        *,
        query: str,
        knowledge_hints: list[str],
        language: str | None,
    ) -> list[
        KnowledgeCandidate
    ]:
        candidates = (
            await self
            .bm25_retriever
            .retrieve(
                query=query,
                knowledge_hints=(
                    knowledge_hints
                ),
                limit=(
                    self.bm25_top_k
                ),
            )
        )

        if language is None:
            return candidates

        normalized_language = (
            language
            .strip()
            .upper()
        )

        return [
            candidate
            for candidate
            in candidates
            if (
                candidate.metadata
                .language.value.upper()
                == normalized_language
            )
        ]

    async def _retrieve_vector(
        self,
        *,
        query: str,
        knowledge_hints: list[str],
        language: str | None,
    ) -> list[
        KnowledgeCandidate
    ]:
        kwargs = {
            "query": query,
            "knowledge_hints":
                knowledge_hints,
            "limit":
                self.vector_top_k,
        }

        if language is not None:
            kwargs["language"] = (
                language
                .strip()
                .upper()
            )

        return (
            await self
            .vector_retriever
            .retrieve(
                **kwargs
            )
        )

    def _apply_eligibility(
        self,
        *,
        candidates: list[
            HybridCandidate
        ],
        context: EligibilityContext,
    ) -> tuple[
        list[HybridCandidate],
        list[EligibilityTrace],
    ]:
        eligible_candidates: list[
            HybridCandidate
        ] = []

        trace: list[
            EligibilityTrace
        ] = []

        for hybrid_candidate in (
            candidates
        ):
            candidate = (
                hybrid_candidate
                .candidate
            )

            result: EligibilityResult = (
                self
                .eligibility_policy
                .evaluate(
                    candidate=candidate,
                    context=context,
                )
            )

            trace.append(
                EligibilityTrace(
                    knowledge_key=(
                        candidate
                        .metadata
                        .knowledge_key
                    ),
                    eligible=(
                        result.eligible
                    ),
                    reasons=list(
                        result.reasons
                    ),
                )
            )

            if result.eligible:
                eligible_candidates.append(
                    hybrid_candidate
                )

        return (
            eligible_candidates,
            trace,
        )
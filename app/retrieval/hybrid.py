from dataclasses import dataclass

from app.retrieval.schemas import KnowledgeCandidate


@dataclass
class HybridCandidate:
    candidate: KnowledgeCandidate

    hybrid_score: float

    bm25_rank: int | None = None
    bm25_score: float | None = None

    vector_rank: int | None = None
    vector_score: float | None = None


class HybridRetriever:
    def __init__(
        self,
        *,
        rrf_k: int = 60,
    ):
        self.rrf_k = rrf_k

    def merge(
        self,
        *,
        bm25_candidates: list[
            KnowledgeCandidate
        ],
        vector_candidates: list[
            KnowledgeCandidate
        ],
        limit: int = 10,
    ) -> list[HybridCandidate]:
        if limit <= 0:
            return []

        merged: dict[
            str,
            HybridCandidate,
        ] = {}

        self._add_bm25(
            merged,
            bm25_candidates,
        )

        self._add_vector(
            merged,
            vector_candidates,
        )

        results = list(
            merged.values()
        )

        results.sort(
            key=lambda item: (
                -item.hybrid_score,
                -(
                    item.vector_score
                    if item.vector_score
                    is not None
                    else -1.0
                ),
                item.candidate.metadata.knowledge_key,
            )
        )

        return results[:limit]

    def _add_bm25(
        self,
        merged: dict[
            str,
            HybridCandidate,
        ],
        candidates: list[
            KnowledgeCandidate
        ],
    ) -> None:
        seen: set[str] = set()

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            key = (
                candidate.metadata.knowledge_key
            )

            if key in seen:
                continue

            seen.add(key)

            item = merged.get(key)

            if item is None:
                item = HybridCandidate(
                    candidate=candidate,
                    hybrid_score=0.0,
                )
                merged[key] = item

            item.bm25_rank = rank
            item.bm25_score = (
                candidate.score
            )

            item.hybrid_score += (
                1.0
                / (self.rrf_k + rank)
            )

    def _add_vector(
        self,
        merged: dict[
            str,
            HybridCandidate,
        ],
        candidates: list[
            KnowledgeCandidate
        ],
    ) -> None:
        seen: set[str] = set()

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            key = (
                candidate.metadata.knowledge_key
            )

            if key in seen:
                continue

            seen.add(key)

            item = merged.get(key)

            if item is None:
                item = HybridCandidate(
                    candidate=candidate,
                    hybrid_score=0.0,
                )
                merged[key] = item

            item.vector_rank = rank
            item.vector_score = (
                candidate.score
            )

            item.hybrid_score += (
                1.0
                / (self.rrf_k + rank)
            )
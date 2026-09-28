from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
)
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from app.core.observability import log_event
from app.knowledge.schemas import KnowledgeMetadata
from app.retrieval.document import KnowledgeChunk
from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.schemas import KnowledgeCandidate


@dataclass
class VectorSyncReport:
    chunks_submitted: int = 0
    chunks_upserted: int = 0
    stale_points_deleted: int = 0
    pages_reconciled: int = 0


class QdrantVectorRetriever:
    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        embedding_provider: EmbeddingProvider,
        collection_name: str,
        vector_size: int,
    ):
        self.client = client
        self.embedding_provider = (
            embedding_provider
        )
        self.collection_name = (
            collection_name
        )
        self.vector_size = vector_size

    async def ensure_collection(
        self,
    ) -> None:
        exists = (
            await self.client
            .collection_exists(
                self.collection_name
            )
        )

        if exists:
            return

        await self.client.create_collection(
            collection_name=(
                self.collection_name
            ),
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    async def sync_chunks(
        self,
        chunks: list[KnowledgeChunk],
        *,
        managed_page_ids: list[str],
        batch_size: int = 32,
    ) -> VectorSyncReport:
        """
        Upsert the current chunks first, then remove stale
        Qdrant points for pages that were successfully processed.

        Pages that failed ingestion must not be included in
        managed_page_ids. This prevents temporary Confluence or
        network failures from deleting previously valid vectors.
        """

        await self.ensure_collection()

        normalized_page_ids = (
            self._normalize_page_ids(
                managed_page_ids
            )
        )

        chunks_by_page: dict[
            str,
            list[KnowledgeChunk],
        ] = {}

        for chunk in chunks:
            page_id = str(
                chunk.page_id
            ).strip()

            chunks_by_page.setdefault(
                page_id,
                [],
            ).append(chunk)

        report = VectorSyncReport(
            chunks_submitted=len(
                chunks
            )
        )

        if chunks:
            report.chunks_upserted = (
                await self.upsert_chunks(
                    chunks,
                    batch_size=batch_size,
                )
            )

        for page_id in normalized_page_ids:
            current_chunks = (
                chunks_by_page.get(
                    page_id,
                    [],
                )
            )

            expected_point_ids = {
                self._point_id(
                    chunk.chunk_id
                )
                for chunk in current_chunks
            }

            existing_point_ids = (
                await self
                ._existing_point_ids_for_page(
                    page_id
                )
            )

            stale_point_ids = (
                existing_point_ids
                - expected_point_ids
            )

            if stale_point_ids:
                await self._delete_point_ids(
                    stale_point_ids
                )

                report.stale_points_deleted += (
                    len(stale_point_ids)
                )

            report.pages_reconciled += 1

        return report

    async def upsert_chunks(
        self,
        chunks: list[KnowledgeChunk],
        *,
        batch_size: int = 32,
    ) -> int:
        if not chunks:
            return 0

        await self.ensure_collection()

        indexed = 0

        for start in range(
            0,
            len(chunks),
            batch_size,
        ):
            batch = chunks[
                start:start + batch_size
            ]

            texts = [
                self._build_embedding_text(
                    chunk
                )
                for chunk in batch
            ]

            vectors = (
                await self
                .embedding_provider
                .embed_texts(
                    texts
                )
            )

            if (
                len(vectors)
                != len(batch)
            ):
                raise RuntimeError(
                    "Embedding count does not "
                    "match chunk count."
                )

            points: list[
                PointStruct
            ] = []

            for chunk, vector in zip(
                batch,
                vectors,
            ):
                if (
                    len(vector)
                    != self.vector_size
                ):
                    raise RuntimeError(
                        "Unexpected embedding "
                        "dimension for "
                        f"{chunk.chunk_id}: "
                        f"{len(vector)}"
                    )

                points.append(
                    PointStruct(
                        id=self._point_id(
                            chunk.chunk_id
                        ),
                        vector=vector,
                        payload=(
                            self._payload(
                                chunk
                            )
                        ),
                    )
                )

            await self.client.upsert(
                collection_name=(
                    self.collection_name
                ),
                points=points,
                wait=True,
            )

            indexed += len(
                points
            )

        return indexed

    # Qdrant living in a container that can be restarted/rescheduled
    # independently of this app means a connection attempt can land in
    # the small window while it is coming back up. Two attempts with a
    # short pause covers that window; anything that fails twice is a
    # real outage, not a blip, and is handled by the caller.
    _QUERY_MAX_ATTEMPTS = 2
    _QUERY_RETRY_DELAY_SECONDS = 0.3

    async def _query_points_with_retry(
        self,
        *,
        query: list[float],
        query_filter: Any,
        limit: int,
    ):
        last_error: Exception | None = None

        for attempt in range(
            1, self._QUERY_MAX_ATTEMPTS + 1
        ):
            try:
                return (
                    await self.client
                    .query_points(
                        collection_name=(
                            self.collection_name
                        ),
                        query=query,
                        query_filter=(
                            query_filter
                        ),
                        limit=limit,
                        with_payload=True,
                    )
                )

            except ResponseHandlingException as exc:
                last_error = exc

                log_event(
                    "retrieval.vector_search_attempt_failed",
                    level=30,
                    attempt=attempt,
                    max_attempts=(
                        self._QUERY_MAX_ATTEMPTS
                    ),
                    error_type=(
                        exc.__class__.__name__
                    ),
                )

                if (
                    attempt
                    < self._QUERY_MAX_ATTEMPTS
                ):
                    await asyncio.sleep(
                        self
                        ._QUERY_RETRY_DELAY_SECONDS
                    )

        log_event(
            "retrieval.vector_search_unavailable",
            level=40,
            attempts=(
                self._QUERY_MAX_ATTEMPTS
            ),
            error_type=(
                last_error.__class__.__name__
                if last_error
                else None
            ),
        )

        return None

    async def retrieve(
        self,
        *,
        query: str,
        knowledge_hints: (
            list[str]
            | None
        ) = None,
        limit: int = 10,
        language: str | None = None,
        market_scope: (
            str | None
        ) = None,
        audience: str | None = None,
    ) -> list[
        KnowledgeCandidate
    ]:
        if limit <= 0:
            return []

        query_text = (
            self._build_query_text(
                query=query,
                knowledge_hints=(
                    knowledge_hints
                    or []
                ),
            )
        )

        if not query_text.strip():
            return []

        vector = (
            await self
            .embedding_provider
            .embed_query(
                query_text
            )
        )

        query_filter = (
            self._build_filter(
                language=language,
                market_scope=(
                    market_scope
                ),
                audience=audience,
            )
        )

        result = (
            await self
            ._query_points_with_retry(
                query=vector,
                query_filter=(
                    query_filter
                ),
                limit=limit,
            )
        )

        if result is None:
            # Qdrant stayed unreachable after retrying (e.g. the
            # container restarting, a network blip). Semantic search is
            # only one of two retrieval signals - degrade to BM25-only
            # for this request rather than failing the whole chat
            # request with a raw 500. Everything downstream already
            # treats a missing vector_score as "no confidence" (the
            # fast-path and no-LLM-reranker floors both require it), so
            # this fails safe: at worst it falls through to the
            # semantic reranker or the existing honest "couldn't find"
            # answer, never a fabricated one.
            return []

        candidates: list[
            KnowledgeCandidate
        ] = []

        for point in (
            result.points
        ):
            payload = (
                point.payload
                or {}
            )

            candidates.append(
                self._to_candidate(
                    payload=payload,
                    score=float(
                        point.score
                    ),
                )
            )

        return candidates

    async def _existing_point_ids_for_page(
        self,
        page_id: str,
    ) -> set[str]:
        point_ids: set[str] = set()

        offset = None

        page_filter = Filter(
            must=[
                FieldCondition(
                    key="page_id",
                    match=MatchValue(
                        value=str(
                            page_id
                        )
                    ),
                )
            ]
        )

        while True:
            records, next_offset = (
                await self.client.scroll(
                    collection_name=(
                        self.collection_name
                    ),
                    scroll_filter=(
                        page_filter
                    ),
                    limit=256,
                    offset=offset,
                    with_payload=False,
                    with_vectors=False,
                )
            )

            for record in records:
                point_ids.add(
                    str(record.id)
                )

            if next_offset is None:
                break

            offset = next_offset

        return point_ids

    async def _delete_point_ids(
        self,
        point_ids: set[str],
    ) -> None:
        if not point_ids:
            return

        await self.client.delete(
            collection_name=(
                self.collection_name
            ),
            points_selector=(
                PointIdsList(
                    points=sorted(
                        point_ids
                    )
                )
            ),
            wait=True,
        )

    @staticmethod
    def _normalize_page_ids(
        page_ids: list[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for raw_page_id in (
            page_ids
        ):
            page_id = str(
                raw_page_id
            ).strip()

            if (
                not page_id
                or page_id in seen
            ):
                continue

            seen.add(
                page_id
            )

            result.append(
                page_id
            )

        return result

    @staticmethod
    def _build_embedding_text(
        chunk: KnowledgeChunk,
    ) -> str:
        metadata = chunk.metadata

        parts = [
            chunk.title,
            chunk.content,
            metadata.category,
            metadata.knowledge_type,
            (
                metadata
                .canonical_knowledge_key
            ),
            metadata.language.value,
        ]

        return "\n".join(
            str(part).strip()
            for part in parts
            if (
                part
                and str(part).strip()
            )
        )

    @staticmethod
    def _build_query_text(
        *,
        query: str,
        knowledge_hints: list[str],
    ) -> str:
        parts = [
            query,
            *knowledge_hints,
        ]

        return "\n".join(
            part.strip()
            for part in parts
            if (
                part
                and part.strip()
            )
        )

    @staticmethod
    def _point_id(
        chunk_id: str,
    ) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"eplanet:{chunk_id}",
            )
        )

    @staticmethod
    def _payload(
        chunk: KnowledgeChunk,
    ) -> dict[
        str,
        Any,
    ]:
        metadata = chunk.metadata

        return {
            "chunk_id":
                chunk.chunk_id,
            "page_id":
                str(chunk.page_id),
            "title":
                chunk.title,
            "content":
                chunk.content,
            "knowledge_key":
                metadata.knowledge_key,
            "canonical_knowledge_key":
                (
                    metadata
                    .canonical_knowledge_key
                ),
            "knowledge_type":
                metadata.knowledge_type,
            "category":
                metadata.category,
            "status":
                metadata.status.value,
            "audience":
                metadata.audience.value,
            "language":
                metadata.language.value,
            "market_scope":
                metadata
                .market_scope
                .value,
            "countries":
                metadata.countries,
            "regions":
                metadata.regions,
            "required_context_fields":
                (
                    metadata
                    .required_context_fields
                ),
            "bot_mode":
                metadata.bot_mode.value,
            "source_ids":
                metadata.source_ids,
            "source_authority":
                (
                    metadata
                    .source_authority
                    .value
                    if (
                        metadata
                        .source_authority
                    )
                    else None
                ),
            "primary_source":
                metadata.primary_source,
            "source_system":
                "CONFLUENCE",
        }

    @staticmethod
    def _build_filter(
        *,
        language: str | None,
        market_scope: (
            str | None
        ),
        audience: str | None,
    ) -> Filter | None:
        conditions = []

        if language:
            conditions.append(
                FieldCondition(
                    key="language",
                    match=MatchValue(
                        value=(
                            language.upper()
                        )
                    ),
                )
            )

        if market_scope:
            conditions.append(
                FieldCondition(
                    key="market_scope",
                    match=MatchValue(
                        value=(
                            market_scope
                            .upper()
                        )
                    ),
                )
            )

        if audience:
            conditions.append(
                FieldCondition(
                    key="audience",
                    match=MatchValue(
                        value=(
                            audience.upper()
                        )
                    ),
                )
            )

        if not conditions:
            return None

        return Filter(
            must=conditions
        )

    @staticmethod
    def _to_candidate(
        *,
        payload: dict[
            str,
            Any,
        ],
        score: float,
    ) -> KnowledgeCandidate:
        metadata = KnowledgeMetadata(
            knowledge_key=(
                payload[
                    "knowledge_key"
                ]
            ),
            canonical_knowledge_key=(
                payload[
                    "canonical_knowledge_key"
                ]
            ),
            knowledge_type=(
                payload[
                    "knowledge_type"
                ]
            ),
            category=(
                payload[
                    "category"
                ]
            ),
            status=(
                payload[
                    "status"
                ]
            ),
            audience=(
                payload[
                    "audience"
                ]
            ),
            primary_source=(
                payload.get(
                    "primary_source"
                )
            ),
            source_ids=(
                payload.get(
                    "source_ids",
                    [],
                )
            ),
            source_authority=(
                payload.get(
                    "source_authority"
                )
            ),
            language=(
                payload[
                    "language"
                ]
            ),
            market_scope=(
                payload[
                    "market_scope"
                ]
            ),
            countries=(
                payload.get(
                    "countries",
                    [],
                )
            ),
            regions=(
                payload.get(
                    "regions",
                    [],
                )
            ),
            required_context_fields=(
                payload.get(
                    "required_context_fields",
                    [],
                )
            ),
            bot_mode=(
                payload[
                    "bot_mode"
                ]
            ),
        )

        return KnowledgeCandidate(
            candidate_id=(
                payload[
                    "chunk_id"
                ]
            ),
            title=(
                payload[
                    "title"
                ]
            ),
            content=(
                payload[
                    "content"
                ]
            ),
            metadata=metadata,
            score=score,
            source_page_id=(
                payload[
                    "page_id"
                ]
            ),
        )
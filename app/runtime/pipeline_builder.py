"""Builds the retrieval runtime (BM25 + Qdrant vector search + hybrid
merge + eligibility + reranking) that the application and the
evaluation/dev scripts all share.

This used to live in scripts/test_pipeline_real_kb.py, so the production
app imported its own pipeline configuration from a script named "test_*".
It lives here so app code depends only on app code.
"""

from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient

from app.ai.openai import OpenAIStructuredLLM
from app.core.config import settings
from app.knowledge.confluence_client import (
    ConfluenceClient,
)
from app.knowledge.confluence_snapshot import (
    ConfluenceSnapshotClient,
)
from app.knowledge.ingestion import KnowledgeIngestionService
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.eligibility import (
    KnowledgeEligibilityPolicy,
)
from app.retrieval.embeddings import OpenAIEmbeddingProvider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.pipeline import KnowledgeRetrievalPipeline
from app.retrieval.reranker import (
    KnowledgeReranker,
    SemanticKnowledgeReranker,
)
from app.retrieval.vector import QdrantVectorRetriever
from app.runtime.context_requirements import (
    ContextRequirementIndex,
    ContextRequirementService,
)


@dataclass
class RetrievalRuntime:
    pipeline: KnowledgeRetrievalPipeline
    qdrant_client: AsyncQdrantClient
    context_requirement_service: ContextRequirementService

    async def close(self) -> None:
        await self.qdrant_client.close()


def load_page_ids() -> list[str]:
    with open(
        "scripts/confluence_page_ids.txt",
        "r",
        encoding="utf-8",
    ) as file:
        return [
            line.strip()
            for line in file
            if line.strip()
        ]


async def build_pipeline() -> RetrievalRuntime:
    page_ids = load_page_ids()

    runtime_source = (
        settings
        .KNOWLEDGE_RUNTIME_SOURCE
        .strip()
        .lower()
    )

    if runtime_source == "snapshot":
        confluence = (
            ConfluenceSnapshotClient(
                snapshot_path=(
                    settings
                    .KNOWLEDGE_SNAPSHOT_PATH
                )
            )
        )

    elif runtime_source == "confluence":
        confluence = (
            ConfluenceClient(
                base_url=(
                    settings
                    .CONFLUENCE_BASE_URL
                ),
                cloud_id=(
                    settings
                    .CONFLUENCE_CLOUD_ID
                ),
                email=(
                    settings
                    .CONFLUENCE_EMAIL
                ),
                api_token=(
                    settings
                    .CONFLUENCE_API_TOKEN
                ),
            )
        )

    else:
        raise ValueError(
            "Unsupported "
            "KNOWLEDGE_RUNTIME_SOURCE: "
            f"{runtime_source}"
        )

    bm25 = BM25Retriever()

    ingestion = (
        KnowledgeIngestionService(
            confluence_client=(
                confluence
            ),
            retriever=bm25,
        )
    )

    print(
        f"Loading {len(page_ids)} "
        f"knowledge pages from "
        f"{runtime_source}..."
    )

    report, chunks = (
        await ingestion
        .collect_indexable_chunks(
            page_ids
        )
    )

    bm25.index(
        chunks
    )

    requirement_index = (
        ContextRequirementIndex
        .from_chunks(
            chunks
        )
    )

    requirement_service = (
        ContextRequirementService(
            requirement_index
        )
    )

    print()
    print(
        "BM25 indexed pages: "
        f"{report.indexed_pages}"
    )
    print(
        "Chunks: "
        f"{report.total_chunks}"
    )
    print(
        "Skipped: "
        f"{report.skipped_pages}"
    )
    print(
        "Failed: "
        f"{report.failed_pages}"
    )
    print(
        "Context requirement entries: "
        f"{len(requirement_index)}"
    )

    qdrant_client = (
        AsyncQdrantClient(
            url=(
                settings
                .QDRANT_URL
            ),
            api_key=(
                settings
                .QDRANT_API_KEY
                or None
            ),
        )
    )

    embeddings = (
        OpenAIEmbeddingProvider(
            api_key=(
                settings
                .OPENAI_API_KEY
            ),
            model=(
                settings
                .EMBEDDING_MODEL
            ),
            base_url=(
                settings
                .EMBEDDING_BASE_URL
            ),
            timeout_seconds=(
                settings
                .EMBEDDING_TIMEOUT_SECONDS
            ),
        )
    )

    vector = (
        QdrantVectorRetriever(
            client=(
                qdrant_client
            ),
            embedding_provider=(
                embeddings
            ),
            collection_name=(
                settings
                .QDRANT_COLLECTION_NAME
            ),
            vector_size=(
                settings
                .QDRANT_VECTOR_SIZE
            ),
        )
    )

    # Kept on the main model, not the fast one: measured directly (via
    # the evaluation suite) that the fast model made noticeably worse
    # relevance judgments on close calls (e.g. ranking an MT5-specific
    # page above the actual trading-account password page), dropping
    # top-1 accuracy from 88.9% to 77.8%. Reranking sits right before
    # the answer is chosen, so a wrong call here is as costly as a wrong
    # generation/grounding call - not a safe place to trade accuracy for
    # speed.
    structured_llm = (
        OpenAIStructuredLLM(
            api_key=(
                settings
                .OPENAI_API_KEY
            ),
            model=(
                settings
                .OPENAI_MODEL
            ),
        )
    )

    semantic_reranker = (
        SemanticKnowledgeReranker(
            llm=(
                structured_llm
            ),
            # Trimmed from 2500: chunked content is already <=1200 chars
            # per KnowledgeChunker's own default, so 2500 was mostly
            # unused headroom that only inflated the prompt/latency of
            # this LLM call without adding information for the vast
            # majority of candidates.
            max_content_chars=1500,
        )
    )

    pipeline = KnowledgeRetrievalPipeline(
        bm25_retriever=bm25,
        vector_retriever=vector,
        hybrid_retriever=HybridRetriever(
            rrf_k=60
        ),
        eligibility_policy=(
            KnowledgeEligibilityPolicy()
        ),
        reranker=KnowledgeReranker(),
        semantic_reranker=(
            semantic_reranker
        ),
        bm25_top_k=20,
        vector_top_k=20,
        hybrid_top_k=20,
        # Trimmed from 8: this is the single biggest latency contributor
        # in the whole request (one LLM call scoring every candidate in
        # the pool). Measured directly: top_k=8/chars=2500 averaged
        # ~11s per call versus ~6.7s at top_k=4/chars=1200. 5 is a more
        # conservative middle ground, verified against the evaluation
        # suite to still recall the right document.
        semantic_rerank_top_k=5,
        final_top_k=5,
        fast_path_enabled=True,
    )

    return RetrievalRuntime(
        pipeline=(
            pipeline
        ),
        qdrant_client=(
            qdrant_client
        ),
        context_requirement_service=(
            requirement_service
        ),
    )

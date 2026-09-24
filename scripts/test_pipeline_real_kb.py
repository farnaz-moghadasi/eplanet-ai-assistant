import asyncio
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from app.knowledge.confluence_snapshot import (
    ConfluenceSnapshotClient,
)
from app.knowledge.confluence_client import (
    ConfluenceClient,
)
from app.ai.openai import OpenAIStructuredLLM
from app.core.config import settings
from app.knowledge.ingestion import KnowledgeIngestionService
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.eligibility import (
    EligibilityContext,
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


TEST_CASES = [
    {
        "language": "FA",
        "query": "رمز پنل کاربری رو میخوام عوض کنم؟",
        "context": EligibilityContext(),
    },
    {
        "language": "EN",
        "query": "How do I reset my trading account password?",
        "context": EligibilityContext(
            is_authenticated=True,
            client_id="TEST-CLIENT-001",
        ),
    },
    {
        "language": "AR",
        "query": "كيف أضيف مؤشر في cTrader؟",
        "context": EligibilityContext(),
    },
    {
        "language": "KU",
        "query": "چۆن دەتوانم بچمە ژوورەوە بۆ cTrader؟",
        "context": EligibilityContext(),
    },
    {
        "language": "FA",
        "query": "مارجین یعنی چی؟",
        "context": EligibilityContext(),
    },
    {
        "language": "EN",
        "query": "What is slippage?",
        "context": EligibilityContext(),
    },
    {
        "language": "FA",
        "query": "پیپ چیه",
        "context": EligibilityContext(),
    },
]


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
            max_content_chars=2500,
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
        semantic_rerank_top_k=8,
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

def print_final_results(
    result,
) -> None:
    print()
    print("FINAL RESULTS")
    print("-" * 100)

    print(
        "Reranker mode: "
        f"{result.reranker_mode}"
    )

    if result.reranker_fallback_reason:
        print(
            "Reranker fallback reason: "
            f"{result.reranker_fallback_reason}"
        )

    if not result.final_candidates:
        print("NO ELIGIBLE RESULTS")
        return

    for rank, item in enumerate(
        result.final_candidates,
        start=1,
    ):
        hybrid = item.candidate
        candidate = hybrid.candidate

        print()
        print(
            f"#{rank} "
            f"{candidate.metadata.knowledge_key}"
        )

        print(
            f"title={candidate.title}"
        )

        print(
            "rerank_score="
            f"{item.rerank_score:.6f}"
        )

        print(
            "hybrid_score="
            f"{hybrid.hybrid_score:.6f}"
        )

        print(
            f"bm25_rank="
            f"{hybrid.bm25_rank}"
        )

        print(
            f"vector_rank="
            f"{hybrid.vector_rank}"
        )

        print(
            f"reasons="
            f"{item.reasons}"
        )


def print_blocked_results(
    result,
) -> None:
    blocked = [
        trace
        for trace
        in result.eligibility_trace
        if not trace.eligible
    ]

    if not blocked:
        return

    print()
    print("BLOCKED BY ELIGIBILITY")
    print("-" * 100)

    for trace in blocked:
        print(
            f"{trace.knowledge_key} "
            f"-> {trace.reasons}"
        )


async def run_test_cases(
    pipeline: KnowledgeRetrievalPipeline,
) -> None:
    for test_case in TEST_CASES:
        language = test_case["language"]
        query = test_case["query"]
        context = test_case["context"]

        print()
        print("=" * 100)
        print(
            f"QUERY: {query}"
        )
        print(
            f"LANGUAGE: {language}"
        )
        print("=" * 100)

        result = await pipeline.retrieve(
            query=query,
            context=context,
            language=language,
            knowledge_hints=[],
        )

        print_final_results(result)
        print_blocked_results(result)


async def main() -> None:
    runtime = await build_pipeline()

    try:
        await run_test_cases(
            runtime.pipeline
        )

    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
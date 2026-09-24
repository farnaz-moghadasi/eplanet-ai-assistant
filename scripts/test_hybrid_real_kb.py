import asyncio

from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.knowledge.ingestion import KnowledgeIngestionService
from app.knowledge.confluence_client import ConfluenceClient
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.embeddings import OpenAIEmbeddingProvider
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import QdrantVectorRetriever


TEST_QUERIES = [
    ("FA", "رمز پنل کاربری رو میخوام عوض کنم؟"),
    ("EN", "How do I reset my trading account password?"),
    ("AR", "كيف أضيف مؤشر في cTrader؟"),
    ("KU", "چۆن دەتوانم بچمە ژوورەوە بۆ cTrader؟"),
    ("FA", "مارجین یعنی چی؟"),
    ("EN", "What is slippage?"),
    ("FA", "پیپ چیه"),
]


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


async def main() -> None:
    page_ids = load_page_ids()

    confluence_client = ConfluenceClient(
        base_url=settings.CONFLUENCE_BASE_URL,
        cloud_id=settings.CONFLUENCE_CLOUD_ID,
        email=settings.CONFLUENCE_EMAIL,
        api_token=settings.CONFLUENCE_API_TOKEN,
    )

    bm25 = BM25Retriever()

    ingestion = KnowledgeIngestionService(
        confluence_client=confluence_client,
        retriever=bm25,
    )

    print("Loading BM25 knowledge...")

    report = await ingestion.sync_pages(
        page_ids
    )

    print(
        f"BM25 indexed pages: "
        f"{report.indexed_pages}"
    )

    qdrant_client = AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=(
            settings.QDRANT_API_KEY
            or None
        ),
    )

    embeddings = OpenAIEmbeddingProvider(
        api_key=settings.OPENAI_API_KEY,
        model=settings.EMBEDDING_MODEL,
        base_url=settings.EMBEDDING_BASE_URL,
        timeout_seconds=(
            settings.EMBEDDING_TIMEOUT_SECONDS
        ),
    )

    vector = QdrantVectorRetriever(
        client=qdrant_client,
        embedding_provider=embeddings,
        collection_name=(
            settings.QDRANT_COLLECTION_NAME
        ),
        vector_size=(
            settings.QDRANT_VECTOR_SIZE
        ),
    )

    hybrid = HybridRetriever(
        rrf_k=60
    )

    for language, query in TEST_QUERIES:
        print()
        print("=" * 90)
        print(f"QUERY: {query}")
        print(f"LANGUAGE: {language}")
        print("=" * 90)

        bm25_results = await bm25.retrieve(
            query=query,
            knowledge_hints=[],
            limit=10,
        )

        bm25_results = [
            item
            for item in bm25_results
            if item.metadata.language.value
            == language
        ]

        vector_results = await vector.retrieve(
            query=query,
            knowledge_hints=[],
            limit=10,
            language=language,
        )

        hybrid_results = hybrid.merge(
            bm25_candidates=bm25_results,
            vector_candidates=vector_results,
            limit=10,
        )

        print()
        print("TOP HYBRID RESULTS")
        print("-" * 90)

        for rank, item in enumerate(
            hybrid_results,
            start=1,
        ):
            candidate = item.candidate

            print()
            print(
                f"#{rank} "
                f"{candidate.metadata.knowledge_key}"
            )

            print(
                f"hybrid_score="
                f"{item.hybrid_score:.6f}"
            )

            print(
                f"bm25_rank="
                f"{item.bm25_rank} "
                f"bm25_score="
                f"{item.bm25_score}"
            )

            print(
                f"vector_rank="
                f"{item.vector_rank} "
                f"vector_score="
                f"{item.vector_score}"
            )

            print(
                f"title="
                f"{candidate.title}"
            )

    await qdrant_client.close()


if __name__ == "__main__":
    asyncio.run(main())
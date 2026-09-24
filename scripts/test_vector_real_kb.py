import asyncio

from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.retrieval.embeddings import (
    OpenAIEmbeddingProvider,
)
from app.retrieval.vector import (
    QdrantVectorRetriever,
)


TEST_QUERIES = [
    ("FA", "رمز پنل کاربری رو میخوام عوض کنم؟"),
    ("EN", "How do I reset my trading account password?"),
    ("AR", "كيف أضيف مؤشر في cTrader؟"),
    ("KU", "چۆن دەتوانم بچمە ژوورەوە بۆ cTrader؟"),
    ("FA", "مارجین یعنی چی؟"),
    ("EN", "What is slippage?"),
    ("FA", "پیپ چیه"),
]


async def main() -> None:
    client = AsyncQdrantClient(
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

    retriever = QdrantVectorRetriever(
        client=client,
        embedding_provider=embeddings,
        collection_name=(
            settings.QDRANT_COLLECTION_NAME
        ),
        vector_size=(
            settings.QDRANT_VECTOR_SIZE
        ),
    )

    for language, query in TEST_QUERIES:
        print()
        print("=" * 80)
        print(f"QUERY: {query}")
        print(f"LANGUAGE FILTER: {language}")
        print("=" * 80)

        results = await retriever.retrieve(
            query=query,
            knowledge_hints=[],
            limit=5,
            language=language,
        )

        if not results:
            print("NO RESULTS")
            continue

        for index, result in enumerate(
            results,
            start=1,
        ):
            print()
            print(
                f"#{index} score="
                f"{result.score:.4f}"
            )
            print(
                f"knowledge_key="
                f"{result.metadata.knowledge_key}"
            )
            print(
                f"language="
                f"{result.metadata.language.value}"
            )
            print(
                f"title={result.title}"
            )
            print(
                f"candidate_id="
                f"{result.candidate_id}"
            )
            print(
                f"source_page_id="
                f"{result.source_page_id}"
            )
            print(
                f"text="
                f"{result.content[:300]}"
            )

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
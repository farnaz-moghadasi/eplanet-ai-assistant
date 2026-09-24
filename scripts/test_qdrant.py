import asyncio

from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.retrieval.embeddings import (
    OpenAIEmbeddingProvider,
)
from app.retrieval.vector import (
    QdrantVectorRetriever,
)


async def main():
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
        base_url=(
            settings.EMBEDDING_BASE_URL
        ),
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

    await retriever.ensure_collection()

    exists = await client.collection_exists(
        settings.QDRANT_COLLECTION_NAME
    )

    print(
        f"collection_exists={exists}"
    )

    collection = await client.get_collection(
        settings.QDRANT_COLLECTION_NAME
    )

    print(
        f"points_count="
        f"{collection.points_count}"
    )

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
import asyncio

from app.core.config import settings
from app.retrieval.embeddings import (
    OpenAIEmbeddingProvider,
)


async def main():
    provider = OpenAIEmbeddingProvider(
        api_key=settings.OPENAI_API_KEY,
        model=settings.EMBEDDING_MODEL,
        base_url=(
            settings.EMBEDDING_BASE_URL
        ),
        timeout_seconds=(
            settings.EMBEDDING_TIMEOUT_SECONDS
        ),
    )

    texts = [
        "How do I reset my password?",
        "چطور رمز عبورم را تغییر بدهم؟",
        "كيف أغير كلمة المرور؟",
        "چۆن وشەی نهێنی بگۆڕم؟",
    ]

    vectors = await provider.embed_texts(
        texts
    )

    print(
        f"vectors={len(vectors)}"
    )

    for index, vector in enumerate(
        vectors,
        start=1,
    ):
        print(
            f"{index}: dimension="
            f"{len(vector)}"
        )


if __name__ == "__main__":
    asyncio.run(main())
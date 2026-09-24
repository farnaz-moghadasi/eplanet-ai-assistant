import asyncio

from app.core.config import settings
from app.knowledge.confluence_client import ConfluenceClient
from app.knowledge.ingestion import KnowledgeIngestionService
from app.retrieval.bm25 import BM25Retriever


PAGE_IDS_FILE = "scripts/confluence_page_ids.txt"


TEST_QUERIES = [
    " رمز پنل کاربری رو میخوام عوض کنم؟",
    "How do I reset my trading account password?",
    "كيف أضيف مؤشر في cTrader؟",
    "چۆن دەتوانم بچمە ژوورەوە بۆ cTrader؟",
    "مارجین یعنی چی؟",
    "What is slippage?",
    "پیپ چیه",
]


def read_page_ids() -> list[str]:
    result = []
    seen = set()

    with open(
        PAGE_IDS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            page_id = line.strip()

            if not page_id:
                continue

            if page_id in seen:
                continue

            seen.add(page_id)
            result.append(page_id)

    return result


async def main():
    retriever = BM25Retriever()

    client = ConfluenceClient(
        base_url=settings.CONFLUENCE_BASE_URL,
        cloud_id=settings.CONFLUENCE_CLOUD_ID,
        email=settings.CONFLUENCE_EMAIL,
        api_token=settings.CONFLUENCE_API_TOKEN,
    )

    ingestion = KnowledgeIngestionService(
        confluence_client=client,
        retriever=retriever,
    )

    page_ids = read_page_ids()

    print(
        f"Ingesting {len(page_ids)} pages..."
    )

    report = await ingestion.sync_pages(
        page_ids
    )

    print(
        f"Indexed pages: {report.indexed_pages}"
    )
    print(
        f"Total chunks: {report.total_chunks}"
    )

    for query in TEST_QUERIES:
        print()
        print("=" * 80)
        print(f"QUERY: {query}")
        print("=" * 80)

        results = await retriever.retrieve(
            query=query,
            knowledge_hints=[],
            limit=5,
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
                f"#{index} score={result.score:.4f}"
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
                f"title="
                f"{result.title}"
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


if __name__ == "__main__":
    asyncio.run(main())
import asyncio
from pathlib import Path

from qdrant_client import (
    AsyncQdrantClient,
)

from app.core.config import settings
from app.knowledge.confluence_client import (
    ConfluenceClient,
)
from app.knowledge.ingestion import (
    KnowledgeIngestionService,
)
from app.retrieval.bm25 import (
    BM25Retriever,
)
from app.retrieval.embeddings import (
    OpenAIEmbeddingProvider,
)
from app.retrieval.vector import (
    QdrantVectorRetriever,
)


PAGE_IDS_FILE = (
    "scripts/confluence_page_ids.txt"
)


def read_page_ids() -> list[str]:
    path = Path(
        PAGE_IDS_FILE
    )

    result: list[str] = []
    seen: set[str] = set()

    for raw_line in (
        path.read_text(
            encoding="utf-8"
        ).splitlines()
    ):
        page_id = (
            raw_line.strip()
        )

        if (
            not page_id
            or page_id.startswith("#")
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


async def main() -> None:
    page_ids = (
        read_page_ids()
    )

    print(
        f"Loading {len(page_ids)} "
        "Confluence pages..."
    )

    confluence_client = (
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

    ingestion = (
        KnowledgeIngestionService(
            confluence_client=(
                confluence_client
            ),
            retriever=(
                BM25Retriever()
            ),
        )
    )

    report, chunks = (
        await ingestion
        .collect_indexable_chunks(
            page_ids
        )
    )

    print()
    print(
        "AI-ready pages: "
        f"{report.indexed_pages}"
    )
    print(
        "Chunks ready: "
        f"{len(chunks)}"
    )
    print(
        "Skipped pages: "
        f"{report.skipped_pages}"
    )
    print(
        "Failed pages: "
        f"{report.failed_pages}"
    )

    successful_page_ids = [
        result.page_id
        for result
        in report.page_results
        if result.success
    ]

    failed_page_ids = [
        result.page_id
        for result
        in report.page_results
        if not result.success
    ]

    print(
        "Pages eligible for "
        "Qdrant reconciliation: "
        f"{len(successful_page_ids)}"
    )

    if failed_page_ids:
        print(
            "Pages excluded from "
            "stale cleanup because "
            "ingestion failed: "
            f"{len(failed_page_ids)}"
        )

    qdrant_client = (
        AsyncQdrantClient(
            url=(
                settings.QDRANT_URL
            ),
            api_key=(
                settings.QDRANT_API_KEY
                or None
            ),
        )
    )

    try:
        embedding_provider = (
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

        vector_retriever = (
            QdrantVectorRetriever(
                client=(
                    qdrant_client
                ),
                embedding_provider=(
                    embedding_provider
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

        print()
        print(
            "Synchronizing Qdrant "
            "vector index..."
        )

        sync_report = (
            await vector_retriever
            .sync_chunks(
                chunks,
                managed_page_ids=(
                    successful_page_ids
                ),
                batch_size=32,
            )
        )

        collection = (
            await qdrant_client
            .get_collection(
                settings
                .QDRANT_COLLECTION_NAME
            )
        )

        print()
        print("=" * 72)
        print(
            "QDRANT VECTOR SYNC"
        )
        print("=" * 72)

        print(
            "Chunks submitted     : "
            f"{sync_report.chunks_submitted}"
        )

        print(
            "Chunks upserted      : "
            f"{sync_report.chunks_upserted}"
        )

        print(
            "Stale points deleted : "
            f"{sync_report.stale_points_deleted}"
        )

        print(
            "Pages reconciled     : "
            f"{sync_report.pages_reconciled}"
        )

        print(
            "Failed pages retained: "
            f"{len(failed_page_ids)}"
        )

        print(
            "Qdrant points        : "
            f"{collection.points_count}"
        )

    finally:
        await qdrant_client.close()


if __name__ == "__main__":
    asyncio.run(
        main()
    )
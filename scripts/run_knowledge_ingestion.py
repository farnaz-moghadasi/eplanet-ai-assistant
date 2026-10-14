import argparse
import asyncio
import json
from pathlib import Path

from app.core.config import settings
from app.knowledge.confluence_client import ConfluenceClient
from app.knowledge.ingestion import KnowledgeIngestionService
from app.retrieval.bm25 import BM25Retriever


def read_page_ids(path: str) -> list[str]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Page ID file not found: {file_path}")

    result: list[str] = []
    seen: set[str] = set()

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        page_id = raw_line.strip()

        if not page_id or page_id.startswith("#"):
            continue

        if page_id in seen:
            continue

        seen.add(page_id)
        result.append(page_id)

    return result


def build_confluence_client() -> ConfluenceClient:
    return ConfluenceClient(
        base_url=settings.CONFLUENCE_BASE_URL,
        cloud_id=settings.CONFLUENCE_CLOUD_ID,
        email=settings.CONFLUENCE_EMAIL,
        api_token=settings.CONFLUENCE_API_TOKEN,
    )


def print_report(report) -> None:
    print()
    print("=" * 72)
    print("KNOWLEDGE INGESTION REPORT")
    print("=" * 72)

    print(f"Requested pages : {report.requested_pages}")
    print(f"Successful pages: {report.successful_pages}")
    print(f"Indexed pages   : {report.indexed_pages}")
    print(f"Skipped pages   : {report.skipped_pages}")
    print(f"Failed pages    : {report.failed_pages}")
    print(f"Total chunks    : {report.total_chunks}")

    skipped = [
        item for item in report.page_results if item.success and not item.ai_ready
    ]

    failed = [item for item in report.page_results if not item.success]

    if skipped:
        print()
        print("Skipped pages:")

        for item in skipped:
            reasons = ", ".join(item.validation_reasons)

            print(f"  {item.page_id} | " f"{item.title or '-'} | " f"{reasons}")

    if failed:
        print()
        print("Failed pages:")

        for item in failed:
            print(f"  {item.page_id} | " f"{item.title or '-'} | " f"{item.error}")


def save_report(report, output_path: str) -> None:
    path = Path(output_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "requested_pages": report.requested_pages,
        "successful_pages": report.successful_pages,
        "indexed_pages": report.indexed_pages,
        "skipped_pages": report.skipped_pages,
        "failed_pages": report.failed_pages,
        "total_chunks": report.total_chunks,
        "pages": [
            {
                "page_id": item.page_id,
                "title": item.title,
                "success": item.success,
                "ai_ready": item.ai_ready,
                "chunk_count": item.chunk_count,
                "validation_reasons": (item.validation_reasons),
                "error": item.error,
            }
            for item in report.page_results
        ],
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--page-ids-file", required=True)

    parser.add_argument(
        "--output", default=("artifacts/" "knowledge_ingestion_report.json")
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    page_ids = read_page_ids(args.page_ids_file)

    if not page_ids:
        raise RuntimeError("No Confluence page IDs provided.")

    print(f"Starting ingestion for " f"{len(page_ids)} unique pages...")

    retriever = BM25Retriever()

    service = KnowledgeIngestionService(
        confluence_client=(build_confluence_client()), retriever=retriever
    )

    report = await service.sync_pages(page_ids)

    print_report(report)

    save_report(report, args.output)

    print()
    print("Report written to: " f"{args.output}")


if __name__ == "__main__":
    asyncio.run(main())

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any
from app.core.config import settings
from app.knowledge.confluence_client import ConfluenceClient
from app.knowledge.parser import KnowledgeParser
from app.knowledge.validator import KnowledgeValidator


def read_page_ids(path: str) -> list[str]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Page ID file not found: {file_path}")

    result: list[str] = []
    seen: set[str] = set()

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line in seen:
            continue

        seen.add(line)
        result.append(line)

    return result


def safe_error(exc: Exception) -> str:
    return (f"{exc.__class__.__name__}: {str(exc)}")[:500]


async def audit_page(
    *,
    page_id: str,
    client: ConfluenceClient,
    parser: KnowledgeParser,
    validator: KnowledgeValidator,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "page_id": page_id,
        "title": None,
        "parsed": False,
        "ai_ready": False,
        "knowledge_key": None,
        "canonical_knowledge_key": None,
        "language": None,
        "status": None,
        "audience": None,
        "market_scope": None,
        "countries": [],
        "regions": [],
        "required_context_fields": [],
        "bot_mode": None,
        "validation_reasons": [],
        "error": None,
    }

    try:
        raw_page = await client.get_page(page_id)

        result["title"] = raw_page.get("title")

        parsed = parser.parse(raw_page)

        result["parsed"] = True
        result["title"] = parsed.title

        metadata = parsed.metadata

        result.update(
            {
                "knowledge_key": metadata.knowledge_key,
                "canonical_knowledge_key": (metadata.canonical_knowledge_key),
                "language": metadata.language.value,
                "status": metadata.status.value,
                "audience": metadata.audience.value,
                "market_scope": (metadata.market_scope.value),
                "countries": metadata.countries,
                "regions": metadata.regions,
                "required_context_fields": (metadata.required_context_fields),
                "bot_mode": metadata.bot_mode.value,
            }
        )

        validation = validator.validate(metadata)

        result["ai_ready"] = validation.ai_ready
        result["validation_reasons"] = validation.reasons

    except Exception as exc:
        result["error"] = safe_error(exc)

    return result


async def run_audit(page_ids: list[str]) -> list[dict[str, Any]]:
    client = ConfluenceClient(
        base_url=settings.CONFLUENCE_BASE_URL,
        cloud_id=settings.CONFLUENCE_CLOUD_ID,
        email=settings.CONFLUENCE_EMAIL,
        api_token=settings.CONFLUENCE_API_TOKEN,
    )

    parser = KnowledgeParser()
    validator = KnowledgeValidator()

    results: list[dict[str, Any]] = []

    for index, page_id in enumerate(page_ids, start=1):
        print(f"[{index}/{len(page_ids)}] " f"Auditing page {page_id}...")

        result = await audit_page(
            page_id=page_id, client=client, parser=parser, validator=validator
        )

        results.append(result)

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    total = len(results)

    parsed = sum(1 for item in results if item["parsed"])

    ai_ready = sum(1 for item in results if item["ai_ready"])

    failed = sum(1 for item in results if item["error"])

    blocked = total - ai_ready - failed

    reason_counter: Counter[str] = Counter()

    for item in results:
        for reason in item["validation_reasons"]:
            reason_counter[reason] += 1

    print()
    print("=" * 72)
    print("CONFLUENCE KNOWLEDGE AUDIT")
    print("=" * 72)
    print(f"Total pages : {total}")
    print(f"Parsed      : {parsed}")
    print(f"AI ready    : {ai_ready}")
    print(f"Blocked     : {blocked}")
    print(f"Failed      : {failed}")

    if reason_counter:
        print()
        print("Validation issues:")

        for reason, count in reason_counter.most_common():
            print(f"  {reason}: {count}")

    failed_items = [item for item in results if item["error"]]

    if failed_items:
        print()
        print("Failed pages:")

        for item in failed_items:
            print(
                f"  {item['page_id']} | "
                f"{item['title'] or '-'} | "
                f"{item['error']}"
            )

    blocked_items = [
        item for item in results if (item["parsed"] and not item["ai_ready"])
    ]

    if blocked_items:
        print()
        print("Blocked pages:")

        for item in blocked_items:
            reasons = ", ".join(item["validation_reasons"])

            print(
                f"  {item['page_id']} | "
                f"{item['knowledge_key'] or '-'} | "
                f"{item['title'] or '-'} | "
                f"{reasons}"
            )


def write_report(*, results: list[dict[str, Any]], output_path: str) -> None:
    path = Path(output_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": {
            "total_pages": len(results),
            "parsed_pages": sum(1 for item in results if item["parsed"]),
            "ai_ready_pages": sum(1 for item in results if item["ai_ready"]),
            "failed_pages": sum(1 for item in results if item["error"]),
        },
        "pages": results,
    }

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Audit Confluence knowledge metadata " "without indexing content.")
    )

    parser.add_argument(
        "--page-ids-file",
        required=True,
        help=("Text file containing one Confluence " "page ID per line."),
    )

    parser.add_argument(
        "--output",
        default="artifacts/confluence_audit.json",
        help="JSON audit report path.",
    )

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    page_ids = read_page_ids(args.page_ids_file)

    if not page_ids:
        raise RuntimeError("No Confluence page IDs were provided.")

    results = await run_audit(page_ids)

    print_summary(results)

    write_report(results=results, output_path=args.output)

    print()
    print(f"Detailed report written to: " f"{args.output}")


if __name__ == "__main__":
    asyncio.run(main())

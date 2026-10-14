from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.core.config import settings
from app.generation.cache import AnswerCacheService
from app.knowledge.confluence_client import ConfluenceClient
from app.knowledge.parser import KnowledgeParser

PAGE_IDS_FILE = Path("scripts/confluence_page_ids.txt")


def read_page_ids() -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for raw_line in PAGE_IDS_FILE.read_text(encoding="utf-8").splitlines():
        page_id = raw_line.strip()

        if not page_id or page_id.startswith("#") or page_id in seen:
            continue

        seen.add(page_id)
        result.append(page_id)

    return result


async def main() -> bool:
    page_ids = read_page_ids()

    output_path = Path(settings.KNOWLEDGE_SNAPSHOT_PATH)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = ConfluenceClient(
        base_url=(settings.CONFLUENCE_BASE_URL),
        cloud_id=(settings.CONFLUENCE_CLOUD_ID),
        email=(settings.CONFLUENCE_EMAIL),
        api_token=(settings.CONFLUENCE_API_TOKEN),
    )

    pages: dict[str, dict] = {}

    failures: list[dict[str, str]] = []

    excluded_by_status: list[dict[str, str]] = []

    print(f"Fetching {len(page_ids)} " "Confluence pages...")

    for index, page_id in enumerate(page_ids, start=1):
        try:
            page = await client.get_page(page_id)


            page_status = page.get("status")

            if page_status is not None and page_status != "current":
                excluded_by_status.append(
                    {
                        "page_id": (str(page_id)),
                        "status": str(page_status),
                        "title": str(page.get("title", "")),
                    }
                )

                print(
                    f"[{index}/{len(page_ids)}] "
                    f"EXCLUDED {page_id} "
                    f"(status={page_status})"
                )

                continue

            pages[str(page_id)] = page

            print(f"[{index}/{len(page_ids)}] " f"OK {page_id}")

        except Exception as exc:
            failures.append(
                {"page_id": (str(page_id)), "error_type": (exc.__class__.__name__)}
            )

            print(
                f"[{index}/{len(page_ids)}] "
                f"FAILED {page_id}: "
                f"{exc.__class__.__name__}"
            )

    duplicate_knowledge_keys = _find_duplicate_knowledge_keys(pages)

    snapshot = {
        "version": 1,
        "generated_at": (datetime.now(timezone.utc).isoformat()),
        "requested_pages": len(page_ids),
        "stored_pages": len(pages),
        "failed_pages": len(failures),
        "failures": failures,
        "excluded_by_status": (excluded_by_status),
        "pages": pages,
    }

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    temporary_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    temporary_path.replace(output_path)


    answer_cache_service = AnswerCacheService(
        redis_url=(settings.REDIS_URL), ttl_seconds=(settings.ANSWER_CACHE_TTL_SECONDS)
    )

    try:
        await answer_cache_service.bump_version()

    except Exception as exc:
        print("WARNING: could not " "invalidate the answer " f"cache: {exc}")

    finally:
        await answer_cache_service.close()

    print()
    print("Knowledge snapshot created:")
    print(output_path)
    print(f"Stored pages: {len(pages)}")
    print(f"Failed pages: " f"{len(failures)}")
    print("Excluded (non-current " f"status): {len(excluded_by_status)}")

    if failures:
        print()
        print("WARNING: snapshot was created " "with missing pages.")

    if duplicate_knowledge_keys:
        print()
        print(
            "WARNING: multiple pages share "
            "the same Knowledge Key. Only "
            "ONE page per key can ever be "
            "used by the bot - retrieval "
            "silently drops the rest, which "
            "can make a real page invisible "
            "with no error. Fix the Knowledge "
            "Key field in Confluence for all "
            "but one page in each group below, "
            "then rebuild the snapshot."
        )

        for knowledge_key, entries in duplicate_knowledge_keys.items():
            print(f"  {knowledge_key}:")

            for page_id, title in entries:
                print(f"    - {page_id}: " f"{title}")

    return not duplicate_knowledge_keys


def _find_duplicate_knowledge_keys(
    pages: dict[str, dict],
) -> dict[str, list[tuple[str, str]]]:
    parser = KnowledgeParser()

    by_key: dict[str, list[tuple[str, str]]] = {}

    for page_id, page in pages.items():
        try:
            parsed = parser.parse(page)
        except Exception:
            continue

        key = parsed.metadata.knowledge_key

        if not key:
            continue

        by_key.setdefault(key, []).append((page_id, str(page.get("title", ""))))

    return {key: entries for key, entries in by_key.items() if len(entries) > 1}


if __name__ == "__main__":
    ok = asyncio.run(main())

    if not ok:
        sys.exit(1)

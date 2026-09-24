from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfluenceSnapshotClient:
    def __init__(
        self,
        *,
        snapshot_path: str | Path,
    ):
        self.snapshot_path = Path(
            snapshot_path
        )

        if not self.snapshot_path.exists():
            raise FileNotFoundError(
                "Knowledge snapshot does not exist: "
                f"{self.snapshot_path}"
            )

        raw = json.loads(
            self.snapshot_path.read_text(
                encoding="utf-8"
            )
        )

        pages = raw.get("pages")

        if not isinstance(pages, dict):
            raise ValueError(
                "Invalid knowledge snapshot: "
                "'pages' must be an object."
            )

        self.pages: dict[
            str,
            dict[str, Any],
        ] = {
            str(page_id): page
            for page_id, page
            in pages.items()
        }

    async def get_page(
        self,
        page_id: str,
    ) -> dict[str, Any]:
        normalized_id = str(
            page_id
        ).strip()

        page = self.pages.get(
            normalized_id
        )

        if page is None:
            raise KeyError(
                "Page is missing from "
                "knowledge snapshot: "
                f"{normalized_id}"
            )

        return page
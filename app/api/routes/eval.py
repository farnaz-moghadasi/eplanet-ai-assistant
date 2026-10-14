from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/api", tags=["evaluation"])


_RESULTS_PATH = (
    Path(__file__).resolve().parents[3] / "evaluation" / "results" / "latest.json"
)


@router.get("/eval/latest")
async def latest_evaluation() -> dict[str, Any]:
    if not _RESULTS_PATH.exists():
        raise HTTPException(
            status_code=(status.HTTP_404_NOT_FOUND),
            detail=(
                "No evaluation results "
                "available yet. Run "
                "scripts/run_evaluation.py "
                "first."
            ),
        )

    with _RESULTS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)

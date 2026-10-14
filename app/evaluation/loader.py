from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.schemas import EvaluationCase


def load_evaluation_file(path: str | Path) -> list[EvaluationCase]:
    file_path = Path(path)

    cases: list[EvaluationCase] = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            normalized = line.strip()

            if not normalized:
                continue

            try:
                payload = json.loads(normalized)

                cases.append(EvaluationCase.model_validate(payload))

            except Exception as exc:
                raise RuntimeError(
                    "Failed to load " f"{file_path} " f"line {line_number}: " f"{exc}"
                ) from exc

    return cases


def load_evaluation_directory(directory: str | Path) -> list[EvaluationCase]:
    root = Path(directory)

    cases: list[EvaluationCase] = []

    for path in sorted(root.glob("*.jsonl")):
        cases.extend(load_evaluation_file(path))

    return cases

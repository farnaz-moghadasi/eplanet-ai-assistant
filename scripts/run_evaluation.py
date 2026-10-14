from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from app.ai.openai import OpenAIStructuredLLM
from app.citations.validator import CitationValidator
from app.conversation.resolver import ConversationResolver
from app.conversation.service import ConversationService
from app.conversation.understanding import ConversationUnderstandingService
from app.core.config import settings
from app.core.logging import configure_logging
from app.evaluation.loader import load_evaluation_directory
from app.evaluation.runner import EvaluationRunner
from app.generation.service import AnswerGenerationService
from app.grounding.service import GroundingValidationService
from app.runtime.context_policy import ContextPolicy
from app.runtime.entity_registry import EntityRegistry
from app.runtime.pipeline_builder import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run the eplanet Brokers " "AI assistant evaluation suite.")
    )

    parser.add_argument("--cases-dir", default="evaluation/cases")

    parser.add_argument("--output", default=("evaluation/results/" "latest.json"))

    return parser.parse_args()


def print_metric(label: str, value) -> None:
    if value is None:
        display = "N/A"

    elif isinstance(value, float):
        display = f"{value:.4f}"

    else:
        display = str(value)

    print(f"{label:<30} {display}")


def print_report(report) -> None:
    metrics = report.metrics

    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print_metric("Total cases", metrics.total_cases)
    print_metric("Enabled cases", metrics.enabled_cases)
    print_metric("Skipped cases", metrics.skipped_cases)
    print_metric("Total turns", metrics.total_turns)
    print_metric("Passed turns", metrics.passed_turns)
    print_metric("Failed turns", metrics.failed_turns)

    print()

    print_metric("Turn pass rate", metrics.turn_pass_rate)
    print_metric("Top-1 accuracy", metrics.top1_accuracy)
    print_metric("Recall@5", metrics.recall_at_5)
    print_metric("MRR", metrics.mrr)
    print_metric("Clarification accuracy", metrics.clarification_accuracy)
    print_metric("Context accuracy", metrics.context_accuracy)
    print_metric("Citation validity rate", metrics.citation_validity_rate)
    print_metric("Grounding pass rate", metrics.grounding_pass_rate)
    print_metric("Abstention accuracy", metrics.abstention_accuracy)
    print_metric("Leakage rate", metrics.leakage_rate)
    print_metric("Average latency ms", metrics.average_latency_ms)

    failed_results = [result for result in report.results if not result.passed]

    if failed_results:
        print()
        print("-" * 80)
        print("FAILED TURNS")
        print("-" * 80)

        for result in failed_results:
            print(f"{result.case_id} " f"turn={result.turn_index}")

            for failure in result.failures:
                print(f"  - {failure}")


async def main() -> None:
    args = parse_args()

    configure_logging(level=settings.LOG_LEVEL)

    entity_registry = EntityRegistry.from_file(settings.ENTITY_ALIASES_PATH)

    resolver = ConversationResolver(entity_registry=(entity_registry))

    structured_llm = OpenAIStructuredLLM(
        api_key=(settings.OPENAI_API_KEY), model=(settings.OPENAI_MODEL)
    )

    understanding_service = ConversationUnderstandingService(llm=structured_llm)

    generation_service = AnswerGenerationService(
        llm=structured_llm, max_candidates=5, max_content_chars=4000
    )

    citation_validator = CitationValidator()

    grounding_service = GroundingValidationService(
        llm=structured_llm, max_content_chars=4000
    )

    cases = load_evaluation_directory(args.cases_dir)

    retrieval_runtime = await build_pipeline()

    service = ConversationService(
        understanding_service=(understanding_service),
        resolver=resolver,
        retrieval_pipeline=(retrieval_runtime.pipeline),
        context_policy=(ContextPolicy()),
        context_requirement_service=(retrieval_runtime.context_requirement_service),
        entity_registry=(entity_registry),
        generation_service=(generation_service),
        citation_validator=(citation_validator),
        grounding_service=(grounding_service),
    )

    runner = EvaluationRunner(service=service)

    try:
        report = await runner.run(cases)

        report.metadata.update(
            {
                "generated_at": (datetime.now(timezone.utc).isoformat()),
                "model": (settings.OPENAI_MODEL),
                "environment": (settings.APP_ENV),
            }
        )

        print_report(report)

        output_path = Path(args.output)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(
            json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print()
        print("Report written to:")
        print(output_path)

    finally:
        await retrieval_runtime.close()


if __name__ == "__main__":
    asyncio.run(main())

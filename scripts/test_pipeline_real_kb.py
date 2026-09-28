import asyncio

from app.retrieval.eligibility import (
    EligibilityContext,
)
from app.retrieval.pipeline import KnowledgeRetrievalPipeline
from app.runtime.pipeline_builder import (
    build_pipeline,
)


TEST_CASES = [
    {
        "language": "FA",
        "query": "رمز پنل کاربری رو میخوام عوض کنم؟",
        "context": EligibilityContext(),
    },
    {
        "language": "EN",
        "query": "How do I reset my trading account password?",
        "context": EligibilityContext(
            is_authenticated=True,
            client_id="TEST-CLIENT-001",
        ),
    },
    {
        "language": "AR",
        "query": "كيف أضيف مؤشر في cTrader؟",
        "context": EligibilityContext(),
    },
    {
        "language": "KU",
        "query": "چۆن دەتوانم بچمە ژوورەوە بۆ cTrader؟",
        "context": EligibilityContext(),
    },
    {
        "language": "FA",
        "query": "مارجین یعنی چی؟",
        "context": EligibilityContext(),
    },
    {
        "language": "EN",
        "query": "What is slippage?",
        "context": EligibilityContext(),
    },
    {
        "language": "FA",
        "query": "پیپ چیه",
        "context": EligibilityContext(),
    },
]


def print_final_results(
    result,
) -> None:
    print()
    print("FINAL RESULTS")
    print("-" * 100)

    print(
        "Reranker mode: "
        f"{result.reranker_mode}"
    )

    if result.reranker_fallback_reason:
        print(
            "Reranker fallback reason: "
            f"{result.reranker_fallback_reason}"
        )

    if not result.final_candidates:
        print("NO ELIGIBLE RESULTS")
        return

    for rank, item in enumerate(
        result.final_candidates,
        start=1,
    ):
        hybrid = item.candidate
        candidate = hybrid.candidate

        print()
        print(
            f"#{rank} "
            f"{candidate.metadata.knowledge_key}"
        )

        print(
            f"title={candidate.title}"
        )

        print(
            "rerank_score="
            f"{item.rerank_score:.6f}"
        )

        print(
            "hybrid_score="
            f"{hybrid.hybrid_score:.6f}"
        )

        print(
            f"bm25_rank="
            f"{hybrid.bm25_rank}"
        )

        print(
            f"vector_rank="
            f"{hybrid.vector_rank}"
        )

        print(
            f"reasons="
            f"{item.reasons}"
        )


def print_blocked_results(
    result,
) -> None:
    blocked = [
        trace
        for trace
        in result.eligibility_trace
        if not trace.eligible
    ]

    if not blocked:
        return

    print()
    print("BLOCKED BY ELIGIBILITY")
    print("-" * 100)

    for trace in blocked:
        print(
            f"{trace.knowledge_key} "
            f"-> {trace.reasons}"
        )


async def run_test_cases(
    pipeline: KnowledgeRetrievalPipeline,
) -> None:
    for test_case in TEST_CASES:
        language = test_case["language"]
        query = test_case["query"]
        context = test_case["context"]

        print()
        print("=" * 100)
        print(
            f"QUERY: {query}"
        )
        print(
            f"LANGUAGE: {language}"
        )
        print("=" * 100)

        result = await pipeline.retrieve(
            query=query,
            context=context,
            language=language,
            knowledge_hints=[],
        )

        print_final_results(result)
        print_blocked_results(result)


async def main() -> None:
    runtime = await build_pipeline()

    try:
        await run_test_cases(
            runtime.pipeline
        )

    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
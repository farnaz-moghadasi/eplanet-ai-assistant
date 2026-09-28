import asyncio

from app.runtime.pipeline_builder import (
    build_pipeline,
)


async def main() -> None:
    runtime = await build_pipeline()

    try:
        service = (
            runtime.context_requirement_service
        )

        print()
        print("CONTEXT REQUIREMENT ENTRIES")
        print("=" * 100)

        for entry in service.index.entries():
            print()
            print(
                f"knowledge_key="
                f"{entry.knowledge_key}"
            )

            print(
                f"canonical_key="
                f"{entry.canonical_knowledge_key}"
            )

            print(
                f"title={entry.title}"
            )

            print(
                f"category={entry.category}"
            )

            print(
                f"knowledge_type="
                f"{entry.knowledge_type}"
            )

            print(
                f"language={entry.language}"
            )

            print(
                "required_context_fields="
                f"{list(entry.required_context_fields)}"
            )

            print(
                f"bot_mode={entry.bot_mode}"
            )

            print(
                f"market_scope="
                f"{entry.market_scope}"
            )

    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
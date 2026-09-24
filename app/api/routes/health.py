from __future__ import annotations

from fastapi import (
    APIRouter,
    Request,
)

from app.api.schemas import (
    HealthResponse,
    ReadyResponse,
)


router = APIRouter(
    tags=[
        "health",
    ]
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
async def health() -> (
    HealthResponse
):
    return HealthResponse(
        status="ok"
    )


@router.get(
    "/ready",
    response_model=ReadyResponse,
)
async def ready(
    request: Request,
) -> ReadyResponse:
    runtime = getattr(
        request.app.state,
        "runtime",
        None,
    )

    if runtime is None:
        return ReadyResponse(
            status="not_ready",
            runtime_initialized=False,
            retrieval_initialized=False,
            redis_ready=False,
        )

    retrieval_runtime = getattr(
        runtime,
        "retrieval_runtime",
        None,
    )

    retrieval_initialized = (
        retrieval_runtime
        is not None
    )

    redis_ready = (
        await runtime
        .redis_ready()
    )

    ready_status = (
        retrieval_initialized
        and redis_ready
    )

    return ReadyResponse(
        status=(
            "ready"
            if ready_status
            else "not_ready"
        ),
        runtime_initialized=True,
        retrieval_initialized=(
            retrieval_initialized
        ),
        redis_ready=(
            redis_ready
        ),
    )
from __future__ import annotations

from fastapi import (
    HTTPException,
    Request,
    status,
)
from redis.asyncio import Redis

from app.core.config import settings


async def enforce_chat_rate_limit(
    *,
    redis: Redis,
    request: Request,
) -> None:
    limit = settings.CHAT_RATE_LIMIT_PER_MINUTE

    if limit <= 0:
        return

    key = (
        "eplanet:ratelimit:chat:"
        f"{_client_identifier(request)}"
    )

    current = await redis.incr(key)

    if current == 1:
        await redis.expire(key, 60)

    if current > limit:
        ttl = await redis.ttl(key)

        retry_after = (
            ttl
            if ttl and ttl > 0
            else 60
        )

        raise HTTPException(
            status_code=(
                status
                .HTTP_429_TOO_MANY_REQUESTS
            ),
            detail=(
                "Too many requests. "
                "Please slow down and "
                "try again shortly."
            ),
            headers={
                "Retry-After": str(
                    retry_after
                )
            },
        )


def _client_identifier(
    request: Request,
) -> str:
    forwarded_for = (
        request.headers.get(
            "x-forwarded-for"
        )
    )

    if forwarded_for:
        return (
            forwarded_for
            .split(",")[0]
            .strip()
        )

    if request.client is not None:
        return request.client.host

    return "unknown"

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any
from contextlib import asynccontextmanager

from redis.asyncio import Redis

from app.conversation.state import ConversationState
from app.conversation.store import ConversationStateStore


class RedisConversationStateStore(ConversationStateStore):
    def __init__(
        self,
        *,
        redis_url: str,
        ttl_seconds: int = 86400,
        key_prefix: str = ("eplanet:conversation:"),
    ):
        self.redis = Redis.from_url(redis_url, decode_responses=True)

        self.ttl_seconds = ttl_seconds

        self.key_prefix = key_prefix

    def _key(self, conversation_id: str) -> str:
        return f"{self.key_prefix}" f"{conversation_id}"

    async def get(self, conversation_id: str) -> ConversationState | None:
        raw = await self.redis.get(self._key(conversation_id))

        if raw is None:
            return None

        data = json.loads(raw)

        return self._restore_state(data)

    async def save(self, state: ConversationState) -> None:
        conversation_id = getattr(state, "conversation_id", None)

        if not conversation_id:
            raise ValueError(
                "ConversationState must have " "conversation_id before saving."
            )

        payload = self._serialize_state(state)

        await self.redis.set(
            self._key(conversation_id),
            json.dumps(payload, ensure_ascii=False),
            ex=self.ttl_seconds,
        )

    @asynccontextmanager
    async def lock(
        self,
        conversation_id: str,
        *,
        timeout_seconds: int = 30,
        blocking_timeout_seconds: int = 10,
    ):
        lock = self.redis.lock(
            f"{self.key_prefix}lock:{conversation_id}",
            timeout=timeout_seconds,
            blocking_timeout=blocking_timeout_seconds,
        )
        acquired = await lock.acquire()
        if not acquired:
            raise TimeoutError("Could not acquire conversation lock.")
        try:
            yield
        finally:
            try:
                await lock.release()
            except Exception:
                pass

    async def delete(self, conversation_id: str) -> None:
        await self.redis.delete(self._key(conversation_id))

    async def ping(self) -> bool:
        try:
            result = await self.redis.ping()

            return bool(result)

        except Exception:
            return False

    async def close(self) -> None:
        await self.redis.aclose()

    @staticmethod
    def _serialize_state(state: ConversationState) -> dict[str, Any]:
        if hasattr(state, "model_dump"):
            return state.model_dump(mode="json")

        if is_dataclass(state):
            return asdict(state)

        return {
            key: value for key, value in vars(state).items() if not key.startswith("_")
        }

    @staticmethod
    def _restore_state(data: dict[str, Any]) -> ConversationState:
        if hasattr(ConversationState, "model_validate"):
            return ConversationState.model_validate(data)

        return ConversationState(**data)

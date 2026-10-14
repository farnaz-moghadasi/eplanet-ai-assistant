from __future__ import annotations

import hashlib
import json
from typing import Mapping

from redis.asyncio import Redis

from app.core.observability import log_event
from app.generation.schemas import GeneratedAnswerPayload
from app.generation.service import GenerationResult


class AnswerCacheService:
    """Caches the generated-and-validated answer for a given set of
    cited knowledge pages, so repeat (or differently-phrased) questions
    that resolve to the same content skip the two most expensive steps
    - LLM generation and grounding validation - entirely.

    Deliberately NOT keyed on the user's raw question text: different
    phrasings of "the same" question ("spread چیه؟" vs "اسپرد یعنی
    چی؟") still converge on the same cited knowledge_key(s) after
    retrieval, which is a far more stable cache key than raw text.

    Understanding, retrieval, and the readiness/triage/disambiguation
    checks always still run fresh - only the final answer text is
    cached, so country-gating, clarifying questions, and per-
    conversation continuity are unaffected by a cache hit.
    """

    _VERSION_KEY = "eplanet:answercache:version"

    def __init__(
        self,
        *,
        redis_url: str,
        ttl_seconds: int = 86400,
        key_prefix: str = ("eplanet:answercache:entry:"),
    ) -> None:
        self.redis = Redis.from_url(redis_url, decode_responses=True)

        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix

    async def get(
        self,
        *,
        knowledge_keys: list[str],
        language: str | None,
        context: Mapping[str, str],
    ) -> GenerationResult | None:
        try:
            key = await self._build_key(
                knowledge_keys=knowledge_keys, language=language, context=context
            )

            raw = await self.redis.get(key)

            if raw is None:
                return None

            return self._deserialize(json.loads(raw))

        except Exception as exc:
            # A cache problem must never break answering - treat it
            # exactly like a miss.
            log_event(
                "answer_cache.get_failed", level=30, error_type=(exc.__class__.__name__)
            )

            return None

    async def set(
        self,
        *,
        knowledge_keys: list[str],
        language: str | None,
        context: Mapping[str, str],
        result: GenerationResult,
    ) -> None:
        try:
            key = await self._build_key(
                knowledge_keys=knowledge_keys, language=language, context=context
            )

            await self.redis.set(
                key,
                json.dumps(self._serialize(result), ensure_ascii=False),
                ex=self.ttl_seconds,
            )

        except Exception as exc:
            log_event(
                "answer_cache.set_failed", level=30, error_type=(exc.__class__.__name__)
            )

    async def bump_version(self) -> None:
        """Invalidates every cached answer at once. Call this after a
        knowledge sync (snapshot rebuild + Qdrant index) so a stale
        answer can never be served after the underlying content
        changes - old entries simply become unreachable and expire via
        their own TTL.
        """
        await self.redis.incr(self._VERSION_KEY)

    async def close(self) -> None:
        await self.redis.aclose()

    async def _current_version(self) -> str:
        try:
            value = await self.redis.get(self._VERSION_KEY)

        except Exception as exc:
            # Every other Redis failure path here logs - this one
            # silently pinned every key to version "0" with no signal.
            log_event(
                "answer_cache.version_read_failed",
                level=30,
                error_type=(exc.__class__.__name__),
            )

            return "0"

        return value or "0"

    async def _build_key(
        self,
        *,
        knowledge_keys: list[str],
        language: str | None,
        context: Mapping[str, str],
    ) -> str:
        version = await self._current_version()

        sorted_keys = ",".join(sorted(key for key in knowledge_keys if key))

        sorted_context = ",".join(
            f"{key}={value}" for key, value in sorted(context.items()) if value
        )

        raw = f"v{version}|" f"{language or ''}|" f"{sorted_keys}|" f"{sorted_context}"

        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

        return f"{self.key_prefix}{digest}"

    @staticmethod
    def _serialize(result: GenerationResult) -> dict:
        return {
            "payload": (result.payload.model_dump(mode="json")),
            "evidence_candidate_ids": (result.evidence_candidate_ids),
            "evidence_knowledge_keys": (result.evidence_knowledge_keys),
            "evidence_pairs": [list(pair) for pair in (result.evidence_pairs)],
        }

    @staticmethod
    def _deserialize(data: dict) -> GenerationResult:
        return GenerationResult(
            payload=(GeneratedAnswerPayload.model_validate(data["payload"])),
            evidence_candidate_ids=(data.get("evidence_candidate_ids", [])),
            evidence_knowledge_keys=(data.get("evidence_knowledge_keys", [])),
            evidence_pairs={tuple(pair) for pair in data.get("evidence_pairs", [])},
        )

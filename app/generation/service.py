from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from app.ai.base import StructuredLLM
from app.generation.prompts import (
    GENERATION_SYSTEM_PROMPT,
    build_generation_prompt,
)
from app.generation.schemas import (
    GeneratedAnswerPayload,
    GeneratedCitation,
)
from app.knowledge.enums import (
    Audience,
    BotMode,
    KnowledgeStatus,
)
from app.retrieval.reranker import (
    RerankedCandidate,
)


@dataclass
class GenerationResult:
    payload: GeneratedAnswerPayload
    evidence_candidate_ids: list[str]
    evidence_knowledge_keys: list[str]
    evidence_pairs: set[tuple[str, str]]


class AnswerGenerationService:
    # Matches KnowledgeChunker's own default max_chars: a chunk at or
    # below this size is expected to already be a single, narrow topic.
    # A candidate longer than this (e.g. a multi-account comparison page,
    # or one that skipped chunking) is likely to cover several distinct
    # items - returning it verbatim would dump everything instead of the
    # one thing the user actually asked about. Route those through LLM
    # generation instead, which can extract just the relevant part.
    _MAX_DETERMINISTIC_ANSWER_CHARS = 1200

    def __init__(
        self,
        *,
        llm: StructuredLLM,
        max_candidates: int = 5,
        max_content_chars: int = 4000,
    ):
        self.llm = llm
        self.max_candidates = max_candidates
        self.max_content_chars = max_content_chars

    def can_generate_deterministically(
        self,
        *,
        candidates: list[RerankedCandidate],
    ) -> bool:
        if not candidates:
            return False

        candidate = (
            candidates[0]
            .candidate
            .candidate
        )

        metadata = candidate.metadata

        if (
            metadata.status
            != KnowledgeStatus.APPROVED
        ):
            return False

        if (
            metadata.bot_mode
            != BotMode.KNOWLEDGE
        ):
            return False

        if metadata.audience not in {
            Audience.VISITOR,
            Audience.CLIENT,
            Audience.BOTH,
        }:
            return False

        content = (
            candidate.content
            or ""
        ).strip()

        if not content:
            return False

        if (
            len(content)
            > self._MAX_DETERMINISTIC_ANSWER_CHARS
        ):
            return False

        return True

    def generate_deterministic(
        self,
        *,
        candidates: list[RerankedCandidate],
    ) -> GenerationResult:
        if not candidates:
            payload = GeneratedAnswerPayload(
                answer="",
                citations=[],
                abstained=True,
                abstain_reason=(
                    "NO_ELIGIBLE_EVIDENCE"
                ),
                needs_handoff=False,
            )

            return GenerationResult(
                payload=payload,
                evidence_candidate_ids=[],
                evidence_knowledge_keys=[],
                evidence_pairs=set(),
            )

        candidate = (
            candidates[0]
            .candidate
            .candidate
        )

        candidate_id = (
            candidate.candidate_id
        )

        knowledge_key = (
            candidate
            .metadata
            .knowledge_key
        )

        content = (
            candidate.content
            or ""
        ).strip()

        if not content:
            payload = GeneratedAnswerPayload(
                answer="",
                citations=[],
                abstained=True,
                abstain_reason=(
                    "EMPTY_KNOWLEDGE_CONTENT"
                ),
                needs_handoff=False,
            )

            return GenerationResult(
                payload=payload,
                evidence_candidate_ids=[
                    candidate_id
                ],
                evidence_knowledge_keys=[
                    knowledge_key
                ],
                evidence_pairs={
                    (
                        candidate_id,
                        knowledge_key,
                    )
                },
            )

        payload = GeneratedAnswerPayload(
            answer=content,
            citations=[
                GeneratedCitation(
                    candidate_id=(
                        candidate_id
                    ),
                    knowledge_key=(
                        knowledge_key
                    ),
                )
            ],
            abstained=False,
            abstain_reason=None,
            needs_handoff=False,
        )

        return GenerationResult(
            payload=payload,
            evidence_candidate_ids=[
                candidate_id
            ],
            evidence_knowledge_keys=[
                knowledge_key
            ],
            evidence_pairs={
                (
                    candidate_id,
                    knowledge_key,
                )
            },
        )

    async def generate(
        self,
        *,
        query: str,
        language: str | None,
        candidates: list[RerankedCandidate],
        context: Mapping[str, str] | None = None,
    ) -> GenerationResult:
        selected = candidates[: self.max_candidates]

        if not selected:
            payload = GeneratedAnswerPayload(
                answer="",
                citations=[],
                abstained=True,
                abstain_reason="NO_ELIGIBLE_EVIDENCE",
                needs_handoff=False,
            )

            return GenerationResult(
                payload=payload,
                evidence_candidate_ids=[],
                evidence_knowledge_keys=[],
                evidence_pairs=set(),
            )

        evidence_candidate_ids = [
            item.candidate.candidate.candidate_id
            for item in selected
        ]

        evidence_knowledge_keys = [
            item.candidate.candidate.metadata.knowledge_key
            for item in selected
        ]

        evidence_pairs = {
            (
                item.candidate.candidate.candidate_id,
                item.candidate.candidate.metadata.knowledge_key,
            )
            for item in selected
        }

        evidence_text = self._build_evidence(
            selected
        )

        context_text = self._build_context_text(
            context or {}
        )

        prompt = build_generation_prompt(
            query=query,
            language=language,
            context_text=context_text,
            evidence_text=evidence_text,
        )

        payload = await self.llm.generate_structured(
            system_prompt=GENERATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_model=GeneratedAnswerPayload,
        )

        return GenerationResult(
            payload=payload,
            evidence_candidate_ids=(
                evidence_candidate_ids
            ),
            evidence_knowledge_keys=(
                evidence_knowledge_keys
            ),
            evidence_pairs=evidence_pairs,
        )

    def _build_evidence(
        self,
        candidates: list[RerankedCandidate],
    ) -> str:
        blocks: list[str] = []

        for index, item in enumerate(
            candidates,
            start=1,
        ):
            hybrid = item.candidate
            candidate = hybrid.candidate
            metadata = candidate.metadata

            content = (
                candidate.content
                or ""
            )

            content = content[
                : self.max_content_chars
            ]

            blocks.append(
                "\n".join(
                    [
                        f"EVIDENCE {index}",
                        (
                            "candidate_id: "
                            f"{candidate.candidate_id}"
                        ),
                        (
                            "knowledge_key: "
                            f"{metadata.knowledge_key}"
                        ),
                        (
                            "title: "
                            f"{candidate.title}"
                        ),
                        (
                            "category: "
                            f"{metadata.category}"
                        ),
                        (
                            "knowledge_type: "
                            f"{metadata.knowledge_type}"
                        ),
                        (
                            "language: "
                            f"{metadata.language.value}"
                        ),
                        (
                            "market_scope: "
                            f"{metadata.market_scope.value}"
                        ),
                        (
                            "countries: "
                            f"{metadata.countries}"
                        ),
                        (
                            "semantic_score: "
                            f"{item.rerank_score}"
                        ),
                        "content:",
                        content,
                    ]
                )
            )

        return "\n\n".join(
            blocks
        )

    @staticmethod
    def _build_context_text(
        context: Mapping[str, str],
    ) -> str:
        safe_keys = {
            "nationality",
            "country_of_residence",
            "kyc_document_country",
            "referenced_country",
        }

        lines: list[str] = []

        for key, value in context.items():
            if key not in safe_keys:
                continue

            if not value:
                continue

            normalized = str(value).strip()

            if not normalized:
                continue

            lines.append(
                f"{key}: {normalized}"
            )

        if not lines:
            return "none"

        return "\n".join(
            lines
        )
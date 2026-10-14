from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from app.ai.base import StructuredLLM
from app.generation.prompts import GENERATION_SYSTEM_PROMPT, build_generation_prompt
from app.generation.schemas import GeneratedAnswerPayload, GeneratedCitation
from app.knowledge.enums import Audience, BotMode, KnowledgeStatus
from app.retrieval.reranker import RerankedCandidate


@dataclass
class GenerationResult:
    payload: GeneratedAnswerPayload
    evidence_candidate_ids: list[str]
    evidence_knowledge_keys: list[str]
    evidence_pairs: set[tuple[str, str]]


class AnswerGenerationService:

    _MAX_DETERMINISTIC_ANSWER_CHARS = 1200

    _MULTI_ITEM_HEADING_PATTERN = re.compile(r"^#{1,6}\s*[^\n]*—", re.MULTILINE)

    _MIN_MULTI_ITEM_HEADINGS = 2

    _MAX_RELEVANCE_GAP_FOR_ADDITIONAL_EVIDENCE = 0.15

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

    _SECTION_HEADING = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$")


    _DISPLAY_DROP = frozenset(
        {
            # Pointers to other pages, meaningless inside a chat answer.
            "related articles",
            "related content",
            "related links",
            "مقالات مرتبط",
            "مقالات ذات صلة",
            "بابەتی پەیوەندیدار",
        }
    )


    _DISPLAY_QUESTION = frozenset({"question", "السؤال", "سؤال", "سوال", "پرسیار"})

    _DISPLAY_ANSWER = frozenset(
        {"answer", "الإجابة", "الاجابة", "إجابة", "پاسخ", "جواب", "وەڵام"}
    )

    # only the heading is dropped; its text (or links) stays.
    _DISPLAY_UNWRAP = frozenset(
        {
            "important notes",
            "ملاحظات مهمة",
            "تێبینی گرنگ",
            "نکات مهم",
            "problem",
            "المشكلة",
            "مشکل",
            "کێشە",
            "useful links",
            "روابط مفيدة",
            "لینک‌های مفید",
            "video tutorial",
            "video tutorials",
            "الفيديو التعليمي",
            "فيديو تعليمي",
            "آموزش ویدیویی",
            "فێرکاری ڤیدیۆیی",
            "resource",
            "resources",
            "web terminal",
        }
    )


    _DISPLAY_PURPOSE = frozenset(
        {
            "what this helps with",
            "purpose",
            "overview",
            "الهدف",
            "نظرة عامة",
            "ئامانج",
            "کورتە باس",
            "هدف",
            "توضیحات",
        }
    )

    _MAX_DROPPABLE_PURPOSE_CHARS = 160


    _DISPLAY_STEPS = frozenset({"steps", "مراحل", "الخطوات", "هەنگاوەکان"})

    _STEPS_LABELS = {"fa": "مراحل", "en": "Steps", "ar": "الخطوات", "ku": "هەنگاوەکان"}

    @staticmethod
    def _normalize_heading(name: str) -> str:
        cleaned = re.sub(r"[*_`:\uff1a]+", "", name)

        return re.sub(r"\s+", " ", cleaned).strip().lower()

    @classmethod
    def to_display_text(cls, content: str, language: str | None = None) -> str:
        """Approved page text as the customer should see it.

        Applies the section-label families above. Headings that are not
        recognised (e.g. "Android", "Desktop", or a page's own topic
        sub-headings) are left exactly as written. If nothing would be
        left, the original text is returned.
        """
        text = (content or "").strip()

        matches = list(cls._SECTION_HEADING.finditer(text))

        if not matches:
            return text

        steps_label = cls._STEPS_LABELS.get(str(language or "").strip().lower())

        sections = []

        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)

            sections.append(
                (
                    match,
                    cls._normalize_heading(match.group(2)),
                    text[match.end() : end].strip(),
                )
            )

        has_answer = any(
            name in cls._DISPLAY_ANSWER and body for _, name, body in sections
        )

        parts: list[str] = []

        preamble = text[: matches[0].start()].strip()

        if preamble:
            parts.append(preamble)

        for match, name, body in sections:
            if name in cls._DISPLAY_DROP:
                continue

            if name in cls._DISPLAY_QUESTION:
                if not has_answer and body:
                    parts.append(body)

                continue

            if name in cls._DISPLAY_ANSWER or name in cls._DISPLAY_UNWRAP:
                if body:
                    parts.append(body)

                continue

            if name in cls._DISPLAY_PURPOSE:
                if body and len(body) > cls._MAX_DROPPABLE_PURPOSE_CHARS:
                    parts.append(body)

                continue

            if name in cls._DISPLAY_STEPS and steps_label:
                parts.append(
                    f"{match.group(1)} "
                    f"{steps_label}" + (f"\n\n{body}" if body else "")
                )

                continue

            parts.append(match.group(0).strip() + (f"\n\n{body}" if body else ""))

        return "\n\n".join(part for part in parts if part).strip() or text

    def can_generate_deterministically(
        self, *, candidates: list[RerankedCandidate]
    ) -> bool:
        if not candidates:
            return False

        candidate = candidates[0].candidate.candidate

        metadata = candidate.metadata

        if metadata.status != KnowledgeStatus.APPROVED:
            return False

        if metadata.bot_mode != BotMode.KNOWLEDGE:
            return False

        if metadata.audience not in {Audience.VISITOR, Audience.CLIENT, Audience.BOTH}:
            return False

        content = (candidate.content or "").strip()

        if not content:
            return False

        if len(content) > self._MAX_DETERMINISTIC_ANSWER_CHARS:
            return False

        multi_item_headings = len(self._MULTI_ITEM_HEADING_PATTERN.findall(content))

        if multi_item_headings >= self._MIN_MULTI_ITEM_HEADINGS:
            return False

        return True

    def generate_deterministic(
        self, *, candidates: list[RerankedCandidate]
    ) -> GenerationResult:
        if not candidates:
            payload = GeneratedAnswerPayload(
                answer="",
                citations=[],
                abstained=True,
                abstain_reason=("NO_ELIGIBLE_EVIDENCE"),
                needs_handoff=False,
            )

            return GenerationResult(
                payload=payload,
                evidence_candidate_ids=[],
                evidence_knowledge_keys=[],
                evidence_pairs=set(),
            )

        candidate = candidates[0].candidate.candidate

        candidate_id = candidate.candidate_id

        knowledge_key = candidate.metadata.knowledge_key

        content = self.to_display_text(
            candidate.content or "",
            language=getattr(candidate.metadata.language, "value", None),
        )

        if not content:
            payload = GeneratedAnswerPayload(
                answer="",
                citations=[],
                abstained=True,
                abstain_reason=("EMPTY_KNOWLEDGE_CONTENT"),
                needs_handoff=False,
            )

            return GenerationResult(
                payload=payload,
                evidence_candidate_ids=[candidate_id],
                evidence_knowledge_keys=[knowledge_key],
                evidence_pairs={(candidate_id, knowledge_key)},
            )

        payload = GeneratedAnswerPayload(
            answer=content,
            citations=[
                GeneratedCitation(
                    candidate_id=(candidate_id), knowledge_key=(knowledge_key)
                )
            ],
            abstained=False,
            abstain_reason=None,
            needs_handoff=False,
        )

        return GenerationResult(
            payload=payload,
            evidence_candidate_ids=[candidate_id],
            evidence_knowledge_keys=[knowledge_key],
            evidence_pairs={(candidate_id, knowledge_key)},
        )

    def _select_evidence_candidates(
        self, candidates: list[RerankedCandidate]
    ) -> list[RerankedCandidate]:
        pool = candidates[: self.max_candidates]

        if not pool:
            return []

        top_score = pool[0].rerank_score

        selected = [pool[0]]

        for candidate in pool[1:]:
            gap = top_score - candidate.rerank_score

            if gap <= self._MAX_RELEVANCE_GAP_FOR_ADDITIONAL_EVIDENCE:
                selected.append(candidate)

        return selected

    async def generate(
        self,
        *,
        query: str,
        language: str | None,
        candidates: list[RerankedCandidate],
        context: Mapping[str, str] | None = None,
        evidence_override: list[RerankedCandidate] | None = None,
    ) -> GenerationResult:

        selected = (
            evidence_override
            if evidence_override is not None
            else self._select_evidence_candidates(candidates)
        )

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
            item.candidate.candidate.candidate_id for item in selected
        ]

        evidence_knowledge_keys = [
            item.candidate.candidate.metadata.knowledge_key for item in selected
        ]

        evidence_pairs = {
            (
                item.candidate.candidate.candidate_id,
                item.candidate.candidate.metadata.knowledge_key,
            )
            for item in selected
        }

        evidence_text = self._build_evidence(selected)

        context_text = self._build_context_text(context or {})

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
            evidence_candidate_ids=(evidence_candidate_ids),
            evidence_knowledge_keys=(evidence_knowledge_keys),
            evidence_pairs=evidence_pairs,
        )

    def _build_evidence(self, candidates: list[RerankedCandidate]) -> str:
        blocks: list[str] = []

        for index, item in enumerate(candidates, start=1):
            hybrid = item.candidate
            candidate = hybrid.candidate
            metadata = candidate.metadata

            content = candidate.content or ""

            content = content[: self.max_content_chars]

            blocks.append(
                "\n".join(
                    [
                        f"EVIDENCE {index}",
                        ("candidate_id: " f"{candidate.candidate_id}"),
                        ("knowledge_key: " f"{metadata.knowledge_key}"),
                        ("title: " f"{candidate.title}"),
                        ("category: " f"{metadata.category}"),
                        ("knowledge_type: " f"{metadata.knowledge_type}"),
                        ("language: " f"{metadata.language.value}"),
                        ("market_scope: " f"{metadata.market_scope.value}"),
                        ("countries: " f"{metadata.countries}"),
                        ("semantic_score: " f"{item.rerank_score}"),
                        "content:",
                        content,
                    ]
                )
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _build_context_text(context: Mapping[str, str]) -> str:
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

            lines.append(f"{key}: {normalized}")

        if not lines:
            return "none"

        return "\n".join(lines)

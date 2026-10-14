from __future__ import annotations

from app.ai.base import StructuredLLM
from app.generation.schemas import GeneratedCitation
from app.grounding.prompts import GROUNDING_SYSTEM_PROMPT, build_grounding_prompt
from app.grounding.schemas import (
    GroundingEvidence,
    GroundingLLMResponse,
    GroundingValidationResult,
)


class GroundingValidationService:
    def __init__(self, *, llm: StructuredLLM, max_content_chars: int = 4000):
        self.llm = llm
        self.max_content_chars = max_content_chars

    async def validate(
        self,
        *,
        answer: str,
        citations: list[GeneratedCitation],
        evidence: list[GroundingEvidence],
        abstained: bool = False,
    ) -> GroundingValidationResult:
        (
            normalized_answer,
            evidence_by_id,
            cited_candidate_ids,
            prevalidation_result,
        ) = self._prevalidate(
            answer=answer, citations=citations, evidence=evidence, abstained=abstained
        )

        if prevalidation_result is not None:
            return prevalidation_result

        cited_evidence = [
            evidence_by_id[candidate_id] for candidate_id in sorted(cited_candidate_ids)
        ]

        evidence_text = self._build_evidence_text(cited_evidence)

        user_prompt = build_grounding_prompt(
            answer=normalized_answer, evidence_text=(evidence_text)
        )

        response = await self.llm.generate_structured(
            system_prompt=(GROUNDING_SYSTEM_PROMPT),
            user_prompt=user_prompt,
            response_model=(GroundingLLMResponse),
        )

        return self._validate_response(
            response=response, cited_candidate_ids=(cited_candidate_ids)
        )

    def validate_deterministic(
        self,
        *,
        answer: str,
        citations: list[GeneratedCitation],
        evidence: list[GroundingEvidence],
        abstained: bool = False,
    ) -> GroundingValidationResult:
        _, evidence_by_id, cited_candidate_ids, prevalidation_result = (
            self._prevalidate(
                answer=answer,
                citations=citations,
                evidence=evidence,
                abstained=abstained,
            )
        )

        if prevalidation_result is not None:
            return prevalidation_result

        issues: list[str] = []

        for citation in citations:
            candidate_id = citation.candidate_id

            evidence_item = evidence_by_id.get(candidate_id)

            if evidence_item is None:
                issues.append("CITED_EVIDENCE_NOT_FOUND:" f"{candidate_id}")
                continue

            citation_knowledge_key = str(citation.knowledge_key or "").strip()

            evidence_knowledge_key = str(evidence_item.knowledge_key or "").strip()

            if citation_knowledge_key != evidence_knowledge_key:
                issues.append("CITATION_KNOWLEDGE_KEY_" "MISMATCH:" f"{candidate_id}")

            if not (evidence_item.content and evidence_item.content.strip()):
                issues.append("EMPTY_CITED_EVIDENCE:" f"{candidate_id}")

        if issues:
            return GroundingValidationResult(
                valid=False,
                skipped=False,
                reason=("DETERMINISTIC_GROUNDING_" "VALIDATION_FAILED"),
                issues=issues,
            )

        return GroundingValidationResult(
            valid=True,
            skipped=True,
            reason=("DETERMINISTIC_LOW_RISK_" "FAST_PATH"),
            issues=[],
        )

    @staticmethod
    def _prevalidate(
        *,
        answer: str,
        citations: list[GeneratedCitation],
        evidence: list[GroundingEvidence],
        abstained: bool,
    ) -> tuple[
        str, dict[str, GroundingEvidence], set[str], GroundingValidationResult | None
    ]:
        normalized_answer = answer.strip()

        evidence_by_id = {item.candidate_id: item for item in evidence}

        cited_candidate_ids = {citation.candidate_id for citation in citations}

        if abstained:
            return (
                normalized_answer,
                evidence_by_id,
                cited_candidate_ids,
                GroundingValidationResult(
                    valid=True, skipped=True, reason=("ANSWER_ABSTAINED")
                ),
            )

        if not normalized_answer:
            return (
                normalized_answer,
                evidence_by_id,
                cited_candidate_ids,
                GroundingValidationResult(
                    valid=True, skipped=True, reason=("EMPTY_ANSWER")
                ),
            )

        if not citations:
            return (
                normalized_answer,
                evidence_by_id,
                cited_candidate_ids,
                GroundingValidationResult(
                    valid=False,
                    reason=("NO_VALIDATED_CITATIONS"),
                    issues=["NO_VALIDATED_CITATIONS"],
                ),
            )

        missing_candidate_ids = cited_candidate_ids - set(evidence_by_id)

        if missing_candidate_ids:
            return (
                normalized_answer,
                evidence_by_id,
                cited_candidate_ids,
                GroundingValidationResult(
                    valid=False,
                    reason=("CITED_EVIDENCE_NOT_FOUND"),
                    issues=[
                        ("CITED_EVIDENCE_NOT_FOUND:" f"{candidate_id}")
                        for candidate_id in sorted(missing_candidate_ids)
                    ],
                ),
            )

        return (normalized_answer, evidence_by_id, cited_candidate_ids, None)

    def _build_evidence_text(self, evidence: list[GroundingEvidence]) -> str:
        blocks: list[str] = []

        for index, item in enumerate(evidence, start=1):
            content = item.content or ""

            content = content[: self.max_content_chars]

            blocks.append(
                "\n".join(
                    [
                        (f"EVIDENCE " f"{index}"),
                        ("candidate_id: " f"{item.candidate_id}"),
                        ("knowledge_key: " f"{item.knowledge_key}"),
                        ("title: " f"{item.title}"),
                        "content:",
                        content,
                    ]
                )
            )

        return "\n\n".join(blocks)

    @staticmethod
    def _validate_response(
        *, response: GroundingLLMResponse, cited_candidate_ids: set[str]
    ) -> GroundingValidationResult:
        issues: list[str] = []

        if not response.claims:
            issues.append("NO_CLAIMS_ANALYZED")

        for claim in response.claims:
            for candidate_id in claim.supporting_candidate_ids:
                if candidate_id not in cited_candidate_ids:
                    issues.append(
                        "VALIDATOR_UNKNOWN_" "CANDIDATE_ID:" f"{candidate_id}"
                    )

            if not claim.supported:
                issues.append("UNSUPPORTED_CLAIM")

        valid = response.overall_supported and not issues

        reason = None

        if not valid:
            reason = "GROUNDING_VALIDATION_FAILED"

        return GroundingValidationResult(
            valid=valid,
            skipped=False,
            reason=reason,
            claims=response.claims,
            issues=issues,
        )

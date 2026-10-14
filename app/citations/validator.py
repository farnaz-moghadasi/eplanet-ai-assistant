from __future__ import annotations

from dataclasses import dataclass, field

from app.generation.schemas import GeneratedCitation


@dataclass
class CitationValidationIssue:
    code: str
    candidate_id: str | None = None
    knowledge_key: str | None = None


@dataclass
class CitationValidationResult:
    valid: bool

    citations: list[GeneratedCitation] = field(default_factory=list)

    issues: list[CitationValidationIssue] = field(default_factory=list)


class CitationValidator:
    def validate(
        self,
        *,
        citations: list[GeneratedCitation],
        evidence_candidate_ids: list[str],
        evidence_knowledge_keys: list[str],
        evidence_pairs: set[tuple[str, str]],
        require_citation: bool = False,
    ) -> CitationValidationResult:
        issues: list[CitationValidationIssue] = []

        valid_citations: list[GeneratedCitation] = []

        candidate_ids = set(evidence_candidate_ids)

        knowledge_keys = set(evidence_knowledge_keys)

        seen_pairs: set[tuple[str, str]] = set()

        for citation in citations:
            candidate_id = citation.candidate_id.strip()

            knowledge_key = citation.knowledge_key.strip()

            pair = (candidate_id, knowledge_key)

            if not candidate_id:
                issues.append(
                    CitationValidationIssue(
                        code=("EMPTY_CANDIDATE_ID"),
                        knowledge_key=(knowledge_key or None),
                    )
                )
                continue

            if not knowledge_key:
                issues.append(
                    CitationValidationIssue(
                        code=("EMPTY_KNOWLEDGE_KEY"), candidate_id=(candidate_id)
                    )
                )
                continue

            if candidate_id not in candidate_ids:
                issues.append(
                    CitationValidationIssue(
                        code=("UNKNOWN_CANDIDATE_ID"),
                        candidate_id=(candidate_id),
                        knowledge_key=(knowledge_key),
                    )
                )
                continue

            if knowledge_key not in knowledge_keys:
                issues.append(
                    CitationValidationIssue(
                        code=("UNKNOWN_KNOWLEDGE_KEY"),
                        candidate_id=(candidate_id),
                        knowledge_key=(knowledge_key),
                    )
                )
                continue

            if pair not in evidence_pairs:
                issues.append(
                    CitationValidationIssue(
                        code=("CITATION_PAIR_MISMATCH"),
                        candidate_id=(candidate_id),
                        knowledge_key=(knowledge_key),
                    )
                )
                continue

            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)

            valid_citations.append(citation)

        if require_citation and not valid_citations:
            issues.append(CitationValidationIssue(code="MISSING_CITATION"))

        return CitationValidationResult(
            valid=not issues, citations=(valid_citations), issues=issues
        )

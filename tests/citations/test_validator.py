from app.citations.validator import CitationValidator
from app.generation.schemas import GeneratedCitation


def test_valid_citation_passes():
    validator = CitationValidator()

    result = validator.validate(
        citations=[GeneratedCitation(candidate_id="C1", knowledge_key="KB-1")],
        evidence_candidate_ids=["C1"],
        evidence_knowledge_keys=["KB-1"],
        evidence_pairs={("C1", "KB-1")},
        require_citation=True,
    )

    assert result.valid is True

    assert len(result.citations) == 1

    assert result.issues == []


def test_unknown_candidate_is_rejected():
    validator = CitationValidator()

    result = validator.validate(
        citations=[GeneratedCitation(candidate_id="FAKE", knowledge_key="KB-1")],
        evidence_candidate_ids=["C1"],
        evidence_knowledge_keys=["KB-1"],
        evidence_pairs={("C1", "KB-1")},
        require_citation=True,
    )

    assert result.valid is False

    codes = [issue.code for issue in result.issues]

    assert "UNKNOWN_CANDIDATE_ID" in codes

    assert "MISSING_CITATION" in codes

    assert result.citations == []


def test_unknown_knowledge_key_is_rejected():
    validator = CitationValidator()

    result = validator.validate(
        citations=[GeneratedCitation(candidate_id="C1", knowledge_key="FAKE-KB")],
        evidence_candidate_ids=["C1"],
        evidence_knowledge_keys=["KB-1"],
        evidence_pairs={("C1", "KB-1")},
        require_citation=True,
    )

    assert result.valid is False

    codes = [issue.code for issue in result.issues]

    assert "UNKNOWN_KNOWLEDGE_KEY" in codes

    assert result.citations == []


def test_mismatched_pair_is_rejected():
    validator = CitationValidator()

    result = validator.validate(
        citations=[GeneratedCitation(candidate_id="C1", knowledge_key="KB-2")],
        evidence_candidate_ids=["C1", "C2"],
        evidence_knowledge_keys=["KB-1", "KB-2"],
        evidence_pairs={("C1", "KB-1"), ("C2", "KB-2")},
        require_citation=True,
    )

    assert result.valid is False

    codes = [issue.code for issue in result.issues]

    assert "CITATION_PAIR_MISMATCH" in codes

    assert result.citations == []


def test_duplicate_valid_citation_is_deduplicated():
    validator = CitationValidator()

    citation = GeneratedCitation(candidate_id="C1", knowledge_key="KB-1")

    result = validator.validate(
        citations=[citation, citation],
        evidence_candidate_ids=["C1"],
        evidence_knowledge_keys=["KB-1"],
        evidence_pairs={("C1", "KB-1")},
        require_citation=True,
    )

    assert result.valid is True

    assert len(result.citations) == 1


def test_missing_citation_is_rejected_when_required():
    validator = CitationValidator()

    result = validator.validate(
        citations=[],
        evidence_candidate_ids=["C1"],
        evidence_knowledge_keys=["KB-1"],
        evidence_pairs={("C1", "KB-1")},
        require_citation=True,
    )

    assert result.valid is False

    assert result.issues[0].code == "MISSING_CITATION"


def test_missing_citation_allowed_when_not_required():
    validator = CitationValidator()

    result = validator.validate(
        citations=[],
        evidence_candidate_ids=[],
        evidence_knowledge_keys=[],
        evidence_pairs=set(),
        require_citation=False,
    )

    assert result.valid is True
    assert result.citations == []
    assert result.issues == []


def test_valid_and_invalid_citations_are_sanitized():
    validator = CitationValidator()

    result = validator.validate(
        citations=[
            GeneratedCitation(candidate_id="C1", knowledge_key="KB-1"),
            GeneratedCitation(candidate_id="FAKE", knowledge_key="KB-FAKE"),
        ],
        evidence_candidate_ids=["C1"],
        evidence_knowledge_keys=["KB-1"],
        evidence_pairs={("C1", "KB-1")},
        require_citation=True,
    )

    assert result.valid is False

    assert len(result.citations) == 1

    assert result.citations[0].candidate_id == "C1"

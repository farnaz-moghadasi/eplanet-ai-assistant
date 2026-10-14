import pytest

from app.generation.schemas import GeneratedCitation
from app.grounding.schemas import (
    GroundingClaimAssessment,
    GroundingEvidence,
    GroundingLLMResponse,
)
from app.grounding.service import GroundingValidationService


class SupportedLLM:
    async def generate_structured(self, *, system_prompt, user_prompt, response_model):
        return GroundingLLMResponse(
            overall_supported=True,
            claims=[
                GroundingClaimAssessment(
                    claim=("The user can change " "the password."),
                    supported=True,
                    supporting_candidate_ids=["C1"],
                    reason=("The evidence directly " "supports the claim."),
                )
            ],
        )


class UnsupportedLLM:
    async def generate_structured(self, *, system_prompt, user_prompt, response_model):
        return GroundingLLMResponse(
            overall_supported=False,
            claims=[
                GroundingClaimAssessment(
                    claim=("The password expires " "after 30 days."),
                    supported=False,
                    supporting_candidate_ids=[],
                    reason=("The evidence does not " "state this."),
                )
            ],
        )


class InvalidCandidateLLM:
    async def generate_structured(self, *, system_prompt, user_prompt, response_model):
        return GroundingLLMResponse(
            overall_supported=True,
            claims=[
                GroundingClaimAssessment(
                    claim="Supported claim",
                    supported=True,
                    supporting_candidate_ids=["FAKE"],
                    reason="test",
                )
            ],
        )


@pytest.mark.asyncio
async def test_supported_answer_passes():
    service = GroundingValidationService(llm=SupportedLLM())

    result = await service.validate(
        answer=("You can change the " "trading password."),
        citations=[GeneratedCitation(candidate_id="C1", knowledge_key="KB-1")],
        evidence=[
            GroundingEvidence(
                candidate_id="C1",
                knowledge_key="KB-1",
                title="Change password",
                content=("Users can change their " "trading password."),
            )
        ],
    )

    assert result.valid is True
    assert result.skipped is False
    assert result.issues == []


@pytest.mark.asyncio
async def test_unsupported_claim_fails():
    service = GroundingValidationService(llm=UnsupportedLLM())

    result = await service.validate(
        answer=("The password expires " "after 30 days."),
        citations=[GeneratedCitation(candidate_id="C1", knowledge_key="KB-1")],
        evidence=[
            GroundingEvidence(
                candidate_id="C1",
                knowledge_key="KB-1",
                title="Password",
                content=("Users can change " "their password."),
            )
        ],
    )

    assert result.valid is False

    assert "UNSUPPORTED_CLAIM" in result.issues


@pytest.mark.asyncio
async def test_missing_citations_fails():
    service = GroundingValidationService(llm=SupportedLLM())

    result = await service.validate(
        answer="Some factual answer.", citations=[], evidence=[]
    )

    assert result.valid is False

    assert result.reason == "NO_VALIDATED_CITATIONS"


@pytest.mark.asyncio
async def test_abstained_answer_skips_validation():
    service = GroundingValidationService(llm=SupportedLLM())

    result = await service.validate(
        answer="", citations=[], evidence=[], abstained=True
    )

    assert result.valid is True
    assert result.skipped is True

    assert result.reason == "ANSWER_ABSTAINED"


@pytest.mark.asyncio
async def test_unknown_validator_candidate_fails():
    service = GroundingValidationService(llm=InvalidCandidateLLM())

    result = await service.validate(
        answer="Supported answer.",
        citations=[GeneratedCitation(candidate_id="C1", knowledge_key="KB-1")],
        evidence=[
            GroundingEvidence(
                candidate_id="C1",
                knowledge_key="KB-1",
                title="Article",
                content="Supported answer.",
            )
        ],
    )

    assert result.valid is False

    assert any(
        issue.startswith("VALIDATOR_UNKNOWN_" "CANDIDATE_ID:")
        for issue in result.issues
    )

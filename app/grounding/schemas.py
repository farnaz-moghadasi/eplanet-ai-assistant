from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
)


class GroundingEvidence(BaseModel):
    candidate_id: str
    knowledge_key: str
    title: str
    content: str


class GroundingClaimAssessment(BaseModel):
    claim: str

    supported: bool

    supporting_candidate_ids: list[str] = (
        Field(
            default_factory=list
        )
    )

    reason: str | None = None


class GroundingLLMResponse(BaseModel):
    overall_supported: bool

    claims: list[
        GroundingClaimAssessment
    ] = Field(
        default_factory=list
    )


class GroundingValidationResult(BaseModel):
    valid: bool

    skipped: bool = False

    reason: str | None = None

    claims: list[
        GroundingClaimAssessment
    ] = Field(
        default_factory=list
    )

    issues: list[str] = Field(
        default_factory=list
    )
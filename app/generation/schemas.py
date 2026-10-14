from pydantic import BaseModel, Field


class GeneratedCitation(BaseModel):
    candidate_id: str
    knowledge_key: str


class GeneratedAnswerPayload(BaseModel):
    answer: str = ""

    citations: list[GeneratedCitation] = Field(default_factory=list)

    abstained: bool = False

    abstain_reason: str | None = None

    needs_handoff: bool = False

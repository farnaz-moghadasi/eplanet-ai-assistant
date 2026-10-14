from pydantic import BaseModel, Field

from app.knowledge.schemas import KnowledgeMetadata


class KnowledgeCandidate(BaseModel):
    candidate_id: str
    title: str
    content: str

    metadata: KnowledgeMetadata

    score: float | None = None

    source_page_id: str | None = None


class RetrievalResult(BaseModel):
    candidates: list[KnowledgeCandidate] = Field(default_factory=list)

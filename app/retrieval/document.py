from pydantic import BaseModel

from app.knowledge.schemas import KnowledgeMetadata


class KnowledgeDocument(BaseModel):
    document_id: str
    page_id: str
    title: str
    content: str
    metadata: KnowledgeMetadata
    version: int | None = None


class KnowledgeChunk(BaseModel):
    chunk_id: str
    document_id: str
    page_id: str

    title: str
    content: str

    chunk_index: int

    metadata: KnowledgeMetadata

    version: int | None = None
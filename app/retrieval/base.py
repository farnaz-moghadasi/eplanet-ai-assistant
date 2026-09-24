from typing import Protocol

from app.retrieval.schemas import KnowledgeCandidate


class KnowledgeRetriever(Protocol):
    async def retrieve(
        self,
        *,
        query: str,
        knowledge_hints: list[str],
        limit: int = 10,
    ) -> list[KnowledgeCandidate]:
        ...
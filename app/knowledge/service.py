# app/knowledge/service.py

from app.knowledge.confluence_client import ConfluenceClient
from app.knowledge.parser import KnowledgeParser
from app.knowledge.validator import KnowledgeValidator


class KnowledgeService:
    def __init__(
        self,
        confluence_client: ConfluenceClient,
    ):
        self.confluence_client = confluence_client
        self.parser = KnowledgeParser()
        self.validator = KnowledgeValidator()

    async def inspect_page(
        self,
        page_id: str,
    ):
        raw_page = await self.confluence_client.get_page(page_id)

        parsed = self.parser.parse(raw_page)

        validation = self.validator.validate(
            parsed.metadata
        )

        return {
            "page": parsed,
            "validation": validation,
        }
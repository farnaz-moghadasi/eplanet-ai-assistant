from dataclasses import dataclass, field

from app.knowledge.confluence_client import ConfluenceClient
from app.knowledge.parser import KnowledgeParser
from app.knowledge.validator import KnowledgeValidator
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.chunker import KnowledgeChunker
from app.retrieval.document import KnowledgeChunk


@dataclass
class IngestionPageResult:
    page_id: str
    title: str | None = None

    success: bool = False
    ai_ready: bool = False

    chunk_count: int = 0

    validation_reasons: list[str] = field(default_factory=list)

    error: str | None = None


@dataclass
class IngestionReport:
    requested_pages: int = 0
    successful_pages: int = 0
    indexed_pages: int = 0
    skipped_pages: int = 0
    failed_pages: int = 0

    total_chunks: int = 0

    page_results: list[IngestionPageResult] = field(default_factory=list)


class KnowledgeIngestionService:
    def __init__(
        self,
        *,
        confluence_client: ConfluenceClient,
        retriever: BM25Retriever,
        parser: KnowledgeParser | None = None,
        validator: KnowledgeValidator | None = None,
        chunker: KnowledgeChunker | None = None,
    ):
        self.confluence_client = confluence_client

        self.parser = parser or KnowledgeParser()

        self.validator = validator or KnowledgeValidator()

        self.chunker = chunker or KnowledgeChunker()

        self.retriever = retriever

    async def collect_indexable_chunks(
        self, page_ids: list[str]
    ) -> tuple[IngestionReport, list[KnowledgeChunk]]:
        unique_page_ids = self._deduplicate_page_ids(page_ids)

        report = IngestionReport(requested_pages=len(unique_page_ids))

        all_chunks: list[KnowledgeChunk] = []

        for page_id in unique_page_ids:
            page_result, chunks = await self._ingest_page(page_id)

            report.page_results.append(page_result)

            if page_result.success:
                report.successful_pages += 1

            if page_result.ai_ready:
                report.indexed_pages += 1

                report.total_chunks += len(chunks)

                all_chunks.extend(chunks)

            elif page_result.success:
                report.skipped_pages += 1

            else:
                report.failed_pages += 1

        return (report, all_chunks)

    async def sync_pages(self, page_ids: list[str]) -> IngestionReport:
        report, chunks = await self.collect_indexable_chunks(page_ids)

        self.retriever.index(chunks)

        return report

    async def _ingest_page(
        self, page_id: str
    ) -> tuple[IngestionPageResult, list[KnowledgeChunk]]:
        try:
            raw_page = await self.confluence_client.get_page(page_id)

            parsed_page = self.parser.parse(raw_page)

            validation = self.validator.validate(parsed_page.metadata)

            if not validation.ai_ready:
                return (
                    IngestionPageResult(
                        page_id=page_id,
                        title=(parsed_page.title),
                        success=True,
                        ai_ready=False,
                        validation_reasons=(validation.reasons),
                    ),
                    [],
                )

            chunks = self.chunker.chunk_page(parsed_page)

            if not chunks:
                return (
                    IngestionPageResult(
                        page_id=page_id,
                        title=(parsed_page.title),
                        success=False,
                        ai_ready=False,
                        error=("NO_INDEXABLE_CONTENT"),
                    ),
                    [],
                )

            return (
                IngestionPageResult(
                    page_id=page_id,
                    title=(parsed_page.title),
                    success=True,
                    ai_ready=True,
                    chunk_count=len(chunks),
                ),
                chunks,
            )

        except Exception as exc:
            return (
                IngestionPageResult(
                    page_id=page_id,
                    success=False,
                    ai_ready=False,
                    error=self._safe_error(exc),
                ),
                [],
            )

    @staticmethod
    def _deduplicate_page_ids(page_ids: list[str]) -> list[str]:
        result: list[str] = []

        seen: set[str] = set()

        for page_id in page_ids:
            normalized = str(page_id).strip()

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)

            result.append(normalized)

        return result

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        return (f"{exc.__class__.__name__}: " f"{str(exc)}")[:500]

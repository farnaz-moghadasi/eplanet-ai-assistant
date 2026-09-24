import pytest

from app.knowledge.ingestion import (
    KnowledgeIngestionService,
)
from app.retrieval.bm25 import BM25Retriever


def make_page(
    *,
    page_id: str,
    title: str,
    status: str = "APPROVED",
    market_scope: str = "GLOBAL",
    countries: str = "",
    required_context_fields: str = "NONE",
    content: str = "Example knowledge content.",
):
    return {
        "id": page_id,
        "title": title,
        "version": {
            "number": 1,
        },
        "body": {
            "storage": {
                "value": f"""
                    <table>
                        <tr>
                            <td>Knowledge Key</td>
                            <td>KB-{page_id}</td>
                        </tr>

                        <tr>
                            <td>Canonical Knowledge Key</td>
                            <td>KB-{page_id}</td>
                        </tr>

                        <tr>
                            <td>Knowledge Type</td>
                            <td>REQUIREMENTS</td>
                        </tr>

                        <tr>
                            <td>Category</td>
                            <td>KYC</td>
                        </tr>

                        <tr>
                            <td>Status</td>
                            <td>{status}</td>
                        </tr>

                        <tr>
                            <td>Audience</td>
                            <td>BOTH</td>
                        </tr>

                        <tr>
                            <td>Language</td>
                            <td>EN</td>
                        </tr>

                        <tr>
                            <td>Market Scope</td>
                            <td>{market_scope}</td>
                        </tr>

                        <tr>
                            <td>Countries</td>
                            <td>{countries}</td>
                        </tr>

                        <tr>
                            <td>Required Context Fields</td>
                            <td>{required_context_fields}</td>
                        </tr>

                        <tr>
                            <td>Bot Mode</td>
                            <td>KNOWLEDGE</td>
                        </tr>

                        <tr>
                            <td>Fallback Policy</td>
                            <td>NONE</td>
                        </tr>
                    </table>

                    <h1>{title}</h1>

                    <p>{content}</p>
                """
            }
        },
    }


class FakeConfluenceClient:
    def __init__(
        self,
        pages: dict[str, dict],
    ):
        self.pages = pages

    async def get_page(
        self,
        page_id: str,
    ):
        if page_id not in self.pages:
            raise RuntimeError(
                "PAGE_NOT_FOUND"
            )

        return self.pages[
            page_id
        ]


@pytest.mark.asyncio
async def test_ingestion_indexes_valid_page():
    pages = {
        "1": make_page(
            page_id="1",
            title="Afghan KYC",
            market_scope="COUNTRY",
            countries="AF",
            required_context_fields="kyc_document_country",
            content=(
                "Afghan clients may provide "
                "approved identity documents."
            ),
        )
    }

    retriever = BM25Retriever()

    service = KnowledgeIngestionService(
        confluence_client=FakeConfluenceClient(
            pages
        ),
        retriever=retriever,
    )

    report = await service.sync_pages(
        ["1"]
    )

    assert report.requested_pages == 1
    assert report.successful_pages == 1
    assert report.indexed_pages == 1
    assert report.skipped_pages == 0
    assert report.failed_pages == 0

    assert report.total_chunks >= 1

    result = report.page_results[0]

    assert result.success is True
    assert result.ai_ready is True
    assert result.chunk_count >= 1


@pytest.mark.asyncio
async def test_ingestion_skips_not_approved_page():
    pages = {
        "2": make_page(
            page_id="2",
            title="Draft Page",
            status="DRAFT",
        )
    }

    retriever = BM25Retriever()

    service = KnowledgeIngestionService(
        confluence_client=FakeConfluenceClient(
            pages
        ),
        retriever=retriever,
    )

    report = await service.sync_pages(
        ["2"]
    )

    assert report.successful_pages == 1
    assert report.indexed_pages == 0
    assert report.skipped_pages == 1
    assert report.failed_pages == 0

    result = report.page_results[0]

    assert result.ai_ready is False

    assert (
        "STATUS_NOT_APPROVED"
        in result.validation_reasons
    )


@pytest.mark.asyncio
async def test_ingestion_accepts_country_metadata_without_context_field_for_now():
    pages = {
        "3": make_page(
            page_id="3",
            title="Country Article",
            market_scope="COUNTRY",
            countries="IR",
            required_context_fields="NONE",
        )
    }

    retriever = BM25Retriever()

    service = KnowledgeIngestionService(
        confluence_client=FakeConfluenceClient(
            pages
        ),
        retriever=retriever,
    )

    report = await service.sync_pages(
        ["3"]
    )

    assert report.requested_pages == 1
    assert report.successful_pages == 1
    assert report.indexed_pages == 1
    assert report.skipped_pages == 0
    assert report.failed_pages == 0

    result = report.page_results[0]

    assert result.success is True
    assert result.ai_ready is True
    assert result.validation_reasons == []


@pytest.mark.asyncio
async def test_ingestion_skips_country_without_countries():
    pages = {
        "4": make_page(
            page_id="4",
            title="Broken Country Article",
            market_scope="COUNTRY",
            countries="",
            required_context_fields="NONE",
        )
    }

    retriever = BM25Retriever()

    service = KnowledgeIngestionService(
        confluence_client=FakeConfluenceClient(
            pages
        ),
        retriever=retriever,
    )

    report = await service.sync_pages(
        ["4"]
    )

    assert report.requested_pages == 1
    assert report.successful_pages == 1
    assert report.indexed_pages == 0
    assert report.skipped_pages == 1
    assert report.failed_pages == 0

    result = report.page_results[0]

    assert result.ai_ready is False

    assert (
        "COUNTRIES_REQUIRED"
        in result.validation_reasons
    )

@pytest.mark.asyncio
async def test_ingestion_failure_does_not_stop_other_pages():
    pages = {
        "1": make_page(
            page_id="1",
            title="Valid Article",
        )
    }

    retriever = BM25Retriever()

    service = KnowledgeIngestionService(
        confluence_client=FakeConfluenceClient(
            pages
        ),
        retriever=retriever,
    )

    report = await service.sync_pages(
        [
            "404",
            "1",
        ]
    )

    assert report.requested_pages == 2

    assert report.failed_pages == 1

    assert report.indexed_pages == 1

    failed_result = (
        report.page_results[0]
    )

    assert failed_result.success is False
    assert "PAGE_NOT_FOUND" in failed_result.error


@pytest.mark.asyncio
async def test_ingestion_deduplicates_page_ids():
    pages = {
        "1": make_page(
            page_id="1",
            title="Valid Article",
        )
    }

    retriever = BM25Retriever()

    service = KnowledgeIngestionService(
        confluence_client=FakeConfluenceClient(
            pages
        ),
        retriever=retriever,
    )

    report = await service.sync_pages(
        [
            "1",
            "1",
            " 1 ",
            "",
        ]
    )

    assert report.requested_pages == 1
    assert report.indexed_pages == 1
    
@pytest.mark.asyncio
async def test_ingested_content_is_searchable():
    pages = {
        "1": make_page(
            page_id="1",
            title="Afghan KYC Documents",
            market_scope="COUNTRY",
            countries="AF",
            required_context_fields="kyc_document_country",
            content=(
                "Afghan clients can provide "
                "identity documents for verification."
            ),
        ),
        "2": make_page(
            page_id="2",
            title="Deposit Information",
            content=(
                "Clients can fund their trading "
                "accounts using supported deposit methods."
            ),
        ),
    }

    retriever = BM25Retriever()

    service = KnowledgeIngestionService(
        confluence_client=FakeConfluenceClient(
            pages
        ),
        retriever=retriever,
    )

    report = await service.sync_pages(
        [
            "1",
            "2",
        ]
    )

    assert report.indexed_pages == 2

    results = await retriever.retrieve(
        query="Afghan identity verification documents",
        knowledge_hints=[
            "KYC",
            "verification",
        ],
        limit=5,
    )

    assert results

    assert (
        results[0].metadata.knowledge_key
        == "KB-1"
    )
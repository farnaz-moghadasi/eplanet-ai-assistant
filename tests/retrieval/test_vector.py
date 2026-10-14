import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from app.knowledge.enums import (
    Audience,
    BotMode,
    KnowledgeStatus,
    Language,
    MarketScope,
)
from app.knowledge.schemas import KnowledgeMetadata
from app.retrieval.document import KnowledgeChunk
from app.retrieval.vector import QdrantVectorRetriever


class FakeEmbeddingProvider:
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakePoint:
    def __init__(self, score: float, payload: dict):
        self.score = score
        self.payload = payload


class FakeQueryResult:
    def __init__(self, points):
        self.points = points


class FlakyQdrantClient:
    """Fails a fixed number of times, then succeeds."""

    def __init__(self, failures_before_success: int):
        self.failures_before_success = failures_before_success
        self.calls = 0

    async def query_points(self, **kwargs):
        self.calls += 1

        if self.calls <= self.failures_before_success:
            raise ResponseHandlingException(
                ConnectionError("all connection attempts failed")
            )

        return FakeQueryResult(points=[])


class AlwaysDownQdrantClient:
    def __init__(self):
        self.calls = 0

    async def query_points(self, **kwargs):
        self.calls += 1

        raise ResponseHandlingException(
            ConnectionError("all connection attempts failed")
        )


def make_retriever(client) -> QdrantVectorRetriever:
    return QdrantVectorRetriever(
        client=client,
        embedding_provider=FakeEmbeddingProvider(),
        collection_name="test_collection",
        vector_size=3,
    )


class FakeScrollRecord:
    def __init__(self, id, payload):
        self.id = id
        self.payload = payload


class FakeSyncQdrantClient:
    """In-memory double covering what sync_chunks needs: upsert, a
    page_id-filtered scroll, an unfiltered scroll, and delete-by-id."""

    def __init__(self, existing_points: dict[str, dict] | None = None):
        self.points: dict[str, dict] = dict(existing_points or {})

    async def collection_exists(self, name: str) -> bool:
        return True

    async def create_collection(self, **kwargs):
        raise AssertionError(
            "collection already exists; create_collection " "should not be called"
        )

    async def upsert(self, *, collection_name, points, wait=True):
        for point in points:
            self.points[str(point.id)] = dict(point.payload)

    async def scroll(
        self,
        *,
        collection_name,
        limit,
        offset=None,
        with_payload=True,
        with_vectors=False,
        scroll_filter=None,
    ):
        page_id_filter = None

        if scroll_filter is not None:
            for condition in scroll_filter.must:
                if condition.key == "page_id":
                    page_id_filter = condition.match.value

        records = [
            FakeScrollRecord(id=point_id, payload=payload)
            for point_id, payload in self.points.items()
            if (page_id_filter is None or payload.get("page_id") == page_id_filter)
        ]

        return records, None

    async def delete(self, *, collection_name, points_selector, wait=True):
        for point_id in points_selector.points:
            self.points.pop(str(point_id), None)


def make_metadata(knowledge_key: str) -> KnowledgeMetadata:
    return KnowledgeMetadata(
        knowledge_key=knowledge_key,
        canonical_knowledge_key=knowledge_key,
        knowledge_type="HOW_TO",
        category="GENERAL",
        status=KnowledgeStatus.APPROVED,
        audience=Audience.BOTH,
        language=Language.EN,
        market_scope=MarketScope.GLOBAL,
        bot_mode=BotMode.KNOWLEDGE,
    )


def make_chunk(*, chunk_id: str, page_id: str, knowledge_key: str) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        document_id=page_id,
        page_id=page_id,
        title="Test title",
        content="Test content",
        chunk_index=0,
        metadata=make_metadata(knowledge_key),
    )


@pytest.mark.asyncio
async def test_transient_qdrant_failure_recovers_on_retry():

    client = FlakyQdrantClient(failures_before_success=1)
    retriever = make_retriever(client)

    candidates = await retriever.retrieve(query="test query", limit=5)

    assert candidates == []
    assert client.calls == 2


@pytest.mark.asyncio
async def test_persistent_qdrant_outage_degrades_to_empty_list():

    client = AlwaysDownQdrantClient()
    retriever = make_retriever(client)

    candidates = await retriever.retrieve(query="test query", limit=5)

    assert candidates == []
    assert client.calls == QdrantVectorRetriever._QUERY_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_sync_removes_vectors_for_pages_dropped_from_managed_list():

    aaa_point_id = QdrantVectorRetriever._point_id("aaa-chunk-1")
    bbb_point_id = QdrantVectorRetriever._point_id("bbb-chunk-1")

    client = FakeSyncQdrantClient(
        existing_points={
            aaa_point_id: {"page_id": "aaa"},
            bbb_point_id: {"page_id": "bbb"},
        }
    )

    retriever = make_retriever(client)

    current_chunks = [
        make_chunk(chunk_id="aaa-chunk-1", page_id="aaa", knowledge_key="KB-A")
    ]

    report = await retriever.sync_chunks(
        current_chunks, managed_page_ids=["aaa"] 
    )

    assert aaa_point_id in client.points
    assert bbb_point_id not in client.points
    assert report.orphaned_pages_removed == 1
    assert report.stale_points_deleted == 1


@pytest.mark.asyncio
async def test_sync_keeps_pages_still_in_managed_list():

    aaa_point_id = QdrantVectorRetriever._point_id("aaa-chunk-1")

    client = FakeSyncQdrantClient(existing_points={aaa_point_id: {"page_id": "aaa"}})

    retriever = make_retriever(client)

    current_chunks = [
        make_chunk(chunk_id="aaa-chunk-1", page_id="aaa", knowledge_key="KB-A")
    ]

    report = await retriever.sync_chunks(current_chunks, managed_page_ids=["aaa"])

    assert aaa_point_id in client.points
    assert report.orphaned_pages_removed == 0
    assert report.stale_points_deleted == 0

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from app.retrieval.vector import QdrantVectorRetriever


class FakeEmbeddingProvider:
    async def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


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


@pytest.mark.asyncio
async def test_transient_qdrant_failure_recovers_on_retry():
    # A single connection blip (e.g. the container restarting) must not
    # surface to the caller at all.
    client = FlakyQdrantClient(failures_before_success=1)
    retriever = make_retriever(client)

    candidates = await retriever.retrieve(query="test query", limit=5)

    assert candidates == []
    assert client.calls == 2


@pytest.mark.asyncio
async def test_persistent_qdrant_outage_degrades_to_empty_list():
    # A real outage (every attempt fails) must not raise - the pipeline
    # falls back to BM25-only rather than the whole chat request
    # crashing with a raw 500.
    client = AlwaysDownQdrantClient()
    retriever = make_retriever(client)

    candidates = await retriever.retrieve(query="test query", limit=5)

    assert candidates == []
    assert (
        client.calls
        == QdrantVectorRetriever._QUERY_MAX_ATTEMPTS
    )

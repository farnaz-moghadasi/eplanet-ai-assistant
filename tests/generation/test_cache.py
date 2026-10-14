import pytest

from app.generation.cache import AnswerCacheService
from app.generation.schemas import GeneratedAnswerPayload, GeneratedCitation
from app.generation.service import GenerationResult


class FakeRedis:
    """In-memory double for the tiny slice of the redis-py async API
    AnswerCacheService actually uses - no real Redis server needed."""

    def __init__(self):
        self.store: dict[str, str] = {}
        self.fail_on_get = False
        self.fail_on_set = False

    async def get(self, key):
        if self.fail_on_get:
            raise ConnectionError("simulated redis outage")

        return self.store.get(key)

    async def set(self, key, value, ex=None):
        if self.fail_on_set:
            raise ConnectionError("simulated redis outage")

        self.store[key] = value

    async def incr(self, key):
        current = int(self.store.get(key, "0"))

        current += 1

        self.store[key] = str(current)

        return current

    async def aclose(self):
        pass


def make_service(fake_redis: FakeRedis) -> AnswerCacheService:
    service = AnswerCacheService(redis_url="redis://unused:6379/0")

    service.redis = fake_redis

    return service


def make_result(
    answer: str = "test answer", knowledge_key: str = "KB-TEST-001-FA"
) -> GenerationResult:
    return GenerationResult(
        payload=GeneratedAnswerPayload(
            answer=answer,
            citations=[
                GeneratedCitation(
                    candidate_id=(f"{knowledge_key}:chunk-1"),
                    knowledge_key=(knowledge_key),
                )
            ],
            abstained=False,
        ),
        evidence_candidate_ids=[f"{knowledge_key}:chunk-1"],
        evidence_knowledge_keys=[knowledge_key],
        evidence_pairs={(f"{knowledge_key}:chunk-1", knowledge_key)},
    )


@pytest.mark.asyncio
async def test_miss_then_hit_round_trip():
    redis = FakeRedis()
    service = make_service(redis)

    miss = await service.get(
        knowledge_keys=["KB-TEST-001-FA"], language="fa", context={}
    )

    assert miss is None

    result = make_result()

    await service.set(
        knowledge_keys=["KB-TEST-001-FA"], language="fa", context={}, result=result
    )

    hit = await service.get(
        knowledge_keys=["KB-TEST-001-FA"], language="fa", context={}
    )

    assert hit is not None
    assert hit.payload.answer == "test answer"
    assert hit.payload.citations[0].knowledge_key == "KB-TEST-001-FA"
    assert hit.evidence_knowledge_keys == ["KB-TEST-001-FA"]
    assert hit.evidence_pairs == {("KB-TEST-001-FA:chunk-1", "KB-TEST-001-FA")}


@pytest.mark.asyncio
async def test_knowledge_key_order_does_not_matter():
    redis = FakeRedis()
    service = make_service(redis)

    result = make_result()

    await service.set(
        knowledge_keys=["KB-A-FA", "KB-B-FA"], language="fa", context={}, result=result
    )

    hit = await service.get(
        knowledge_keys=["KB-B-FA", "KB-A-FA"], language="fa", context={}
    )

    assert hit is not None


@pytest.mark.asyncio
async def test_different_language_is_a_different_entry():
    redis = FakeRedis()
    service = make_service(redis)

    await service.set(
        knowledge_keys=["KB-A-FA"], language="fa", context={}, result=make_result()
    )

    miss = await service.get(knowledge_keys=["KB-A-FA"], language="en", context={})

    assert miss is None


@pytest.mark.asyncio
async def test_different_context_is_a_different_entry():
    redis = FakeRedis()
    service = make_service(redis)

    await service.set(
        knowledge_keys=["KB-KYC-FA"],
        language="fa",
        context={"kyc_document_country": "IR"},
        result=make_result(),
    )

    miss = await service.get(
        knowledge_keys=["KB-KYC-FA"],
        language="fa",
        context={"kyc_document_country": "AF"},
    )

    same_country_hit = await service.get(
        knowledge_keys=["KB-KYC-FA"],
        language="fa",
        context={"kyc_document_country": "IR"},
    )

    assert miss is None
    assert same_country_hit is not None


@pytest.mark.asyncio
async def test_bump_version_invalidates_existing_entries():
    redis = FakeRedis()
    service = make_service(redis)

    await service.set(
        knowledge_keys=["KB-A-FA"], language="fa", context={}, result=make_result()
    )

    await service.bump_version()

    miss = await service.get(knowledge_keys=["KB-A-FA"], language="fa", context={})

    assert miss is None


@pytest.mark.asyncio
async def test_redis_outage_on_get_is_treated_as_a_miss_not_a_crash():
    redis = FakeRedis()
    redis.fail_on_get = True
    service = make_service(redis)

    result = await service.get(knowledge_keys=["KB-A-FA"], language="fa", context={})

    assert result is None


@pytest.mark.asyncio
async def test_redis_outage_on_set_does_not_raise():
    redis = FakeRedis()
    redis.fail_on_set = True
    service = make_service(redis)


    await service.set(
        knowledge_keys=["KB-A-FA"], language="fa", context={}, result=make_result()
    )

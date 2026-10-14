import pytest
from fastapi import HTTPException

from app.api.rate_limit import enforce_chat_rate_limit
from app.core.config import settings


class FakeClient:
    def __init__(self, host: str):
        self.host = host


class FakeRequest:
    def __init__(self, *, headers=None, host="1.2.3.4"):
        self.headers = headers or {}
        self.client = FakeClient(host)


class FakeRedis:
    def __init__(self, *, counts=None, fail_on_incr=False):
        self.counts = counts or {}
        self.expirations = {}
        self.fail_on_incr = fail_on_incr

    async def incr(self, key):
        if self.fail_on_incr:
            raise ConnectionError("redis unreachable")

        self.counts[key] = self.counts.get(key, 0) + 1

        return self.counts[key]

    async def expire(self, key, seconds):
        self.expirations[key] = seconds

    async def ttl(self, key):
        return 42


@pytest.fixture(autouse=True)
def _enable_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "CHAT_RATE_LIMIT_PER_MINUTE", 2)


@pytest.mark.asyncio
async def test_requests_under_the_limit_pass():
    redis = FakeRedis()
    request = FakeRequest()

    await enforce_chat_rate_limit(redis=redis, request=request)
    await enforce_chat_rate_limit(redis=redis, request=request)


@pytest.mark.asyncio
async def test_requests_over_the_limit_are_rejected():
    redis = FakeRedis()
    request = FakeRequest()

    await enforce_chat_rate_limit(redis=redis, request=request)
    await enforce_chat_rate_limit(redis=redis, request=request)

    with pytest.raises(HTTPException) as exc_info:
        await enforce_chat_rate_limit(redis=redis, request=request)

    assert exc_info.value.status_code == 429
    assert "Retry-After" in exc_info.value.headers


@pytest.mark.asyncio
async def test_redis_outage_still_fails_closed():

    redis = FakeRedis(fail_on_incr=True)
    request = FakeRequest()

    with pytest.raises(ConnectionError):
        await enforce_chat_rate_limit(redis=redis, request=request)


def test_client_identifier_prefers_x_forwarded_for():
    from app.api.rate_limit import _client_identifier

    request = FakeRequest(
        headers={"x-forwarded-for": "9.9.9.9, 1.1.1.1"}, host="1.2.3.4"
    )

    assert _client_identifier(request) == "9.9.9.9"


def test_client_identifier_falls_back_to_socket_host():
    from app.api.rate_limit import _client_identifier

    request = FakeRequest(headers={}, host="1.2.3.4")

    assert _client_identifier(request) == "1.2.3.4"

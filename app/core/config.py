from pydantic_settings import BaseSettings, SettingsConfigDict

from app.knowledge.confluence_client import ConfluenceClient


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    CONFLUENCE_BASE_URL: str = "https://api.atlassian.com"
    CONFLUENCE_CLOUD_ID: str = ""
    CONFLUENCE_EMAIL: str = ""
    CONFLUENCE_API_TOKEN: str = ""
    ENTITY_ALIASES_PATH: str = "app/core/entity_aliases.json"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.6-terra"
    OPENAI_FAST_MODEL: str = ""
    OPENAI_TIMEOUT_SECONDS: float = 20.0

    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-5"

    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_BASE_URL: str = "https://api.openai.com/v1"
    EMBEDDING_TIMEOUT_SECONDS: float = 30.0

    QDRANT_URL: str = "http://localhost:6335"
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "eplanet_knowledge_local"
    QDRANT_VECTOR_SIZE: int = 3072
    VECTOR_TOP_K: int = 10
    APP_ENV: str = "local"

    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    DEMO_DEBUG_TRACE_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"
    CONVERSATION_STATE_TTL_SECONDS: int = 86400
    KNOWLEDGE_RUNTIME_SOURCE: str = "snapshot"
    KNOWLEDGE_SNAPSHOT_PATH: str = "data/knowledge/" "confluence_snapshot.json"

    CHAT_RATE_LIMIT_PER_MINUTE: int = 20
    ANSWER_CACHE_TTL_SECONDS: int = 86400

    @property
    def OPENAI_FAST_MODEL_RESOLVED(self) -> str:
        return self.OPENAI_FAST_MODEL or self.OPENAI_MODEL


settings = Settings()


def get_confluence_client() -> ConfluenceClient:
    return ConfluenceClient(
        base_url=settings.CONFLUENCE_BASE_URL,
        cloud_id=settings.CONFLUENCE_CLOUD_ID,
        email=settings.CONFLUENCE_EMAIL,
        api_token=settings.CONFLUENCE_API_TOKEN,
    )

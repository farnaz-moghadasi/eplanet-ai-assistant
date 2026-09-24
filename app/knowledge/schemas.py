# app/knowledge/schemas.py

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from app.knowledge.enums import (
    Audience,
    BotMode,
    FallbackPolicy,
    KnowledgeStatus,
    Language,
    MarketScope,
    SourceAuthority,
)


class KnowledgeMetadata(BaseModel):
    knowledge_key: str
    canonical_knowledge_key: str

    knowledge_type: str
    category: str

    status: KnowledgeStatus
    audience: Audience

    owner: Optional[str] = None
    primary_source: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)
    source_authority: Optional[SourceAuthority] = None

    language: Language

    market_scope: MarketScope
    countries: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)

    required_context_fields: list[str] = Field(default_factory=list)

    bot_mode: BotMode
    fallback_policy: FallbackPolicy = FallbackPolicy.NONE

    effective_from: Optional[date] = None
    valid_until: Optional[date] = None
    last_reviewed: Optional[date] = None
    review_frequency_days: Optional[int] = None


class ParsedKnowledgePage(BaseModel):
    page_id: str
    title: str
    version: Optional[int] = None

    metadata: KnowledgeMetadata

    content: str


class KnowledgeValidationResult(BaseModel):
    ai_ready: bool
    reasons: list[str] = Field(default_factory=list)
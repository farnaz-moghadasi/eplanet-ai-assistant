# app/knowledge/enums.py

from enum import Enum


class KnowledgeStatus(str, Enum):
    DRAFT = "DRAFT"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    APPROVED = "APPROVED"
    DEPRECATED = "DEPRECATED"


class MarketScope(str, Enum):
    GLOBAL = "GLOBAL"
    COUNTRY = "COUNTRY"
    REGION = "REGION"
    UNSPECIFIED = "UNSPECIFIED"


class Audience(str, Enum):
    VISITOR = "VISITOR"
    CLIENT = "CLIENT"
    BOTH = "BOTH"
    INTERNAL = "INTERNAL"


class BotMode(str, Enum):
    KNOWLEDGE = "KNOWLEDGE"
    LOOKUP = "LOOKUP"
    GUIDED = "GUIDED"
    HUMAN = "HUMAN"
    LINK_ONLY = "LINK_ONLY"
    NOT_APPLICABLE = "N/A"


class SourceAuthority(str, Enum):
    OPERATIONAL_PRIMARY = "OPERATIONAL_PRIMARY"
    STRUCTURED_PRODUCT = "STRUCTURED_PRODUCT"
    TRUSTED_CLIENT_STATE = "TRUSTED_CLIENT_STATE"
    APPROVED_DOCUMENT = "APPROVED_DOCUMENT"
    OBSERVATIONAL = "OBSERVATIONAL"


class Language(str, Enum):
    FA = "FA"
    EN = "EN"
    AR = "AR"
    KU = "KU"


class FallbackPolicy(str, Enum):
    NONE = "NONE"

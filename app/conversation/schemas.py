from enum import Enum

from pydantic import BaseModel, Field


class ConversationDomain(str, Enum):
    KYC = "KYC"
    ACCOUNT = "ACCOUNT"
    DEPOSIT = "DEPOSIT"
    WITHDRAWAL = "WITHDRAWAL"
    TRANSFER = "TRANSFER"
    MESSAGES = "MESSAGES"
    PROFILE = "PROFILE"
    ACCOUNT_MANAGER = "ACCOUNT_MANAGER"
    PLATFORM = "PLATFORM"
    IB = "IB"
    GENERAL = "GENERAL"
    UNKNOWN = "UNKNOWN"


class ConversationTask(str, Enum):
    REQUIREMENTS = "REQUIREMENTS"
    STATUS = "STATUS"
    TROUBLESHOOTING = "TROUBLESHOOTING"
    HOW_TO = "HOW_TO"
    LIST = "LIST"
    BALANCE = "BALANCE"
    CONTACT = "CONTACT"
    EXPLANATION = "EXPLANATION"
    COMPARISON = "COMPARISON"
    UNKNOWN = "UNKNOWN"


class TopicRelation(str, Enum):
    NEW_TOPIC = "NEW_TOPIC"
    CONTINUATION = "CONTINUATION"
    FOLLOW_UP = "FOLLOW_UP"
    UNCLEAR = "UNCLEAR"


class ConversationUnderstanding(BaseModel):
    language: str | None = None

    domain: ConversationDomain = ConversationDomain.UNKNOWN
    task: ConversationTask = ConversationTask.UNKNOWN
    client_specific: bool = False
    topic_relation: TopicRelation = TopicRelation.UNCLEAR

    referenced_countries: list[str] = Field(default_factory=list)
    referenced_entities: list[str] = Field(default_factory=list)
    requires_previous_context: bool = False


    is_off_topic: bool = False


    is_advice_request: bool = False


    is_small_talk: bool = False


    small_talk_kind: str | None = None

    self_document_country: str | None = None
    self_residence_country: str | None = None

    standalone_query: str | None = None
    retrieval_concepts: list[str] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    classification_summary: str | None = None
    needs_user_clarification: bool = False
    clarification_question: str | None = None


class ClarificationQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

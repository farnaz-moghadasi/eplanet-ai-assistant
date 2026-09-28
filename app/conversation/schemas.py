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

    # True only when the message has nothing to do with brokerage
    # services (weather, jokes, general knowledge, ...). Used together
    # with retrieval evidence to pick the "outside my scope" reply.
    is_off_topic: bool = False

    # True when the user asks for a trading/investment recommendation
    # (buy/sell/hold, which instrument to pick, price forecasts,
    # signals) rather than how the service itself works.
    is_advice_request: bool = False

    # True when the user is correcting or restating a fact about
    # THEMSELVES (their own nationality, residence, or ID document's
    # country) that was already established earlier in this
    # conversation - not asking about a country in general. See rule 5
    # in the system prompt: a country mentioned in a question is not
    # automatically the user's own. This flag exists so a later
    # correction ("actually no, I'm Iranian") can update the
    # already-stored value instead of being silently ignored while the
    # stale country keeps being used.
    corrects_self_country: bool = False

    standalone_query: str | None = None
    retrieval_concepts: list[str] = Field(default_factory=list)

    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    classification_summary: str | None = None

    # Semantic ambiguity is different from missing market/context fields.
    # Example: "ECN" names a topic but does not tell us what the user
    # wants to know about it. In that case retrieval should not guess.
    needs_user_clarification: bool = False
    clarification_question: str | None = None


class ClarificationQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=1000)

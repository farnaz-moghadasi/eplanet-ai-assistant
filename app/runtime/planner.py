from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.conversation.schemas import (
    ConversationDomain,
    ConversationTask,
    ConversationUnderstanding,
)


class LookupCapability(str, Enum):
    KYC_STATUS = "KYC_STATUS"
    ACCOUNTS = "ACCOUNTS"
    ACCOUNT_MANAGER = "ACCOUNT_MANAGER"
    UNREAD_MESSAGES = "UNREAD_MESSAGES"
    MESSAGES = "MESSAGES"


class ExecutionMode(str, Enum):
    KNOWLEDGE = "KNOWLEDGE"
    LOOKUP = "LOOKUP"
    LOOKUP_PLUS_KNOWLEDGE = "LOOKUP_PLUS_KNOWLEDGE"
    CLARIFICATION = "CLARIFICATION"


class RequestPlan(BaseModel):
    mode: ExecutionMode

    requires_authentication: bool = False
    requires_live_state: bool = False
    requires_knowledge: bool = False

    required_context: list[str] = Field(default_factory=list)

    lookup_capability: Optional[LookupCapability] = None

    knowledge_hints: list[str] = Field(default_factory=list)


def build_request_plan(
    understanding: ConversationUnderstanding,
) -> RequestPlan:
    domain = understanding.domain
    task = understanding.task

    if domain == ConversationDomain.KYC:
        if task == ConversationTask.REQUIREMENTS:
            return RequestPlan(
                mode=ExecutionMode.KNOWLEDGE,
                requires_knowledge=True,
                knowledge_hints=[
                    "KYC",
                    "identity verification",
                    "requirements",
                ],
            )

        if task == ConversationTask.STATUS:
            return RequestPlan(
                mode=ExecutionMode.LOOKUP,
                requires_authentication=True,
                requires_live_state=True,
                required_context=["client_id"],
                lookup_capability=LookupCapability.KYC_STATUS,
            )

        if task == ConversationTask.TROUBLESHOOTING:
            return RequestPlan(
                mode=ExecutionMode.LOOKUP_PLUS_KNOWLEDGE,
                requires_authentication=True,
                requires_live_state=True,
                requires_knowledge=True,
                required_context=["client_id"],
                lookup_capability=LookupCapability.KYC_STATUS,
                knowledge_hints=[
                    "KYC",
                    "verification troubleshooting",
                ],
            )

        if task == ConversationTask.HOW_TO:
            return RequestPlan(
                mode=ExecutionMode.KNOWLEDGE,
                requires_knowledge=True,
                knowledge_hints=[
                    "KYC",
                    "verification",
                    "how to",
                ],
            )

    if domain == ConversationDomain.ACCOUNT:
        if task in {
            ConversationTask.BALANCE,
            ConversationTask.LIST,
        }:
            return RequestPlan(
                mode=ExecutionMode.LOOKUP,
                requires_authentication=True,
                requires_live_state=True,
                required_context=["client_id"],
                lookup_capability=LookupCapability.ACCOUNTS,
            )

    if domain == ConversationDomain.ACCOUNT_MANAGER:
        return RequestPlan(
            mode=ExecutionMode.LOOKUP,
            requires_authentication=True,
            requires_live_state=True,
            required_context=["client_id"],
            lookup_capability=LookupCapability.ACCOUNT_MANAGER,
        )

    if domain == ConversationDomain.MESSAGES:
        if task == ConversationTask.STATUS:
            return RequestPlan(
                mode=ExecutionMode.LOOKUP,
                requires_authentication=True,
                requires_live_state=True,
                required_context=["client_id"],
                lookup_capability=LookupCapability.UNREAD_MESSAGES,
            )

        if task == ConversationTask.LIST:
            return RequestPlan(
                mode=ExecutionMode.LOOKUP,
                requires_authentication=True,
                requires_live_state=True,
                required_context=["client_id"],
                lookup_capability=LookupCapability.MESSAGES,
            )

    return RequestPlan(
        mode=ExecutionMode.KNOWLEDGE,
        requires_knowledge=True,
    )
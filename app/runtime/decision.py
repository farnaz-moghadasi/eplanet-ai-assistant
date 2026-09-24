from enum import Enum
from typing import Optional

from pydantic import BaseModel , Field

from app.runtime.context import (
    RuntimeContext,
    has_verified_client_identity,
)
from app.runtime.planner import RequestPlan


class RuntimeDecisionState(str, Enum):
    READY = "READY"

    AUTHENTICATION_REQUIRED = "AUTHENTICATION_REQUIRED"

    CONTEXT_REQUIRED = "CONTEXT_REQUIRED"


class RuntimeDecision(BaseModel):
    state: RuntimeDecisionState

    allowed: bool

    reason: str

    missing_context: list[str] = Field(default_factory=list)

    user_message: Optional[str] = None


def evaluate_request_plan(
    plan: RequestPlan,
    context: RuntimeContext,
) -> RuntimeDecision:

    if plan.requires_authentication:

        if not has_verified_client_identity(context):
            return RuntimeDecision(
                state=RuntimeDecisionState.AUTHENTICATION_REQUIRED,
                allowed=False,
                reason="VERIFIED_CLIENT_IDENTITY_REQUIRED",
                missing_context=["client_id"],
                user_message=(
                    "Please log in to your Client Portal "
                    "so I can check your account-specific information."
                ),
            )

    missing_context: list[str] = []

    for field_name in plan.required_context:

        if field_name == "client_id":
            continue

        context_value = getattr(
            context,
            field_name,
            None,
        )

        if not context_value or not context_value.value:
            missing_context.append(field_name)

    if missing_context:
        return RuntimeDecision(
            state=RuntimeDecisionState.CONTEXT_REQUIRED,
            allowed=False,
            reason="REQUIRED_CONTEXT_MISSING",
            missing_context=missing_context,
        )

    return RuntimeDecision(
        state=RuntimeDecisionState.READY,
        allowed=True,
        reason="REQUEST_READY",
    )
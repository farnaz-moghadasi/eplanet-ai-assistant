from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class IdentityState(str, Enum):
    ANONYMOUS = "ANONYMOUS"
    AUTHENTICATED = "AUTHENTICATED"


class ContextTrust(str, Enum):
    VERIFIED = "VERIFIED"
    DECLARED = "DECLARED"
    QUERY_CONTEXT = "QUERY_CONTEXT"
    WEAK = "WEAK"


class ContextValue(BaseModel):
    value: Optional[str] = None
    source: Optional[str] = None
    trust: Optional[ContextTrust] = None


class RuntimeContext(BaseModel):
    identity_state: IdentityState

    client_id: ContextValue = Field(default_factory=ContextValue)

    nationality: ContextValue = Field(default_factory=ContextValue)

    country_of_residence: ContextValue = Field(
        default_factory=ContextValue
    )

    ip_country: ContextValue = Field(default_factory=ContextValue)

    language: ContextValue = Field(default_factory=ContextValue)

    referenced_country: ContextValue = Field(
        default_factory=ContextValue
    )

    profile_country: ContextValue = Field(
        default_factory=ContextValue
    )

    profile_verified: ContextValue = Field(
        default_factory=ContextValue
    )

    client_type: ContextValue = Field(
        default_factory=ContextValue
    )

    is_ib: ContextValue = Field(
        default_factory=ContextValue
    )


def has_verified_client_identity(
    context: RuntimeContext,
) -> bool:
    return (
        context.identity_state == IdentityState.AUTHENTICATED
        and bool(context.client_id.value)
        and context.client_id.trust == ContextTrust.VERIFIED
    )
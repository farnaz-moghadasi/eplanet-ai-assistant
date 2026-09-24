from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from app.conversation.state import (
    ConversationState,
)


class ConversationStateStore(ABC):

    @abstractmethod
    async def get(
        self,
        conversation_id: str,
    ) -> ConversationState | None:
        raise NotImplementedError

    @abstractmethod
    async def save(
        self,
        state: ConversationState,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def delete(
        self,
        conversation_id: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def ping(
        self,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def close(
        self,
    ) -> None:
        raise NotImplementedError
from typing import Protocol, TypeVar, Type

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredLLM(Protocol):
    async def generate_structured(
        self, *, system_prompt: str, user_prompt: str, response_model: Type[T]
    ) -> T: ...

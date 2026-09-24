from typing import Type, TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from app.ai.base import StructuredLLM
from app.core.config import settings


T = TypeVar("T", bound=BaseModel)


class AnthropicStructuredLLM(StructuredLLM):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 1200,
    ):
        resolved_api_key = (
            api_key
            or settings.ANTHROPIC_API_KEY
        )

        if not resolved_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not configured."
            )

        self.client = AsyncAnthropic(
            api_key=resolved_api_key
        )

        self.model = (
            model
            or settings.ANTHROPIC_MODEL
        )

        self.max_tokens = max_tokens

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> T:

        response = await self.client.messages.parse(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            output_format=response_model,
        )

        parsed = response.parsed_output

        if parsed is None:
            raise RuntimeError(
                "Anthropic returned no parsed structured output."
            )

        return parsed
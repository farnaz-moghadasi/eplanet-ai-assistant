from typing import Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.ai.base import StructuredLLM
from app.core.config import settings


T = TypeVar("T", bound=BaseModel)


class OpenAIStructuredLLM(StructuredLLM):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        resolved_api_key = api_key or settings.OPENAI_API_KEY

        if not resolved_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        self.client = AsyncOpenAI(
            api_key=resolved_api_key,
            timeout=settings.OPENAI_TIMEOUT_SECONDS,
        )

        self.model = (
            model
            or settings.OPENAI_MODEL
        )

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> T:

        response = await self.client.responses.parse(
            model=self.model,
            instructions=system_prompt,
            input=user_prompt,
            text_format=response_model,
        )

        parsed = response.output_parsed

        if parsed is None:
            raise RuntimeError(
                "OpenAI returned no parsed structured output."
            )

        return parsed
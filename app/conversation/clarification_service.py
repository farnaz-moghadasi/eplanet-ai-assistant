from __future__ import annotations

from app.ai.base import StructuredLLM
from app.conversation.schemas import (
    ClarificationQuestion,
    ConversationUnderstanding,
)


CLARIFICATION_SYSTEM_PROMPT = """
You are the clarification-question component for a governed
financial-services customer-support assistant.

The backend has already decided that exactly one context field is
missing. Your only job is to ask the user one natural, concise
follow-up question that obtains that field.

Rules:
1. Ask only for the supplied missing field.
2. Do not answer the user's original question yet.
3. Do not invent requirements, policies, countries, account details,
   eligibility rules, or client data.
4. Do not infer nationality, residence, market, or document country
   from language, name, IP address, location, or wording.
5. Use the user's language when it is supplied.
6. Keep the question short and conversational.
7. Ask one question only.
8. Do not mention internal implementation terms such as field names,
   retrieval, eligibility, metadata, knowledge keys, or Confluence.
9. If the missing field is client_id, do not ask the user to type or
   reveal a client ID. Instead, tell the user that authentication or
   sign-in is required before account-specific information can be
   checked.
10. Return only the requested structured response.
""".strip()


class ClarificationGenerationService:
    def __init__(
        self,
        *,
        llm: StructuredLLM,
    ) -> None:
        self.llm = llm

    async def generate(
        self,
        *,
        original_message: str,
        missing_field: str,
        reason: str | None,
        understanding: ConversationUnderstanding,
        language: str | None,
    ) -> ClarificationQuestion:
        normalized_field = str(
            missing_field
            or ""
        ).strip()

        if not normalized_field:
            raise ValueError(
                "missing_field cannot be empty."
            )

        user_prompt = self._build_prompt(
            original_message=original_message,
            missing_field=normalized_field,
            reason=reason,
            understanding=understanding,
            language=language,
        )

        result = (
            await self.llm.generate_structured(
                system_prompt=(
                    CLARIFICATION_SYSTEM_PROMPT
                ),
                user_prompt=user_prompt,
                response_model=(
                    ClarificationQuestion
                ),
            )
        )

        question = (
            result.question
            or ""
        ).strip()

        if not question:
            raise RuntimeError(
                "Clarification generation returned "
                "an empty question."
            )

        return result.model_copy(
            update={
                "question": question,
            }
        )

    @staticmethod
    def _build_prompt(
        *,
        original_message: str,
        missing_field: str,
        reason: str | None,
        understanding: ConversationUnderstanding,
        language: str | None,
    ) -> str:
        domain = getattr(
            understanding.domain,
            "value",
            understanding.domain,
        )

        task = getattr(
            understanding.task,
            "value",
            understanding.task,
        )

        return f"""
CURRENT USER MESSAGE
--------------------
{original_message}

USER LANGUAGE
-------------
{language or understanding.language or "unknown"}

UNDERSTOOD DOMAIN
-----------------
{domain}

UNDERSTOOD TASK
---------------
{task}

MISSING CONTEXT FIELD
---------------------
{missing_field}

BACKEND REASON
--------------
{reason or "MISSING_REQUIRED_CONTEXT"}

Ask one natural customer-facing follow-up question that obtains only
the missing context required to continue safely.
""".strip()

import pytest

from app.generation.schemas import (
    GeneratedAnswerPayload,
)
from app.generation.service import (
    AnswerGenerationService,
)


class FakeLLM:
    async def generate_structured(
        self,
        *,
        system_prompt,
        user_prompt,
        response_model,
    ):
        assert (
            response_model
            is GeneratedAnswerPayload
        )

        return GeneratedAnswerPayload(
            answer=(
                "برای احراز هویت کاربران "
                "افغانستان، کارت ملی یا "
                "پاسپورت قابل استفاده است."
            ),
            citations=[],
            abstained=False,
            needs_handoff=False,
        )


@pytest.mark.asyncio
async def test_no_candidates_abstains_without_llm():
    service = AnswerGenerationService(
        llm=FakeLLM()
    )

    result = await service.generate(
        query="test",
        language="FA",
        candidates=[],
    )

    assert (
        result.payload.abstained
        is True
    )

    assert (
        result.payload.abstain_reason
        == "NO_ELIGIBLE_EVIDENCE"
    )


def test_context_builder_excludes_sensitive_identity():
    context = {
        "kyc_document_country":
            "AF",
        "referenced_country":
            "AF",
        "client_id":
            "CLIENT-123",
        "email":
            "client@example.com",
    }

    result = (
        AnswerGenerationService
        ._build_context_text(
            context
        )
    )

    assert (
        "kyc_document_country: AF"
        in result
    )

    assert (
        "referenced_country: AF"
        in result
    )

    assert "CLIENT-123" not in result
    assert "client@example.com" not in result
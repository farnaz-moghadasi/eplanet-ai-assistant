import pytest

from app.generation.schemas import (
    GeneratedAnswerPayload,
)
from app.generation.service import (
    AnswerGenerationService,
)
from app.knowledge.enums import (
    Audience,
    BotMode,
    FallbackPolicy,
    KnowledgeStatus,
    Language,
    MarketScope,
    SourceAuthority,
)
from app.knowledge.schemas import (
    KnowledgeMetadata,
)
from app.retrieval.hybrid import (
    HybridCandidate,
)
from app.retrieval.reranker import (
    RerankedCandidate,
)
from app.retrieval.schemas import (
    KnowledgeCandidate,
)


def make_reranked_candidate(
    *,
    knowledge_key: str,
    content: str,
    rerank_score: float = 0.9,
) -> RerankedCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key=knowledge_key,
        canonical_knowledge_key=knowledge_key,
        knowledge_type="PRODUCT_INFORMATION",
        category="TEST",
        status=KnowledgeStatus.APPROVED,
        audience=Audience.BOTH,
        owner="Test",
        primary_source="Test",
        source_ids=["TEST"],
        source_authority=(
            SourceAuthority.APPROVED_DOCUMENT
        ),
        language=Language.FA,
        market_scope=MarketScope.GLOBAL,
        countries=[],
        regions=[],
        required_context_fields=[],
        bot_mode=BotMode.KNOWLEDGE,
        fallback_policy=FallbackPolicy.NONE,
    )

    candidate = KnowledgeCandidate(
        candidate_id=f"{knowledge_key}:chunk-1",
        title="Test",
        content=content,
        metadata=metadata,
        score=1.0,
        source_page_id="1",
    )

    hybrid = HybridCandidate(
        candidate=candidate,
        hybrid_score=0.9,
    )

    return RerankedCandidate(
        candidate=hybrid,
        rerank_score=rerank_score,
        reasons=[],
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


def test_deterministic_generation_rejects_multi_item_comparison_chunk():
    # Mirrors a real bug: a single chunk under the length gate can still
    # bundle several distinct account types' full spec sheets (Standard,
    # ECN, ECN PRO, Gold, ...), each introduced by an "Item — descriptor"
    # heading. Dumping it verbatim answers with every item instead of the
    # one the user actually asked about.
    content = (
        "## انواع حساب‌های معاملاتی\n\n"
        "## Standard — حساب استاندارد\n\n"
        "کمیسیون: 0 دلار\n\n"
        "## ECN — مناسب معامله‌گران فعال\n\n"
        "کمیسیون: 5 دلار\n\n"
        "## ECN PRO — برای معامله‌گران حرفه‌ای\n\n"
        "کمیسیون: 4 دلار\n"
    )

    service = AnswerGenerationService(
        llm=FakeLLM()
    )

    candidate = make_reranked_candidate(
        knowledge_key="KB-ACC-005-IR-FA",
        content=content,
    )

    assert (
        service.can_generate_deterministically(
            candidates=[candidate]
        )
        is False
    )


def test_deterministic_generation_allows_single_topic_chunk():
    content = (
        "## هدف\n\n"
        "تغییر لوریج حساب معاملاتی.\n\n"
        "## مراحل\n\n"
        "برای تغییر لوریج به پنل کاربری مراجعه کنید.\n"
    )

    service = AnswerGenerationService(
        llm=FakeLLM()
    )

    candidate = make_reranked_candidate(
        knowledge_key="KB-ACC-004-FA",
        content=content,
    )

    assert (
        service.can_generate_deterministically(
            candidates=[candidate]
        )
        is True
    )


def test_generate_excludes_trailing_low_relevance_candidates():
    # Grounding rejects the whole answer if even one of its claims comes
    # back unsupported, and merging more candidates means more claims to
    # independently pass that check. A candidate trailing far behind the
    # top semantic relevance score should not be pulled into the answer
    # just because it fit under max_candidates.
    service = AnswerGenerationService(
        llm=FakeLLM(),
        max_candidates=5,
    )

    strong = make_reranked_candidate(
        knowledge_key="KB-KYC-DOCS-AF-FA",
        content="AF content",
        rerank_score=0.96,
    )

    close = make_reranked_candidate(
        knowledge_key="KB-KYC-DOCS-GENERAL-FA",
        content="General content",
        rerank_score=0.88,
    )

    far = make_reranked_candidate(
        knowledge_key="KB-KYC-UPLOAD-FA",
        content="Upload content",
        rerank_score=0.66,
    )

    selected = service._select_evidence_candidates(
        [strong, close, far]
    )

    selected_keys = {
        item.candidate.candidate.metadata.knowledge_key
        for item in selected
    }

    assert selected_keys == {
        "KB-KYC-DOCS-AF-FA",
        "KB-KYC-DOCS-GENERAL-FA",
    }


def test_generate_evidence_selection_unaffected_by_deterministic_scale():
    # Deterministic-reranker scores are all close together on a small
    # RRF-derived scale (see HybridRetriever), so the relevance-gap
    # filter should never trim anything there - only semantic (0-1)
    # scores carry a meaningful gap.
    service = AnswerGenerationService(
        llm=FakeLLM(),
        max_candidates=5,
    )

    first = make_reranked_candidate(
        knowledge_key="KB-A",
        content="A",
        rerank_score=0.0330,
    )

    second = make_reranked_candidate(
        knowledge_key="KB-B",
        content="B",
        rerank_score=0.0180,
    )

    selected = service._select_evidence_candidates(
        [first, second]
    )

    assert len(selected) == 2


def test_display_text_drops_question_and_unwraps_answer_notes_links():
    content = (
        "## Question\n\n"
        "مارجین کال به چه معناست؟\n\n"
        "## Answer\n\n"
        "مارجین کال به وضعیت کمبود مارجین در حساب مربوط است.\n\n"
        "## Important Notes\n\n"
        "شرایط هر حساب را بررسی کنید.\n\n"
        "## Useful Links\n\n"
        "[راهنما](https://example.com/guide)\n"
    )

    assert AnswerGenerationService.to_display_text(content) == (
        "مارجین کال به وضعیت کمبود مارجین در حساب مربوط است.\n\n"
        "شرایط هر حساب را بررسی کنید.\n\n"
        "[راهنما](https://example.com/guide)"
    )


@pytest.mark.parametrize(
    "answer_heading",
    ["Answer", "الإجابة", "وەڵام", "پاسخ"],
)
def test_display_text_handles_localised_answer_headings(
    answer_heading,
):
    content = (
        "## Question\n\nسؤال الصفحة\n\n"
        f"## {answer_heading}\n\nنص الإجابة."
    )

    assert (
        AnswerGenerationService.to_display_text(content)
        == "نص الإجابة."
    )


def test_display_text_drops_related_articles_completely():
    content = (
        "## Answer\n\nمتن پاسخ.\n\n"
        "## Related Articles\n\n"
        "- [مقاله ۱](https://example.com/1)\n"
        "- [مقاله ۲](https://example.com/2)\n"
    )

    result = AnswerGenerationService.to_display_text(content)

    assert result == "متن پاسخ."
    assert "Related" not in result
    assert "example.com" not in result


@pytest.mark.parametrize(
    "video_heading",
    [
        "Video Tutorial",
        "الفيديو التعليمي",
        "فێرکاری ڤیدیۆیی",
        "آموزش ویدیویی",
        "**Useful Links**",
        "Web Terminal",
    ],
)
def test_display_text_keeps_only_links_under_link_headings(
    video_heading,
):
    content = (
        "## Answer\n\nمتن.\n\n"
        f"## {video_heading}\n\n"
        "[ویدیو](https://example.com/v)\n"
    )

    assert AnswerGenerationService.to_display_text(content) == (
        "متن.\n\n[ویدیو](https://example.com/v)"
    )


@pytest.mark.parametrize(
    "purpose_heading",
    [
        "What this helps with",
        "Purpose",
        "Overview",
        "الهدف",
        "نظرة عامة",
        "ئامانج",
        "کورتە باس",
        "هدف",
        "توضیحات",
    ],
)
def test_display_text_drops_short_purpose_sections(
    purpose_heading,
):
    content = (
        f"## {purpose_heading}\n\nتغییر لوریج حساب.\n\n"
        "## Steps\n\nبه پنل بروید."
    )

    result = AnswerGenerationService.to_display_text(
        content, language="FA"
    )

    assert result == "## مراحل\n\nبه پنل بروید."


def test_display_text_keeps_text_of_long_purpose_sections():
    # A long "Purpose" holds real instructions (as on the KYC upload
    # page) - only its heading goes.
    long_body = "To complete verification, follow these steps: " + (
        "do the thing. " * 20
    )

    result = AnswerGenerationService.to_display_text(
        f"## Purpose\n\n{long_body}\n\n## Steps\n\nstep one"
    )

    assert result.startswith(long_body.strip())
    assert "Purpose" not in result


@pytest.mark.parametrize(
    "language,label",
    [
        ("FA", "مراحل"),
        ("EN", "Steps"),
        ("AR", "الخطوات"),
        ("KU", "هەنگاوەکان"),
    ],
)
def test_display_text_labels_steps_in_the_page_language(
    language,
    label,
):
    result = AnswerGenerationService.to_display_text(
        "## Steps\n\nمتن مراحل", language=language
    )

    assert result == f"## {label}\n\nمتن مراحل"


def test_display_text_keeps_unknown_headings_and_falls_back():
    content = (
        "## Steps\n\nمراحل کلی\n\n"
        "### Android\n\nمرحله اندروید\n\n"
        "### درخواست مدرک برداشت\n\nبرای دریافت مدرک..."
    )

    result = AnswerGenerationService.to_display_text(content)

    assert "### Android" in result
    assert "### درخواست مدرک برداشت" in result

    # Nothing but droppable scaffolding: return the original rather
    # than an empty answer.
    only_scaffolding = "## Related Articles\n\n- [x](https://e.com)"

    assert (
        AnswerGenerationService.to_display_text(only_scaffolding)
        == only_scaffolding
    )


def test_lone_question_section_keeps_its_text():
    # A page with no answer section (its whole answer sits under a
    # single "Question" heading) must not lose that text.
    content = (
        "## Question\n\nحساب دمو و بررسی شرایط معامله\n\n"
        "کمیسیون در حساب استاندارد ۱۴ صدم درصد است."
    )

    result = AnswerGenerationService.to_display_text(content)

    assert "Question" not in result
    assert "کمیسیون در حساب استاندارد" in result


def test_display_text_on_real_withdrawal_troubleshooting_page():
    content = (
        "## Problem\n\n"
        "درخواست برداشت هنوز تکمیل نشده یا با تأخیر همراه است.\n\n"
        "## Steps\n\n"
        "وضعیت درخواست را بررسی کنید.\n\n"
        "### درخواست مدرک برداشت\n\n"
        "شناسه تراکنش CRM را در تیکت بنویسید.\n\n"
        "## Important Notes\n\n"
        "زمان تکمیل به وضعیت همان درخواست وابسته است.\n"
    )

    assert AnswerGenerationService.to_display_text(
        content, language="FA"
    ) == (
        "درخواست برداشت هنوز تکمیل نشده یا با تأخیر همراه است.\n\n"
        "## مراحل\n\n"
        "وضعیت درخواست را بررسی کنید.\n\n"
        "### درخواست مدرک برداشت\n\n"
        "شناسه تراکنش CRM را در تیکت بنویسید.\n\n"
        "زمان تکمیل به وضعیت همان درخواست وابسته است."
    )


def test_deterministic_answer_uses_display_text():
    service = AnswerGenerationService(llm=FakeLLM())

    candidate = make_reranked_candidate(
        knowledge_key="KB-GEN-MARGIN-CALL-FA",
        content=(
            "## Question\n\nمارجین کال چیست؟\n\n"
            "## Answer\n\nکمبود مارجین در حساب."
        ),
    )

    result = service.generate_deterministic(
        candidates=[candidate]
    )

    assert result.payload.answer == "کمبود مارجین در حساب."
    assert "Question" not in result.payload.answer
    assert "Answer" not in result.payload.answer

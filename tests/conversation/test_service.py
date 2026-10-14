import json

import pytest

from app.conversation.resolver import ConversationResolver
from app.generation.service import AnswerGenerationService
from app.conversation.service import ConversationService, TrustedSessionContext
from app.conversation.state import ConversationState
from app.knowledge.enums import (
    Audience,
    BotMode,
    FallbackPolicy,
    KnowledgeStatus,
    Language,
    MarketScope,
    SourceAuthority,
)
from app.knowledge.schemas import KnowledgeMetadata
from app.retrieval.hybrid import HybridCandidate
from app.retrieval.pipeline import EligibilityTrace, RetrievalPipelineResult
from app.retrieval.reranker import RerankedCandidate
from app.retrieval.schemas import KnowledgeCandidate
from app.runtime.context_policy import ContextPolicy
from app.runtime.entity_registry import EntityRegistry
from app.conversation.schemas import (
    ConversationDomain,
    ConversationTask,
    ConversationUnderstanding,
    TopicRelation,
)


class FakeRetrievalPipeline:
    def __init__(self, result_factory):
        self.result_factory = result_factory
        self.calls = []

    async def retrieve(
        self, *, query, context, language=None, knowledge_hints=None, rerank_query=None
    ):
        self.calls.append(
            {
                "query": query,
                "context": context,
                "language": language,
                "rerank_query": rerank_query,
            }
        )

        return self.result_factory(query, context)


class FakeUnderstandingService:
    def __init__(
        self,
        domain=ConversationDomain.GENERAL,
        is_off_topic=False,
        is_advice_request=False,
        is_small_talk=False,
        small_talk_kind=None,
        task=ConversationTask.HOW_TO,
        sub_questions=None,
    ):
        self.domain = domain
        self.is_off_topic = is_off_topic
        self.is_advice_request = is_advice_request
        self.is_small_talk = is_small_talk
        self.small_talk_kind = small_talk_kind
        self.client_specific_override = False
        self.task = task
        self.sub_questions = sub_questions or []

    async def understand(self, *, message, state):
        return ConversationUnderstanding(
            language=(state.language or "FA"),
            domain=self.domain,
            is_off_topic=self.is_off_topic,
            is_advice_request=self.is_advice_request,
            is_small_talk=self.is_small_talk,
            small_talk_kind=(self.small_talk_kind),
            client_specific=(self.client_specific_override),
            task=self.task,
            topic_relation=(TopicRelation.NEW_TOPIC),
            referenced_countries=[],
            referenced_entities=[],
            requires_previous_context=False,
            confidence=0.95,
            sub_questions=self.sub_questions,
        )


class SequencedLanguageUnderstandingService:

    def __init__(self, languages):
        self.languages = list(languages)
        self.calls = 0

    async def understand(self, *, message, state):
        language = self.languages[min(self.calls, len(self.languages) - 1)]
        self.calls += 1

        return ConversationUnderstanding(
            language=language,
            domain=ConversationDomain.GENERAL,
            is_off_topic=False,
            is_advice_request=False,
            is_small_talk=False,
            small_talk_kind=None,
            client_specific=False,
            task=ConversationTask.HOW_TO,
            topic_relation=TopicRelation.NEW_TOPIC,
            referenced_countries=[],
            referenced_entities=[],
            requires_previous_context=False,
            confidence=0.95,
        )


def empty_result(query: str):
    return RetrievalPipelineResult(
        query=query,
        bm25_candidates=[],
        vector_candidates=[],
        hybrid_candidates=[],
        eligible_candidates=[],
        final_candidates=[],
        eligibility_trace=[],
    )


def make_hybrid_candidate(
    knowledge_key: str, category: str = "TEST"
) -> HybridCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key=knowledge_key,
        canonical_knowledge_key=(knowledge_key),
        knowledge_type="HOW_TO",
        category=category,
        status=(KnowledgeStatus.APPROVED),
        audience=Audience.BOTH,
        owner="Test",
        primary_source="Test",
        source_ids=["TEST"],
        source_authority=(SourceAuthority.APPROVED_DOCUMENT),
        language=Language.FA,
        market_scope=(MarketScope.COUNTRY),
        countries=["AF"],
        regions=[],
        required_context_fields=["kyc_document_country"],
        bot_mode=(BotMode.KNOWLEDGE),
        fallback_policy=(FallbackPolicy.NONE),
    )

    candidate = KnowledgeCandidate(
        candidate_id=(f"{knowledge_key}:chunk-1"),
        title="Test knowledge",
        content="Test content",
        metadata=metadata,
        score=1.0,
        source_page_id="1",
    )

    return HybridCandidate(
        candidate=candidate,
        hybrid_score=0.04,
        bm25_rank=1,
        bm25_score=10.0,
        vector_rank=1,
        vector_score=0.8,
    )


def result_with_final_candidate(
    query: str, knowledge_key: str = "KB-TEST-001"
) -> RetrievalPipelineResult:
    hybrid = make_hybrid_candidate(knowledge_key)

    reranked = RerankedCandidate(candidate=hybrid, rerank_score=0.9, reasons=[])

    return RetrievalPipelineResult(
        query=query,
        bm25_candidates=[hybrid.candidate],
        vector_candidates=[hybrid.candidate],
        hybrid_candidates=[hybrid],
        eligible_candidates=[hybrid],
        final_candidates=[reranked],
        eligibility_trace=[],
    )


def result_with_eligible_but_no_final(
    query: str, knowledge_key: str = "KB-TEST-002"
) -> RetrievalPipelineResult:

    hybrid = make_hybrid_candidate(knowledge_key)

    return RetrievalPipelineResult(
        query=query,
        bm25_candidates=[hybrid.candidate],
        vector_candidates=[hybrid.candidate],
        hybrid_candidates=[hybrid],
        eligible_candidates=[hybrid],
        final_candidates=[],
        eligibility_trace=[],
    )


class FakeReadinessService:
    def __init__(
        self,
        *,
        needs_clarification: bool,
        question: str = "",
        triage_needs_clarification: bool = False,
        triage_question: str = "",
    ):
        self.needs_clarification = needs_clarification

        self.question = question

        self.triage_needs_clarification = triage_needs_clarification

        self.triage_question = triage_question

        self.calls: list[dict] = []

        self.triage_calls: list[dict] = []

    async def check(
        self, *, query, language, understanding, candidates, conversation_context
    ):
        from app.conversation.readiness_service import ReadinessCheckResponse

        self.calls.append(
            {"query": query, "conversation_context": conversation_context}
        )

        return ReadinessCheckResponse(
            needs_clarification=(self.needs_clarification), question=self.question
        )

    async def check_before_abstaining(
        self, *, query, language, understanding, near_candidates, conversation_context
    ):
        from app.conversation.readiness_service import ReadinessCheckResponse

        self.triage_calls.append(
            {
                "query": query,
                "near_candidates": near_candidates,
                "conversation_context": conversation_context,
            }
        )

        return ReadinessCheckResponse(
            needs_clarification=(self.triage_needs_clarification),
            question=(self.triage_question),
        )


def build_registry(tmp_path):
    config = {
        "entity_types": {
            "platform": {
                "fuzzy_enabled": True,
                "fuzzy_threshold": 0.70,
                "aliases": {"ctrader": "cTrader", "mt5": "MT5", "metatrader": "MT5"},
            },
            "country": {
                "fuzzy_enabled": False,
                "fuzzy_threshold": 0.90,
                "aliases": {
                    "afghanistan": "AF",
                    "افغانستان": "AF",
                    "iran": "IR",
                    "ایران": "IR",
                },
            },
        },
        "context_field_entity_types": {
            "kyc_document_country": "country",
            "country_of_residence": "country",
            "nationality": "country",
            "referenced_country": "country",
        },
        "conversation": {
            "short_followup_max_tokens": 4,
            "correction_markers": ["نه منظورم", "i mean"],
        },
    }

    path = tmp_path / "entities.json"

    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    return EntityRegistry.from_file(path)


def build_service(
    tmp_path,
    result_factory,
    domain=ConversationDomain.GENERAL,
    is_off_topic=False,
    is_advice_request=False,
    is_small_talk=False,
    small_talk_kind=None,
    task=ConversationTask.HOW_TO,
    readiness_service=None,
    generation_service=None,
    answer_cache_service=None,
    sub_questions=None,
):
    registry = build_registry(tmp_path)

    resolver = ConversationResolver(entity_registry=registry)

    pipeline = FakeRetrievalPipeline(result_factory)

    service = ConversationService(
        understanding_service=(
            FakeUnderstandingService(
                domain=domain,
                is_off_topic=is_off_topic,
                is_advice_request=is_advice_request,
                is_small_talk=is_small_talk,
                small_talk_kind=small_talk_kind,
                task=task,
                sub_questions=sub_questions,
            )
        ),
        resolver=resolver,
        retrieval_pipeline=pipeline,
        context_policy=ContextPolicy(),
        readiness_service=(readiness_service),
        generation_service=(generation_service),
        answer_cache_service=(answer_cache_service),
    )

    return service, pipeline


@pytest.mark.asyncio
async def test_service_sends_resolved_query_to_pipeline(tmp_path):
    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    state = ConversationState()

    await service.handle_message(message="cTrader", state=state, language="FA")

    assert len(pipeline.calls) == 1

    assert pipeline.calls[0]["query"] == "ctrader"


@pytest.mark.asyncio
async def test_followup_changes_platform_using_conversation_context(tmp_path):
    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    state = ConversationState()

    await service.handle_message(
        message=("cTrader چجوری نصب میشه؟"), state=state, language="FA"
    )

    await service.handle_message(message="MT5 چی؟", state=state, language="FA")

    assert len(pipeline.calls) == 2

    second_query = pipeline.calls[1]["query"]

    assert "MT5" in second_query

    assert "cTrader" not in second_query


@pytest.mark.asyncio
async def test_missing_context_sets_pending_field(tmp_path):
    blocked_candidate = make_hybrid_candidate("KB-COUNTRY-001", category="KYC")

    def factory(query, context):
        return RetrievalPipelineResult(
            query=query,
            bm25_candidates=[],
            vector_candidates=[],
            hybrid_candidates=[blocked_candidate],
            eligible_candidates=[],
            final_candidates=[],
            eligibility_trace=[
                EligibilityTrace(
                    knowledge_key=("KB-COUNTRY-001"),
                    eligible=False,
                    reasons=[
                        ("MISSING_CONTEXT:" "kyc_document_country"),
                        ("COUNTRY_CONTEXT_" "MISSING"),
                    ],
                )
            ],
        )

    service, _ = build_service(tmp_path, factory, domain=ConversationDomain.KYC)

    state = ConversationState()

    result = await service.handle_message(
        message=("مدارک احراز هویت"), state=state, language="FA"
    )

    assert result.should_generate_answer is False

    assert result.resolved_turn.needs_clarification is True

    assert result.resolved_turn.missing_context_fields == ["kyc_document_country"]

    assert state.awaiting_context_field == "kyc_document_country"

    assert result.clarification is not None

    assert result.clarification.field_name == "kyc_document_country"


@pytest.mark.asyncio
async def test_pending_context_answer_is_stored(tmp_path):
    def factory(query, context):
        return empty_result(query)

    service, pipeline = build_service(tmp_path, factory)

    state = ConversationState(
        language="FA",
        last_resolved_query=("KYC document requirements"),
        awaiting_context_field=("kyc_document_country"),
    )

    await service.handle_message(message="افغانستان", state=state, language="FA")

    assert state.kyc_document_country == "AF"

    assert state.awaiting_context_field is None

    assert pipeline.calls[-1]["context"].kyc_document_country == "AF"


@pytest.mark.asyncio
async def test_trusted_session_sets_client_id(tmp_path):
    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    state = ConversationState()

    await service.handle_message(
        message="what is my balance",
        state=state,
        language="EN",
        trusted_session=(
            TrustedSessionContext(is_authenticated=True, client_id="CLIENT-123")
        ),
    )

    assert state.is_authenticated is True

    assert state.client_id == "CLIENT-123"

    context = pipeline.calls[-1]["context"]

    assert context.is_authenticated is True

    assert context.client_id == "CLIENT-123"


@pytest.mark.asyncio
async def test_authenticated_session_without_client_id_is_allowed(tmp_path):
    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    state = ConversationState()

    result = await service.handle_message(
        message="what is my balance",
        state=state,
        language="EN",
        trusted_session=(TrustedSessionContext(is_authenticated=True, client_id=None)),
    )

    assert state.is_authenticated is True

    assert state.client_id is None

    context = pipeline.calls[-1]["context"]

    assert context.is_authenticated is True

    assert context.client_id is None

    assert result.should_generate_answer is False


@pytest.mark.asyncio
async def test_user_text_cannot_set_client_id(tmp_path):
    service, _ = build_service(tmp_path, lambda query, context: empty_result(query))

    state = ConversationState()

    await service.handle_message(
        message=("my client id is 12345"), state=state, language="EN"
    )

    assert state.is_authenticated is False

    assert state.client_id is None


@pytest.mark.asyncio
async def test_fuzzy_greeting_slm_skips_retrieval(tmp_path):
    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    state = ConversationState()

    result = await service.handle_message(message="slm", state=state, language="EN")

    assert pipeline.calls == []
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is False
    assert result.generation_result.payload.answer


@pytest.mark.asyncio
async def test_no_candidates_returns_safe_answer_instead_of_blank(tmp_path):

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        domain=ConversationDomain.ACCOUNT,
    )

    state = ConversationState()

    result = await service.handle_message(
        message="unknown unsupported topic", state=state, language="EN"
    )

    assert len(pipeline.calls) == 1
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is True
    assert result.generation_result.payload.abstain_reason == "NO_ELIGIBLE_KNOWLEDGE"
    assert result.generation_result.payload.needs_handoff is True
    assert result.generation_result.payload.answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language,expected_fragment",
    [
        ("FA", "خارج از حوزهٔ کاری"),
        ("EN", "outside what I can help with"),
        ("AR", "خارج نطاق"),
        ("KU", "دەرەوەی بواری"),
    ],
)
async def test_off_topic_question_gets_scope_message_not_generic_fallback(
    tmp_path, language, expected_fragment
):

    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        domain=ConversationDomain.GENERAL,
        is_off_topic=True,
    )

    result = await service.handle_message(
        message="what is the weather like",
        state=ConversationState(language=language.lower()),
        language=language,
    )

    payload = result.generation_result.payload

    assert payload.abstained is True
    assert payload.abstain_reason == "OUT_OF_SCOPE"
    assert payload.needs_handoff is False
    assert expected_fragment in payload.answer
    assert not payload.citations


@pytest.mark.asyncio
async def test_general_question_with_similar_knowledge_is_not_out_of_scope(tmp_path):

    similar = make_hybrid_candidate("KB-GEN-SPREAD-FA")
    similar.vector_score = 0.6

    def factory(query, context):
        result = empty_result(query)
        result.hybrid_candidates = [similar]
        return result

    service, _ = build_service(tmp_path, factory, domain=ConversationDomain.GENERAL)

    result = await service.handle_message(
        message="what is a spread", state=ConversationState(), language="EN"
    )

    payload = result.generation_result.payload

    assert payload.abstain_reason == "NO_ELIGIBLE_KNOWLEDGE"
    assert payload.needs_handoff is True


@pytest.mark.asyncio
async def test_unflagged_general_question_keeps_generic_fallback(tmp_path):

    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        domain=ConversationDomain.GENERAL,
        is_off_topic=False,
    )

    result = await service.handle_message(
        message="what are your support hours", state=ConversationState(), language="EN"
    )

    payload = result.generation_result.payload

    assert payload.abstain_reason == "NO_ELIGIBLE_KNOWLEDGE"
    assert payload.needs_handoff is True


@pytest.mark.asyncio
async def test_language_name_from_understanding_selects_matching_message(tmp_path):

    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        domain=ConversationDomain.GENERAL,
        is_off_topic=True,
    )

    result = await service.handle_message(
        message="what is the capital of france",
        state=ConversationState(language="English"),
        language="EN",
    )

    assert "outside what I can help with" in result.generation_result.payload.answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language,expected_fragment",
    [
        ("FA", "توصیهٔ سرمایه‌گذاری"),
        ("EN", "can't give investment advice"),
        ("AR", "نصائح استثمارية"),
        ("KU", "ئامۆژگاری وەبەرهێنان"),
    ],
)
async def test_investment_advice_request_gets_dedicated_message(
    tmp_path, language, expected_fragment
):

    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        domain=ConversationDomain.GENERAL,
        is_advice_request=True,
    )

    result = await service.handle_message(
        message="should i buy gold today",
        state=ConversationState(language=language.lower()),
        language=language,
    )

    payload = result.generation_result.payload

    assert payload.abstained is True
    assert payload.abstain_reason == "INVESTMENT_ADVICE_NOT_SUPPORTED"
    assert payload.needs_handoff is False
    assert expected_fragment in payload.answer
    assert not payload.citations


def _grounding_skip_inputs(*, answer, client_specific=False, reranker_mode="semantic"):
    from app.citations.validator import CitationValidationResult
    from app.generation.schemas import GeneratedAnswerPayload, GeneratedCitation
    from app.generation.service import GenerationResult
    from app.retrieval.reranker import RerankedCandidate


    hybrid = make_hybrid_candidate("KB-KYC-DOCS-AF-FA", category="KYC")

    candidate = hybrid.candidate

    retrieval_result = RetrievalPipelineResult(
        query="q",
        bm25_candidates=[],
        vector_candidates=[],
        hybrid_candidates=[hybrid],
        eligible_candidates=[hybrid],
        final_candidates=[
            RerankedCandidate(candidate=hybrid, rerank_score=0.9, reasons=[])
        ],
        reranker_mode=reranker_mode,
    )

    generation_result = GenerationResult(
        payload=GeneratedAnswerPayload(
            answer=answer,
            citations=[
                GeneratedCitation(
                    candidate_id=candidate.candidate_id,
                    knowledge_key=(candidate.metadata.knowledge_key),
                )
            ],
            abstained=False,
            needs_handoff=False,
        ),
        evidence_candidate_ids=[candidate.candidate_id],
        evidence_knowledge_keys=[candidate.metadata.knowledge_key],
        evidence_pairs={(candidate.candidate_id, candidate.metadata.knowledge_key)},
    )

    understanding = ConversationUnderstanding(
        domain=ConversationDomain.KYC, client_specific=client_specific
    )

    return dict(
        understanding=understanding,
        retrieval_result=retrieval_result,
        generation_result=generation_result,
        citation_validation_result=(CitationValidationResult(valid=True)),
    )


def test_verbatim_answer_skips_llm_grounding_after_semantic_reranker(tmp_path):

    service, _ = build_service(tmp_path, lambda query, context: empty_result(query))

    assert (
        service._can_use_deterministic_grounding(
            **_grounding_skip_inputs(answer="Test content")
        )
        is True
    )


def test_rewritten_answer_still_needs_llm_grounding(tmp_path):

    service, _ = build_service(tmp_path, lambda query, context: empty_result(query))

    assert (
        service._can_use_deterministic_grounding(
            **_grounding_skip_inputs(
                answer="Test content, plus a claim that is not in it."
            )
        )
        is False
    )


def test_client_specific_verbatim_answer_keeps_llm_grounding(tmp_path):
    service, _ = build_service(tmp_path, lambda query, context: empty_result(query))

    assert (
        service._can_use_deterministic_grounding(
            **_grounding_skip_inputs(answer="Test content", client_specific=True)
        )
        is False
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("fa", "fa"),
        ("FA", "fa"),
        ("en", "en"),
        ("EN", "en"),
        ("English", "en"),
        ("english", "en"),
        ("Persian", "fa"),
        ("Persian (Farsi)", "fa"),
        ("Farsi", "fa"),
        ("Arabic", "ar"),
        ("Kurdish", "ku"),
        ("Sorani", "ku"),
        ("  ", None),
        (None, None),
        ("de", "de"),
    ],
)
def test_language_names_map_to_page_language_codes(raw, expected):
    assert ConversationService._canonical_language(raw) == expected


@pytest.mark.asyncio
async def test_language_name_from_model_reaches_retrieval_as_code(tmp_path):
    # "English" from the understanding step used to be sent to the
    # language filters as-is and matched no page.
    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    await service.handle_message(
        message="how do i create a demo account",
        state=ConversationState(language="English"),
        language="EN",
    )

    assert pipeline.calls
    assert pipeline.calls[0]["language"] == "en"


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["what is my balance", "موجودی من چقدره؟"])
async def test_unauthenticated_personal_data_question_short_circuits(tmp_path, message):

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        task=ConversationTask.BALANCE,
    )

    result = await service.handle_message(
        message=message, state=ConversationState(), language="EN"
    )

    assert len(pipeline.calls) == 1
    assert result.retrieval_result is not None
    payload = result.generation_result.payload
    assert payload.abstained is True
    assert payload.abstain_reason == "CLIENT_DATA_NOT_AVAILABLE"
    assert payload.needs_handoff is True
    assert not payload.citations


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message", ["آخرین واریزم چقدر بود؟", "ایمیل ثبت‌شده روی حسابم چیه؟"]
)
async def test_account_deposit_withdrawal_possessive_no_longer_short_circuits(
    tmp_path, message
):

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        task=ConversationTask.BALANCE,
    )

    result = await service.handle_message(
        message=message, state=ConversationState(), language="FA"
    )

    if result.generation_result is not None:
        assert (
            result.generation_result.payload.abstain_reason
            != "CLIENT_DATA_NOT_AVAILABLE"
        )
    else:
        assert result.clarification_question is not None


@pytest.mark.asyncio
async def test_model_flagged_client_specific_also_short_circuits(tmp_path):

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        task=ConversationTask.EXPLANATION,
    )
    service.understanding_service.client_specific_override = True

    result = await service.handle_message(
        message="something with no keyword match",
        state=ConversationState(),
        language="EN",
    )

    assert len(pipeline.calls) == 1
    assert (
        result.generation_result.payload.abstain_reason == "CLIENT_DATA_NOT_AVAILABLE"
    )


@pytest.mark.asyncio
async def test_client_specific_question_still_answers_when_kb_has_it(tmp_path):

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: result_with_final_candidate(query),
        task=ConversationTask.TROUBLESHOOTING,
    )
    service.understanding_service.client_specific_override = True

    result = await service.handle_message(
        message="چرا صفحه برداشت من از دیروز غیرفعال شده",
        state=ConversationState(),
        language="FA",
    )

    assert len(pipeline.calls) == 1

    assert (
        result.generation_result is None
        or result.generation_result.payload.abstain_reason
        != "CLIENT_DATA_NOT_AVAILABLE"
    )


@pytest.mark.asyncio
async def test_generic_feature_disabled_report_overrides_false_client_specific(
    tmp_path,
):

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        task=ConversationTask.TROUBLESHOOTING,
    )
    service.understanding_service.client_specific_override = True

    result = await service.handle_message(
        message="چرا صفحه برداشت از دیروز غیرفعال شده",
        state=ConversationState(),
        language="FA",
    )

    assert len(pipeline.calls) == 1

    assert (
        result.generation_result is None
        or result.generation_result.payload.abstain_reason
        != "CLIENT_DATA_NOT_AVAILABLE"
    )


@pytest.mark.asyncio
async def test_generic_feature_disabled_override_does_not_catch_named_personal_complaint(
    tmp_path,
):

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        task=ConversationTask.TROUBLESHOOTING,
    )
    service.understanding_service.client_specific_override = True

    result = await service.handle_message(
        message=("my balance shows withdrawal is disabled, what is going " "on"),
        state=ConversationState(),
        language="EN",
    )

    assert len(pipeline.calls) == 1
    assert (
        result.generation_result.payload.abstain_reason == "CLIENT_DATA_NOT_AVAILABLE"
    )


@pytest.mark.asyncio
async def test_authenticated_client_specific_question_still_short_circuits(tmp_path):

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        task=ConversationTask.EXPLANATION,
    )
    service.understanding_service.client_specific_override = True

    result = await service.handle_message(
        message="what is my current balance",
        state=ConversationState(),
        language="EN",
        trusted_session=TrustedSessionContext(
            is_authenticated=True, client_id="CLIENT-123"
        ),
    )

    assert len(pipeline.calls) == 1
    assert (
        result.generation_result.payload.abstain_reason == "CLIENT_DATA_NOT_AVAILABLE"
    )


@pytest.mark.asyncio
async def test_authenticated_personal_data_question_still_reaches_retrieval(tmp_path):

    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    await service.handle_message(
        message="what is my balance",
        state=ConversationState(),
        language="EN",
        trusted_session=TrustedSessionContext(
            is_authenticated=True, client_id="CLIENT-123"
        ),
    )

    assert len(pipeline.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "چطور حسابمو تأیید کنم؟",
        "رمز حسابم رو چجوری عوض کنم؟",
        "how do i verify my account?",
    ],
)
async def test_howto_question_about_own_account_is_not_blocked(tmp_path, message):

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        task=ConversationTask.HOW_TO,
    )

    result = await service.handle_message(
        message=message, state=ConversationState(), language="FA"
    )

    assert len(pipeline.calls) == 1
    assert result.retrieval_result is not None
    assert (
        result.generation_result.payload.abstain_reason != "CLIENT_DATA_NOT_AVAILABLE"
    )


@pytest.mark.asyncio
async def test_readiness_check_asks_one_question_before_answering(tmp_path):
    readiness = FakeReadinessService(
        needs_clarification=True, question="آیا از فیلترشکن استفاده می‌کنید؟"
    )

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: result_with_final_candidate(query),
        readiness_service=readiness,
    )

    state = ConversationState()

    result = await service.handle_message(
        message="چرا نمودار من دیر آپدیت میشه؟", state=state, language="FA"
    )

    assert len(readiness.calls) == 1
    assert result.generation_result is None
    assert result.should_generate_answer is False
    assert result.clarification_question == "آیا از فیلترشکن استفاده می‌کنید؟"
    assert state.awaiting_context_field == "open_clarification"
    assert (
        state.pending_open_clarification_question == "آیا از فیلترشکن استفاده می‌کنید؟"
    )


@pytest.mark.asyncio
async def test_readiness_check_is_not_repeated_right_after_it_fires(tmp_path):
    readiness = FakeReadinessService(
        needs_clarification=True, question="آیا از فیلترشکن استفاده می‌کنید؟"
    )

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: result_with_final_candidate(query),
        readiness_service=readiness,
    )

    state = ConversationState()

    await service.handle_message(
        message="چرا نمودار من دیر آپدیت میشه؟", state=state, language="FA"
    )

    await service.handle_message(
        message="بله، از فیلترشکن استفاده می‌کنم", state=state, language="FA"
    )

    assert len(readiness.calls) == 1

    assert state.awaiting_context_field is None

    second_call_query = pipeline.calls[-1]["query"]

    assert second_call_query == "چرا نمودار من دیر آپدیت میشه؟"


@pytest.mark.asyncio
async def test_open_clarification_reply_enriches_rerank_but_not_search(tmp_path):

    readiness = FakeReadinessService(
        needs_clarification=True, question="کدام نوع حساب معاملاتی دارید؟"
    )

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: result_with_final_candidate(query),
        readiness_service=readiness,
    )

    state = ConversationState()

    await service.handle_message(message="چطور معامله کنم", state=state, language="FA")

    await service.handle_message(message="استاندارد", state=state, language="FA")

    second_call = pipeline.calls[-1]

    assert second_call["query"] == "چطور معامله کنم"

    assert second_call["rerank_query"] is not None

    assert "چطور معامله کنم" in second_call["rerank_query"]

    assert "استاندارد" in second_call["rerank_query"]


@pytest.mark.asyncio
async def test_readiness_check_skipped_when_no_candidates(tmp_path):
    readiness = FakeReadinessService(
        needs_clarification=True, question="should not be asked"
    )

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        readiness_service=readiness,
    )

    result = await service.handle_message(
        message="چرا نمودار من دیر آپدیت میشه؟",
        state=ConversationState(),
        language="FA",
    )


    assert len(readiness.calls) == 0


@pytest.mark.asyncio
async def test_pending_clarification_reply_does_not_flip_conversation_language(
    tmp_path,
):

    registry = build_registry(tmp_path)

    resolver = ConversationResolver(entity_registry=registry)

    pipeline = FakeRetrievalPipeline(
        lambda query, context: result_with_final_candidate(query)
    )

    readiness = FakeReadinessService(
        needs_clarification=True, question="کشور محل اقامت شما کجاست؟"
    )

    understanding = SequencedLanguageUnderstandingService(["FA", "EN"])

    service = ConversationService(
        understanding_service=understanding,
        resolver=resolver,
        retrieval_pipeline=pipeline,
        context_policy=ContextPolicy(),
        readiness_service=readiness,
    )

    state = ConversationState()

    await service.handle_message(
        message="چرا صفحه برداشت از دیروز غیرفعال شده؟", state=state, language="FA"
    )

    assert state.language == "fa"

    result = await service.handle_message(message="iran", state=state, language="FA")

    assert understanding.calls == 2

    assert state.language == "fa"

    assert result.resolved_turn.language == "fa"


@pytest.mark.asyncio
@pytest.mark.parametrize("message,expected_language", [("مرحبا", "ar"), ("سڵاو", "ku")])
async def test_arabic_and_kurdish_greetings_skip_retrieval(
    tmp_path, message, expected_language
):
    # The greeting fast-path previously only recognized FA/EN openers -
    # an Arabic or Kurdish greeting fell through to the full pipeline
    # instead of getting an instant reply.
    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    result = await service.handle_message(
        message=message, state=ConversationState(), language=None
    )

    assert pipeline.calls == []
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is False
    assert result.understanding.language == expected_language


@pytest.mark.asyncio
async def test_greeting_response_does_not_list_topics(tmp_path):

    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    result = await service.handle_message(
        message="hi", state=ConversationState(), language="EN"
    )

    answer = result.generation_result.payload.answer

    for leaked_topic in ("KYC", "deposit", "withdrawal", "MT5", "cTrader"):
        assert leaked_topic.lower() not in answer.lower()


@pytest.mark.asyncio
async def test_triage_asks_before_abstaining_when_nothing_matches(tmp_path):
    readiness = FakeReadinessService(
        needs_clarification=False,
        triage_needs_clarification=True,
        triage_question="آیا از فیلترشکن استفاده می‌کنید؟",
    )

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: result_with_eligible_but_no_final(query),
        readiness_service=readiness,
    )

    state = ConversationState()

    result = await service.handle_message(
        message="چرا نمودار من دیر آپدیت میشه؟", state=state, language="FA"
    )

    assert len(readiness.calls) == 0
    assert len(readiness.triage_calls) == 1
    assert result.generation_result is None
    assert result.should_generate_answer is False
    assert result.clarification_question == "آیا از فیلترشکن استفاده می‌کنید؟"
    assert state.awaiting_context_field == "open_clarification"


@pytest.mark.asyncio
async def test_triage_not_called_when_no_eligible_candidates_at_all(tmp_path):
    readiness = FakeReadinessService(
        needs_clarification=False,
        triage_needs_clarification=True,
        triage_question="should not be asked",
    )

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        readiness_service=readiness,
        domain=ConversationDomain.ACCOUNT,
    )

    result = await service.handle_message(
        message="چرا نمودار من دیر آپدیت میشه؟",
        state=ConversationState(),
        language="FA",
    )

    # Nothing to ground a triage question in - must fall through to the
    # ordinary honest "couldn't find" answer instead of inventing one.
    assert len(readiness.triage_calls) == 0
    assert result.generation_result is not None


@pytest.mark.asyncio
async def test_triage_declines_falls_through_to_no_candidates_answer(tmp_path):
    readiness = FakeReadinessService(
        needs_clarification=False, triage_needs_clarification=False
    )

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: result_with_eligible_but_no_final(query),
        readiness_service=readiness,
    )

    result = await service.handle_message(
        message="چرا نمودار من دیر آپدیت میشه؟",
        state=ConversationState(),
        language="FA",
    )

    assert len(readiness.triage_calls) == 1
    assert result.clarification_question is None
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is True


@pytest.mark.asyncio
async def test_triage_not_repeated_right_after_it_fires(tmp_path):
    readiness = FakeReadinessService(
        needs_clarification=False,
        triage_needs_clarification=True,
        triage_question="آیا از فیلترشکن استفاده می‌کنید؟",
    )

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: result_with_eligible_but_no_final(query),
        readiness_service=readiness,
    )

    state = ConversationState()

    await service.handle_message(
        message="چرا نمودار من دیر آپدیت میشه؟", state=state, language="FA"
    )

    await service.handle_message(
        message="بله، از فیلترشکن استفاده می‌کنم", state=state, language="FA"
    )

    assert len(readiness.triage_calls) == 1

    second_call_query = pipeline.calls[-1]["query"]

    assert second_call_query == "چرا نمودار من دیر آپدیت میشه؟"


def test_build_generation_query_folds_in_clarification_reply():


    from app.conversation.state import ResolvedTurn

    resolved_turn = ResolvedTurn(
        original_message="بله، از فیلترشکن استفاده می‌کنم",
        resolved_query="چرا نمودار من دیر آپدیت میشه؟",
        inherited_context={
            "resolved_context_field": "open_clarification",
            "clarification_answer": "بله، از فیلترشکن استفاده می‌کنم",
        },
    )

    generation_query = ConversationService._build_generation_query(
        resolved_query="چرا نمودار من دیر آپدیت میشه؟", resolved_turn=resolved_turn
    )

    assert "چرا نمودار من دیر آپدیت میشه؟" in generation_query
    assert "فیلترشکن" in generation_query


def test_build_generation_query_unchanged_outside_open_clarification():
    from app.conversation.state import ResolvedTurn

    resolved_turn = ResolvedTurn(
        original_message="ایران",
        resolved_query="مدارک احراز هویت برای کدام کشور؟",
        inherited_context={"resolved_context_field": "kyc_document_country"},
    )

    generation_query = ConversationService._build_generation_query(
        resolved_query="مدارک احراز هویت برای کدام کشور؟", resolved_turn=resolved_turn
    )

    assert generation_query == "مدارک احراز هویت برای کدام کشور؟"


@pytest.mark.asyncio
async def test_password_content_uses_deterministic_generation(tmp_path):

    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        generation_service=AnswerGenerationService(llm=None),
    )

    candidate = make_hybrid_candidate("KB-ACCESS-002-FA")

    candidate.candidate.title = "Change your trading password"

    candidate.candidate.content = (
        "Enter your current password, " "then your new password."
    )

    reranked = RerankedCandidate(candidate=candidate, rerank_score=0.9, reasons=[])

    retrieval_result = result_with_final_candidate("how do I change my password")

    retrieval_result.final_candidates = [reranked]

    understanding = ConversationUnderstanding(
        language="en",
        domain=ConversationDomain.ACCOUNT,
        task=ConversationTask.HOW_TO,
        client_specific=False,
        topic_relation=TopicRelation.NEW_TOPIC,
        referenced_countries=[],
        referenced_entities=[],
        requires_previous_context=False,
        confidence=0.9,
    )

    assert (
        service._can_use_deterministic_generation(
            understanding=understanding, retrieval_result=retrieval_result
        )
        is True
    )


@pytest.mark.asyncio
async def test_non_password_content_uses_llm_generation(tmp_path):

    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        generation_service=AnswerGenerationService(llm=None),
    )

    candidate = make_hybrid_candidate("KB-IB-002-FA")

    candidate.candidate.title = "Change your IB referral code"

    candidate.candidate.content = (
        "Go to Settings, then Profile, " "then update your referral code."
    )

    reranked = RerankedCandidate(candidate=candidate, rerank_score=0.9, reasons=[])

    retrieval_result = result_with_final_candidate("how do I change my referral code")

    retrieval_result.final_candidates = [reranked]

    understanding = ConversationUnderstanding(
        language="en",
        domain=ConversationDomain.IB,
        task=ConversationTask.HOW_TO,
        client_specific=False,
        topic_relation=TopicRelation.NEW_TOPIC,
        referenced_countries=[],
        referenced_entities=[],
        requires_previous_context=False,
        confidence=0.9,
    )

    assert (
        service._can_use_deterministic_generation(
            understanding=understanding, retrieval_result=retrieval_result
        )
        is False
    )


def _deterministic_routing_understanding(
    *, domain=ConversationDomain.ACCOUNT
) -> ConversationUnderstanding:
    return ConversationUnderstanding(
        language="fa",
        domain=domain,
        task=ConversationTask.HOW_TO,
        client_specific=False,
        topic_relation=TopicRelation.NEW_TOPIC,
        referenced_countries=[],
        referenced_entities=[],
        requires_previous_context=False,
        confidence=0.9,
    )


async def _assert_deterministic_routing(
    tmp_path, *, title: str, content: str, expected: bool
):
    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        generation_service=AnswerGenerationService(llm=None),
    )

    candidate = make_hybrid_candidate("KB-TEST-001")

    candidate.candidate.title = title
    candidate.candidate.content = content

    reranked = RerankedCandidate(candidate=candidate, rerank_score=0.9, reasons=[])

    retrieval_result = result_with_final_candidate(title)

    retrieval_result.final_candidates = [reranked]

    assert (
        service._can_use_deterministic_generation(
            understanding=(_deterministic_routing_understanding()),
            retrieval_result=retrieval_result,
        )
        is expected
    )


@pytest.mark.asyncio
async def test_arabic_password_title_uses_deterministic_generation(tmp_path):

    await _assert_deterministic_routing(
        tmp_path,
        title="إعادة تعيين كلمة مرور حساب التداول",
        content="أدخل كلمة المرور الحالية ثم الجديدة.",
        expected=True,
    )


@pytest.mark.asyncio
async def test_kurdish_password_title_uses_deterministic_generation(tmp_path):

    await _assert_deterministic_routing(
        tmp_path,
        title="نوێکردنەوەی وشەی نهێنی پۆرتاڵی کڕیار",
        content="وشەی نهێنی ئێستات بنووسە، پاشان وشەی نهێنی نوێ.",
        expected=True,
    )


@pytest.mark.asyncio
async def test_arabic_ib_referral_code_title_uses_llm_generation(tmp_path):

    await _assert_deterministic_routing(
        tmp_path,
        title="تغيير رمز المُعرّف IB",
        content="انتقل إلى الإعدادات ثم أدخل رمز المُعرّف الجديد.",
        expected=False,
    )


@pytest.mark.asyncio
async def test_persian_withdrawal_fee_title_uses_llm_generation(tmp_path):

    await _assert_deterministic_routing(
        tmp_path,
        title="کارمزد برداشت",
        content="کارمزد به روش برداشت بستگی دارد.",
        expected=False,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message,expected_language",
    [
        ("ممنون", "fa"),
        ("خیلی ممنون", "fa"),
        ("thanks", "en"),
        ("thank you", "en"),
        ("شكرا", "ar"),
        ("سوپاس", "ku"),
    ],
)
async def test_thanks_message_skips_retrieval(tmp_path, message, expected_language):

    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    result = await service.handle_message(
        message=message, state=ConversationState(), language=None
    )

    assert pipeline.calls == []
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is False
    assert result.understanding.language == expected_language


@pytest.mark.asyncio
async def test_thanks_as_part_of_a_real_request_is_not_shortcut(tmp_path):

    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    await service.handle_message(
        message="ممنون میشم راهنمایی کنید", state=ConversationState(), language="FA"
    )

    assert len(pipeline.calls) == 1


class FakeGenerationServiceForCache:
    def __init__(self, result):
        self.result = result
        self.generate_calls = []
        self.max_candidates = 2

    def can_generate_deterministically(self, *, candidates):
        return False

    async def generate(
        self, *, query, language, candidates, context, evidence_override=None
    ):
        self.generate_calls.append({"query": query, "language": language})

        return self.result


class FakeAnswerCacheService:
    def __init__(self, *, cached_result=None):
        self.cached_result = cached_result

        self.get_calls = []
        self.set_calls = []

    async def get(self, *, knowledge_keys, language, context):
        self.get_calls.append(
            {"knowledge_keys": knowledge_keys, "language": language, "context": context}
        )

        return self.cached_result

    async def set(self, *, knowledge_keys, language, context, result):
        self.set_calls.append(
            {
                "knowledge_keys": knowledge_keys,
                "language": language,
                "context": context,
                "result": result,
            }
        )


def make_llm_generation_result(
    answer="cached or fresh answer", knowledge_key="KB-TEST-001"
):
    from app.generation.schemas import GeneratedAnswerPayload, GeneratedCitation
    from app.generation.service import GenerationResult

    return GenerationResult(
        payload=GeneratedAnswerPayload(
            answer=answer,
            citations=[
                GeneratedCitation(
                    candidate_id=(f"{knowledge_key}:chunk-1"),
                    knowledge_key=(knowledge_key),
                )
            ],
            abstained=False,
        ),
        evidence_candidate_ids=[f"{knowledge_key}:chunk-1"],
        evidence_knowledge_keys=[knowledge_key],
        evidence_pairs={(f"{knowledge_key}:chunk-1", knowledge_key)},
    )


@pytest.mark.asyncio
async def test_cache_hit_skips_llm_generation(tmp_path):
    cache = FakeAnswerCacheService(
        cached_result=(make_llm_generation_result(answer="from cache"))
    )

    generation = FakeGenerationServiceForCache(
        make_llm_generation_result(answer="freshly generated")
    )

    service, _ = build_service(
        tmp_path,
        lambda query, context: (result_with_final_candidate(query)),
        generation_service=generation,
        answer_cache_service=cache,
    )

    result = await service.handle_message(
        message="how do I change my IB referral code",
        state=ConversationState(),
        language="EN",
    )

    assert len(cache.get_calls) == 1
    assert generation.generate_calls == []
    assert result.generation_result.payload.answer == "from cache"


@pytest.mark.asyncio
async def test_cache_miss_calls_llm_and_stores_result(tmp_path):
    cache = FakeAnswerCacheService(cached_result=None)

    generation = FakeGenerationServiceForCache(
        make_llm_generation_result(answer="freshly generated")
    )

    service, _ = build_service(
        tmp_path,
        lambda query, context: (result_with_final_candidate(query)),
        generation_service=generation,
        answer_cache_service=cache,
    )

    result = await service.handle_message(
        message="how do I change my IB referral code",
        state=ConversationState(),
        language="EN",
    )

    assert len(generation.generate_calls) == 1
    assert len(cache.set_calls) == 1
    assert result.generation_result.payload.answer == "freshly generated"


@pytest.mark.asyncio
async def test_abstained_result_is_not_cached(tmp_path):
    cache = FakeAnswerCacheService(cached_result=None)

    from app.generation.schemas import GeneratedAnswerPayload
    from app.generation.service import GenerationResult

    abstained_result = GenerationResult(
        payload=GeneratedAnswerPayload(
            answer="",
            citations=[],
            abstained=True,
            abstain_reason=("INSUFFICIENT_EVIDENCE"),
        ),
        evidence_candidate_ids=[],
        evidence_knowledge_keys=[],
        evidence_pairs=set(),
    )

    generation = FakeGenerationServiceForCache(abstained_result)

    service, _ = build_service(
        tmp_path,
        lambda query, context: (result_with_final_candidate(query)),
        generation_service=generation,
        answer_cache_service=cache,
    )

    await service.handle_message(
        message="how do I change my IB referral code",
        state=ConversationState(),
        language="EN",
    )

    assert cache.set_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["وقت بخیر", "روز بخیر", "ظهر بخیر"])
async def test_standalone_time_greeting_skips_retrieval(tmp_path, message):

    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query)
    )

    result = await service.handle_message(
        message=message, state=ConversationState(), language="FA"
    )

    assert pipeline.calls == []
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is False


@pytest.mark.asyncio
async def test_small_talk_flag_skips_retrieval_for_unlisted_phrasing(tmp_path):

    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query), is_small_talk=True
    )

    result = await service.handle_message(
        message="یه عبارت کاملاً جدید که تا حالا ندیدیم",
        state=ConversationState(),
        language="FA",
    )

    assert pipeline.calls == []
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is False
    assert result.should_generate_answer is True


@pytest.mark.asyncio
async def test_small_talk_false_still_reaches_retrieval(tmp_path):
    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query), is_small_talk=False
    )

    await service.handle_message(
        message="چطور می‌تونم کد معرفم رو عوض کنم؟",
        state=ConversationState(),
        language="FA",
    )

    assert len(pipeline.calls) == 1


@pytest.mark.asyncio
async def test_exact_thanks_phrase_overrides_false_small_talk(tmp_path):

    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query), is_small_talk=False
    )

    result = await service.handle_message(
        message="سپاس از شما", state=ConversationState(), language="FA"
    )

    assert pipeline.calls == []
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is False


@pytest.mark.asyncio
async def test_exact_acknowledgment_phrase_overrides_false_small_talk(tmp_path):

    service, pipeline = build_service(
        tmp_path, lambda query, context: empty_result(query), is_small_talk=False
    )

    result = await service.handle_message(
        message="اکی", state=ConversationState(), language="FA"
    )

    assert pipeline.calls == []
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is False


@pytest.mark.asyncio
async def test_small_talk_flag_is_ignored_while_context_is_pending(tmp_path):

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: result_with_final_candidate(query),
        is_small_talk=True,
        small_talk_kind="opener",
    )

    state = ConversationState()
    state.awaiting_context_field = "open_clarification"
    state.pending_open_clarification_question = (
        "آیا قبلاً کیف پول بانکی خود را ثبت کرده‌اید؟"
    )
    state.pending_open_clarification_query = "چطور می‌تونم پول برداشت کنم؟"

    result = await service.handle_message(message="بله", state=state, language="FA")


    assert len(pipeline.calls) == 1
    assert result.understanding.is_small_talk is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,expected_substring",
    [
        ("thanks", "خواهش می‌کنم"),
        ("farewell", "خدانگهدار"),
        ("greeting", "چطور می‌تونم کمکتون کنم"),
        (None, "چطور می‌تونم کمکتون کنم"),
    ],
)
async def test_small_talk_reply_matches_its_kind(tmp_path, kind, expected_substring):

    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        is_small_talk=True,
        small_talk_kind=kind,
    )

    result = await service.handle_message(
        message="خیلی لطف کردید", state=ConversationState(), language="FA"
    )

    assert expected_substring in result.generation_result.payload.answer


@pytest.mark.asyncio
async def test_small_talk_opener_kind_invites_the_question(tmp_path):

    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        is_small_talk=True,
        small_talk_kind="opener",
    )

    result = await service.handle_message(
        message="اها یه سوال دیگه دارم", state=ConversationState(), language="FA"
    )

    assert pipeline.calls == []
    assert "سؤالتون رو بپرسید" in result.generation_result.payload.answer


def _sub_question_result_factory(query, context):

    if query == "sub-question-1":
        return result_with_final_candidate(query, knowledge_key="KB-SUB-001")
    if query == "sub-question-2":
        return result_with_final_candidate(query, knowledge_key="KB-SUB-002")
    return result_with_final_candidate(query, knowledge_key="KB-PRIMARY")


@pytest.mark.asyncio
async def test_sub_questions_merge_candidates_from_each_sub_question(tmp_path):

    service, pipeline = build_service(
        tmp_path,
        _sub_question_result_factory,
        sub_questions=["sub-question-1", "sub-question-2"],
    )

    result = await service.handle_message(
        message="compound question with two sub-questions",
        state=ConversationState(),
        language="FA",
    )

    # One retrieve() call for the combined/primary query, plus one per
    # sub-question.
    assert len(pipeline.calls) == 3

    final_keys = {
        rc.candidate.candidate.metadata.knowledge_key
        for rc in result.retrieval_result.final_candidates
    }

    assert final_keys == {"KB-PRIMARY", "KB-SUB-001", "KB-SUB-002"}


@pytest.mark.asyncio
async def test_sub_questions_do_not_trigger_disambiguation(tmp_path):

    def factory(query, context):
        result = _sub_question_result_factory(query, context)
        result.distinct_options = ["option a", "option b"]
        return result

    service, pipeline = build_service(
        tmp_path, factory, sub_questions=["sub-question-1", "sub-question-2"]
    )

    result = await service.handle_message(
        message="compound question with two sub-questions",
        state=ConversationState(),
        language="FA",
    )

    assert result.clarification_question is None

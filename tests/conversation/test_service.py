import json

import pytest

from app.conversation.resolver import (ConversationResolver,)
from app.conversation.service import (ConversationService,TrustedSessionContext,)
from app.conversation.state import (ConversationState,)
from app.knowledge.enums import (Audience,BotMode,FallbackPolicy,KnowledgeStatus,Language,MarketScope,SourceAuthority,)
from app.knowledge.schemas import (KnowledgeMetadata,)
from app.retrieval.hybrid import (HybridCandidate,)
from app.retrieval.pipeline import (EligibilityTrace,RetrievalPipelineResult,)
from app.retrieval.schemas import (KnowledgeCandidate,)
from app.runtime.context_policy import (ContextPolicy,)
from app.runtime.entity_registry import (EntityRegistry,)
from app.conversation.schemas import (ConversationDomain,ConversationTask,ConversationUnderstanding,TopicRelation,)

class FakeRetrievalPipeline:
    def __init__(
        self,
        result_factory,
    ):
        self.result_factory = (
            result_factory
        )
        self.calls = []

    async def retrieve(
        self,
        *,
        query,
        context,
        language=None,
        knowledge_hints=None,
    ):
        self.calls.append(
            {
                "query": query,
                "context": context,
                "language": language,
            }
        )

        return self.result_factory(
            query,
            context,
        )



class FakeUnderstandingService:
    def __init__(
        self,
        domain=ConversationDomain.GENERAL,
        is_off_topic=False,
        is_advice_request=False,
        task=ConversationTask.HOW_TO,
    ):
        self.domain = domain
        self.is_off_topic = is_off_topic
        self.is_advice_request = is_advice_request
        self.client_specific_override = False
        self.task = task

    async def understand(
        self,
        *,
        message,
        state,
    ):
        return ConversationUnderstanding(
            language=(
                state.language
                or "FA"
            ),
            domain=self.domain,
            is_off_topic=self.is_off_topic,
            is_advice_request=self.is_advice_request,
            client_specific=(
                self.client_specific_override
            ),
            task=self.task,
            topic_relation=(
                TopicRelation.NEW_TOPIC
            ),
            referenced_countries=[],
            referenced_entities=[],
            requires_previous_context=False,
            confidence=0.95,
        )

def empty_result(
    query: str,
):
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
    knowledge_key: str,
    category: str = "TEST",
) -> HybridCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key=knowledge_key,
        canonical_knowledge_key=(
            knowledge_key
        ),
        knowledge_type="HOW_TO",
        category=category,
        status=(
            KnowledgeStatus.APPROVED
        ),
        audience=Audience.BOTH,
        owner="Test",
        primary_source="Test",
        source_ids=["TEST"],
        source_authority=(
            SourceAuthority.APPROVED_DOCUMENT
        ),
        language=Language.FA,
        market_scope=(
            MarketScope.COUNTRY
        ),
        countries=["AF"],
        regions=[],
        required_context_fields=[
            "kyc_document_country"
        ],
        bot_mode=(
            BotMode.KNOWLEDGE
        ),
        fallback_policy=(
            FallbackPolicy.NONE
        ),
    )

    candidate = KnowledgeCandidate(
        candidate_id=(
            f"{knowledge_key}:chunk-1"
        ),
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


def build_registry(
    tmp_path,
):
    config = {
        "entity_types": {
            "platform": {
                "fuzzy_enabled": True,
                "fuzzy_threshold": 0.70,
                "aliases": {
                    "ctrader": "cTrader",
                    "mt5": "MT5",
                    "metatrader": "MT5",
                },
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
            "kyc_document_country":
                "country",
            "country_of_residence":
                "country",
            "nationality":
                "country",
            "referenced_country":
                "country",
        },
        "conversation": {
            "short_followup_max_tokens": 4,
            "correction_markers": [
                "نه منظورم",
                "i mean",
            ],
        },
    }

    path = (
        tmp_path
        / "entities.json"
    )

    path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return (
        EntityRegistry.from_file(
            path
        )
    )


def build_service(
    tmp_path,
    result_factory,
    domain=ConversationDomain.GENERAL,
    is_off_topic=False,
    is_advice_request=False,
    task=ConversationTask.HOW_TO,
):
    registry = build_registry(
        tmp_path
    )

    resolver = (
        ConversationResolver(
            entity_registry=registry
        )
    )

    pipeline = (
        FakeRetrievalPipeline(
            result_factory
        )
    )

    service = ConversationService(
        understanding_service=(
            FakeUnderstandingService(
                domain=domain,
                is_off_topic=is_off_topic,
                is_advice_request=is_advice_request,
                task=task,
            )
        ),
        resolver=resolver,
        retrieval_pipeline=pipeline,
        context_policy=ContextPolicy(),
    )

    return service, pipeline


@pytest.mark.asyncio
async def test_service_sends_resolved_query_to_pipeline(
    tmp_path,
):
    service, pipeline = (
        build_service(
            tmp_path,
            lambda query, context:
                empty_result(query),
        )
    )

    state = ConversationState()

    await service.handle_message(
        message="cTrader",
        state=state,
        language="FA",
    )

    assert len(
        pipeline.calls
    ) == 1

    assert (
        pipeline.calls[0]["query"]
        == "ctrader"
    )


@pytest.mark.asyncio
async def test_followup_changes_platform_using_conversation_context(
    tmp_path,
):
    service, pipeline = (
        build_service(
            tmp_path,
            lambda query, context:
                empty_result(query),
        )
    )

    state = ConversationState()

    await service.handle_message(
        message=(
            "cTrader چجوری نصب میشه؟"
        ),
        state=state,
        language="FA",
    )

    await service.handle_message(
        message="MT5 چی؟",
        state=state,
        language="FA",
    )

    assert len(
        pipeline.calls
    ) == 2

    second_query = (
        pipeline.calls[1]["query"]
    )

    assert "MT5" in second_query

    assert (
        "cTrader"
        not in second_query
    )


@pytest.mark.asyncio
async def test_missing_context_sets_pending_field(
    tmp_path,
):
    blocked_candidate = (
        make_hybrid_candidate(
            "KB-COUNTRY-001",
            category="KYC",
        )
    )

    def factory(
        query,
        context,
    ):
        return RetrievalPipelineResult(
            query=query,
            bm25_candidates=[],
            vector_candidates=[],
            hybrid_candidates=[
                blocked_candidate
            ],
            eligible_candidates=[],
            final_candidates=[],
            eligibility_trace=[
                EligibilityTrace(
                    knowledge_key=(
                        "KB-COUNTRY-001"
                    ),
                    eligible=False,
                    reasons=[
                        (
                            "MISSING_CONTEXT:"
                            "kyc_document_country"
                        ),
                        (
                            "COUNTRY_CONTEXT_"
                            "MISSING"
                        ),
                    ],
                )
            ],
        )

    service, _ = (
        build_service(
            tmp_path,
            factory,
            domain=ConversationDomain.KYC,
        )
    )

    state = ConversationState()

    result = (
        await service.handle_message(
            message=(
                "مدارک احراز هویت"
            ),
            state=state,
            language="FA",
        )
    )

    assert (
        result.should_generate_answer
        is False
    )

    assert (
        result.resolved_turn
        .needs_clarification
        is True
    )

    assert (
        result.resolved_turn
        .missing_context_fields
        == [
            "kyc_document_country"
        ]
    )

    assert (
        state.awaiting_context_field
        == "kyc_document_country"
    )

    assert (
        result.clarification
        is not None
    )

    assert (
        result.clarification
        .field_name
        == "kyc_document_country"
    )


@pytest.mark.asyncio
async def test_pending_context_answer_is_stored(
    tmp_path,
):
    def factory(
        query,
        context,
    ):
        return empty_result(
            query
        )

    service, pipeline = (
        build_service(
            tmp_path,
            factory,
        )
    )

    state = ConversationState(
        language="FA",
        last_resolved_query=(
            "KYC document requirements"
        ),
        awaiting_context_field=(
            "kyc_document_country"
        ),
    )

    await service.handle_message(
        message="افغانستان",
        state=state,
        language="FA",
    )

    assert (
        state.kyc_document_country
        == "AF"
    )

    assert (
        state.awaiting_context_field
        is None
    )

    assert (
        pipeline.calls[-1]
        ["context"]
        .kyc_document_country
        == "AF"
    )


@pytest.mark.asyncio
async def test_trusted_session_sets_client_id(
    tmp_path,
):
    service, pipeline = (
        build_service(
            tmp_path,
            lambda query, context:
                empty_result(query),
        )
    )

    state = ConversationState()

    await service.handle_message(
        message="what is my balance",
        state=state,
        language="EN",
        trusted_session=(
            TrustedSessionContext(
                is_authenticated=True,
                client_id="CLIENT-123",
            )
        ),
    )

    assert (
        state.is_authenticated
        is True
    )

    assert (
        state.client_id
        == "CLIENT-123"
    )

    context = (
        pipeline.calls[-1][
            "context"
        ]
    )

    assert (
        context.is_authenticated
        is True
    )

    assert (
        context.client_id
        == "CLIENT-123"
    )

@pytest.mark.asyncio
async def test_authenticated_session_without_client_id_is_allowed(
    tmp_path,
):
    service, pipeline = (
        build_service(
            tmp_path,
            lambda query, context:
                empty_result(query),
        )
    )

    state = ConversationState()

    result = await service.handle_message(
        message="what is my balance",
        state=state,
        language="EN",
        trusted_session=(
            TrustedSessionContext(
                is_authenticated=True,
                client_id=None,
            )
        ),
    )

    assert (
        state.is_authenticated
        is True
    )

    assert (
        state.client_id
        is None
    )

    context = (
        pipeline.calls[-1][
            "context"
        ]
    )

    assert (
        context.is_authenticated
        is True
    )

    assert (
        context.client_id
        is None
    )

    assert (
        result.should_generate_answer
        is False
    )
@pytest.mark.asyncio
async def test_user_text_cannot_set_client_id(
    tmp_path,
):
    service, _ = (
        build_service(
            tmp_path,
            lambda query, context:
                empty_result(query),
        )
    )

    state = ConversationState()

    await service.handle_message(
        message=(
            "my client id is 12345"
        ),
        state=state,
        language="EN",
    )

    assert (
        state.is_authenticated
        is False
    )

    assert (
        state.client_id
        is None
    )

@pytest.mark.asyncio
async def test_fuzzy_greeting_slm_skips_retrieval(
    tmp_path,
):
    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
    )

    state = ConversationState()

    result = await service.handle_message(
        message="slm",
        state=state,
        language="EN",
    )

    assert pipeline.calls == []
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is False
    assert result.generation_result.payload.answer


@pytest.mark.asyncio
async def test_no_candidates_returns_safe_answer_instead_of_blank(
    tmp_path,
):
    # An in-scope product area (ACCOUNT) with nothing eligible in the
    # knowledge base gets the generic "couldn't find" message + handoff.
    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        domain=ConversationDomain.ACCOUNT,
    )

    state = ConversationState()

    result = await service.handle_message(
        message="unknown unsupported topic",
        state=state,
        language="EN",
    )

    assert len(pipeline.calls) == 1
    assert result.generation_result is not None
    assert result.generation_result.payload.abstained is True
    assert (
        result.generation_result.payload.abstain_reason
        == "NO_ELIGIBLE_KNOWLEDGE"
    )
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
    tmp_path,
    language,
    expected_fragment,
):
    # GENERAL/UNKNOWN domain + nothing even vaguely similar in the
    # knowledge base = clearly off-topic (weather, jokes, ...): say what
    # the assistant covers instead of implying the answer might exist.
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
async def test_general_question_with_similar_knowledge_is_not_out_of_scope(
    tmp_path,
):
    # A GENERAL-domain question that does resemble knowledge-base content
    # (top vector similarity above the off-topic ceiling) but produced no
    # eligible answer is a knowledge gap, not an off-topic question.
    similar = make_hybrid_candidate("KB-GEN-SPREAD-FA")
    similar.vector_score = 0.6

    def factory(query, context):
        result = empty_result(query)
        result.hybrid_candidates = [similar]
        return result

    service, _ = build_service(
        tmp_path,
        factory,
        domain=ConversationDomain.GENERAL,
    )

    result = await service.handle_message(
        message="what is a spread",
        state=ConversationState(),
        language="EN",
    )

    payload = result.generation_result.payload

    assert payload.abstain_reason == "NO_ELIGIBLE_KNOWLEDGE"
    assert payload.needs_handoff is True


@pytest.mark.asyncio
async def test_unflagged_general_question_keeps_generic_fallback(
    tmp_path,
):
    # A question about eplanet that simply is not in the knowledge base
    # (e.g. support hours) is not flagged off-topic by the understanding
    # step, so it must not be told it is outside the assistant's scope.
    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        domain=ConversationDomain.GENERAL,
        is_off_topic=False,
    )

    result = await service.handle_message(
        message="what are your support hours",
        state=ConversationState(),
        language="EN",
    )

    payload = result.generation_result.payload

    assert payload.abstain_reason == "NO_ELIGIBLE_KNOWLEDGE"
    assert payload.needs_handoff is True


@pytest.mark.asyncio
async def test_language_name_from_understanding_selects_matching_message(
    tmp_path,
):
    # The understanding model sometimes returns the language as a name
    # ("English") instead of a code ("en"); that must still pick the
    # English message rather than the Persian default.
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

    assert (
        "outside what I can help with"
        in result.generation_result.payload.answer
    )


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
    tmp_path,
    language,
    expected_fragment,
):
    # "Should I buy gold today?" is in the trading world but is a request
    # for a recommendation, not something the knowledge base answers:
    # say so plainly instead of implying the answer just was not found.
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
    assert (
        payload.abstain_reason
        == "INVESTMENT_ADVICE_NOT_SUPPORTED"
    )
    assert payload.needs_handoff is False
    assert expected_fragment in payload.answer
    assert not payload.citations


def _grounding_skip_inputs(
    *,
    answer,
    client_specific=False,
    reranker_mode="semantic",
):
    from app.citations.validator import (
        CitationValidationResult,
    )
    from app.generation.schemas import (
        GeneratedAnswerPayload,
        GeneratedCitation,
    )
    from app.generation.service import (
        GenerationResult,
    )
    from app.retrieval.reranker import (
        RerankedCandidate,
    )

    # A country-scoped page with a required context field: exactly the
    # kind of "sensitive" candidate the old fast-path-only skip refused.
    hybrid = make_hybrid_candidate(
        "KB-KYC-DOCS-AF-FA",
        category="KYC",
    )

    candidate = hybrid.candidate

    retrieval_result = RetrievalPipelineResult(
        query="q",
        bm25_candidates=[],
        vector_candidates=[],
        hybrid_candidates=[hybrid],
        eligible_candidates=[hybrid],
        final_candidates=[
            RerankedCandidate(
                candidate=hybrid,
                rerank_score=0.9,
                reasons=[],
            )
        ],
        reranker_mode=reranker_mode,
    )

    generation_result = GenerationResult(
        payload=GeneratedAnswerPayload(
            answer=answer,
            citations=[
                GeneratedCitation(
                    candidate_id=candidate.candidate_id,
                    knowledge_key=(
                        candidate.metadata.knowledge_key
                    ),
                )
            ],
            abstained=False,
            needs_handoff=False,
        ),
        evidence_candidate_ids=[
            candidate.candidate_id
        ],
        evidence_knowledge_keys=[
            candidate.metadata.knowledge_key
        ],
        evidence_pairs={
            (
                candidate.candidate_id,
                candidate.metadata.knowledge_key,
            )
        },
    )

    understanding = ConversationUnderstanding(
        domain=ConversationDomain.KYC,
        client_specific=client_specific,
    )

    return dict(
        understanding=understanding,
        retrieval_result=retrieval_result,
        generation_result=generation_result,
        citation_validation_result=(
            CitationValidationResult(valid=True)
        ),
    )


def test_verbatim_answer_skips_llm_grounding_after_semantic_reranker(
    tmp_path,
):
    # The answer IS the cited approved text ("Test content"), so LLM
    # grounding cannot catch anything - even for a sensitive, scoped page
    # and even when the semantic reranker (not the fast path) picked it.
    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
    )

    assert (
        service._can_use_deterministic_grounding(
            **_grounding_skip_inputs(
                answer="Test content"
            )
        )
        is True
    )


def test_rewritten_answer_still_needs_llm_grounding(
    tmp_path,
):
    # An LLM-composed answer (not the literal source text) on a sensitive
    # page must still go through the real grounding check.
    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
    )

    assert (
        service._can_use_deterministic_grounding(
            **_grounding_skip_inputs(
                answer="Test content, plus a claim that is not in it."
            )
        )
        is False
    )


def test_client_specific_verbatim_answer_keeps_llm_grounding(
    tmp_path,
):
    service, _ = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
    )

    assert (
        service._can_use_deterministic_grounding(
            **_grounding_skip_inputs(
                answer="Test content",
                client_specific=True,
            )
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
def test_language_names_map_to_page_language_codes(
    raw,
    expected,
):
    assert (
        ConversationService._canonical_language(raw)
        == expected
    )


@pytest.mark.asyncio
async def test_language_name_from_model_reaches_retrieval_as_code(
    tmp_path,
):
    # "English" from the understanding step used to be sent to the
    # language filters as-is and matched no page.
    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
    )

    await service.handle_message(
        message="how do i create a demo account",
        state=ConversationState(language="English"),
        language="EN",
    )

    assert pipeline.calls
    assert pipeline.calls[0]["language"] == "en"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "what is my balance",
        "موجودی من چقدره؟",
        "آخرین واریزم چقدر بود؟",
        "ایمیل ثبت‌شده روی حسابم چیه؟",
    ],
)
async def test_unauthenticated_personal_data_question_short_circuits(
    tmp_path,
    message,
):
    # This project has no live account-data integration for anyone, so
    # a personal-data question from an unauthenticated session must not
    # reach retrieval at all - the nearest KB match would be a generic
    # explanation that reads like an answer without being one.
    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        task=ConversationTask.BALANCE,
    )

    result = await service.handle_message(
        message=message,
        state=ConversationState(),
        language="EN",
    )

    assert pipeline.calls == []
    assert result.retrieval_result is None
    payload = result.generation_result.payload
    assert payload.abstained is True
    assert payload.abstain_reason == "CLIENT_DATA_NOT_AVAILABLE"
    assert payload.needs_handoff is True
    assert not payload.citations


@pytest.mark.asyncio
async def test_model_flagged_client_specific_also_short_circuits(
    tmp_path,
):
    # Even when the message has no matching keyword, the model's own
    # client_specific=True flag must still short-circuit retrieval.
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

    assert pipeline.calls == []
    assert (
        result.generation_result.payload.abstain_reason
        == "CLIENT_DATA_NOT_AVAILABLE"
    )


@pytest.mark.asyncio
async def test_authenticated_personal_data_question_still_reaches_retrieval(
    tmp_path,
):
    # An authenticated session may have a real client-only KB page (e.g.
    # KYC status tracking) that eligibility should still get a chance to
    # serve - only the unauthenticated case is a guaranteed dead end.
    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
    )

    await service.handle_message(
        message="what is my balance",
        state=ConversationState(),
        language="EN",
        trusted_session=TrustedSessionContext(
            is_authenticated=True,
            client_id="CLIENT-123",
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
async def test_howto_question_about_own_account_is_not_blocked(
    tmp_path,
    message,
):
    # Regression: "how do I verify/change MY account?" names the
    # account but asks for a procedure the knowledge base can answer -
    # it must not be caught by the same gate as "what IS my balance?".
    service, pipeline = build_service(
        tmp_path,
        lambda query, context: empty_result(query),
        task=ConversationTask.HOW_TO,
    )

    result = await service.handle_message(
        message=message,
        state=ConversationState(),
        language="FA",
    )

    assert len(pipeline.calls) == 1
    assert result.retrieval_result is not None
    assert (
        result.generation_result.payload.abstain_reason
        != "CLIENT_DATA_NOT_AVAILABLE"
    )

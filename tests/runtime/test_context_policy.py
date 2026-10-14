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
from app.retrieval.schemas import KnowledgeCandidate
from app.runtime.context_policy import ContextPolicy


def make_candidate(
    *, knowledge_key: str, hybrid_score: float = 0.03
) -> HybridCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key=knowledge_key,
        canonical_knowledge_key=(knowledge_key.rsplit("-", 1)[0]),
        knowledge_type="HOW_TO",
        category="TEST",
        status=(KnowledgeStatus.APPROVED),
        audience=Audience.BOTH,
        owner="Test",
        primary_source="Test",
        source_ids=["TEST"],
        source_authority=(SourceAuthority.APPROVED_DOCUMENT),
        language=Language.EN,
        market_scope=MarketScope.GLOBAL,
        countries=[],
        regions=[],
        required_context_fields=[],
        bot_mode=BotMode.KNOWLEDGE,
        fallback_policy=(FallbackPolicy.NONE),
    )

    candidate = KnowledgeCandidate(
        candidate_id=(f"{knowledge_key}:chunk-1"),
        title="Test",
        content="Test",
        metadata=metadata,
        score=1.0,
        source_page_id="1",
    )

    return HybridCandidate(
        candidate=candidate,
        hybrid_score=hybrid_score,
        bm25_rank=1,
        bm25_score=10.0,
        vector_rank=1,
        vector_score=0.7,
    )


def build_result(*, hybrid_candidates, eligible_candidates, traces):
    return RetrievalPipelineResult(
        query="test",
        bm25_candidates=[],
        vector_candidates=[],
        hybrid_candidates=(hybrid_candidates),
        eligible_candidates=(eligible_candidates),
        final_candidates=[],
        eligibility_trace=traces,
    )


def test_no_missing_context_requires_no_clarification():
    candidate = make_candidate(knowledge_key="KB-GLOBAL-EN")

    result = build_result(
        hybrid_candidates=[candidate],
        eligible_candidates=[candidate],
        traces=[
            EligibilityTrace(knowledge_key=("KB-GLOBAL-EN"), eligible=True, reasons=[])
        ],
    )

    decision = ContextPolicy().decide(retrieval_result=result)

    assert decision.required is False


def test_missing_context_requests_clarification():
    candidate = make_candidate(knowledge_key="KB-KYC-AF-EN")

    result = build_result(
        hybrid_candidates=[candidate],
        eligible_candidates=[],
        traces=[
            EligibilityTrace(
                knowledge_key=("KB-KYC-AF-EN"),
                eligible=False,
                reasons=[
                    ("MISSING_CONTEXT:" "kyc_document_country"),
                    ("COUNTRY_CONTEXT_" "MISSING"),
                ],
            )
        ],
    )

    decision = ContextPolicy().decide(retrieval_result=result)

    assert decision.required is True

    assert decision.field_name == "kyc_document_country"

    assert "KB-KYC-AF-EN" in decision.candidate_keys


def test_higher_ranked_blocked_candidate_can_trigger_clarification():
    blocked = make_candidate(knowledge_key="KB-BLOCKED-EN")

    allowed = make_candidate(knowledge_key="KB-ALLOWED-EN")

    result = build_result(
        hybrid_candidates=[blocked, allowed],
        eligible_candidates=[allowed],
        traces=[
            EligibilityTrace(
                knowledge_key=("KB-BLOCKED-EN"),
                eligible=False,
                reasons=["MISSING_CONTEXT:" "country_of_residence"],
            ),
            EligibilityTrace(
                knowledge_key=("KB-ALLOWED-EN"), eligible=True, reasons=[]
            ),
        ],
    )

    decision = ContextPolicy().decide(retrieval_result=result)

    assert decision.required is True

    assert decision.field_name == "country_of_residence"


def test_lower_ranked_blocked_candidate_does_not_interrupt_good_result():
    allowed = make_candidate(knowledge_key="KB-ALLOWED-EN")

    blocked = make_candidate(knowledge_key="KB-BLOCKED-EN")

    result = build_result(
        hybrid_candidates=[allowed, blocked],
        eligible_candidates=[allowed],
        traces=[
            EligibilityTrace(
                knowledge_key=("KB-ALLOWED-EN"), eligible=True, reasons=[]
            ),
            EligibilityTrace(
                knowledge_key=("KB-BLOCKED-EN"),
                eligible=False,
                reasons=["MISSING_CONTEXT:" "country_of_residence"],
            ),
        ],
    )

    decision = ContextPolicy().decide(retrieval_result=result)

    assert decision.required is False


def _general_domain_result(vector_score):
    candidate = make_candidate(knowledge_key="KB-KYC-AF-EN")
    candidate.vector_score = vector_score

    return build_result(
        hybrid_candidates=[candidate],
        eligible_candidates=[],
        traces=[
            EligibilityTrace(
                knowledge_key="KB-KYC-AF-EN",
                eligible=False,
                reasons=[
                    "MISSING_CONTEXT:kyc_document_country",
                    "COUNTRY_CONTEXT_MISSING",
                ],
            )
        ],
    )


def test_off_topic_or_advice_never_requests_clarification():

    from app.conversation.schemas import ConversationDomain, ConversationUnderstanding

    result = _general_domain_result(0.7)

    for flags in ({"is_off_topic": True}, {"is_advice_request": True}):
        decision = ContextPolicy().decide(
            retrieval_result=result,
            understanding=ConversationUnderstanding(
                domain=ConversationDomain.GENERAL, **flags
            ),
        )

        assert decision.required is False


def test_general_domain_weak_neighbour_cannot_block():
    from app.conversation.schemas import ConversationDomain, ConversationUnderstanding

    for domain in (ConversationDomain.GENERAL, ConversationDomain.UNKNOWN):
        decision = ContextPolicy().decide(
            retrieval_result=_general_domain_result(0.30),
            understanding=ConversationUnderstanding(domain=domain),
        )

        assert decision.required is False


def test_general_domain_genuinely_similar_candidate_still_clarifies():

    from app.conversation.schemas import ConversationDomain, ConversationUnderstanding

    decision = ContextPolicy().decide(
        retrieval_result=_general_domain_result(0.62),
        understanding=ConversationUnderstanding(domain=ConversationDomain.GENERAL),
    )

    assert decision.required is True
    assert decision.field_name == "kyc_document_country"

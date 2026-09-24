from app.conversation.schemas import (
    ConversationDomain,
    ConversationTask,
    ConversationUnderstanding,
    TopicRelation,
)
from app.conversation.state import (
    ConversationState,
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
from app.retrieval.schemas import (
    KnowledgeCandidate,
)
from app.runtime.context_requirements import (
    ContextRequirementIndex,
    ContextRequirementService,
)


def make_candidate(
    *,
    knowledge_key: str,
    category: str,
    knowledge_type: str,
    required_context_fields: list[str],
    language: Language = Language.EN,
    bot_mode: BotMode = BotMode.KNOWLEDGE,
    audience: Audience = Audience.BOTH,
) -> KnowledgeCandidate:
    metadata = KnowledgeMetadata(
        knowledge_key=knowledge_key,
        canonical_knowledge_key=(
            knowledge_key.rsplit(
                "-",
                1,
            )[0]
        ),
        knowledge_type=(
            knowledge_type
        ),
        category=category,
        status=(
            KnowledgeStatus.APPROVED
        ),
        audience=audience,
        owner="Test",
        primary_source="Test",
        source_ids=["TEST"],
        source_authority=(
            SourceAuthority.APPROVED_DOCUMENT
        ),
        language=language,
        market_scope=(
            MarketScope.COUNTRY
            if required_context_fields
            else MarketScope.GLOBAL
        ),
        countries=(
            ["XX"]
            if required_context_fields
            else []
        ),
        regions=[],
        required_context_fields=(
            required_context_fields
        ),
        bot_mode=bot_mode,
        fallback_policy=(
            FallbackPolicy.NONE
        ),
    )

    return KnowledgeCandidate(
        candidate_id=(
            f"{knowledge_key}:chunk-1"
        ),
        title=(
            f"Test {knowledge_key}"
        ),
        content="Test content",
        metadata=metadata,
        score=1.0,
        source_page_id="1",
    )


def make_understanding(
    *,
    domain: ConversationDomain,
    task: ConversationTask,
    client_specific: bool = False,
    language: str = "EN",
) -> ConversationUnderstanding:
    return ConversationUnderstanding(
        language=language,
        domain=domain,
        task=task,
        client_specific=(
            client_specific
        ),
        topic_relation=(
            TopicRelation.NEW_TOPIC
        ),
        referenced_countries=[],
        referenced_entities=[],
        requires_previous_context=False,
        confidence=0.95,
    )


def test_index_ignores_candidates_without_required_context():
    candidate = make_candidate(
        knowledge_key="KB-GLOBAL-EN",
        category="ACCOUNT",
        knowledge_type="HOW_TO",
        required_context_fields=[],
    )

    index = (
        ContextRequirementIndex
        .from_candidates(
            [candidate]
        )
    )

    assert len(index) == 0


def test_matching_category_returns_required_context():
    candidate = make_candidate(
        knowledge_key=(
            "KB-DEPOSIT-LOCAL-EN"
        ),
        category="DEPOSIT",
        knowledge_type="HOW_TO",
        required_context_fields=[
            "country_of_residence"
        ],
    )

    index = (
        ContextRequirementIndex
        .from_candidates(
            [candidate]
        )
    )

    service = (
        ContextRequirementService(
            index
        )
    )

    result = service.decide(
        understanding=(
            make_understanding(
                domain=(
                    ConversationDomain
                    .DEPOSIT
                ),
                task=(
                    ConversationTask
                    .HOW_TO
                ),
            )
        ),
        state=ConversationState(),
        language="EN",
    )

    assert result.required is True

    assert (
        result.missing_fields
        == [
            "country_of_residence"
        ]
    )


def test_existing_context_satisfies_requirement():
    candidate = make_candidate(
        knowledge_key=(
            "KB-DEPOSIT-LOCAL-EN"
        ),
        category="DEPOSIT",
        knowledge_type="HOW_TO",
        required_context_fields=[
            "country_of_residence"
        ],
    )

    index = (
        ContextRequirementIndex
        .from_candidates(
            [candidate]
        )
    )

    service = (
        ContextRequirementService(
            index
        )
    )

    state = ConversationState()

    state.set_context_field(
        field_name=(
            "country_of_residence"
        ),
        value="IR",
    )

    result = service.decide(
        understanding=(
            make_understanding(
                domain=(
                    ConversationDomain
                    .DEPOSIT
                ),
                task=(
                    ConversationTask
                    .HOW_TO
                ),
            )
        ),
        state=state,
        language="EN",
    )

    assert result.required is False

    assert (
        result.missing_fields
        == []
    )

    assert (
        result.required_fields
        == [
            "country_of_residence"
        ]
    )


def test_unrelated_domain_does_not_trigger_requirement():
    candidate = make_candidate(
        knowledge_key=(
            "KB-DEPOSIT-LOCAL-EN"
        ),
        category="DEPOSIT",
        knowledge_type="HOW_TO",
        required_context_fields=[
            "country_of_residence"
        ],
    )

    index = (
        ContextRequirementIndex
        .from_candidates(
            [candidate]
        )
    )

    service = (
        ContextRequirementService(
            index
        )
    )

    result = service.decide(
        understanding=(
            make_understanding(
                domain=(
                    ConversationDomain
                    .WITHDRAWAL
                ),
                task=(
                    ConversationTask
                    .HOW_TO
                ),
            )
        ),
        state=ConversationState(),
        language="EN",
    )

    assert result.required is False


def test_non_client_specific_request_skips_lookup_requirements():
    lookup_candidate = make_candidate(
        knowledge_key=(
            "KB-ACCOUNT-STATUS-EN"
        ),
        category="ACCOUNT",
        knowledge_type="STATUS",
        required_context_fields=[
            "client_id"
        ],
        bot_mode=BotMode.LOOKUP,
        audience=Audience.CLIENT,
    )

    index = (
        ContextRequirementIndex
        .from_candidates(
            [lookup_candidate]
        )
    )

    service = (
        ContextRequirementService(
            index
        )
    )

    result = service.decide(
        understanding=(
            make_understanding(
                domain=(
                    ConversationDomain
                    .ACCOUNT
                ),
                task=(
                    ConversationTask
                    .EXPLANATION
                ),
                client_specific=False,
            )
        ),
        state=ConversationState(),
        language="EN",
    )

    assert result.required is False


def test_client_specific_lookup_can_require_client_id():
    lookup_candidate = make_candidate(
        knowledge_key=(
            "KB-ACCOUNT-STATUS-EN"
        ),
        category="ACCOUNT",
        knowledge_type="STATUS",
        required_context_fields=[
            "client_id"
        ],
        bot_mode=BotMode.LOOKUP,
        audience=Audience.CLIENT,
    )

    index = (
        ContextRequirementIndex
        .from_candidates(
            [lookup_candidate]
        )
    )

    service = (
        ContextRequirementService(
            index
        )
    )

    result = service.decide(
        understanding=(
            make_understanding(
                domain=(
                    ConversationDomain
                    .ACCOUNT
                ),
                task=(
                    ConversationTask
                    .STATUS
                ),
                client_specific=True,
            )
        ),
        state=ConversationState(
            is_authenticated=True,
            client_id=None,
        ),
        language="EN",
    )

    assert result.required is True

    assert (
        result.missing_fields
        == ["client_id"]
    )


def test_client_id_from_trusted_state_satisfies_requirement():
    lookup_candidate = make_candidate(
        knowledge_key=(
            "KB-ACCOUNT-STATUS-EN"
        ),
        category="ACCOUNT",
        knowledge_type="STATUS",
        required_context_fields=[
            "client_id"
        ],
        bot_mode=BotMode.LOOKUP,
        audience=Audience.CLIENT,
    )

    index = (
        ContextRequirementIndex
        .from_candidates(
            [lookup_candidate]
        )
    )

    service = (
        ContextRequirementService(
            index
        )
    )

    state = ConversationState(
        is_authenticated=True,
        client_id="CLIENT-123",
    )

    result = service.decide(
        understanding=(
            make_understanding(
                domain=(
                    ConversationDomain
                    .ACCOUNT
                ),
                task=(
                    ConversationTask
                    .STATUS
                ),
                client_specific=True,
            )
        ),
        state=state,
        language="EN",
    )

    assert result.required is False


def test_exact_task_match_has_priority_over_category_only_match():
    how_to = make_candidate(
        knowledge_key=(
            "KB-TRANSFER-HOWTO-EN"
        ),
        category="TRANSFER",
        knowledge_type="HOW_TO",
        required_context_fields=[
            "country_of_residence"
        ],
    )

    status = make_candidate(
        knowledge_key=(
            "KB-TRANSFER-STATUS-EN"
        ),
        category="TRANSFER",
        knowledge_type="STATUS",
        required_context_fields=[
            "client_id"
        ],
        bot_mode=BotMode.LOOKUP,
        audience=Audience.CLIENT,
    )

    index = (
        ContextRequirementIndex
        .from_candidates(
            [
                how_to,
                status,
            ]
        )
    )

    service = (
        ContextRequirementService(
            index
        )
    )

    result = service.decide(
        understanding=(
            make_understanding(
                domain=(
                    ConversationDomain
                    .TRANSFER
                ),
                task=(
                    ConversationTask
                    .HOW_TO
                ),
            )
        ),
        state=ConversationState(),
        language="EN",
    )

    assert result.required is True

    assert (
        result.missing_fields
        == [
            "country_of_residence"
        ]
    )

    assert (
        "client_id"
        not in result.required_fields
    )


def test_language_mismatch_is_ignored():
    candidate = make_candidate(
        knowledge_key=(
            "KB-DEPOSIT-FA"
        ),
        category="DEPOSIT",
        knowledge_type="HOW_TO",
        required_context_fields=[
            "country_of_residence"
        ],
        language=Language.FA,
    )

    index = (
        ContextRequirementIndex
        .from_candidates(
            [candidate]
        )
    )

    service = (
        ContextRequirementService(
            index
        )
    )

    result = service.decide(
        understanding=(
            make_understanding(
                domain=(
                    ConversationDomain
                    .DEPOSIT
                ),
                task=(
                    ConversationTask
                    .HOW_TO
                ),
                language="EN",
            )
        ),
        state=ConversationState(),
        language="EN",
    )

    assert result.required is False
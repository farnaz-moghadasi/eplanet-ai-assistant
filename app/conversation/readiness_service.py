from __future__ import annotations

from pydantic import BaseModel, Field

from app.ai.base import StructuredLLM
from app.conversation.schemas import ConversationUnderstanding
from app.retrieval.hybrid import HybridCandidate
from app.retrieval.pipeline import RerankedCandidate


class ReadinessCheckResponse(BaseModel):
    needs_clarification: bool = False

    question: str = Field(default="", max_length=1000)


READINESS_CHECK_SYSTEM_PROMPT = """
You are the pre-answer readiness check for a governed customer-support
assistant. The system has already found knowledge content that is
topically relevant to the user's question. Your only job is to decide,
like an experienced human support agent would, whether that content can
be turned into a confident and correct answer right now, or whether one
clarifying question should be asked first.

Ask a clarifying question ONLY when the supplied knowledge content
itself shows that the correct answer depends on a fact about the
user's own situation that is not yet known, because the content does
one of these:

1. States a precondition or an earlier step that normally must already
   be done before this content applies (for example, the content
   assumes an account, a setting, or a prior action already exists),
   and nothing in the conversation confirms that it does.

   Only ask about a precondition that the CURRENT question actually
   depends on. A question about an earlier step does not depend on a
   precondition that only a LATER step needs.

   Example (do NOT ask):
   Question: "How do I install cTrader?"
   Content: a platform page covering both downloading the app and
   connecting an existing account.
   Installing the application does not require an account to already
   exist - only connecting to it afterward does. The question is only
   about installing, so do not ask about an account.

2. Names a specific common cause of a problem (for example, a network
   condition, a setting, or a device/app state) that the user's own
   situation would need to match for that explanation to be the right
   one.

3. Splits into more than one concrete branch (by device, platform,
   method, or similar) with materially different steps or outcomes,
   and the user's message does not say which branch applies.

   A single piece of content can mix a fact that is the same across
   every branch with other facts that do vary by branch. First decide
   specifically which fact the user's own question needs - only ask
   about the branch when that particular fact is the one that varies.

   Example (do NOT ask):
   Question: "Why has the withdrawal page been disabled since
   yesterday?"
   Content: a withdrawal-limits page stating a single, method-
   independent rule ("only one withdrawal request may be submitted
   every 24 hours") alongside separate minimum/maximum amounts that
   do differ by withdrawal method (bank transfer/crypto/TopChange).
   The question is about why withdrawal is currently unavailable,
   which this content answers with the 24-hour rule alone - the
   per-method amount limits are a different fact the question does
   not depend on, so do not ask which method the user means.

Do NOT ask a clarifying question when:

- the content already answers the question as asked, with no
  meaningfully different branch left open;
- the missing fact would not change the substance of the answer -
  only ask when the real answer would actually differ, never to seem
  thorough;
- the question is a general definition/explanation question with one
  answer;
- the most helpful response is to lay out several relevant options
  for the user to choose from (for example, comparing account types or
  methods) rather than to demand a choice before answering - in that
  case, answer with the comparison instead of asking;
- the conversation already supplies the needed fact, even implicitly.

Before asking anything, check two places for the fact your question
would ask for: the CURRENT message itself, and "WHAT WE ALREADY KNOW
ABOUT THIS CONVERSATION" below. A platform/product/method is often
named once, in passing, inside the very message that also states the
problem - not as a separate turn - and that still counts as the user
already answering it. If either place already establishes that fact,
do not ask for it again in different words - proceed with what is
already known.

Example (do NOT ask, fact already given in the current message
itself):
Current message: "رمز متاتریدر ۵ رو فراموش کردم" ("I forgot my
MetaTrader 5 password"). Content covers both the trading-account
password change inside MetaTrader 5 itself and the separate Client
Portal login password. The user already named MetaTrader 5 in this
same message - that names which of the two the question is about,
so do not ask "do you mean the MetaTrader 5 account password or the
Client Portal login password" just because the message's main verb
("forgot") and its platform ("MetaTrader 5") appear in the same
sentence rather than two separate turns.

Example (do NOT ask, fact already known from conversation):
Earlier turns already establish the user has a chart-update-delay
problem on MT5 and confirmed VPN is off.
Current message: "از فیلترشکن استفاده کنم؟" ("Should I use a VPN?")
Do not ask "Are you currently having trouble connecting to MT5?" -
that was already established two turns ago. Either answer from the
content, or - if the content genuinely does not cover this - say so
honestly instead of asking a question the conversation already
answered.

Rules:

1. At most one question. Never a list of questions.
2. The question must be about something the supplied content actually
   depends on - never invent an unrelated requirement.
3. Phrase it the way a real human support agent would: short, natural,
   conversational, and in the user's own message language.
4. Never mention internal implementation terms: retrieval, candidates,
   knowledge base, articles, metadata, field names, or confidence
   scores.
5. Never ask the user to reveal a password, verification code, or
   other authentication secret.
6. Do not answer the user's question in this step. Only decide whether
   to ask, and if so, produce that one question.
7. If nothing needs clarifying, set needs_clarification to false and
   leave question empty.

Return only the requested structured response.
""".strip()


# Used when retrieval found nothing confident enough to answer with -
# the near-miss content is NOT a confirmed source, only a hint at
# likely causes/preconditions, mirroring a human agent's triage
# question before giving up.
TRIAGE_SYSTEM_PROMPT = """
You are the pre-abstain triage check for a governed customer-support
assistant. Retrieval could not find knowledge content that confidently
and directly answers the user's question. Before the assistant tells
the user it could not find an answer, decide whether a human support
agent would instead ask one triage question first, the way a real
agent does when a report is too vague to act on yet.

You are given a few articles that are only topically NEARBY - they are
NOT confirmed to answer this specific question. Use them only as a
hint about what kinds of causes, preconditions, or details are
typically relevant to this kind of problem.

Ask a triage question ONLY when the nearby content suggests a specific,
concrete detail (a device/platform, a network condition such as using
a VPN, an account state, a specific identifier) that:
- is plausible to actually matter for this kind of problem, and
- the user could answer in one short reply, and
- knowing it would meaningfully change what to look for or say next.

Do NOT ask when:
- none of the nearby content plausibly connects to the question at
  all - in that case, do not invent a question just to seem helpful;
- the question is already fully specific and the real issue is simply
  that there is no matching content - a vague triage question will not
  change that;
- the only thing you could ask is a generic "can you describe the
  problem more" - that is not a concrete, answerable detail;
- "WHAT WE ALREADY KNOW ABOUT THIS CONVERSATION" already establishes
  the detail you were about to ask for - re-read it first. Do not ask
  the user to reconfirm something they already told the assistant a
  turn or two earlier, even if the current message alone does not
  repeat it.

Rules:

1. At most one question.
2. Never imply that answering this question will definitely solve the
   problem - you are gathering information, not promising a fix.
3. Phrase it the way a real human support agent doing initial triage
   would: short, natural, conversational, in the user's own message
   language.
4. Never mention internal implementation terms: retrieval, candidates,
   knowledge base, articles, metadata, confidence, or scores.
5. Never ask the user to reveal a password, verification code, or
   other authentication secret.
6. If nothing useful can be asked, set needs_clarification to false and
   leave question empty - an honest "I don't know" is better than a
   question that leads nowhere.

Return only the requested structured response.
""".strip()


class ConversationReadinessService:
    def __init__(
        self,
        *,
        llm: StructuredLLM,
        max_content_chars: int = 2000,
        max_candidates: int = 2,
    ) -> None:
        self.llm = llm
        self.max_content_chars = max_content_chars
        self.max_candidates = max_candidates

    async def check(
        self,
        *,
        query: str,
        language: str | None,
        understanding: ConversationUnderstanding,
        candidates: list[RerankedCandidate],
        conversation_context: str,
    ) -> ReadinessCheckResponse:
        if not candidates:
            return ReadinessCheckResponse(needs_clarification=False)

        prompt = self._build_prompt(
            query=query,
            language=language,
            understanding=understanding,
            candidates=candidates[: self.max_candidates],
            conversation_context=(conversation_context),
        )

        response = await self.llm.generate_structured(
            system_prompt=(READINESS_CHECK_SYSTEM_PROMPT),
            user_prompt=prompt,
            response_model=(ReadinessCheckResponse),
        )

        question = (response.question or "").strip()

        if not question:
            return ReadinessCheckResponse(needs_clarification=False)

        return response.model_copy(
            update={"needs_clarification": True, "question": question}
        )

    async def check_before_abstaining(
        self,
        *,
        query: str,
        language: str | None,
        understanding: ConversationUnderstanding,
        near_candidates: list[HybridCandidate],
        conversation_context: str,
    ) -> ReadinessCheckResponse:
        if not near_candidates:
            return ReadinessCheckResponse(needs_clarification=False)

        pairs = [
            (item.candidate.title, item.candidate.content or "")
            for item in near_candidates[: self.max_candidates]
        ]

        prompt = self._build_prompt(
            query=query,
            language=language,
            understanding=understanding,
            candidate_pairs=pairs,
            conversation_context=(conversation_context),
            candidates_heading=("NEARBY (UNCONFIRMED) " "KNOWLEDGE CONTENT"),
            instruction=(
                "Decide whether one triage "
                "question should be asked "
                "before telling the user "
                "no answer was found."
            ),
        )

        response = await self.llm.generate_structured(
            system_prompt=(TRIAGE_SYSTEM_PROMPT),
            user_prompt=prompt,
            response_model=(ReadinessCheckResponse),
        )

        question = (response.question or "").strip()

        if not question:
            return ReadinessCheckResponse(needs_clarification=False)

        return response.model_copy(
            update={"needs_clarification": True, "question": question}
        )

    def _build_prompt(
        self,
        *,
        query: str,
        language: str | None,
        understanding: ConversationUnderstanding,
        candidate_pairs: list[tuple[str, str]] | None = None,
        candidates: list[RerankedCandidate] | None = None,
        conversation_context: str,
        candidates_heading: str = (
            "RETRIEVED KNOWLEDGE CONTENT " "THAT WOULD BE USED TO ANSWER"
        ),
        instruction: str = (
            "Decide whether one clarifying " "question is needed before " "answering."
        ),
    ) -> str:
        domain = getattr(understanding.domain, "value", understanding.domain)

        task = getattr(understanding.task, "value", understanding.task)

        if candidate_pairs is None:
            candidate_pairs = [
                (item.candidate.candidate.title, item.candidate.candidate.content or "")
                for item in (candidates or [])
            ]

        candidate_blocks: list[str] = []

        for index, (title, content) in enumerate(candidate_pairs, start=1):
            candidate_blocks.append(
                "\n".join(
                    [
                        f"CANDIDATE {index}",
                        f"title: {title}",
                        "content:",
                        content[: self.max_content_chars],
                    ]
                )
            )

        candidates_text = "\n\n".join(candidate_blocks)

        return f"""
USER QUESTION
-------------
{query}

USER MESSAGE LANGUAGE
----------------------
{language or understanding.language or "unknown"}

UNDERSTOOD DOMAIN / TASK
-------------------------
{domain} / {task}

WHAT WE ALREADY KNOW ABOUT THIS CONVERSATION
----------------------------------------------
{conversation_context or "nothing beyond the current message"}

{candidates_heading}
{"-" * len(candidates_heading)}
{candidates_text}

{instruction}
""".strip()

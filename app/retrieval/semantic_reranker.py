from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.ai.base import StructuredLLM
from app.retrieval.hybrid import HybridCandidate


class SemanticRerankItem(BaseModel):
    candidate_id: str

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    rationale: str = ""


class SemanticRerankResponse(BaseModel):
    items: list[SemanticRerankItem] = Field(
        default_factory=list
    )


@dataclass(frozen=True)
class SemanticRerankedCandidate:
    candidate: HybridCandidate
    rerank_score: float
    reasons: list[str]


SEMANTIC_RERANK_SYSTEM_PROMPT = """
You are a semantic relevance reranker for a governed brokerage
knowledge retrieval system.

Your only task is to rank already-eligible knowledge candidates by
how well they answer the user's retrieval query.

You are NOT responsible for:
- authentication
- authorization
- deciding whether a user is a client
- market eligibility
- country access rules
- policy enforcement
- deciding whether content is approved
- answering the user

Those decisions have already been made by deterministic backend rules.

Evaluate semantic relevance only.

Important ranking rules:

1. Prefer a candidate that directly answers the requested task over
a candidate that only discusses the same broad topic.

Example:

Query:
"How do I download and install cTrader?"

Better:
"Download and Login to cTrader"

Worse:
"Add Indicators in cTrader"

2. Respect the requested action.

"download", "install", "reset", "change", "check status", "requirements",
"fees", and "troubleshooting" are different intents even when they refer
to the same product.

3. Use the candidate title and content together.

4. Market-specific relevance may be used only when the backend-provided
context explicitly contains that market.

Example:

Query:
"What KYC documents are required in Afghanistan?"

Context:
kyc_document_country = AF

A candidate specifically about Afghanistan KYC documents should normally
rank above a generic KYC article when both are otherwise relevant.

5. Never infer nationality, residence, or market from language.

6. Do not penalize valid technical product names written in English
inside Persian, Arabic, or Kurdish text.

7. Score each supplied candidate independently from 0.0 to 1.0.

Suggested interpretation:

0.90 - 1.00:
direct answer to the exact request

0.75 - 0.89:
strongly relevant

0.50 - 0.74:
related but incomplete

0.25 - 0.49:
weakly related

0.00 - 0.24:
not useful for this request

8. Return each candidate exactly once using its supplied candidate_id.

Do not create candidate IDs.
Do not omit candidates because of policy concerns.
Do not answer the user.
"""


class SemanticKnowledgeReranker:
    def __init__(
        self,
        *,
        llm: StructuredLLM,
        max_content_chars: int = 2500,
    ):
        self.llm = llm
        self.max_content_chars = max_content_chars

    async def rerank(
        self,
        *,
        query: str,
        candidates: list[HybridCandidate],
        language: str | None = None,
        context: dict[str, str] | None = None,
        top_k: int = 5,
    ) -> list[SemanticRerankedCandidate]:
        if not candidates:
            return []

        prompt = self._build_prompt(
            query=query,
            candidates=candidates,
            language=language,
            context=context or {},
        )

        response = await self.llm.generate_structured(
            system_prompt=(
                SEMANTIC_RERANK_SYSTEM_PROMPT
            ),
            user_prompt=prompt,
            response_model=(
                SemanticRerankResponse
            ),
        )

        score_map = {
            item.candidate_id: item
            for item in response.items
        }

        reranked: list[
            SemanticRerankedCandidate
        ] = []

        for hybrid_candidate in candidates:
            candidate = (
                hybrid_candidate.candidate
            )

            item = score_map.get(
                candidate.candidate_id
            )

            if item is None:
                score = 0.0

                reasons = [
                    "SEMANTIC_RERANK_RESULT_MISSING"
                ]

            else:
                score = (
                    item.relevance_score
                )

                reasons = [
                    (
                        "SEMANTIC_RELEVANCE:"
                        f"{score:.4f}"
                    )
                ]

                rationale = (
                    item.rationale.strip()
                )

                if rationale:
                    reasons.append(
                        (
                            "SEMANTIC_REASON:"
                            f"{rationale}"
                        )
                    )

            reranked.append(
                SemanticRerankedCandidate(
                    candidate=(
                        hybrid_candidate
                    ),
                    rerank_score=score,
                    reasons=reasons,
                )
            )

        reranked.sort(
            key=lambda item: (
                item.rerank_score,
                item.candidate.hybrid_score,
            ),
            reverse=True,
        )

        return reranked[:top_k]

    def _build_prompt(
        self,
        *,
        query: str,
        candidates: list[HybridCandidate],
        language: str | None,
        context: dict[str, str],
    ) -> str:
        context_lines = []

        for key, value in context.items():
            if not value:
                continue

            context_lines.append(
                f"{key}: {value}"
            )

        context_text = (
            "\n".join(context_lines)
            if context_lines
            else "none"
        )

        candidate_blocks: list[str] = []

        for index, hybrid in enumerate(
            candidates,
            start=1,
        ):
            candidate = hybrid.candidate

            metadata = candidate.metadata

            content = (
                candidate.content
                or ""
            )

            content = content[
                : self.max_content_chars
            ]

            candidate_blocks.append(
                "\n".join(
                    [
                        (
                            f"CANDIDATE {index}"
                        ),
                        (
                            "candidate_id: "
                            f"{candidate.candidate_id}"
                        ),
                        (
                            "knowledge_key: "
                            f"{metadata.knowledge_key}"
                        ),
                        (
                            "canonical_knowledge_key: "
                            f"{metadata.canonical_knowledge_key}"
                        ),
                        (
                            "title: "
                            f"{candidate.title}"
                        ),
                        (
                            "category: "
                            f"{metadata.category}"
                        ),
                        (
                            "knowledge_type: "
                            f"{metadata.knowledge_type}"
                        ),
                        (
                            "market_scope: "
                            f"{metadata.market_scope.value}"
                        ),
                        (
                            "countries: "
                            f"{metadata.countries}"
                        ),
                        (
                            "hybrid_score: "
                            f"{hybrid.hybrid_score}"
                        ),
                        (
                            "bm25_rank: "
                            f"{hybrid.bm25_rank}"
                        ),
                        (
                            "vector_rank: "
                            f"{hybrid.vector_rank}"
                        ),
                        "content:",
                        content,
                    ]
                )
            )

        return (
            f"""
USER QUERY
----------
{query}

USER MESSAGE LANGUAGE
---------------------
{language or "unknown"}

TRUSTED / RESOLVED RETRIEVAL CONTEXT
------------------------------------
{context_text}

ELIGIBLE CANDIDATES
-------------------
{chr(10).join(candidate_blocks)}

Return semantic relevance scores for every candidate.
""".strip()
        )
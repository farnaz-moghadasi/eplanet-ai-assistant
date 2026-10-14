import asyncio
import math
import re
from collections import Counter

from app.retrieval.base import KnowledgeRetriever
from app.retrieval.document import KnowledgeChunk
from app.retrieval.schemas import KnowledgeCandidate


class BM25Retriever(KnowledgeRetriever):
    def __init__(
        self,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b

        self._chunks: list[KnowledgeChunk] = []

        self._tokenized_documents: list[list[str]] = []

        self._term_frequencies: list[Counter[str]] = []

        self._document_frequencies: Counter[str] = Counter()

        self._document_lengths: list[int] = []

        self._average_document_length: float = 0.0

        if chunks:
            self.index(chunks)

    def index(self, chunks: list[KnowledgeChunk]) -> None:
        self._chunks = list(chunks)

        self._tokenized_documents = []
        self._term_frequencies = []
        self._document_frequencies = Counter()
        self._document_lengths = []

        for chunk in self._chunks:
            searchable_text = self._build_searchable_text(chunk)

            tokens = self._tokenize(searchable_text)

            self._tokenized_documents.append(tokens)

            frequencies = Counter(tokens)

            self._term_frequencies.append(frequencies)

            document_length = len(tokens)

            self._document_lengths.append(document_length)

            for term in frequencies:
                self._document_frequencies[term] += 1

        if self._document_lengths:
            self._average_document_length = sum(self._document_lengths) / len(
                self._document_lengths
            )
        else:
            self._average_document_length = 0.0

    async def retrieve(
        self, *, query: str, knowledge_hints: list[str], limit: int = 10
    ) -> list[KnowledgeCandidate]:
        if limit <= 0:
            return []

        if not self._chunks:
            return []

        query_text = self._build_query_text(
            query=query, knowledge_hints=knowledge_hints
        )

        query_tokens = self._tokenize(query_text)

        if not query_tokens:
            return []


        scored_chunks = await asyncio.to_thread(self._score_all_chunks, query_tokens)

        return [
            self._to_candidate(chunk=chunk, score=score)
            for score, chunk in scored_chunks[:limit]
        ]

    def _score_all_chunks(
        self, query_tokens: list[str]
    ) -> list[tuple[float, KnowledgeChunk]]:
        scored_chunks: list[tuple[float, KnowledgeChunk]] = []

        for index, chunk in enumerate(self._chunks):
            score = self._score_document(
                query_tokens=query_tokens, document_index=index
            )

            if score <= 0:
                continue

            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda item: (-item[0], item[1].chunk_id))

        return scored_chunks

    def _score_document(self, *, query_tokens: list[str], document_index: int) -> float:
        if self._average_document_length <= 0:
            return 0.0

        frequencies = self._term_frequencies[document_index]

        document_length = self._document_lengths[document_index]

        score = 0.0

        for term in set(query_tokens):
            term_frequency = frequencies.get(term, 0)

            if term_frequency == 0:
                continue

            inverse_document_frequency = self._idf(term)

            numerator = term_frequency * (self.k1 + 1)

            length_normalization = (
                1 - self.b + self.b * (document_length / self._average_document_length)
            )

            denominator = term_frequency + self.k1 * length_normalization

            score += inverse_document_frequency * numerator / denominator

        return score

    def _idf(self, term: str) -> float:
        total_documents = len(self._chunks)

        matching_documents = self._document_frequencies.get(term, 0)

        return math.log(
            1
            + (total_documents - matching_documents + 0.5) / (matching_documents + 0.5)
        )

    @staticmethod
    def _build_searchable_text(chunk: KnowledgeChunk) -> str:
        metadata = chunk.metadata

        parts = [
            chunk.title,
            chunk.content,
            metadata.knowledge_key,
            metadata.canonical_knowledge_key,
            metadata.knowledge_type,
            metadata.category,
            metadata.language.value,
        ]

        return " ".join(str(part) for part in parts if part)

    @staticmethod
    def _build_query_text(*, query: str, knowledge_hints: list[str]) -> str:
        parts = [query, *knowledge_hints]

        return " ".join(part.strip() for part in parts if part and part.strip())

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        normalized = text.lower()

        normalized = (
            normalized.replace("ي", "ی").replace("ك", "ک").replace("\u200c", " ")
        )

        return re.findall(r"[a-z0-9_]+|[\u0600-\u06ff]+", normalized)

    @staticmethod
    def _to_candidate(*, chunk: KnowledgeChunk, score: float) -> KnowledgeCandidate:
        return KnowledgeCandidate(
            candidate_id=chunk.chunk_id,
            title=chunk.title,
            content=chunk.content,
            metadata=chunk.metadata,
            score=score,
            source_page_id=chunk.page_id,
        )

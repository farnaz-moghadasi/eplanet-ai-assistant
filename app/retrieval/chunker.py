import hashlib
import re

from app.knowledge.schemas import ParsedKnowledgePage
from app.retrieval.document import (
    KnowledgeChunk,
    KnowledgeDocument,
)


class KnowledgeChunker:
    def __init__(
        self,
        *,
        max_chars: int = 1200,
        overlap_chars: int = 150,
    ):
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")

        if overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative")

        if overlap_chars >= max_chars:
            raise ValueError(
                "overlap_chars must be smaller than max_chars"
            )

        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def page_to_document(
        self,
        page: ParsedKnowledgePage,
    ) -> KnowledgeDocument:
        return KnowledgeDocument(
            document_id=self._document_id(page),
            page_id=page.page_id,
            title=page.title.strip(),
            content=self._normalize_text(page.content),
            metadata=page.metadata,
            version=page.version,
        )

    def chunk_page(
        self,
        page: ParsedKnowledgePage,
    ) -> list[KnowledgeChunk]:
        document = self.page_to_document(page)

        return self.chunk_document(document)

    def chunk_document(
        self,
        document: KnowledgeDocument,
    ) -> list[KnowledgeChunk]:
        content = self._normalize_text(
            document.content
        )

        if not content:
            return []

        text_chunks = self._split_text(content)

        chunks: list[KnowledgeChunk] = []

        for index, text in enumerate(text_chunks):
            chunk_id = self._chunk_id(
                document=document,
                chunk_index=index,
                content=text,
            )

            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    page_id=document.page_id,
                    title=document.title,
                    content=text,
                    chunk_index=index,
                    metadata=document.metadata,
                    version=document.version,
                )
            )

        return chunks

    def _split_text(
        self,
        text: str,
    ) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]

        paragraphs = self._paragraphs(text)

        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.max_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""

                large_parts = self._split_large_paragraph(
                    paragraph
                )

                for part in large_parts:
                    chunks.append(part)

                continue

            candidate = (
                paragraph
                if not current
                else f"{current}\n\n{paragraph}"
            )

            if len(candidate) <= self.max_chars:
                current = candidate
                continue

            chunks.append(current.strip())

            overlap = self._build_overlap(current)

            current = (
                f"{overlap}\n\n{paragraph}".strip()
                if overlap
                else paragraph
            )

        if current:
            chunks.append(current.strip())

        return self._deduplicate_chunks(chunks)

    def _split_large_paragraph(
        self,
        paragraph: str,
    ) -> list[str]:
        sentences = self._sentences(paragraph)

        if len(sentences) <= 1:
            return self._sliding_window(paragraph)

        result: list[str] = []
        current = ""

        for sentence in sentences:
            candidate = (
                sentence
                if not current
                else f"{current} {sentence}"
            )

            if len(candidate) <= self.max_chars:
                current = candidate
                continue

            if current:
                result.append(current.strip())

            overlap = self._build_overlap(current)

            current = (
                f"{overlap} {sentence}".strip()
                if overlap
                else sentence
            )

            if len(current) > self.max_chars:
                result.extend(
                    self._sliding_window(current)
                )
                current = ""

        if current:
            result.append(current.strip())

        return result

    def _sliding_window(
        self,
        text: str,
    ) -> list[str]:
        chunks: list[str] = []

        start = 0
        text_length = len(text)

        step = self.max_chars - self.overlap_chars

        while start < text_length:
            end = min(
                start + self.max_chars,
                text_length,
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start += step

        return chunks

    def _build_overlap(
        self,
        text: str,
    ) -> str:
        if (
            not text
            or self.overlap_chars == 0
        ):
            return ""

        overlap = text[-self.overlap_chars:]

        first_space = overlap.find(" ")

        if first_space != -1:
            overlap = overlap[first_space + 1:]

        return overlap.strip()

    @staticmethod
    def _paragraphs(
        text: str,
    ) -> list[str]:
        return [
            paragraph.strip()
            for paragraph in re.split(
                r"\n\s*\n",
                text,
            )
            if paragraph.strip()
        ]

    @staticmethod
    def _sentences(
        text: str,
    ) -> list[str]:
        parts = re.split(
            r"(?<=[.!?؟])\s+",
            text.strip(),
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        lines = [
            line.strip()
            for line in text.split("\n")
        ]

        normalized_lines: list[str] = []

        empty_seen = False

        for line in lines:
            if not line:
                if (
                    normalized_lines
                    and not empty_seen
                ):
                    normalized_lines.append("")

                empty_seen = True
                continue

            normalized_lines.append(
                re.sub(r"\s+", " ", line)
            )

            empty_seen = False

        return "\n".join(
            normalized_lines
        ).strip()

    @staticmethod
    def _deduplicate_chunks(
        chunks: list[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for chunk in chunks:
            normalized = chunk.strip()

            if (
                not normalized
                or normalized in seen
            ):
                continue

            seen.add(normalized)
            result.append(normalized)

        return result

    @staticmethod
    def _document_id(
        page: ParsedKnowledgePage,
    ) -> str:
        canonical_key = (
            page.metadata.canonical_knowledge_key
            or page.metadata.knowledge_key
        )

        raw = (
            f"{canonical_key}|"
            f"{page.page_id}"
        )

        digest = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:16]

        return (
            f"{canonical_key}:{digest}"
        )

    @staticmethod
    def _chunk_id(
        *,
        document: KnowledgeDocument,
        chunk_index: int,
        content: str,
    ) -> str:
        raw = (
            f"{document.document_id}|"
            f"{document.version}|"
            f"{chunk_index}|"
            f"{content}"
        )

        digest = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:20]

        return (
            f"{document.document_id}:"
            f"{chunk_index}:{digest}"
        )
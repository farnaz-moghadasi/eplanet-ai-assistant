# app/knowledge/parser.py

from __future__ import annotations

from datetime import date
from typing import Any

from bs4 import (
    BeautifulSoup,
    NavigableString,
    Tag,
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
    ParsedKnowledgePage,
)


class KnowledgeParser:
    # Headings that mark internal editorial/migration scaffolding, never
    # customer-facing content. A page carrying one of these headings had it
    # left in the body after an automated migration pass; everything from
    # this heading to the end of the page is stripped before parsing.
    EDITORIAL_SECTION_HEADINGS = {
        "migration review notes",
    }

    # Some pages carry a single reviewer instruction under a heading
    # ("Important Notes") that is otherwise legitimately used for real
    # customer-facing warnings on other pages. Only this exact sentence is
    # editorial scaffolding, so it is matched by content, not by heading
    # name, and only the matching heading+paragraph pair is removed.
    EDITORIAL_HEADING_PARAGRAPH_PAIRS = {
        (
            "important notes",
            "confirm that links, labels, and "
            "steps are still current before "
            "approving this page.",
        ),
    }

    FIELD_MAP = {
        "knowledge key": "knowledge_key",
        "canonical knowledge key": (
            "canonical_knowledge_key"
        ),
        "knowledge type": "knowledge_type",
        "category": "category",
        "status": "status",
        "audience": "audience",
        "owner": "owner",
        "primary source": "primary_source",
        "source ids": "source_ids",
        "source authority": "source_authority",
        "language": "language",
        "languages": "language",
        "market scope": "market_scope",
        "countries": "countries",
        "regions": "regions",
        "required context fields": (
            "required_context_fields"
        ),
        "bot mode": "bot_mode",
        "fallback policy": "fallback_policy",
        "effective from": "effective_from",
        "valid until": "valid_until",
        "last reviewed": "last_reviewed",
        "review frequency": (
            "review_frequency_days"
        ),
    }

    def parse(
        self,
        page: dict[str, Any],
    ) -> ParsedKnowledgePage:
        html = (
            page[
                "body"
            ][
                "storage"
            ][
                "value"
            ]
        )

        soup = BeautifulSoup(
            html,
            "html.parser",
        )

        table = soup.find(
            "table"
        )

        if table is None:
            raise ValueError(
                "METADATA_TABLE_NOT_FOUND"
            )

        raw_metadata = (
            self._parse_metadata_table(
                table
            )
        )

        # Metadata must never become
        # customer-facing knowledge content.
        table.decompose()

        self._strip_editorial_sections(
            soup
        )

        content = (
            self._clean_content(
                soup
            )
        )

        metadata = (
            self._build_metadata(
                raw_metadata
            )
        )

        return ParsedKnowledgePage(
            page_id=str(
                page["id"]
            ),
            title=page["title"],
            version=(
                page
                .get(
                    "version",
                    {},
                )
                .get(
                    "number"
                )
            ),
            metadata=metadata,
            content=content,
        )

    def _parse_metadata_table(
        self,
        table: Tag,
    ) -> dict[str, str]:
        result: dict[
            str,
            str,
        ] = {}

        for row in table.find_all(
            "tr"
        ):
            cells = row.find_all(
                [
                    "td",
                    "th",
                ]
            )

            if len(cells) < 2:
                continue

            field = (
                cells[0]
                .get_text(
                    " ",
                    strip=True,
                )
            )

            value = (
                cells[1]
                .get_text(
                    " ",
                    strip=True,
                )
            )

            normalized_field = (
                field
                .strip()
                .lower()
            )

            mapped_field = (
                self.FIELD_MAP
                .get(
                    normalized_field
                )
            )

            if mapped_field:
                result[
                    mapped_field
                ] = value

        return result

    def _build_metadata(
        self,
        raw: dict[
            str,
            str,
        ],
    ) -> KnowledgeMetadata:
        return KnowledgeMetadata(
            knowledge_key=(
                raw
                .get(
                    "knowledge_key",
                    "",
                )
                .strip()
            ),
            canonical_knowledge_key=(
                (
                    raw.get(
                        "canonical_knowledge_key"
                    )
                    or raw.get(
                        "knowledge_key",
                        "",
                    )
                )
                .strip()
            ),
            knowledge_type=(
                raw
                .get(
                    "knowledge_type",
                    "",
                )
                .strip()
                .upper()
            ),
            category=(
                raw
                .get(
                    "category",
                    "",
                )
                .strip()
                .upper()
            ),
            status=(
                self._parse_status(
                    raw.get(
                        "status"
                    )
                )
            ),
            audience=(
                self._parse_audience(
                    raw.get(
                        "audience"
                    )
                )
            ),
            owner=(
                self._empty_to_none(
                    raw.get(
                        "owner"
                    )
                )
            ),
            primary_source=(
                self._empty_to_none(
                    raw.get(
                        "primary_source"
                    )
                )
            ),
            source_ids=(
                self._parse_list(
                    raw.get(
                        "source_ids"
                    )
                )
            ),
            source_authority=(
                self
                ._parse_source_authority(
                    raw.get(
                        "source_authority"
                    )
                )
            ),
            language=(
                self._parse_language(
                    raw.get(
                        "language"
                    )
                )
            ),
            market_scope=(
                self
                ._parse_market_scope(
                    raw.get(
                        "market_scope"
                    )
                )
            ),
            countries=[
                item.upper()
                for item
                in self._parse_list(
                    raw.get(
                        "countries"
                    )
                )
            ],
            regions=[
                item.upper()
                for item
                in self._parse_list(
                    raw.get(
                        "regions"
                    )
                )
            ],
            required_context_fields=[
                item.lower()
                for item
                in self._parse_list(
                    raw.get(
                        "required_context_fields"
                    )
                )
                if (
                    item.upper()
                    != "NONE"
                )
            ],
            bot_mode=(
                self._parse_bot_mode(
                    raw.get(
                        "bot_mode"
                    )
                )
            ),
            fallback_policy=(
                self
                ._parse_fallback_policy(
                    raw.get(
                        "fallback_policy"
                    )
                )
            ),
            effective_from=(
                self._parse_date(
                    raw.get(
                        "effective_from"
                    )
                )
            ),
            valid_until=(
                self._parse_date(
                    raw.get(
                        "valid_until"
                    )
                )
            ),
            last_reviewed=(
                self._parse_date(
                    raw.get(
                        "last_reviewed"
                    )
                )
            ),
            review_frequency_days=(
                self
                ._parse_review_frequency(
                    raw.get(
                        "review_frequency_days"
                    )
                )
            ),
        )

    def _strip_editorial_sections(
        self,
        soup: BeautifulSoup,
    ) -> None:
        headings = soup.find_all(
            [
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
            ]
        )

        for heading in headings:
            heading_text = (
                heading
                .get_text(strip=True)
                .strip()
                .lower()
            )

            if (
                heading_text
                in self.EDITORIAL_SECTION_HEADINGS
            ):
                trailing = list(
                    heading.next_siblings
                )

                heading.extract()

                for node in trailing:
                    node.extract()

                break

            next_tag = heading.find_next_sibling()

            if (
                next_tag is not None
                and next_tag.name == "p"
            ):
                pair = (
                    heading_text,
                    next_tag
                    .get_text(strip=True)
                    .strip()
                    .lower(),
                )

                if (
                    pair
                    in self
                    .EDITORIAL_HEADING_PARAGRAPH_PAIRS
                ):
                    heading.extract()
                    next_tag.extract()

    def _clean_content(
        self,
        soup: BeautifulSoup,
    ) -> str:
        self._remove_non_content_tags(
            soup
        )

        lines: list[str] = []

        for child in soup.contents:
            self._render_block(
                child,
                lines,
            )

        return self._normalize_lines(
            lines
        )

    def _render_block(
        self,
        node,
        lines: list[str],
    ) -> None:
        if isinstance(
            node,
            NavigableString,
        ):
            text = (
                str(node)
                .strip()
            )

            if text:
                lines.append(
                    text
                )

            return

        if not isinstance(
            node,
            Tag,
        ):
            return

        name = (
            node.name
            or ""
        ).lower()

        if name in {
            "script",
            "style",
            "noscript",
        }:
            return

        if name in {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:
            text = (
                self
                ._render_inline(
                    node
                )
                .strip()
            )

            if not text:
                return

            level = int(
                name[1]
            )

            # Keep headings readable without
            # making Confluence H1 dominate
            # the customer response.
            level = max(
                2,
                min(
                    level,
                    4,
                ),
            )

            lines.append(
                (
                    "#"
                    * level
                )
                + " "
                + text
            )

            lines.append(
                ""
            )

            return

        if name == "ol":
            self._render_list(
                node=node,
                lines=lines,
                ordered=True,
            )

            lines.append(
                ""
            )

            return

        if name == "ul":
            self._render_list(
                node=node,
                lines=lines,
                ordered=False,
            )

            lines.append(
                ""
            )

            return

        if name == "table":
            self._render_table(
                node=node,
                lines=lines,
            )

            lines.append(
                ""
            )

            return

        if name in {
            "p",
            "blockquote",
        }:
            text = (
                self
                ._render_inline(
                    node
                )
                .strip()
            )

            if text:
                if (
                    name
                    == "blockquote"
                ):
                    lines.append(
                        f"> {text}"
                    )
                else:
                    lines.append(
                        text
                    )

                lines.append(
                    ""
                )

            return

        if name == "hr":
            lines.append(
                "---"
            )

            lines.append(
                ""
            )

            return

        if name == "br":
            lines.append(
                ""
            )

            return

        if name in {
            "div",
            "section",
            "article",
            "main",
            "body",
        }:
            for child in (
                node.contents
            ):
                self._render_block(
                    child,
                    lines,
                )

            return

        # Confluence storage-format elements
        # and other unknown wrappers are
        # recursively traversed so useful
        # customer-facing content is not lost.
        block_children = [
            child
            for child
            in node.contents
            if isinstance(
                child,
                Tag,
            )
            and (
                child.name
                or ""
            ).lower()
            in {
                "p",
                "div",
                "section",
                "article",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "ol",
                "ul",
                "table",
                "blockquote",
            }
        ]

        if block_children:
            for child in (
                node.contents
            ):
                self._render_block(
                    child,
                    lines,
                )

            return

        text = (
            self
            ._render_inline(
                node
            )
            .strip()
        )

        if text:
            lines.append(
                text
            )

    def _render_list(
        self,
        *,
        node: Tag,
        lines: list[str],
        ordered: bool,
    ) -> None:
        items = node.find_all(
            "li",
            recursive=False,
        )

        for index, item in enumerate(
            items,
            start=1,
        ):
            text = (
                self
                ._render_list_item_text(
                    item
                )
                .strip()
            )

            if not text:
                continue

            prefix = (
                f"{index}."
                if ordered
                else "-"
            )

            lines.append(
                f"{prefix} {text}"
            )

            nested_lists = (
                item.find_all(
                    [
                        "ol",
                        "ul",
                    ],
                    recursive=False,
                )
            )

            for nested in (
                nested_lists
            ):
                nested_lines: list[
                    str
                ] = []

                self._render_list(
                    node=nested,
                    lines=nested_lines,
                    ordered=(
                        nested.name
                        == "ol"
                    ),
                )

                for nested_line in (
                    nested_lines
                ):
                    if (
                        nested_line
                        .strip()
                    ):
                        lines.append(
                            (
                                "   "
                                + nested_line
                            )
                        )

    def _render_list_item_text(
        self,
        item: Tag,
    ) -> str:
        parts: list[str] = []

        for child in item.contents:
            if (
                isinstance(
                    child,
                    Tag,
                )
                and (
                    child.name
                    or ""
                ).lower()
                in {
                    "ol",
                    "ul",
                }
            ):
                continue

            value = (
                self
                ._render_inline(
                    child
                )
                .strip()
            )

            if value:
                parts.append(
                    value
                )

        return self._normalize_inline_text(
            " ".join(
                parts
            )
        )

    def _render_inline(
        self,
        node,
    ) -> str:
        if isinstance(
            node,
            NavigableString,
        ):
            return str(
                node
            )

        if not isinstance(
            node,
            Tag,
        ):
            return ""

        name = (
            node.name
            or ""
        ).lower()

        if name == "br":
            return "\n"

        if name == "a":
            label = (
                node
                .get_text(
                    " ",
                    strip=True,
                )
            )

            href = (
                node
                .get(
                    "href"
                )
            )

            if not label:
                label = (
                    str(
                        href
                        or ""
                    )
                    .strip()
                )

            if (
                href
                and label
            ):
                return (
                    f"[{label}]"
                    f"({href})"
                )

            return label

        if name in {
            "strong",
            "b",
        }:
            text = self._render_children_inline(
                node
            )

            if not text:
                return ""

            return (
                f"**{text}**"
            )

        if name in {
            "em",
            "i",
        }:
            text = self._render_children_inline(
                node
            )

            if not text:
                return ""

            return (
                f"*{text}*"
            )

        if name == "code":
            text = (
                node
                .get_text(
                    " ",
                    strip=True,
                )
            )

            if not text:
                return ""

            return (
                f"`{text}`"
            )

        # Preserve Confluence links whose
        # display body is nested in custom
        # storage-format tags.
        if name == "ac:link":
            return self._render_confluence_link(
                node
            )

        return self._render_children_inline(
            node
        )

    def _render_children_inline(
        self,
        node: Tag,
    ) -> str:
        parts = [
            self._render_inline(
                child
            )
            for child
            in node.contents
        ]

        return self._normalize_inline_text(
            "".join(
                parts
            )
        )

    def _render_confluence_link(
        self,
        node: Tag,
    ) -> str:
        body = (
            node.find(
                "ac:plain-text-link-body"
            )
            or node.find(
                "ac:link-body"
            )
        )

        label = ""

        if body is not None:
            label = (
                body
                .get_text(
                    " ",
                    strip=True,
                )
            )

        if not label:
            page_ref = node.find(
                "ri:page"
            )

            if page_ref is not None:
                label = (
                    page_ref.get(
                        "ri:content-title"
                    )
                    or page_ref.get(
                        "content-title"
                    )
                    or ""
                )

        if not label:
            label = (
                node
                .get_text(
                    " ",
                    strip=True,
                )
            )

        return label.strip()

    def _render_table(
        self,
        *,
        node: Tag,
        lines: list[str],
    ) -> None:
        rows: list[
            list[str]
        ] = []

        for row in node.find_all(
            "tr"
        ):
            cells = row.find_all(
                [
                    "th",
                    "td",
                ],
                recursive=False,
            )

            values = [
                self
                ._render_inline(
                    cell
                )
                .strip()
                .replace(
                    "\n",
                    " ",
                )
                .replace(
                    "|",
                    "\\|",
                )
                for cell
                in cells
            ]

            if values:
                rows.append(
                    values
                )

        if not rows:
            return

        column_count = max(
            len(row)
            for row
            in rows
        )

        normalized_rows: list[
            list[str]
        ] = []

        for row in rows:
            normalized_rows.append(
                row
                + [
                    ""
                ]
                * (
                    column_count
                    - len(row)
                )
            )

        header = (
            normalized_rows[0]
        )

        lines.append(
            "| "
            + " | ".join(
                header
            )
            + " |"
        )

        lines.append(
            "| "
            + " | ".join(
                [
                    "---"
                    for _ in range(
                        column_count
                    )
                ]
            )
            + " |"
        )

        for row in (
            normalized_rows[1:]
        ):
            lines.append(
                "| "
                + " | ".join(
                    row
                )
                + " |"
            )

    @staticmethod
    def _remove_non_content_tags(
        soup: BeautifulSoup,
    ) -> None:
        for tag in soup.find_all(
            [
                "script",
                "style",
                "noscript",
            ]
        ):
            tag.decompose()

    @staticmethod
    def _normalize_inline_text(
        value: str,
    ) -> str:
        value = (
            value
            .replace(
                "\u00a0",
                " ",
            )
            .replace(
                "\u200c",
                "\u200c",
            )
        )

        lines = [
            " ".join(
                line.split()
            )
            for line
            in value.splitlines()
        ]

        return "\n".join(
            line
            for line
            in lines
            if line
        ).strip()

    @staticmethod
    def _normalize_lines(
        lines: list[str],
    ) -> str:
        result: list[str] = []

        previous_blank = True

        for raw_line in lines:
            line = (
                str(
                    raw_line
                )
                .strip()
            )

            if not line:
                if (
                    result
                    and not previous_blank
                ):
                    result.append(
                        ""
                    )

                previous_blank = True
                continue

            result.append(
                line
            )

            previous_blank = False

        while (
            result
            and not result[-1]
        ):
            result.pop()

        return "\n".join(
            result
        ).strip()

    @staticmethod
    def _parse_list(
        value: str | None,
    ) -> list[str]:
        if not value:
            return []

        normalized = value.replace(
            ";",
            ",",
        )

        return [
            item.strip()
            for item
            in normalized.split(
                ","
            )
            if item.strip()
        ]

    @staticmethod
    def _empty_to_none(
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        return value or None

    @staticmethod
    def _parse_status(
        value: str | None,
    ) -> KnowledgeStatus:
        normalized = (
            (
                value
                or ""
            )
            .strip()
            .upper()
            .replace(
                " ",
                "_",
            )
        )

        return KnowledgeStatus(
            normalized
        )

    @staticmethod
    def _parse_audience(
        value: str | None,
    ) -> Audience:
        normalized = (
            (
                value
                or ""
            )
            .strip()
            .upper()
        )

        return Audience(
            normalized
        )

    @staticmethod
    def _parse_language(
        value: str | None,
    ) -> Language:
        normalized = (
            (
                value
                or ""
            )
            .strip()
            .upper()
        )

        return Language(
            normalized
        )

    @staticmethod
    def _parse_market_scope(
        value: str | None,
    ) -> MarketScope:
        normalized = (
            (
                value
                or "UNSPECIFIED"
            )
            .strip()
            .upper()
        )

        return MarketScope(
            normalized
        )

    @staticmethod
    def _parse_bot_mode(
        value: str | None,
    ) -> BotMode:
        normalized = (
            (
                value
                or ""
            )
            .strip()
            .upper()
        )

        return BotMode(
            normalized
        )

    @staticmethod
    def _parse_source_authority(
        value: str | None,
    ) -> SourceAuthority | None:
        if not value:
            return None

        normalized = (
            value
            .strip()
            .upper()
            .replace(
                "-",
                "_",
            )
            .replace(
                " ",
                "_",
            )
        )

        # Optional[SourceAuthority] in the schema - an unrecognized
        # draft/placeholder value (e.g. "UNVERIFIED_CURATED_CONTENT",
        # not a real enum member) should degrade to unset rather than
        # crash the whole page's ingestion.
        try:
            return SourceAuthority(
                normalized
            )

        except ValueError:
            return None

    @staticmethod
    def _parse_fallback_policy(
        value: str | None,
    ) -> FallbackPolicy:
        normalized = (
            (
                value
                or "NONE"
            )
            .strip()
            .upper()
        )

        return FallbackPolicy(
            normalized
        )

    @staticmethod
    def _parse_date(
        value: str | None,
    ) -> date | None:
        if not value:
            return None

        normalized = (
            value.strip()
        )

        if (
            normalized.lower()
            in {
                "",
                "not specified",
                "none",
                "n/a",
            }
        ):
            return None

        # This field is Optional[date] in the schema - any free-text
        # placeholder a content author leaves here ("not yet reviewed",
        # "pending", "TBD", ...) should degrade to unset rather than
        # crash the whole page's ingestion. Maintaining an allowlist of
        # every phrase someone might type is a losing game; catching
        # the parse failure is the general fix.
        try:
            return date.fromisoformat(
                normalized
            )

        except ValueError:
            return None

    @staticmethod
    def _parse_review_frequency(
        value: str | None,
    ) -> int | None:
        if not value:
            return None

        normalized = (
            value
            .strip()
            .lower()
        )

        normalized = (
            normalized
            .replace(
                "days",
                "",
            )
            .replace(
                "day",
                "",
            )
            .strip()
        )

        return int(
            normalized
        )
from app.knowledge.parser import KnowledgeParser


def make_page(*, page_id: str, title: str, body_html: str):
    return {
        "id": page_id,
        "title": title,
        "version": {"number": 1},
        "body": {"storage": {"value": body_html}},
    }


METADATA_TABLE = """
    <table>
        <tr><td>Knowledge Key</td><td>KB-TEST-001-EN</td></tr>
        <tr><td>Canonical Knowledge Key</td><td>KB-TEST-001</td></tr>
        <tr><td>Knowledge Type</td><td>HOW_TO</td></tr>
        <tr><td>Category</td><td>GENERAL</td></tr>
        <tr><td>Status</td><td>APPROVED</td></tr>
        <tr><td>Audience</td><td>BOTH</td></tr>
        <tr><td>Language</td><td>EN</td></tr>
        <tr><td>Market Scope</td><td>GLOBAL</td></tr>
        <tr><td>Bot Mode</td><td>KNOWLEDGE</td></tr>
    </table>
"""


def test_finds_metadata_table_as_the_first_table():
    page = make_page(
        page_id="1",
        title="Test page",
        body_html=f"{METADATA_TABLE}<p>Example content.</p>",
    )

    parsed = KnowledgeParser().parse(page)

    assert parsed.metadata.knowledge_key == "KB-TEST-001-EN"
    assert "Knowledge Key" not in parsed.content
    assert "Example content." in parsed.content


def test_finds_metadata_table_when_a_content_table_comes_first():

    content_table = (
        "<table>"
        "<tr><td>Account Type</td><td>Commission</td></tr>"
        "<tr><td>ECN</td><td>$3.5/lot</td></tr>"
        "</table>"
    )

    page = make_page(
        page_id="1",
        title="Test page",
        body_html=(
            f"<p>Intro.</p>{content_table}{METADATA_TABLE}" "<p>Example content.</p>"
        ),
    )

    parsed = KnowledgeParser().parse(page)

    assert parsed.metadata.knowledge_key == "KB-TEST-001-EN"
    assert "Knowledge Key" not in parsed.content
    assert "Knowledge Type" not in parsed.content
    assert "Example content." in parsed.content
    assert "Commission" in parsed.content
    assert "ECN" in parsed.content


def test_raises_when_no_table_has_a_knowledge_key():
    page = make_page(
        page_id="1",
        title="Broken page",
        body_html="<table><tr><td>Not metadata</td><td>x</td></tr></table>",
    )

    try:
        KnowledgeParser().parse(page)

        raise AssertionError("expected ValueError for a page with no " "metadata table")

    except ValueError as exc:
        assert "METADATA_TABLE_NOT_FOUND" in str(exc)

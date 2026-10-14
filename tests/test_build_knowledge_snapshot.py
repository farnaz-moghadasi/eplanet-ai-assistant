from scripts.build_knowledge_snapshot import _find_duplicate_knowledge_keys


def make_page(*, page_id: str, title: str, knowledge_key: str):
    return {
        "id": page_id,
        "title": title,
        "version": {"number": 1},
        "body": {"storage": {"value": f"""
                    <table>
                        <tr>
                            <td>Knowledge Key</td>
                            <td>{knowledge_key}</td>
                        </tr>

                        <tr>
                            <td>Canonical Knowledge Key</td>
                            <td>{knowledge_key}</td>
                        </tr>

                        <tr>
                            <td>Knowledge Type</td>
                            <td>HOW_TO</td>
                        </tr>

                        <tr>
                            <td>Category</td>
                            <td>DEPOSITS</td>
                        </tr>

                        <tr>
                            <td>Status</td>
                            <td>APPROVED</td>
                        </tr>

                        <tr>
                            <td>Audience</td>
                            <td>BOTH</td>
                        </tr>

                        <tr>
                            <td>Language</td>
                            <td>EN</td>
                        </tr>

                        <tr>
                            <td>Market Scope</td>
                            <td>GLOBAL</td>
                        </tr>

                        <tr>
                            <td>Countries</td>
                            <td></td>
                        </tr>

                        <tr>
                            <td>Required Context Fields</td>
                            <td>NONE</td>
                        </tr>

                        <tr>
                            <td>Bot Mode</td>
                            <td>KNOWLEDGE</td>
                        </tr>

                        <tr>
                            <td>Fallback Policy</td>
                            <td>NONE</td>
                        </tr>
                    </table>

                    <h1>{title}</h1>

                    <p>Example content.</p>
                """}},
    }


def test_flags_pages_sharing_one_knowledge_key():

    pages = {
        "292061205": make_page(
            page_id="292061205",
            title="Deposit via bank transfer",
            knowledge_key="KB-DEP-IR-003-FA",
        ),
        "292290561": make_page(
            page_id="292290561",
            title="Deposit not credited",
            knowledge_key="KB-DEP-IR-003-FA",
        ),
        "111": make_page(
            page_id="111", title="Unrelated page", knowledge_key="KB-OTHER-001-FA"
        ),
    }

    duplicates = _find_duplicate_knowledge_keys(pages)

    assert list(duplicates.keys()) == ["KB-DEP-IR-003-FA"]

    page_ids = {entry[0] for entry in duplicates["KB-DEP-IR-003-FA"]}

    assert page_ids == {"292061205", "292290561"}


def test_no_duplicates_when_all_keys_unique():
    pages = {
        "1": make_page(page_id="1", title="Page one", knowledge_key="KB-ONE-001-FA"),
        "2": make_page(page_id="2", title="Page two", knowledge_key="KB-TWO-001-FA"),
    }

    assert _find_duplicate_knowledge_keys(pages) == {}


def test_unparseable_page_is_skipped_not_crashed_on():
    pages = {
        "1": make_page(page_id="1", title="Page one", knowledge_key="KB-ONE-001-FA"),
        "2": {
            "id": "2",
            "title": "Broken page",
            "version": {"number": 1},
            "body": {"storage": {"value": "<p>no table here</p>"}},
        },
    }

    assert _find_duplicate_knowledge_keys(pages) == {}

import ast
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))

    modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)

        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    return modules


def test_application_code_does_not_depend_on_scripts_or_tests():
    # scripts/ holds one-off dev, sync and evaluation tools and tests/ is
    # not shipped in the production image. The app used to build its own
    # retrieval pipeline by importing a script called test_*.py, so
    # deleting or renaming a "test" script would have broken production.
    offenders = []

    for path in APP_DIR.rglob("*.py"):
        for module in _imported_modules(path):
            top_level = module.split(".")[0]

            if top_level in {"scripts", "tests"}:
                offenders.append(
                    f"{path.relative_to(APP_DIR.parent)} imports {module}"
                )

    assert offenders == []

"""
Structural leakage guard (Phase 4): production code under app/ must
never import from the evaluation package, or read evaluation labels,
annotation CSVs, hidden strata, or blind keys at runtime.

This is enforced here structurally (an AST scan), not merely by
convention/docstring, the same discipline the Phase 3 evaluation
scripts applied at runtime via a builtins.open guard -- this test is
the app/-side, always-on equivalent.

Deliberately AST-based rather than a plain text/word scan: several
app/ docstrings already reference "evaluation/PROJECT_RUBRIC.md" in
prose, by design, to document exactly why a given field is NOT read by
production code (see e.g. CandidateProject's docstring). A text scan
would false-positive on that documentation. What actually matters is
(1) an import statement naming the evaluation package, and (2) a
string literal that looks like an evaluation/ file path passed as an
argument to a call (open(...), Path(...), etc.) -- i.e. code that would
actually execute a read, not prose that mentions the directory.
"""
import ast
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent / "app"


def _imports_evaluation(tree: ast.AST) -> list[str]:
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "evaluation" or alias.name.startswith("evaluation."):
                    offenders.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "evaluation" or node.module.startswith("evaluation.")):
                offenders.append(f"line {node.lineno}: from {node.module} import ...")
    return offenders


def _call_arguments_pointing_into_evaluation(tree: ast.AST) -> list[str]:
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if "evaluation/" in arg.value or "evaluation\\" in arg.value:
                    offenders.append(f"line {node.lineno}: {arg.value!r}")
    return offenders


def test_no_app_module_imports_the_evaluation_package():
    offenders = []
    for path in sorted(APP_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for hit in _imports_evaluation(tree):
            offenders.append(f"{path.name}: {hit}")

    assert offenders == [], f"app/ must never import from evaluation/: {offenders}"


def test_no_app_module_passes_an_evaluation_path_to_a_call():
    """
    Catches code that would actually READ from evaluation/ at runtime
    (open("evaluation/..."), Path("evaluation/..."), etc.) without
    flagging docstring prose that merely mentions the directory.
    """
    offenders = []
    for path in sorted(APP_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for hit in _call_arguments_pointing_into_evaluation(tree):
            offenders.append(f"{path.name}: {hit}")

    assert offenders == [], (
        f"app/ files pass an evaluation/-shaped path to a call -- production "
        f"code must never read evaluation artifacts at runtime: {offenders}"
    )

"""The walkthrough notebook copies package code verbatim; fail when the two drift apart."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "bevformer_e2e_walkthrough.ipynb"


def find_drift(notebook: Path, package_root: Path) -> tuple[int, list[str]]:
    """(definitions compared, ["cell N: name", ...] that differ from the package's version).

    A top-level function or class in a notebook cell is compared, by syntax tree (so formatting
    and comments do not matter), with every definition of the same name under `package_root`.
    Definitions that exist only in the notebook (probe helpers) are ignored.
    """
    package: dict[str, set[str]] = {}
    for path in package_root.rglob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                package.setdefault(node.name, set()).add(ast.dump(node))
    compared, drifted = 0, []
    for number, cell in enumerate(json.loads(notebook.read_text())["cells"]):
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell["source"]))
        except SyntaxError:  # e.g. a cell with IPython magics
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in package:
                compared += 1
                if ast.dump(node) not in package[node.name]:
                    drifted.append(f"cell {number}: {node.name}")
    return compared, drifted


def test_notebook_definitions_match_the_package():
    compared, drifted = find_drift(NOTEBOOK, ROOT / "bevformer")
    assert compared >= 25, "the notebook should copy most of the model code from the package"
    assert not drifted, "notebook code differs from bevformer/ (re-sync the cells): " + ", ".join(drifted)


def _write_notebook(path: Path, source: str) -> None:
    cell = {"cell_type": "code", "source": source.splitlines(keepends=True), "outputs": [], "metadata": {}}
    path.write_text(json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))


def test_find_drift_ignores_formatting_flags_changes_and_skips_notebook_only_code(tmp_path):
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "mod.py").write_text("def f(x):\n    return x + 1\n")
    notebook = tmp_path / "nb.ipynb"

    _write_notebook(notebook, "def f(x):\n    # a comment\n    return x +  1\n\ndef helper():\n    return 0\n")
    assert find_drift(notebook, package) == (1, [])  # same syntax tree; `helper` is notebook-only

    _write_notebook(notebook, "def f(x):\n    return x + 2\n")
    assert find_drift(notebook, package) == (1, ["cell 0: f"])

from __future__ import annotations

import re
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_quarto_docs_snapshot_contains_expected_pages() -> None:
    root = _repo_root()

    expected = {
        "docs/index.qmd",
        "docs/getting-started.qmd",
        "docs/ai-quickstart.qmd",
        "docs/core-concepts/model-tree.qmd",
        "docs/core-concepts/querypaths.qmd",
        "docs/core-concepts/data-types.qmd",
        "docs/core-concepts/embeddings.qmd",
        "docs/core-concepts/dynamic-masking.qmd",
        "docs/data-types/branch.qmd",
        "docs/data-types/category.qmd",
        "docs/data-types/dateparts.qmd",
        "docs/data-types/entity.qmd",
        "docs/data-types/number.qmd",
        "docs/data-types/set.qmd",
        "docs/data-types/text.qmd",
        "docs/data-types/vector.qmd",
        "docs/guides/batch-inference.qmd",
        "docs/guides/data-modules.qmd",
        "docs/guides/field-importance.qmd",
        "docs/guides/field-stacking.qmd",
        "docs/guides/lightning.qmd",
        "docs/guides/postprocessors.qmd",
        "docs/case-studies/device-tenure.qmd",
    }

    missing = {path for path in expected if not (root / path).is_file()}
    assert missing == set()


def test_quarto_docs_use_current_public_api_style() -> None:
    root = _repo_root()
    sources = "\n".join(path.read_text() for path in sorted((root / "docs").rglob("*.qmd")))

    assert "jv.Model(" in sources
    assert "jv.Model.from_tree(" not in sources
    assert "jv.Branch(" in sources
    assert "Struct(" not in sources
    assert not re.search(r"Model\.from_tree\([^)]*\broot\s*=", sources, re.DOTALL)


def test_only_allowed_standalone_examples_are_present() -> None:
    root = _repo_root()
    examples = root / "examples"
    if not examples.exists():
        return

    discovered = {
        path.relative_to(root).as_posix()
        for path in examples.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    assert discovered == {
        "examples/dynamic-masking/run.py",
        "examples/inference-masking/run.py",
        "examples/realtime-serving/run.py",
        "examples/text-encoder/run.py",
    }

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _execute_notebook(path: str) -> nbformat.NotebookNode:
    root = _repo_root()
    run_dirs = {
        "docs/guides/tensorfields.ipynb": root / "docs/guides",
    }
    notebook = nbformat.read(root / path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(run_dirs.get(path, root))}},
    )
    client.execute()
    return notebook


def test_pretraining_example_runs() -> None:
    notebook = _execute_notebook("docs/tutorials/pretraining.ipynb")

    assert _plot_output(notebook)


def test_supervised_tabular_example_runs() -> None:
    notebook = _execute_notebook("docs/tutorials/supervised-tabular-training.ipynb")

    assert _plot_output(notebook)


def test_serving_example_configures_without_starting_server() -> None:
    notebook = _execute_notebook("docs/tutorials/serving.ipynb")

    assert _plot_output(notebook)


def test_custom_tensorfield_example_runs() -> None:
    notebook = _execute_notebook("docs/guides/tensorfields.ipynb")
    source = "\n".join(cell.source for cell in notebook.cells)

    assert 'Plugin(name="bucket")' in source
    assert "Boolean" not in source
    assert _plot_output(notebook)


def test_field_ablation_example_runs() -> None:
    notebook = _execute_notebook("docs/guides/field-ablation.ipynb")
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "active=False" in source
    assert "trainer.test" in source


def test_examples_live_under_docs() -> None:
    examples_path = _repo_root() / "examples"

    assert not examples_path.exists()


def _plot_output(notebook: nbformat.NotebookNode) -> str:
    for cell in notebook.cells:
        if cell.cell_type != "code" or "model.plot(" not in cell.source:
            continue
        text = "\n".join(_output_text(output) for output in cell.get("outputs", []))
        if "Schema" in text:
            return text
    return ""


def _output_text(output: nbformat.NotebookNode) -> str:
    if "text" in output:
        return output.text

    data = output.get("data", {})
    return "\n".join(str(data.get(mime, "")) for mime in ("text/plain", "text/html"))

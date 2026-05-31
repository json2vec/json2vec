from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _execute_notebook(path: str) -> nbformat.NotebookNode:
    root = _repo_root()
    run_dirs = {
        "docs/data-types/tensorfields.ipynb": root / "docs/data-types",
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
    notebook = _execute_notebook("docs/data-types/tensorfields.ipynb")
    source = "\n".join(cell.source for cell in notebook.cells)

    assert 'Plugin(name="bucket")' in source
    assert "Boolean" not in source
    assert _plot_output(notebook)


def test_field_importance_example_runs() -> None:
    notebook = _execute_notebook("docs/guides/field-importance.ipynb")
    source = "\n".join(cell.source for cell in notebook.cells)

    assert "active=False" in source
    assert "trainer.test" in source


def test_standalone_examples_directory_is_absent() -> None:
    assert not (_repo_root() / "examples").exists()


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

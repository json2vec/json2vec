.PHONY: notebooks

NOTEBOOKS := $(shell find docs -type f -name '*.ipynb' | sort)
export NOTEBOOKS

define RUN_NOTEBOOKS
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

root = Path.cwd()
notebooks = [root / notebook for notebook in os.environ["NOTEBOOKS"].split()]
run_dirs = {
    Path("docs/guides/tensorfields.ipynb"): Path("docs/guides"),
}

for path in notebooks:
    notebook_path = path.relative_to(root)
    cwd = root / run_dirs.get(notebook_path, Path("."))
    print(f"executing {notebook_path} from {cwd}", flush=True)
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(cwd)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"wrote {notebook_path}", flush=True)
endef
export RUN_NOTEBOOKS

notebooks:
	uv run python -c "$$RUN_NOTEBOOKS"

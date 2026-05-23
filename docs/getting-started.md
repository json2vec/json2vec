# Getting Started

Install the development environment:

```bash
uv sync
```

The base install is enough for the library and test suite. The documentation uses notebooks, MkDocs, and a few bundled scikit-learn datasets, so docs work uses the optional `docs` extra.

Serve the notebook-backed docs locally:

```bash
uv sync --extra docs
uv run --extra docs mkdocs serve
```

The examples use direct tensorfield constructors and the minimal model API:

```python
import json2vec as j2v

model = j2v.Model.from_schema(
    j2v.Number("sepal_length"),
    j2v.Number("petal_length"),
    j2v.Category("species", target=True, max_vocab_size=4),
    d_model=16,
    n_layers=1,
    n_heads=4,
    batch_size=2,
)
```

The schema above creates a root `record` array containing two numeric inputs and one categorical target. The target field is hidden from model input during supervised training and decoded from the remaining context.

`target=True` is shorthand for setting the field as a supervised target. `model.set(...)` can mutate selected nodes later:

```python
model.set(j2v.where("name") == "record", embed=True)
model.set(j2v.where("type") == "category", p_mask=0.10)
```

The preprocessor is optional. If no preprocessor is passed to a dataset or deployment, records are encoded unchanged.

Most examples use explicit `query=...` values because notebook records are passed through a batch wrapper before encoding. For simple top-level records, omitting `query` lets JSON2Vec infer the source path from the field name.

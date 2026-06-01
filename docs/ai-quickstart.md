# AI Quickstart

This page gives AI agents and experienced Python users the shortest reliable
path through `json2vec`. If you are new to the package, start with
[Getting Started](getting-started.md).

## Import Style

Use the package root for normal work:

```python
import json2vec as j2v
```

## Build A Model

Top-level records use field names directly. `json2vec` infers request queries from those names.

```python
model = j2v.Model.from_schema(
    j2v.Number("amount"),
    j2v.Category("merchant", max_vocab_size=4096),
    j2v.Category("label", target=True, max_vocab_size=2),
    d_model=64,
    n_layers=2,
    n_heads=4,
    batch_size=32,
)
```

`target=True` is exact shorthand for `p_prune=1.0`: it withholds `label`
from the input and trains the decoder to reconstruct it from the remaining
fields. It is the "always hidden" supervised-target case; use `p_mask` for
stochastic self-supervised masking with rates lower than `1.0`.

## Add Nested Data

Use `Array` for repeated child objects.

```python
model = j2v.Model.from_schema(
    j2v.Array(
        j2v.Category("sku", max_vocab_size=2048),
        j2v.Number("quantity"),
        j2v.Number("price"),
        name="line_items",
        max_length=32,
    ),
    j2v.Category("returned", target=True, max_vocab_size=2),
    d_model=64,
    n_layers=2,
    n_heads=4,
)
```

For ordered histories, set `overflow="tail"` on the array when the newest
records should be retained after `max_length` is reached. Use
`overflow="error"` when truncation should fail.

This reads records shaped like:

```python
{
    "line_items": [
        {"sku": "ABC-123", "quantity": 2, "price": 19.99},
        {"sku": "XYZ-999", "quantity": 1, "price": 7.50},
    ],
    "returned": "false",
}
```

## Name The Root Context

The generated root array is named `record` by default. Override it with `name=...`.

```python
model = j2v.Model.from_schema(
    j2v.Number("amount"),
    name="event",
    d_model=32,
    n_layers=1,
    n_heads=4,
    embed=True,
)
```

## Train Quickly

Use `PolarsDataModule(...)` for in-memory examples. `j2v.Model` is a Lightning
module, so training uses the normal Lightning `Trainer.fit(...)` loop.

```python
import lightning.pytorch as lit
import polars as pl
import torch

records = pl.read_ndjson("docs/data/iris.jsonl").head(36)

model = j2v.Model.from_schema(
    j2v.Number("sepal_length"),
    j2v.Number("petal_length"),
    j2v.Category("species", target=True, max_vocab_size=4, topk=[2]),
    d_model=16,
    n_layers=1,
    n_heads=4,
    batch_size=8,
    embed=True,
    optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-2),
)

datamodule = j2v.PolarsDataModule(
    model=model,
    train=records,
    validate=records,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)

trainer = lit.Trainer(
    accelerator="cpu",
    max_epochs=1,
    logger=False,
    enable_progress_bar=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    limit_train_batches=1,
    limit_val_batches=1,
)
trainer.fit(model=model, datamodule=datamodule)
```

For larger file-backed datasets, use
[Data Modules](guides/data-modules.md) to choose `PolarsDataModule` or
`StreamingDataModule`.

## Predict

`model.predict(...)` accepts a list of raw dictionaries. Configured embeddings appear in the same address-keyed output.

```python
predictions = model.predict(records.to_dicts()[:3])

species = predictions[j2v.Address("record", "species")]
record = predictions[j2v.Address("record")]
```

For batch prediction jobs, use
[Batch Inference](guides/batch-inference.md): configure a `predict` split on a
data module, attach `j2v.Writer(...)` to `Trainer(callbacks=[...])`, and call
`trainer.predict(...)`.

## Shape Raw Data

Use `query=...` when the source shape is stable but does not match the schema name.

```python
model = j2v.Model.from_schema(
    j2v.Number("amount", query='[*].transaction."amount_usd"'),
    j2v.Category("merchant", query="[*].transaction.merchant_name", max_vocab_size=1024),
    j2v.Category("label", query="[*].outcome", target=True, max_vocab_size=2),
    d_model=32,
    n_layers=1,
    n_heads=4,
)
```

Use a preprocessor when the source needs Python logic before querying. The
decorator registers the callable for dataset and deployment configuration, and
the same callable can be passed directly to `model.encode(...)` or
`model.predict(...)`.

```python
@j2v.preprocess
def normalize_request(record: dict) -> dict:
    return {
        "amount": float(record["amount_usd"]),
        "merchant": record["merchant"].strip().lower(),
        "label": record["label"],
    }
```

Then pass it to data modules, `model.encode(...)`, `model.predict(...)`, or
deployments where supported.

## Reshape Outputs

Use a [postprocessor](guides/postprocessors.md) to turn address-keyed predictions into an API or warehouse shape.

```python
def risk_response(context, predictions):
    label = {}
    for address, values in predictions.items():
        if str(address) == "record/label":
            label = values.get("content", {})
            break

    return {
        "risk": {
            "prediction": label.get("value"),
            "scores": label.get("probability"),
        }
    }
```

## Serve

Serving helpers are top-level exports when the serving extra is installed.

```python
deployment = (
    j2v.Deployment(model=model, accelerator="cpu", max_batch_size=16)
    .postprocess(risk_response)
    .update(j2v.where("name") == "label", topk=[2])
)
```

Use `model=...` for an already constructed model during local setup, or
`checkpoint="model.ckpt"` when loading a trained checkpoint. Call
`deployment.serve()` from an application entry point.

## Rules For Agents

- Prefer `import json2vec as j2v`.
- Prefer `Model.from_schema(...)` and top-level exports.
- Use `name=...` to name the generated root context.
- Set `max_vocab_size` on categorical fields where necessary.
- Only put `Entity` under repeated contexts.
- Use `embed=True` for embeddings and `target=True` for supervised targets.
- Use `p_mask` for stochastic self-supervised reconstruction; use `target=True`
  or `p_prune=1.0` when a field should always be hidden and decoded.
- Use Lightning `Trainer` with `PolarsDataModule` or `StreamingDataModule` for
  training and batch prediction.
- Use `j2v.Writer` for Parquet batch inference output.

Use [Getting Started](getting-started.md) and the tutorial notebooks for
runnable patterns.

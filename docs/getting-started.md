# Getting Started

This page is the first practical pass through `json2vec`: define the record
shape, train a tiny model, inspect predictions, then extend the same idea
to nested arrays. The goal is not model quality. It is to make the package's
core loop concrete.

## Start With The Record Shape

`json2vec` models dictionaries and lists of dictionaries directly. For a simple
record, the schema field names can match the source keys:

```python
{
    "sepal_length": 5.1,
    "petal_length": 1.4,
    "species": "setosa",
}
```

This is the contract you want the model to see:

- `sepal_length` and `petal_length` are numeric inputs.
- `species` is a categorical target.
- the root record should also emit an embedding during prediction.

## Build The Model

Use the package root import. `Model.from_schema(...)` turns the field
declarations into a model tree.

```python
import json2vec as j2v
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
```

Just like that, a model has been created.

`target=True` defines a field as supervised target: it withholds `species` from the input during the training loop and asks the model to decode it from the remaining fields.

You may use `p_mask` to randomly "mask" a random portion of field values. This allows for a powerful "masked language modeling" like self-supervised learning task. `p_mask` is defined such that random values of nested arrays are masked.

You may also define `p_prune` to randomly "mask" all values of a field within a batch. For tabular data, this is the same thing as `p_mask`, but for nested data (a multi-dimensional context of input tokens), it will mask all available data within an observation with probability `p_prune`.

!!! Note
    Use `target=True` or `p_prune=1.0` when a field should always be hidden.
    `p_mask` is stochastic and must be lower than `1.0`.

Lastly, `embed=True` is separate. It asks prediction to include an embedding for any node. You may embed the entire observation, individual fields, or branches of the observation.

## Train One Batch

For small in-memory examples, `PolarsDataModule(...)` ties the configured model to sample observations from a Polars dataframe.

```python
datamodule = j2v.PolarsDataModule(
    model=model,
    train=records,
    validate=records,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    observation_buffer_size=32,
    sample_rate=1.0,
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

## Inspect Tensor Representations


`model.encode(...)` accepts raw dictionaries and will return a `torch.TensorDict` of nested data structures. `json2vec` tensor data structures are defined by nested collections of [`tensorfields`](data-types/tensorfields.ipynb), which define the `content` and `state` of type-specific multi-dimensional arrays. These are complex, but may be viewed for validation or parsing logic.

```python
tensors = model.encode(records.to_dicts()[:3])

print(tensors)
```

For messy data, you may define custom [preprocessors](guides/preprocessors.ipynb) to reshape training observations, filter some out, or yield multiple training observations from a data record.

Alternatively, you may choose to define [custom queries](core-concepts/querypaths.md) to extract data from awkwardly data structures.


## Inspect Predictions

`model.predict(...)` accepts raw dictionaries. It returns a dictionary keyed by
schema address, so decoded targets and embeddings stay attached to the part of
the schema that produced them.

```python
predictions = model.predict(records.to_dicts()[:3])

species = predictions[j2v.Address("record", "species")]
record = predictions[j2v.Address("record")]

print(species["content"]["value"])
print(species["content"]["probability"])
print(record["embedding"])
```

For API responses or warehouse rows, keep the model output stable and add a
[postprocessor](guides/postprocessors.md) to reshape the address-keyed
dictionary.

## Add Nested Arrays

Flat examples are useful for mechanics, but `json2vec` is designed for predictive modeling with hierarchical data structure in which nested structures carry signal. Use `Array` when a record contains a list of objects:

```python
{
    "measurements": [
        {"name": "mean_radius", "value": 17.99},
        {"name": "mean_texture", "value": 10.38},
    ],
    "diagnosis": "malignant",
}
```

The matching schema gives `measurements` its own repeated context encoder:

```python
model = j2v.Model.from_schema(
    j2v.Array(
        j2v.Category("name", max_vocab_size=16),
        j2v.Number("value"),
        name="measurements",
        max_length=8,
        embed=True,
    ),
    j2v.Category("diagnosis", target=True, max_vocab_size=2),
    d_model=24,
    n_layers=1,
    n_heads=4,
    batch_size=8,
    embed=True,
    optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-2),
)
```

The inferred child queries are `[*].measurements[*].name` and
`[*].measurements[*].value`. During prediction, configured embeddings can appear
at both `record` and `record/measurements`. These queries are inferred from the schema, but you may alternatively define [custom queries](core-concepts/querypaths.md).

Run the complete nested example with:

```python
nested_records = pl.read_ndjson("docs/data/breast-cancer.jsonl").head(32)

nested_datamodule = j2v.PolarsDataModule(
    model=model,
    train=nested_records,
    validate=nested_records,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
    observation_buffer_size=32,
    sample_rate=1.0,
)

nested_trainer = lit.Trainer(
    accelerator="cpu",
    max_epochs=1,
    logger=False,
    enable_progress_bar=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    limit_train_batches=1,
    limit_val_batches=1,
)

nested_trainer.fit(model=model, datamodule=nested_datamodule)
nested_predictions = model.predict(nested_records.to_dicts()[:2])
print(nested_predictions.keys())
```

## Next Steps

- **Understand the modeling idea:** [Why `json2vec`](motivation.md)
- **Map source records to schemas:** [Query Paths](core-concepts/querypaths.md)
- **Choose field types:** [Built-In Data Types](core-concepts/data-types.md)
- **Run a notebook walkthrough:** [Hello World](tutorials/hello-world.ipynb)
- **Train without labels:** [Masked Pretraining](tutorials/pretraining.ipynb)
- **Export embeddings:** [Embeddings & Self-Supervised Learning](core-concepts/embeddings.md)
- **Change schemas after construction:** [Mutations](core-concepts/mutations.ipynb)
- **Apply the nested-data pattern:** [Device Tenure](case-studies/device-tenure.md)

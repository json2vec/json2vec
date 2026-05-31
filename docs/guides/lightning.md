# Training With Lightning

`json2vec` builds on top of PyTorch Lightning. A `j2v.Model` is a Lightning
`LightningModule`, and the data modules in `json2vec` are Lightning
`LightningDataModule` implementations. The schema defines the model tree,
typed inputs, losses, targets, and embeddings; Lightning runs the training,
validation, test, and prediction loops.

That means normal Lightning tools apply: `Trainer`, callbacks, checkpointing,
logging, device placement, precision settings, and distributed strategies.

This design decision was not taken lightly: the requirements with extensible 
callback management, complex loop management,and device configurations
aligned really well with the abstractions provided by Lightning.
`json2vec` aims to expose as much flexibility and customization to users as possible.

## Ownership

| Concern | Owned by |
| --- | --- |
| Schema, addresses, query paths, targets, embeddings | `json2vec` |
| Tensorization, typed value state, datatype-specific losses | `json2vec` |
| Optimizer and scheduler hooks | `j2v.Model` through Lightning |
| Epochs, devices, precision, callbacks, logging, checkpoints | Lightning `Trainer` |
| Data loading and tensorization batches | `j2v.PolarsDataModule` or `j2v.StreamingDataModule` |
| Batch prediction writing | Lightning `Trainer.predict(...)` plus `j2v.Writer` |

## Minimal Training Loop

```python
import lightning.pytorch as lit
import polars as pl

import json2vec as j2v

records = pl.DataFrame(
    {
        "amount": [12.5, 8.0, 19.0],
        "merchant": ["books", "coffee", "books"],
        "fraud": ["no", "no", "yes"],
    }
)

model = j2v.Model.from_schema(
    j2v.Number("amount"),
    j2v.Category("merchant", max_vocab_size=128),
    j2v.Category("fraud", target=True, max_vocab_size=2),
    d_model=32,
    n_layers=1,
    n_heads=4,
    batch_size=2,
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
    enable_checkpointing=False,
)

trainer.fit(model=model, datamodule=datamodule)
```

The same model and data module can be passed to other Lightning loops:

```python
trainer.validate(model=model, datamodule=datamodule)
trainer.test(model=model, datamodule=datamodule)
trainer.predict(model=model, datamodule=datamodule)
```

Configure the matching split on the data module before using each loop.

## Callbacks

Lightning callbacks work normally:

```python
checkpoint = lit.callbacks.ModelCheckpoint(
    monitor="loss/validate",
    mode="min",
)

trainer = lit.Trainer(callbacks=[checkpoint])
```

`json2vec` also installs internal callbacks when needed. For example, schema
mutation is blocked while Lightning owns an active training, validation, test,
or prediction loop.

!!! Warning
    Do not call `model.update(...)`, `model.extend(...)`, `model.delete(...)`,
    or other schema mutations from inside an active Lightning loop. Mutate the
    schema between runs, or use `model.override(...)` around a complete
    evaluation call.

## Interactive Prediction

For small batches, `model.predict(...)` is a convenience method that accepts raw
dictionaries and returns the written prediction payload directly:

```python
predictions = model.predict(
    [
        {"amount": 12.5, "merchant": "books"},
        {"amount": 8.0, "merchant": "coffee"},
    ]
)
```

Use this for debugging, notebooks, and small request batches. For high-volume
prediction, use `trainer.predict(...)` with a data module and `j2v.Writer`.

## Production Notes

Examples often set `num_workers=0`, `persistent_workers=False`, and
`pin_memory=False` so they run reliably in notebooks and documentation builds.
Those settings are not production recommendations. For larger jobs, tune
workers, sharding, buffers, accelerator settings, logging, and checkpointing
through Lightning and the data module options.

## Where Next

- Use [Data Modules](data-modules.md) to choose `PolarsDataModule` or
  `StreamingDataModule`.
- Use [Batch Inference](batch-inference.md) to write predictions with
  `j2v.Writer`.
- Use [Preprocessors](preprocessors.ipynb) for input-side Python logic.
- Use the [API Reference](../reference/api.md) for constructor details.

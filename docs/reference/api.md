# API Reference

This page is generated from public docstrings and is meant as a lookup companion to the tutorials. Start with the notebooks when learning the workflow, then use this page to inspect constructor options, mutation methods, and extension base classes.

## Common Entry Points

- `Model.from_schema(...)` builds the model tree from field constructors and arrays.
- `Array(...)` declares a repeated nested context.
- `Overflow` enumerates `Array` overflow policies: `head`, `tail`, and `error`.
- `Number`, `Category`, `Set`, `DateParts`, `Entity`, `Vector`, and `Text` declare typed fields.
- `PolarsDataModule(...)` builds data loaders from a configured model.
- `StreamingDataModule(...)` streams local or S3-backed files into Lightning loops.
- `Model.predict(...)` returns configured target predictions and embeddings.
- `Writer(...)` writes batch prediction output from `Trainer.predict(...)`.
- `Postprocessor` reshapes predictions after decoding; see
  [Postprocessors](../guides/postprocessors.md).
- `Deployment` wraps a checkpoint or model instance for serving; install the
  `serving` extra for LitServe-backed deployment paths.

Learning-oriented entry points:

- [Getting Started](../getting-started.md)
- [Model Tree](../core-concepts/model-tree.md)
- [Query Paths](../core-concepts/querypaths.md)
- [Built-In Data Types](../core-concepts/data-types.md)
- [Training With Lightning](../guides/lightning.md)
- [Data Modules](../guides/data-modules.md)
- [Batch Inference](../guides/batch-inference.md)
- [Serving](../tutorials/serving.ipynb)

## Package

::: json2vec
    options:
      show_root_heading: true
      show_root_full_path: false

## Model

::: json2vec.Model
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - from_schema
        - select
        - update
        - extend
        - delete
        - reset
        - override
        - plot
        - save
        - load
        - predict

## Schema

::: json2vec.Array
    options:
      show_root_heading: true
      show_root_full_path: false

::: json2vec.Hyperparameters
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - from_schema
        - target
        - embed
        - select
        - update
        - extend
        - delete
        - override

::: json2vec.where
    options:
      show_root_heading: true
      show_root_full_path: false

## Tensorfield Constructors

::: json2vec.Number
    options:
      show_root_heading: true
      show_root_full_path: false

::: json2vec.Category
    options:
      show_root_heading: true
      show_root_full_path: false

::: json2vec.Set
    options:
      show_root_heading: true
      show_root_full_path: false

::: json2vec.DateParts
    options:
      show_root_heading: true
      show_root_full_path: false

::: json2vec.Entity
    options:
      show_root_heading: true
      show_root_full_path: false

::: json2vec.Vector
    options:
      show_root_heading: true
      show_root_full_path: false

::: json2vec.Text
    options:
      show_root_heading: true
      show_root_full_path: false

## Data

Use [Data Modules](../guides/data-modules.md) for the workflow-level guide to
`PolarsDataModule` and `StreamingDataModule`.

::: json2vec.PolarsDataModule
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - dataloader
        - train_dataloader
        - val_dataloader
        - test_dataloader
        - predict_dataloader

::: json2vec.StreamingDataModule
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - dataloader
        - train_dataloader
        - val_dataloader
        - test_dataloader
        - predict_dataloader

## Batch Inference

Use [Batch Inference](../guides/batch-inference.md) for the workflow-level guide
to `Trainer.predict(...)`, `Writer`, and postprocessed Parquet output.

::: json2vec.Writer
    options:
      show_root_heading: true
      show_root_full_path: false

## Preprocessing

::: json2vec.preprocess
    options:
      show_root_heading: true
      show_root_full_path: false

::: json2vec.Preprocessor
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - outputs

## Serving

::: json2vec.Deployment
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - forge
        - preprocess
        - postprocess
        - update
        - serve

## Tensorfield Extension API

::: json2vec.tensorfields.base.Plugin
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - register
        - callback
        - callbacks

::: json2vec.TensorFieldBase
    options:
      show_root_heading: true
      show_root_full_path: false

::: json2vec.EmbedderBase
    options:
      show_root_heading: true
      show_root_full_path: false

::: json2vec.DecoderBase
    options:
      show_root_heading: true
      show_root_full_path: false

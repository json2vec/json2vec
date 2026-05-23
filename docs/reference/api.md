# API Reference

This page is generated from public docstrings and is meant as a lookup companion to the tutorials. Start with the notebooks when learning the workflow, then use this page to inspect constructor options, mutation methods, and extension base classes.

## Common Entry Points

- `Model.from_schema(...)` builds the model tree from field constructors and arrays.
- `Array(...)` declares a repeated nested context.
- `Number`, `Category`, `Set`, `DateParts`, `Entity`, `Vector`, and `Text` declare typed fields.
- `PolarsDataModule.from_model(...)` builds data loaders from a configured model.
- `Model.predict(...)` returns configured target predictions.
- `Model.embed(...)` returns embeddings from nodes configured with `embed=True`.
- `Deployment` wraps a saved model for serving.

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
        - nodes_matching
        - select
        - set
        - plot
        - save
        - load
        - evaluate
        - predict
        - embed

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
        - set
        - mutation_history

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

::: json2vec.Dataset
    options:
      show_root_heading: true
      show_root_full_path: false

::: json2vec.PolarsDataModule
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - from_model
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
        - from_model
        - dataloader
        - train_dataloader
        - val_dataloader
        - test_dataloader
        - predict_dataloader

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

::: json2vec.inference.deployment.Deployment
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - forge
        - preprocess
        - postprocess
        - set
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

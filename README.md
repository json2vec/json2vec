# JSON2Vec

`json2vec` is a Python library for learning embeddings directly from nested, semi-structured records without flattening them into a fixed feature table first.

The model is defined as a tree of contexts and typed fields. Leaf tensorfield plugins encode raw values, context nodes aggregate them with attention, and the same configured pipeline is used for training, batch prediction, and online inference.

## What Is In This Repository

This repository currently contains:

- the core library under `src/json2vec/`
- tensorfield plugins for `number`, `category`, `dateparts`, `entity`, and `vector`
- a processor registry for dataset-specific preprocessing
- a LitServe deployment entrypoint for serving from checkpoints
- tests covering structure loading, data processing, tensorfields, training helpers, logging, and inference
- diagrams plus longer design docs in `docs/`

It does not currently ship maintained example experiments or `make` shortcuts. Older references to `experiments/`, `examples/`, and `make train` were removed because they no longer reflect the checked-in code.

## Install

For local development:

```bash
uv sync
```

If you want an editable install:

```bash
pip install -e .
```

The package requires Python `>=3.12`.

## Core Concepts

- `Structure` defines the model tree.
- `Context` nodes describe hierarchical grouping and aggregation.
- Field `Request` nodes declare a `type`, a `query`, and type-specific options.
- `jmespath` queries extract values from each observation.
- `Session` combines a dataset, structure, task, and runtime controls.
- `Experiment` is an ordered list of sessions loaded from config files.

Supported session tasks are:

- `fit`
- `validate`
- `test`
- `predict`

Supported dataset suffixes are:

- `ndjson`
- `parquet`
- `feather`
- `avro`
- `csv`
- `orc`
- `json`

Supported dataset roots are local paths and `s3://...` URIs. If `dataset.root` is `null`, the pipeline runs in processor-driven mode and expects the configured processor to generate observations.

## Minimal Training Workflow

The CLI entrypoint is:

```bash
uv run python -m json2vec --experiments /path/to/configs --experiment demo --name local-dev --notes "first run"
```

The same function is also exposed as the `train` console script after installation.

Config discovery is directory-based. `json2vec` can load `.json`, `.yaml`, `.yml`, `.toml`, and `.jsonnet` experiment files. If a config directory contains exactly one experiment file, `--experiment` can be omitted.

A minimal YAML experiment looks like this:

```yaml
project: demo
sessions:
  - name: train
    task: fit
    learning_rate: 0.001
    dataset:
      root: /path/to/data
      sample_rate: 1.0
      file_buffer_size: 16
      observation_buffer_size: 16
      processor: default
      kwargs: {}
      suffix: ndjson
      patterns:
        train: .*
        validate: .*
        test: .*
        predict: .*
    structure:
      name: demo-structure
      type: structure
      batch_size: 2
      dropout: 0.1
      d_model: 16
      fields:
        name: root
        type: context
        context_size: 1
        n_outputs: 1
        fields:
          - name: identifier
            type: category
            query: "[*].id"
            max_vocab_size: 1024
```

`fit` sessions write checkpoints to `models/`. In multi-session experiments, the output checkpoint from a `fit` session is automatically passed to later `validate`, `test`, or `predict` sessions.

## Inference And Serving

Batch prediction uses the same experiment/session machinery as training. Prediction outputs are written to `tmp/predictions/`.

For online serving, the repository exposes `json2vec.inference.deployment.Deployment`, which wraps a checkpoint-backed model in LitServe. Runtime configuration is environment-driven:

- `JSON2VEC_CHECKPOINT` or `CHECKPOINT`
- `JSON2VEC_MAX_BATCH_SIZE`
- `JSON2VEC_BATCH_TIMEOUT`
- `JSON2VEC_WORKERS_PER_DEVICE`
- `JSON2VEC_ACCELERATOR`
- `JSON2VEC_TRACK_REQUESTS`

A minimal serve entrypoint is:

```python
from json2vec.inference.deployment import Deployment

Deployment.serve()
```

## Processor Model

Processors are registered Python callables. The built-in `default` processor returns each observation unchanged.

Custom processors live under `src/json2vec/processors/extensions/` and are registered with either `@register.transformation` or `@register.generator`.

- transformation processors must return a single `dict`
- generator processors may yield `dict` objects or return a `list[dict]`
- every emitted object is wrapped as a single-item root context before tensorization

Configured `dataset.kwargs` are passed into the processor, with unsupported keyword arguments automatically ignored.

## Tensorfield Plugins

The current built-in tensorfield types are:

- `number`
- `category`
- `dateparts`
- `entity`
- `vector`

Each tensorfield plugin provides a request schema plus the model components needed to encode values, decode predictions, compute losses, and optionally serialize outputs.

## Runtime Environment

Training and dataloading behavior is controlled with environment variables such as:

- `JSON2VEC_LOGGER`
- `WANDB_API_KEY`
- `NEPTUNE_API_TOKEN`
- `COMET_API_KEY`
- `MLFLOW_TRACKING_URI`
- `JSON2VEC_TENSORBOARD_LOG_DIR`
- `JSON2VEC_CSV_LOG_DIR`
- `JSON2VEC_NUM_WORKERS`
- `JSON2VEC_PERSISTENT_WORKERS`
- `JSON2VEC_PIN_MEMORY`
- `JSON2VEC_SHARDING`
- `JSON2VEC_CHUNK_BATCH_SIZE`

Supported sharding strategies are `file`, `chunk`, and `record`.

## Repository Layout

- `src/json2vec/architecture`: model assembly, attention, pooling, and parcel routing
- `src/json2vec/data`: dataset fetch/read/process/batch/encode pipeline
- `src/json2vec/entrypoints`: training and evaluation orchestration
- `src/json2vec/inference`: serving and prediction callbacks
- `src/json2vec/logging`: tracking and runtime logging helpers
- `src/json2vec/processors`: processor registry and built-in extensions
- `src/json2vec/structs`: pydantic config models, enums, tree structures, and environment settings
- `src/json2vec/tensorfields`: tensorfield plugin system and built-in field types
- `tests/`: package test suite
- `docs/summary.typ` and `docs/whitepaper.typ`: longer written documentation

## Diagrams

The repository includes architecture and pipeline diagrams:

![Tree of encoding modules](docs/diagrams/tree.drawio.svg)

![Single context node](docs/diagrams/node.drawio.svg)

![Pipeline stages](docs/diagrams/pipeline.drawio.svg)

![Example configured module tree](docs/diagrams/modules.drawio.svg)

## Development

Run the test suite with:

```bash
uv run pytest
```

Run lint checks with:

```bash
uv run ruff check
```

## License

Licensed under the Apache License, Version 2.0. See `LICENSE` and `NOTICE`.

## References

- `BIBLIOGRAPHY.md`
- `CITATION.bib`

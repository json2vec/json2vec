# JSON2Vec

`json2vec` is a schema-driven framework for learning embeddings and task
heads directly from nested, semi-structured records without flattening them
into a fixed feature table first.

The central idea is that the schema is the encoder. A declared tree of
contexts and typed fields becomes an addressable neural graph: leaf tensorfield
plugins encode raw values, context nodes aggregate child embeddings with
rotary self-attention and learned-query cross-attention pooling, and
datatype-specific decoders reconstruct masked, pruned, or supervised targets
from the surrounding hierarchy.

This makes `json2vec` a factory for structure-aware encoders rather than a
single domain model. Customer/account/transaction data, flight itineraries,
order fulfillment events, clickstream sessions, and other nested records can
all use the same machinery while keeping their proprietary data, schemas, and
trained checkpoints private.

## What Makes This Different

- **Attributed-distance embeddings.** The model can emit embeddings at any
  configured field or context, not only at the root. That means two observations
  can be similar overall while still exposing which branch of the hierarchy
  accounts for the difference: customer profile, monthly statement, login
  session, transaction history, or any other declared context.
- **Prune-trained counterfactuals.** Training can periodically remove whole
  fields, not just mask individual values. At inference time, the
  same mechanism supports zero-shot ablation questions such as "what changes if
  device data is unavailable?" without retraining a separate model for every
  feature-removal scenario.
- **One path for self-supervised and supervised learning.** Masked values,
  pruned fields, and explicit supervised targets all flow through the same
  datatype-specific heads. A new tensorfield type brings its own embedding,
  decoding, loss, and writing logic, so the framework stays reusable as schemas
  grow.
- **Schema evolution is a first-class workflow.** Because modules are addressed
  by the schema tree, structures can be patched, fields can be added or
  removed, and selected fields can be pruned across sessions without rebuilding
  a separate feature pipeline.
- **Production semantics for missingness.** `null`, `padded`, `masked`,
  `pruned`, and `valued` are distinct states in the tensorfield type system.
  They are not collapsed into one generic missing-value bucket.
- **Online state lives with the model.** Stateful components such as category
  vocabularies, counters, and numeric normalization state are learned during
  streaming training and serialized with checkpoints, so deployment does not
  depend on a parallel tokenizer or normalizer artifact.
- **Training-serving parity.** The same configured graph is used for fitting,
  validation, testing, batch prediction, and LitServe-backed online inference.

The attributed embeddings and prune-trained ablations are model-level
explanation primitives. They help answer where two records differ and how a
prediction changes when an information source is withheld. They are not a
complete compliance story by themselves, but they make governance and audit
layers easier to build on top of the representation layer.

## Where It Fits

Use `json2vec` when the hierarchy is part of the signal:

- customer, account, transaction, statement, device, and session records
- flight itineraries, legs, segments, and events
- orders, shipments, fulfillment events, and support histories
- entities with repeated sub-objects, evolving schemas, and mixed datatypes
- embedding retrieval, anomaly detection, counterfactual ablation, and
  multi-target prediction over nested records

## What It Does Not Do

`json2vec` stops at the representation and typed prediction layer. It does not
try to be a feature store, governance system, rule engine, authorization layer,
decision-capture system, or audit platform. Those systems can consume
`json2vec` embeddings and predictions, but their policies and operational
controls remain separate concerns.

It also does not require users to publish data, schemas, checkpoints, or model
parameters. The open-source layer is the reusable encoder and runtime
infrastructure. Your data stays yours, as does your parameters.
The framework works under the assumption that model parameters will not be shared.

## What Is In This Repository

This repository currently contains:

- the core library under `src/json2vec/`
- tensorfield plugins for `number`, `category`, `dateparts`, `entity`, `vector`, and `text`
- a processor registry for dataset-specific preprocessing
- a LitServe deployment entrypoint for serving from checkpoints
- tests covering structure loading, data processing, tensorfields, training helpers, logging, and inference
- diagrams plus longer design docs in `docs/`

It does not currently ship maintained example experiments or `make` shortcuts. Older references to `experiments/`, `examples/`, and `make train` were removed because they no longer reflect the checked-in code.

More examples based on publicly available will soon be included to showcase implementation and expected behavior.

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
- `Address` values are stable paths such as `root/account/transaction/amount`.
- `jmespath` queries extract values from each observation.
- `TensorField` instances preserve typed content plus state tokens such as
  `valued`, `null`, `padded`, `masked`, and `pruned`.
- `Parcel` objects carry embeddings from leaves to parent contexts and then up
  the tree.
- `heritage` is the path from a leaf to the root; decoders use that path as
  context when reconstructing masked, pruned, or supervised targets.
- `Session` combines a dataset, structure, task, masking/pruning controls, and
  selected embedding outputs.
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
This will likely expand to support `@register` based UDFs for arbitrary data sourcing and file format support ...

## How The Graph Runs

For each batch:

1. Each field request extracts values with its `jmespath` query.
2. The matching tensorfield plugin tensorizes those values, updates any online
   state allowed for the current split, and records trainable targets when
   masking or pruning occurs.
3. Leaf embedders emit parcels to their parent contexts.
4. Context nodes run bottom-up. Each context concatenates available child
   parcels, applies rotary transformer layers, compresses with learned-query
   cross-attention, and emits a new parcel to its parent.
5. Leaf decoders consume the parcel sequence along their heritage path to
   reconstruct trainable targets.

Random `p_mask` corrupts individual values. Random `p_prune` removes whole
field instances across an observation. Session-level `pruned` fields are always
withheld and become supervised targets; session-level `output` addresses are
serialized as embeddings during prediction.

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
    p_mask: 0.15
    p_prune: 0.05
    output:
      - root
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
        n_layers: 1
        n_heads: 4
        n_linear: 1
        fields:
          - name: identifier
            type: category
            query: "[*].id"
            max_vocab_size: 1024
```

`fit` sessions write checkpoints to `models/`. In multi-session experiments, the output checkpoint from a `fit` session is automatically passed to later `validate`, `test`, or `predict` sessions.

To turn a field into a supervised target, include its address in
`session.pruned` for a fit, validate, test, or predict session. The model will
withhold that field from the encoder and use the same datatype-specific decoder
that is used for masked/pruned reconstruction. To export embeddings, include
field or context addresses in `session.output`.

## Inference And Serving

Batch prediction uses the same experiment/session machinery as training. Prediction outputs are written to `tmp/predictions/`.

Checkpoints carry the Lightning weights, serialized session configuration, and
stateful tensorfield state such as online category vocabularies, numeric
normalization buffers, and class-frequency counters. This tight coupling is
intentional: the deployed model should not depend on a separate, manually
synchronized tokenizer or normalizer artifact.

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
- `text`

Each tensorfield plugin provides a request schema plus the model components needed to encode values, decode predictions, compute losses, and optionally serialize outputs.

The `text` tensorfield requires the optional `transformers` dependency and is not installed by default.

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

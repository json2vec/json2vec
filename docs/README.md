# JSON2Vec

`json2vec` is a framework for learning embeddings directly from nested, semi-structured records (JSON, Parquet-like objects, and similar shapes) without flattening them into static feature tables.

It treats each dataset as a tree of contexts and fields, learns datatype-specific leaf embeddings, and routes those embeddings upward through context encoders to produce representations at multiple levels.

## Why this repository exists

Most production ML systems on nested business data eventually accumulate:
- brittle, duplicated feature engineering code,
- train/serve skew between offline and online transforms,
- heavy coupling to a static schema.

`json2vec` aims to make the model itself responsible for value encoding, masking, pruning, and reconstruction so the same logic can run in training and inference.

## Ambition and scope

### Ambition
- Provide a reusable representation layer for hierarchical data.
- Support changing schemas without rebuilding a separate feature store pipeline.
- Make intermediate embeddings available for diagnostics and downstream modeling.
- Real-time serving created solely from checkpoint to perfectly duplicate model training logic.

### Scope
- Structured and semi-structured domains (finance, travel, operational telemetry, ecommerce, etc.).
- Inputs that can be described as nested contexts with typed fields.

### Current restrictions
- Not a general multimodal system (images/audio/video are currently out of scope, but you may include pre-encoded embeddings).
- Not schema-free; you must define structure and `jmespath` queries explicitly.
- Field plugins currently implemented: `number`, `category`, `dateparts`, `entity`, `vector`.

## Core features

- Hierarchical modeling from structure definitions:
  You declare contexts and fields in a configuration file akin to a `jsonschema`; the model compiles this tree into addressable modules.
- N-dimensional / ragged nested value support:
  Field tensorization handles arbitrarily nested list-like values, pads them to fixed shapes (`ndarray`-style tensors), and tracks value state (`valued`, `null`, `padded`, `masked`, `pruned`).
- Featureless training flow:
  The model learns value encoding/normalization/tokenization within field plugins instead of depending on a separate handcrafted feature pipeline.
- `jmespath` query extraction:
  Each field request has a `query` powered by `jmespath`, letting you pull values from deeply nested JSON-like records without flattening upstream.
- SHIM processor support:
  Dataset processors (registered via `@register`) can mutate, filter, explode, or enrich observations before tensorization. This supports domain-specific logic without offline batch feature jobs.
- Masking and pruning controls:
  `p_mask` and `p_prune` support self-supervised reconstruction and robustness to missing branches; permanent field pruning is supported per training session.
- Pruning-based feature importance:
  Because pruning is native to the model path, you can run controlled ablations (field/context removal) and measure impact as an intrinsic importance signal..
- Multi-level embedding outputs:
  You can emit intermediate embeddings at leaf/context/root addresses (`session.output`), not only final decoded predictions.
- Shared train/serve logic:
  Training and online inference both use the same structure, field plugins, and processors, reducing train/serve skew risk.

## Architecture at a glance

The model is a tree of modules:
- Leaf nodes: datatype-specific embedders/decoders.
- Context nodes: stacked rotary self-attention + learned-query cross-attention pooling.
- Routing unit: "parcels" carrying tensor payloads with `origin` and `destination` addresses.

![Tree of encoding modules](diagrams/tree.drawio.svg)

![Single context node](diagrams/node.drawio.svg)

The repository also includes a full example architecture diagram used in the TaxML configuration.

![Example configured module tree](diagrams/modules.drawio.svg)

## Data path

The data path is iterable/streaming and designed for large datasets:
- `fetch -> read -> process -> shuffle -> batch -> transform -> mask -> prune`

Supported sources/formats in the current code:
- Local filesystem and S3.
- `ndjson`, `parquet`, `feather`, `avro`, `csv`, `orc`, `json`.

![Pipeline stages](diagrams/pipeline.drawio.svg)

## Potential use cases

- Financial services:
  Customer/account/transaction/statement hierarchies for fraud detection, risk scoring, customer similarity, and anomaly detection.
- Travel and pricing:
  Itinerary/flight/segment structures for offer quality modeling, tax/fee behavior, conversion propensity, and partner/carrier analysis.
- E-commerce and marketplaces:
  User/session/order/item/event trees for ranking, return-risk prediction, abuse detection, and behavioral clustering.
- Product telemetry and operations:
  Device/session/event streams for reliability monitoring, failure prediction, and root-cause-oriented embedding analysis.
- Insurance and claims:
  Policy/claim/line-item/event structures for triage, severity estimation, and outlier detection.
- Healthcare administration data:
  Patient/encounter/claim/procedure trees for cohort modeling and utilization pattern analysis (subject to compliance constraints).

Common task patterns across these domains:
- Supervised prediction from nested records without flattening.
- Similarity search and clustering on entity embeddings.
- Counterfactual analysis via context/field pruning.
- Robust multi-target inference when branches or fields are missing at runtime.

## Repository layout

- `src/json2vec/architecture`: model, encoders, attention/pooling, parcel flow.
- `src/json2vec/tensorfields`: plugin system and datatype implementations.
- `src/json2vec/data`: streaming dataset pipeline and tensor instantiation.
- `src/json2vec/processors`: dataset-specific shims/transforms.
- `experiments/`: self-contained Jsonnet experiment configs.
- `docs/summary.typ`: short conceptual overview.
- `docs/whitepaper.typ`: extended technical write-up.

## Quickstart

### 1. Install

```bash
uv sync
```

### 2. Run a training workflow

```bash
uv run python -m json2vec --experiment taxml --name local-dev --notes "baseline run"
```

`make train` is a shorthand for launching the same workflow.

### 3. Run serving API

```bash
CHECKPOINT=/path/to/model.ckpt uv run python src/json2vec/inference/deployment.py
```

`make serve` runs the same deployment entrypoint.

## Synthetic Examples

The `examples/` directory contains runnable, shim-first tutorials where `dataset.root` is `null` and observations are generated by a registered processor.

Each use case has:
- `config.jsonnet`: schema + session config.
- `run.py`: shim registration and pipeline execution.

Try any of these:

```bash
uv run python examples/finance-risk/run.py --batches 2
uv run python examples/travel-pricing/run.py --batches 2
uv run python examples/operations-telemetry/run.py --batches 2
```

## Configuration model

Experiment configuration is Jsonnet-based:
- `experiments/<name>.jsonnet`: project-level settings and ordered sessions, with dataset and structure definitions inline per session.
- `dataset.root` may be `null` when observations are generated entirely by the configured processor (useful for tutorials/examples).
- runtime behavior is environment-driven:
  `WANDB_API_KEY`, `NEPTUNE_API_TOKEN`, `COMET_API_KEY`, `MLFLOW_TRACKING_URI`,
  `JSON2VEC_LOGGER`, `JSON2VEC_NUM_WORKERS`, `JSON2VEC_PERSISTENT_WORKERS`, `JSON2VEC_PIN_MEMORY`,
  `JSON2VEC_SHARDING` (`file|chunk|record`, default `chunk`) and
  `JSON2VEC_CHUNK_BATCH_SIZE` (default `4096`).

Sessions support staged workflows (`fit`, `validate`, `test`, `predict`) and per-session controls:
- `p_mask`, `p_prune`, permanent `pruned` addresses,
- LR/scheduler parameters,
- trainer args and early stopping.

## Extensibility

### Add a new field type
Implement and register:
- `Request`
- `TensorField`
- `Embedder`
- `Decoder`
- `loss`
- optional `write`

in `src/json2vec/tensorfields/extensions/`.

### Add dataset-specific preprocessing
Register a processor in `src/json2vec/processors/extensions/` with `@register`, then reference it from dataset config.

## Maturity notes

This repository is actively evolving. The design is stable enough for experimentation and internal workloads, and additional improvements are expected as plugin coverage and deployment ergonomics continue to mature.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.
Attribution details are in `NOTICE`.

## Bibliography

Reference material is listed in `BIBLIOGRAPHY.md`.
Project citation metadata is available in `CITATION.bib`.

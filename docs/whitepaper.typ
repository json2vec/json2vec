#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/treet:1.0.0": *

// Optional paper template metadata.
// Uncomment if you want IEEE-style rendering from this file.
// #show: ieee.with(
//   title: [JSON2Vec: Hierarchical Representation Learning for Semi-Structured Data],
//   abstract: [
//     JSON2Vec is a modeling framework for nested business data that replaces static feature
//     engineering pipelines with an end-to-end, datatype-aware architecture. Inputs are declared
//     as a tree of contexts and typed fields, leaf values are embedded by plugin modules, and
//     intermediate tensors are routed upward through context encoders to produce multi-resolution
//     embeddings. The same execution graph is used for training and inference, enabling masking,
//     pruning, and schema evolution without maintaining a separate train/serve feature stack.
//   ],
//   authors: (
//     (
//       name: "Grantham Taylor",
//       location: [Arlington, Virginia],
//     ),
//   ),
//   index-terms: ("hierarchical modeling", "semi-structured data", "transformers", "embeddings"),
// )

= JSON2Vec: Hierarchical Representation Learning for Semi-Structured Data

== Abstract

`JSON2Vec` is a framework for learning embeddings directly from nested, semi-structured records without flattening them into static feature tables.

The core idea is to model data in its native hierarchy. A user declares a tree of contexts and typed fields, leaf values are encoded by datatype plugins, and contextual information is aggregated upward through transformer-based context encoders. This yields representations at multiple resolutions (field, context, root) and allows the same modeling graph to run in both training and inference.

The repository emphasizes operational pragmatism: a streaming data path, explicit schema definitions, plugin extensibility, masking/pruning objectives, and real-time serving compatibility.

= 1. Motivation

Most production ML systems on nested data rely on flattening, hand-crafted aggregates, and duplicated logic across training and serving paths. This creates persistent friction:

- schema changes require feature pipeline rewrites,
- preprocessing logic drifts between offline and online systems,
- representation quality is capped by manual aggregation choices.

For many operational datasets, structure is not incidental. It is where important signal lives:

- customer -> account -> transaction,
- itinerary -> flight -> segment,
- order -> shipment -> event,
- device -> session -> action.

`JSON2Vec` treats this as the primary modeling object rather than something to flatten away.

== 1.1 Relationship to Feature Stores

Feature stores remain useful for governance, lineage, and serving reliability. `JSON2Vec` is not a replacement for those systems.

Instead, it addresses a different gap: learning robust representations directly from evolving nested structures without requiring a large parallel stack of handcrafted feature transforms.

== 1.2 Problem Target

The intended class of problems has all of the following characteristics:

- semantically meaningful hierarchy,
- heterogeneous field types,
- frequent schema evolution,
- need for both prediction and representation-level analysis.

= 2. Problem Formulation

Let an observation be a nested record `x` and let a declared structure `S` define:

- context nodes `C` (internal nodes),
- leaf fields `F` (typed leaves),
- addresses `a` that uniquely identify each node.

A model must:

1. map each leaf field value set into embeddings,
2. aggregate those embeddings through the context hierarchy,
3. decode targets for masked/pruned leaves,
4. expose embeddings and predictions for downstream use.

This yields two coupled objectives:

- reconstruction of masked/pruned targets,
- learning reusable embeddings that preserve hierarchical information.

= 3. System Overview

`JSON2Vec` is built around four core subsystems:

- structure compiler (tree -> addressable modules),
- plugin-defined field modeling,
- parcel-based routing and aggregation,
- streaming data pipeline for training/inference.

#figure(
  image("diagrams/tree.drawio.svg", width: 92%),
  caption: [
    Tree of encoding modules. Typed leaves emit parcels, context nodes aggregate, and outputs route upward.
  ],
)

== 3.1 Structure as a First-Class Configuration

A model is declared in YAML. Contexts specify `context_size`, `n_layers`, `n_heads`, and `n_outputs`; leaves specify datatype-specific request parameters and `jmespath` queries.

A minimal example:

#text(tree-list[
  - Observation
    - X (type: number)
])

A nested example:

#text(tree-list[
  - Customer
    - Account
      - Transaction
        - Amount (type: number)
        - Type (type: category)
      - Statement
        - Date (type: dateparts)
])

The structure compiler materializes nodes in a module map keyed by addresses such as `customer/account/transaction/amount`.

== 3.2 Addresses and Heritage

Every node has an address and each leaf has a heritage path (leaf -> ... -> root). Heritage is the decoding context for that leaf.

This explicit addressing model is critical for:

- deterministic routing,
- pruning by address,
- selective embedding export,
- loss accounting per field.

= 4. Core Architecture

== 4.1 Leaf Encoding

Each active leaf field receives a `TensorField` instance produced by its plugin. The plugin embedder outputs a parcel with:

- payload tensor,
- `origin` (leaf address),
- `destination` (parent context address).

Fields in the session-level permanent prune set are skipped during embedding.

== 4.2 Context Aggregation

Context nodes run bottom-up by depth:

- concatenate incoming child payloads,
- apply stacked rotary self-attention encoder layers,
- apply learned-query cross-attention pooling,
- emit a new parcel to the parent context.

This design combines interaction modeling (self-attention) with controllable compression (`n_outputs`).

#figure(
  image("diagrams/node.drawio.svg", width: 82%),
  caption: [
    Context node internals with transformer interaction and learned-query pooling.
  ],
)

#figure(
  image("diagrams/modules.drawio.svg", width: 96%),
  caption: [
    Example instantiated architecture (TaxML) showing concrete contexts and leaves.
  ],
)

== 4.3 Parcel Routing and Decoding

Forward computation has three stages:

1. leaf embedding,
2. context encoding,
3. leaf decoding from heritage parcels.

Each leaf decoder consumes the stacked parcel sequence along its heritage path, excluding permanently pruned nodes.

== 4.4 Loss Aggregation

Loss is computed only for trainable targets (values touched by masking/pruning logic). For each leaf:

- plugin loss computes field-specific reconstruction loss,
- field `weight` scales contribution,
- global loss is the sum across active leaves.

Field-level metrics (for example accuracy, MAE, RMSE) are logged during execution.

== 4.5 Optimization and Scheduling

Training uses `AdamW` with parameter-group decay rules and a warmup + cosine decay schedule. Session settings configure:

- learning rate,
- weight decay,
- warmup ratio,
- minimum LR ratio,
- trainer limits and early stopping behavior.

= 5. Modular Field Management

== 5.1 Plugin Contract

Each field type is a plugin that must provide:

- `Request`,
- `TensorField`,
- `Embedder`,
- `Decoder`,
- `loss`,
- optional `write`.

Plugins are auto-discovered from `src/json2vec/tensorfields/extensions/`.

This keeps the core model stable while letting datatypes evolve independently.

== 5.2 TensorField Responsibilities

`TensorField` implementations are responsible for:

- tensorizing raw query outputs,
- handling ragged nested values,
- preserving state tokens (`valued`, `null`, `padded`, `masked`, `pruned`),
- recording targets for supervised reconstruction when values are masked/pruned,
- producing an `empty` field for fully pruned cases.

This is where ndarray-style and ragged input support is anchored in the system.

== 5.3 Implemented Field Types

=== Number

`number` fields currently implement:

- padding/tensorization from nested numeric inputs,
- online mean/variance normalization,
- Fourier feature projection (`n_bands`, `offset`, optional jitter),
- joint decoding of state logits and scalar content,
- configurable content objectives (`mae`, `mse`, `huber`).

=== Category

`category` fields implement:

- online vocabulary growth with special state tokens,
- direct token embedding,
- categorical decoding with optional top-k tracking,
- class reweighting through an online counter.

=== Timestamp

`dateparts` fields implement:

- optional pattern-based parsing,
- datepart extraction (`day_of_week`, `day_of_month`, `week_of_month`, etc.),
- summed datepart + state embeddings,
- state loss plus per-datepart classification losses.

=== Entity

`entity` fields implement:

- deterministic hashing for hashable scalar values (for example IPs, phone numbers, emails),
- fixed-size hash-bucket tokenization (no per-value vocabulary growth),
- categorical decoding with optional top-k tracking,
- strict validation that per-observation entity count is at least 2.

== 5.4 Shared Online State

Field plugins can maintain evolving state in training:

- vocabulary state for category,
- normalization state for number,
- class-frequency counters for reweighting.

State is serialized with checkpoints so resumed training/inference remains consistent.

== 5.5 Extension Surface

The architecture is intentionally open to additional datatypes (for example quantiles or text adapters). Media-heavy modalities (image/audio/video) are currently non-goals due to cost and different modeling constraints.

= 6. Data Pipeline

== 6.1 Streaming Design

The data pipeline is iterable and streaming-oriented, not full-materialization based.

Current stages:

- `fetch` paths,
- `read` records,
- `process` via optional shim,
- buffered shuffle,
- `batch`,
- `transform` (query + tensorization),
- `mask`,
- `prune`.

#figure(
  image("diagrams/pipeline.drawio.svg", width: 98%),
  caption: [
    Streaming pipeline stages from path listing and reading to tensorization, masking, and pruning.
  ],
)

== 6.2 Sources and Formats

Implemented sources:

- local filesystem,
- S3.

Implemented record formats:

- `ndjson`, `parquet`, `feather`, `avro`, `csv`, `orc`, `json`.

== 6.3 SHIM Processors

Dataset-level SHIM processors are registered with `@register` and executed before tensorization. They can:

- mutate fields,
- filter records,
- split/combine values,
- derive auxiliary fields.

This supports domain-specific transformations while preserving one execution path for training and serving.

== 6.4 Query and Tensorization

Each leaf request carries a `jmespath` query. During transform:

- query extracts nested values,
- `TensorField.new` converts values to typed tensors,
- plugin state can be consulted/updated.

The pipeline also performs periodic spot checks to catch consistently empty query results.

== 6.5 Masking vs Pruning

Masking and pruning are related but not equivalent:

- masking: value-level corruption for reconstruction,
- pruning: observation-level branch removal.

In the pipeline, masking is applied before stochastic pruning. Permanent address pruning can also be configured per session.

= 7. Experiment Configuration and Execution

== 7.1 YAML Composition Model

Experiments are composed from reusable YAML units:

- top-level experiment (`project`, sessions),
- dataset config,
- structure config,
- operation config.

`!include` references allow composition without duplicating definitions.

== 7.2 Sessions and Stages

A run is a sequence of sessions. Each session defines:

- task stage (`fit`, `validate`, `test`, `predict`),
- dataset + structure,
- masking/pruning rates,
- trainer arguments,
- optimization settings where applicable.

This makes pretrain/finetune/polish/refit workflows explicit and reproducible.

== 7.3 Runtime Patching

JSON patch operations can modify session config at execution time. This is used for dynamic overrides (for example dataset root updates during scheduled refits).

== 7.4 Local and Remote Execution

Execution is initiated through a prompt-driven task entrypoint and can run:

- locally,
- remotely via Flyte task environments.

Remote environments define resources (including GPU profile), cache policy, and secrets.

== 7.5 Logging and Throughput

The training loop logs per-stage metrics and includes a throughput callback that tracks observations/sec using session batch size and batch timing.

= 8. Inference, Serving, and Outputs

== 8.1 Online Inference API

The repository includes a LitServe wrapper:

- decode request -> dataset processor -> encode to tensor fields,
- optional batching,
- model forward pass,
- typed response encoding.

Serving uses the same structure + plugin stack as training, reducing transformation drift risk.

== 8.2 Prediction Writing

Batch prediction output is written to parquet through a dedicated writer callback. The output schema supports:

- original inputs metadata,
- supervised predictions,
- optional embeddings.

This format supports downstream analysis and offline evaluation pipelines.

= 9. Interpretability and Diagnostics

`JSON2Vec` supports practical interpretability via architecture design rather than post-hoc wrappers alone.

== 9.1 Multi-Resolution Embeddings

Sessions can request embedding outputs from selected addresses (`session.output`), making it possible to inspect representation geometry at leaf, context, and higher levels.

== 9.2 Pruning-Based Ablations

Because pruning is native, users can run controlled counterfactuals by removing fields/branches and measuring degradation. This provides a direct importance signal tied to model behavior.

== 9.3 Per-Field Metrics

Plugin losses and metrics are logged by address, helping localize error concentration and identify unstable or low-signal fields.

= 10. Potential Use Cases

Representative deployment patterns include:

- finance: customer/account/transaction modeling for fraud and risk,
- travel: itinerary and fare/tax structures for pricing and quality tasks,
- e-commerce: user/session/order trees for ranking and anomaly detection,
- operations telemetry: device/session/event trees for reliability and failure prediction,
- insurance and claims: policy/claim/line-item hierarchies for triage and severity.

Common task types:

- supervised prediction,
- embedding retrieval and clustering,
- anomaly detection,
- counterfactual branch ablation.

= 11. Ambitions, Constraints, and Non-Goals

== 11.1 Ambitions

- provide a reusable representation layer for nested operational data,
- reduce dependence on brittle handcrafted feature pipelines,
- improve train/serve consistency through a shared model path,
- preserve structure-aware signal in learned embeddings.

== 11.2 Current Constraints

- explicit structure declaration is required,
- plugin coverage is still early-stage (`number`, `category`, `dateparts`, `entity`),
- production hardening is ongoing (performance tuning, packaging ergonomics, broader tests).

== 11.3 Non-Goals

- general multimodal training for image/audio/video,
- replacing governance/lineage systems such as feature stores,
- automatic schema inference as the primary workflow.

= 12. Future Directions

Likely high-value extensions include:

- additional datatypes (quantile/text adapters),
- richer evaluation harnesses for pruning-based importance,
- performance work on streaming and tensorization hotspots,
- deployment packaging and operational tooling upgrades.

= Conclusion

`JSON2Vec` is a modular architecture for hierarchical structured data that combines:

- typed leaf modeling,
- context-aware attention routing,
- streaming ingestion,
- unified train/serve execution.

For teams working with deeply nested records and evolving schemas, this offers a practical alternative to repeatedly rebuilding flattening pipelines while retaining interpretability and extensibility.

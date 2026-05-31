# Home

`json2vec` is for predictive modeling on records that are not naturally flat.
Customers have transactions, orders have line items, sessions have clickstream
events, devices recur across histories, and every level can carry signal.

Most ML pipelines handle that shape by flattening it first: build rolling
aggregates, hand-pick windows, maintain feature stores, and then train a model
on one fixed row. That can work, but it makes representation a separate
engineering system. `json2vec` takes the opposite path: describe the structured
record, and the schema becomes the model.

The framework is designed for large production datasets, not just toy nested
examples: training and batch inference over billions of observations, with
throughput-oriented data paths and logging for pipelines that can process
100k+ observations per second on appropriately sized hardware.

## Core Idea

A `json2vec` schema is both a data contract and an architecture blueprint.

- Leaf fields such as `Number`, `Category`, `Set`, `Entity`, `Text`, and
  `Vector` become datatype-specific tensorfields.
- `Array` nodes become local context encoders for repeated child objects.
- Targets, masks, pruning, and embeddings are configured on the same schema
  tree.
- Prediction output is keyed by schema address, so decoded values and
  embeddings remain attached to the part of the record that produced them.

That lets one model surface support supervised prediction, self-supervised
reconstruction, embedding export, schema mutation, field importance, and serving
without rebuilding the data representation for each workflow.

## A Schema Defines A Model

```python
import json2vec as j2v

model = j2v.Model.from_schema(
    j2v.Category("customer_tier", max_vocab_size=16),
    j2v.Array(
        j2v.Category("sku", max_vocab_size=2048),
        j2v.Number("quantity"),
        j2v.Number("price"),
        name="line_items",
        max_length=32,
        embed=True,
    ),
    j2v.Category("returned", target=True, max_vocab_size=2),
    name="order",
    d_model=64,
    n_layers=2,
    n_heads=4,
    embed=True,
)
```

This model reads records shaped like:

```python
{
    "customer_tier": "gold",
    "line_items": [
        {"sku": "A12", "quantity": 2, "price": 19.99},
        {"sku": "B07", "quantity": 1, "price": 45.50},
    ],
    "returned": "false",
}
```

The model learns from the order as a structured object. The `line_items` branch
has its own repeated context, `returned` is withheld and decoded as a supervised
target, and `embed=True` asks prediction to emit embeddings at configured
addresses.

## What Is Different

- **Hierarchical context encoding:** child records interact locally before their
  representation flows upward. A login session, transaction list, or order
  item list can keep its own window instead of competing inside one flat
  sequence.
- **Extensible datatypes:** each field type owns validation, tensorization,
  missing-state handling, masking, decoding, loss, metrics, and output writing.
- **Unified training roles:** `target=True`, `p_prune`, and `p_mask` all use the
  same reconstruction path, so supervised and self-supervised objectives share
  the model tree.
- **Embedding trees:** embeddings can come from the root, arrays, or selected
  leaves, making branch-level retrieval and diagnostics possible.
- **Schema evolution:** models can be updated, extended, deleted, reset, or
  temporarily overridden after construction.
- **Scale-oriented runtime:** online preprocessing state, streaming data
  modules, throughput logging, and checkpointed tensorfield state are designed
  for high-volume training and prediction jobs.
- **One path for training and serving:** queries, preprocessors, tensorization,
  model execution, prediction writing, and postprocessors stay in the same
  configured path.

## When It Fits

Use `json2vec` when relationships inside the record matter: account histories,
fraud or risk snapshots, order and fulfillment events, flight itineraries,
operations telemetry, user sessions, repeated measurements, or any mixed
datatype object where flattening would discard useful structure.

Use a simpler tabular model when flattening loses no meaningful context. The
point is not to replace every table. The point is to model nested business data
without making a feature table the only representation the model can see.

## Where To Start

- New to the package: [Getting Started](getting-started.md)
- Short API map: [AI Quickstart](ai-quickstart.md)
- Modeling rationale: [Why `json2vec`](motivation.md)
- Record shapes and query rules: [Query Paths](core-concepts/querypaths.md)
- Built-in fields: [Data Types](core-concepts/data-types.md)
- Self-supervised embeddings: [Embeddings & Self-Supervised Learning](core-concepts/embeddings.md)
- Serving-time output shaping: [Postprocessors](guides/postprocessors.md)
- Applied risk example: [Device Tenure](case-studies/device-tenure.md)

## Tutorials

The tutorials are ordered by workflow:

- **Hello World** runs the smallest supervised training loop.
- **Masked Pretraining** introduces nested arrays and self-supervised masking.
- **Nested Supervised Training** uses repeated measurement objects plus a root target.
- **Supervised Tabular Training** shows a compact flat classifier for comparison.
- **Serving** turns a saved model into a deployment wrapper.

## Community

Join the [`json2vec` Discord](https://discord.gg/DVyZUkvTFA) for questions,
design discussion, and release notes.

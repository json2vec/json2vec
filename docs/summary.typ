= JSON2Vec: Hierarchical Modeling for Semi-Structured Data

`JSON2Vec` is an open-source framework for learning representations directly from nested records (JSON, parquet-like structures, and similar formats) without a feature-engineering pipeline.

The goal is simple: model the structure that already exists in the data instead of flattening it into a table and rebuilding context by hand.

= Core Ideas

== 1. Hierarchical Contexts

Data is defined as a tree of contexts and fields. Each node in that tree has an address, and each address maps to a trainable module.

Leaf fields are embedded first. Their embeddings are routed upward and pooled at parent contexts. This produces intermediate representations at multiple levels (field, sub-context, root), not just a single final vector.

This matters because many real datasets are hierarchical by construction:
- customer -> account -> transaction
- itinerary -> leg -> segment
- order -> shipment -> event

Flattening removes this structure. JSON2Vec keeps it.

Contexts may have multiple child fields and sub-contexts. The model learns how to pool those children into a single embedding, which is then routed up to the next level. A customer may have multiple accounts, each with multiple transactions and multiple monthly statements and multiple clickstream login sessions, and each login session may have multiple events. The model learns how to pool each of those levels, and the final customer embedding is informed by all of that context.

== 2. Featureless Modeling

The framework does not require a static feature store or handcrafted derived columns before training.

Instead, raw values are encoded on the fly by datatype-specific tensor fields. Masking, numerical normalization, and tokenization happen inside the model pipeline. The model learns how to use each field from data, rather than relying on manually curated feature logic.

The data may be queried and processed in batches from any data source (files, databases, APIs, etc.), any data format, and any data structure via `jsonpath`-like querying syntax.

This reduces coupling between:
- offline preprocessing
- online inference logic
- schema evolution over time (adding / removing / modifying fields or contexts)

Additional wrangling logic (mutations, filters, sampling, etc.) may be encoded in a `shim` function registered to a dataset as a plugin. This allows for custom loading and processing without batch preprocessing.

The exact same logic is used for both batch training and batch / real-time inference, so there is no risk of training-serving skew from different feature pipelines.

== 3. Modularity via Datatype Plugins

Datatypes are implemented as plugins. A plugin defines the components needed for one field type:
- field definition and configuration (`Request`)
- tensor field implementation (`TensorField`)
- input embedder (`Embedder`)
- embedding decoder (`Decoder`)
- loss function (`loss`)
- writer (optional) (`write`)

Built-in types can cover common primitives (category, number, dateparts, entity, vector), and new types can be added by registering another plugin. This keeps the core architecture stable while allowing domain-specific extensions.

Plugins may be authored to leverage additional state. For example, the category plugin can maintain a global vocabulary that is shared across all nodes and updated during training. This allows it to handle new categories without manual intervention.

== 4. Scalability as a First-Class Constraint

The data path is streaming and iterable. Batches are read, processed, encoded, and trained without requiring full dataset materialization in memory.

The architecture is designed for:
- large observation counts
- multi-worker loading
- distributed training/inference environments

The important point is operational: the same architecture is used for both representation learning and downstream prediction tasks, so deployment does not require a separate handcrafted encoding system.

== 5. Explainability from Intermediate Embeddings

Because each node in the tree has an explicit embedding, analysis can be done at multiple resolutions:
- compare two observations at root level
- localize where they diverge in child contexts
- inspect distances between node embeddings along the tree

This provides a practical explanation path: not only "these two samples are far apart," but also "which branches in the hierarchy account for that distance."

Combined with feature masking/pruning experiments, this supports interpretable diagnostics without treating the model as a single opaque vectorizer.

= Scope

JSON2Vec is not a general replacement for all modeling workflows. It is a focused approach for datasets where:
- structure is nested and meaningful
- schema changes are frequent
- manual feature engineering is a bottleneck

For that class of problems, it offers a direct, modular, and inspectable alternative.

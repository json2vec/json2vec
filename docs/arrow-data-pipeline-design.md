# Arrow-Native Pipeline Design And Roadmap

- Status: Implemented Arrow foundation plus explicitly deferred roadmap
- Date: 2026-09-02
- Audit baseline: current `dev/awkward-arrays` implementation
- Scope: Ingestion, preprocessing, querying, sampling, shuffling, sharding,
  batching, coalescing, plugin preparation, prediction writing,
  postprocessing, persistence, and deployment serialization
- Related designs:
  [Awkward Array Preprocessing](awkward-preprocessing-design.md) and
  [Unified Mask](unified-mask-design-spec.md)

This spec supersedes the persistent Awkward `RaggedField`, Python-row dataset,
JMESPath, and plugin-boundary decisions in the Awkward design. That document
remains the record of the first coalescing implementation and its loop audit.
The `"<MASK>"` data-literal transport is removed: that string is ordinary
content. The unified masking spec supersedes every mask-selection, skipping,
external-selection, and reconstruction-objective proposal in this document.
Those changes are not shipped by this Arrow implementation.

## Implementation Status

This document records both the implemented foundation and the larger target
architecture. The rendered user guides are the authoritative current API. Use
this legend when reading the detailed design below:

| Status | Meaning |
| --- | --- |
| **Shipped** | Implemented and documented as a current public contract. |
| **Limited** | Implemented only with the constraints listed here. |
| **Roadmap** | Normative design intent or acceptance work; not a callable API yet. |

**Shipped:**

- `rf.Batch` with aligned Arrow data and identity, including `slice`, `take`,
  `filter`, `replace`, `order`, `expand`, `explode`, and ordered `group`;
- one Arrow loader behind `ArrowDataModule`, `PolarsDataModule`,
  `CustomDataModule`, and `SyntheticDataModule`;
- in-memory Arrow, `pyarrow.dataset.Dataset`, and restartable Arrow factory
  sources, plus bounded mapping conversion in the custom and synthetic
  adapters;
- Arrow `Preprocessor` and same-row `Postprocessor` contracts;
- the node-relative structural query grammar and Arrow-backed `RaggedField`;
- plugin-declared value families, Arrow codecs, stable output types, and Arrow
  writers;
- `Model.predict(...) -> pa.Table`, `Model.write(...) -> rf.Batch`, direct
  Parquet `Writer`, and one terminal Python conversion in JSON deployment.

**Limited in the current implementation:**

- one process and `num_workers=0`; distributed ownership and DataLoader workers
  fail explicitly;
- sampling is without replacement; `replacement=True` fails explicitly;
- Dataset and factory shuffle use a row-bounded mixer. There is no byte ceiling;
- Dataset-scoped preprocessing is global for finite Tables and Datasets. It is
  rejected for callable factories until coordinated materialization exists;
- `requires` and `produces` are validated dependency metadata, but `requires`
  does not yet drive Dataset Scanner projection pushdown;
- branch and leaf queries bind and cache by expression/type/address. Each active
  branch layout and overflow decision is materialized once and reused by its
  descendants;
- the data-module seed controls sampling and row order. Legacy Torch masking,
  pruning, branch masks, and unavailable simulation are not identity- or
  chunk-invariant;
- `pin_memory` is accepted for loader compatibility, but the encoded carrier
  does not currently expose recursive PyTorch pinning.

**Roadmap:** coordinated source caches and indices, projection pushdown,
replacement sampling, worker/rank scheduling, equal-step distributed plans,
byte-bounded shuffling, and the separately specified unified masking design.
Any later section that describes one of these mechanisms is a target design,
not a statement that it ships. The implementation plan,
required tests, benchmark plan, and acceptance criteria are forward-looking
unless an item is explicitly present in the shipped list above.

## Decision Summary

Arrow owns RelFlow's CPU data plane from ingestion through prediction output.

- Polars is an input adapter only. A `DataFrame` is converted to Arrow once;
  no writer, postprocessor, callback, or deployment runtime imports Polars.
  PyArrow remains a core dependency; Polars moves to an optional ingress extra.
- RelFlow exposes four Lightning data modules but only one loader implementation:
  `ArrowDataModule` is the canonical implementation; `PolarsDataModule`,
  `CustomDataModule`, and `SyntheticDataModule` are ingress-only adapters.
- File-backed datasets emit Arrow record batches directly.
- `ArrowDataModule` never accepts Python records. `CustomDataModule` and
  `SyntheticDataModule` convert bounded mapping batches at ingress. For small
  interactive calls only, `Model.predict(...)` accepts a nonempty mapping
  sequence and converts it to one Arrow table; empty input requires a typed
  Arrow object.
- Dataset stages exchange a thin `Batch` carrier whose per-observation data and
  lineage are Arrow tables and arrays. Static schema and execution plans remain
  ordinary Python metadata.
- Awkward may be used as a transient view inside a user preprocessor. It is not
  used by coalescing and is not a dataset interchange format.
- Torch begins only at the tensorfield codec boundary. NumPy is used for
  vectorized lineage/scheduling kernels and as a short-lived contiguous bridge
  between Arrow and Torch; it never becomes a row carrier.
- Plugin `write` hooks emit typed Arrow arrays. `Model.write` returns an
  identity-aligned `Batch`, and postprocessors transform that same carrier.
- `Writer` sends Arrow directly to Parquet. Deployment converts Arrow to Python
  exactly once only when a JSON response requires it.

RelFlow keeps `query`, but removes JMESPath. `query` becomes a small,
node-relative, Arrow-native path language for structural selection. It supports
struct fields, list traversal, list indexing and slicing, quoted members, and
map lookup. It deliberately excludes filters, functions, expressions, joins,
and sorting. Those operations belong in preprocessors.

Training data is shuffled by default. An in-memory post-preprocessor table
receives a full permutation each epoch. Dataset and factory sources use a
bounded cross-chunk Arrow shuffle. Sampling and shuffling operate on logical
observations after preprocessing and before model batches are formed.

There is one pipeline:

```text
adapt → scan → align → preprocess
      → sample → shuffle → batch → coalesce(query + geometry)
      → plugin → tensor → mask → forward
      → write Arrow → postprocess → persist or serialize
```

The roadmap inserts physical projection planning, indexing, and distributed
logical-owner scheduling without changing these phase boundaries.

Direct selection, explicit queries, in-memory data, streaming data, interactive
prediction, callbacks, and deployment do not fork into separate data or output
implementations.

## Motivation

Before this redesign, Arrow-backed sources were converted to Python rows before
tensorization:

- `data/datasets/polars.py` uses `to_dicts()` and
  `iter_rows(named=True)`;
- `data/datasets/streaming.py` receives Arrow `RecordBatch` objects and calls
  `to_pylist()`;
- `data/iterables.py` samples, shuffles, and batches Python observations;
- `data/ragged.py` recursively rebuilds value and state trees before creating
  Awkward arrays;
- explicit JMESPath queries require Python objects and independently traverse
  them for each queried leaf;
- plugin writers emit NumPy arrays, object arrays, dictionaries, and recursive
  candidate lists;
- `Model.write` recursively squeezes tensors and materializes Python lists;
- postprocessors transform mutable mappings with runtime-specific Python
  providers;
- the prediction callback rebuilds one Polars frame per address, concatenates
  them, and converts the result back to Arrow only for Parquet;
- deployment recursively splits those collections and, when a postprocessor is
  present, repeats model writing once per request instead of once per batch.

The dataset surface repeats the same loader around different ingress forms.
`CustomDataModule`, `PolarsDataModule`, and `StreamingDataModule` duplicate
split normalization, model references, worker configuration, preprocessing,
sampling, shuffling, batching, and encoding. `SyntheticDataModule` only wraps a
generator in the custom module. The duplication has also allowed replacement,
sharding, and shuffle defaults to diverge by source type.

This pays for columnar decoding, discards the columnar representation, creates
large Python object graphs, and then reconstructs columnar buffers. It also
makes direct selection and queries behave like unrelated features.

The output path pays the same cost in reverse. It also makes schema depend on
runtime values: singleton dimensions are recursively squeezed, Set labels can
become dynamic field names, and an empty or all-null first batch cannot reliably
declare the later Parquet schema.

The current training defaults add a correctness risk: a shuffle buffer of one
does not shuffle. Large contiguous runs of one entity, time period, source, or
target can therefore dominate consecutive optimization steps and contribute to
collapse. Streaming replacement sampling is also file-based while Polars
replacement sampling is record-based, so the statistical unit varies with the
source adapter.

## Goals

- Preserve Arrow buffers through ingestion, preprocessing, selection, and
  batching, then restore predictions to Arrow without Python rows.
- Make Arrow the only persistent CPU data contract.
- Give direct bindings and explicit queries one projection engine.
- Replace source-specific loader implementations with one Arrow loader while
  retaining Polars, custom, and synthetic inputs as focused ingress adapters.
- Resolve complex list, fixed-size-list, struct, and map layouts without
  requiring a preprocessor for simple structural navigation.
- Let preprocessors filter, derive, join, explode, regroup, and emit zero or
  many observations while returning to one Arrow contract.
- Make training randomization safe by default, reproducible, distributed, and
  independent of incidental scanner chunking.
- Keep datatype compatibility and normalization owned by plugins.
- Keep datatype-specific prediction schemas and writing owned by plugins while
  consolidating shared state, inferred, embedding, and shape handling.
- Give interactive prediction, postprocessors, callbacks, Parquet writing, and
  deployment one identity-aligned Arrow output contract.
- Preserve bounded memory for observations containing thousands of nested
  values.
- Remove automatic Python row materialization from the standard path.
- Isolate Polars to its ingress adapter and JSON-compatible Python values to an
  explicit external serialization boundary.

## Non-Goals

- Recreating all of JMESPath, JSONPath, or SQL.
- Supporting query filters, arithmetic, functions, pipes, recursive descent,
  unions, sorting, joins, or computed values.
- Making arbitrary Python objects part of the canonical dataset format.
- Claiming zero-copy through padding or dense tensor construction. Ragged to
  dense conversion necessarily allocates.
- Claiming zero-copy from GPU tensors to CPU Arrow buffers.
- Designing Mask selection, `skip`, `dropout`, `reconstruct`, or presence
  semantics; the unified masking spec owns that work. Arrow remains the CPU
  selection plane and Torch remains the model execution plane.
- Guaranteeing identical order after the world size or source snapshot changes.
- Maintaining permanent Arrow and Python dataset engines.
- Reimplementing filesystem discovery, file-format inference, or scan planning
  already owned by `pyarrow.dataset`.
- Eliminating small Python control structures used for plugin dispatch,
  configuration, exceptions, and HTTP routing.
- Making JSON or Pydantic operate directly on Arrow. They remain explicit
  deployment-edge conversions.
- Building a general sink framework or Arrow Flight service in the first
  implementation.

## Format Ownership

Each library has one role:

| Layer | Owner | Contract |
| --- | --- | --- |
| Ingress | PyArrow or Polars/custom/synthetic adapters | Produces Arrow |
| Dataset transport | RelFlow `Batch` | Arrow table plus aligned Arrow identity |
| Structural query | Arrow | Preserves offsets, validity, and coordinates |
| Ragged regularization | Arrow with NumPy offset/placement geometry | Produces `RaggedField` |
| Datatype conversion | Plugin | Consumes whole Arrow leaf columns |
| Model runtime | Torch | Dense content, state, targets, and routing presence |
| Prediction conversion | Plugin plus shared writer | Produces typed Arrow arrays |
| Postprocessing | RelFlow `Batch` | Arrow table plus aligned identity |
| Persistence | PyArrow | Writes Arrow directly to Parquet |
| HTTP JSON edge | Deployment serializer | One explicit Arrow-to-Python conversion |

An Awkward array created inside a user preprocessor is a computational view,
not another pipeline representation. It must not cause Python materialization,
and it must return to Arrow before leaving the preprocessor.

## Canonical Arrow Units

### Batch

Every stage after adaptation exchanges one carrier:

```python
@dataclass(frozen=True, slots=True)
class Batch:
    data: pa.Table
    identity: pa.Array | pa.ChunkedArray
```

`identity` has exactly one non-null struct value per logical row:

```text
struct<
    logical: fixed_size_binary[32] not null,
    instance: fixed_size_binary[32] not null,
    order: large_binary not null
>
```

`logical` uniquely identifies a post-preprocessor observation in the split.
`instance` equals `logical` without replacement and uniquely identifies a draw
when replacement repeats that observation. `order` is a canonical,
lexicographically sortable processed-source position used for stable merges.
Duplicate `logical` values after preprocessing are an error. Deliberate
preprocessor filtering or reordering establishes a new processed order; ordered
prediction does not undo it to recover raw source order.

Slicing, filtering, taking, shuffling, and batching always transform data and
identity together. Static configuration, bound schemas, and query plans are
intentionally not forced into Arrow.

`Batch` is an alignment invariant, not a second data engine. Its dynamic
contents remain Arrow-native and it never exposes Python observations.

`len(batch)` is defined by `identity`. Every present data column must have that
length. Arrow cannot record a positive row count in a zero-column table, so
`data.num_rows` may be zero when `data.num_columns` is zero even though the
batch is not empty. `Batch` operations handle that case from identity and
validate length as soon as a column is added. This avoids a hidden sentinel
column and still permits reconstruction-only prediction with no source
projection.

The same carrier crosses the prediction boundary. Before the model,
`Batch.data` contains processed inputs. After `Model.write`, it contains the
canonical output envelope while retaining the same identity. Preprocessors and
postprocessors therefore share lineage operations without sharing data
semantics.

### RecordBatch

A `pa.RecordBatch` is the physical transport unit emitted by a scanner. Its
columns contain one non-chunked Arrow array rather than a `ChunkedArray`.
Scanner batch size controls I/O and working memory; it does not define model
batch size or statistical order.

### Table

A `pa.Table` is the logical data payload inside `Batch`. It may contain zero,
one, or many chunks. Exact-size model batches can be assembled from zero-copy
`RecordBatch.slice(...)` operations and `Table.from_batches(...)` without
globally combining source chunks.

Coalescing operates chunk-wise. It never calls `combine_chunks()` on the whole
table. A plugin may combine only its selected leaf column when its external
codec requires one contiguous buffer.

### Observation

One Arrow row is one processed RelFlow observation. The generated root
singleton is implicit in the table and is introduced only as model geometry
during coalescing.

A repeated branch is represented by Arrow list-of-struct storage:

```text
line_items: large_list<
    struct<
        sku: string,
        quantity: double,
        discounts: large_list<struct<amount: double>>
    >
>
```

`list`, `large_list`, and `fixed_size_list` are accepted. Source offsets remain
source structure; configured branch `length` remains model capacity. The
dataset does not pad source lists.

### Identity

Every source row carries a stable Arrow-native identity that is framework
metadata, not a modeled field.

- In-memory identity derives from split identity and original row offset.
- File-backed identity derives from canonical fragment identity, row group or
  stable row range, and source offset.
- A one-to-many preprocessor output derives `logical` and `order` from its
  parent identity and an explicit stable output ordinal.
- Replacement retains `logical` and assigns a separate unique `instance` for
  every draw.

Identity is the protected Arrow sidecar in `Batch`, not a user column.
Filtering, taking, slicing, exploding, and rebatching apply to records and
identity together. It is excluded from modeled projections but retained
through model masking and any distributed ordered write.

Stable identity makes sampling membership and shard ownership independent of
record payloads and incidental scanner batch boundaries. It is excluded from
ordinary interactive and JSON output. `Writer` persists it as a protected
top-level Arrow column so distributed shards remain joinable and orderable.

## Public Data Modules

RelFlow has four public Lightning data modules. They share one loader
implementation; the distinction exists only at the input boundary.

The four classes and their basic source adapters below are shipped. References
in this section to coordinated caches, source indices, multiple workers/ranks,
or byte-bounded shuffling describe the roadmap. The current constructor surface
is shown under [Constructor Contract](#constructor-contract).

The root package exports `rf.ArrowDataModule`, `rf.PolarsDataModule`,
`rf.CustomDataModule`, `rf.SyntheticDataModule`, and the `rf.Batch` carrier
needed by Arrow preprocessors and identity-aware factories. Dataset
implementation classes and loader functions remain internal.

### ArrowDataModule

`ArrowDataModule` is the canonical data module and owns preprocessing,
identity, sampling, scheduling, shuffling, batching, query execution,
coalescing, plugin dispatch, and prediction metadata.

```python
import pyarrow.dataset as ds
import relflow as rf

train = ds.dataset("data/train", format="parquet")
validate = ds.dataset("data/validate", format="parquet")

datamodule = rf.ArrowDataModule(
    model=model,
    train=train,
    validate=validate,
    preprocessor=prepare,
    seed=42,
)
```

Each named split accepts one `ArrowSource`:

```python
ArrowUnit = rf.Batch | pa.Table | pa.RecordBatch
ArrowStream = pa.RecordBatchReader | Iterable[ArrowUnit]
ArrowSource = ArrowUnit | ds.Dataset | Callable[[], ArrowStream]
```

The source rules are strict:

- `Batch`, `Table`, and `RecordBatch` are finite and indexable.
- `Dataset` is restartable and scanned without Python row conversion.
- A callable is a restartable source factory. It is invoked once per source
  iterator and returns a `RecordBatchReader` or iterable of Arrow units.
- A user-produced `rf.Batch` may supply stable identity. Otherwise RelFlow
  assigns identity from the source and logical position.
- A factory is currently single-reader: it requires `world_size=1` and
  `num_workers=0`. Coordinated cache/index support for multiple ranks or workers
  is roadmap work.
- An uncached repeatable factory must emit the same observations in canonical
  order on every invocation. Identified batches must additionally have
  monotonic `identity.order`. RelFlow trusts this source contract because it
  cannot prove repeatability without materializing the stream.
- An infinite training factory requires `epoch_size`, `replacement=False`, one
  process, and `num_workers=0`. Validation, test, and prediction factories must
  terminate. Replacement requires a finite indexed eligible population, so it
  rejects an infinite factory.
- Bare readers and iterables are rejected because RelFlow cannot know whether
  they can restart for the next epoch or worker.
- A configured `Scanner` is rejected because it prevents RelFlow from owning
  projection and scan planning; pass its source `Dataset` instead.
- Paths, Polars objects, and Python mappings are rejected. Construct a Dataset,
  use `PolarsDataModule`, use `CustomDataModule` for an `IterableDataset`, or
  use `SyntheticDataModule` for a mapping generator. Advanced producers may
  convert directly to Arrow in a factory.

Python producers remain possible, but conversion is explicit and local:

```python
def generate():
    for rows in source():
        yield pa.RecordBatch.from_pylist(rows, schema=source_schema)

datamodule = rf.ArrowDataModule(model=model, train=generate)
```

The source must produce one declared, stable Arrow schema. Schema inference is
never repeated independently for each chunk, so query validity cannot depend
on batch size. An iterable factory that has no rows must still yield an empty
Arrow unit carrying its schema, or return a `RecordBatchReader` whose schema is
known before iteration.

File discovery, filesystems, formats, source filters, and partitioning belong
to `pyarrow.dataset`. RelFlow does not retain `root`, `suffix`, regex pattern,
or format-specific reader arguments. A configured `Dataset` carries those
decisions into `ArrowDataModule`.

The current Dataset path reads the supplied Dataset schema without RelFlow
projection pushdown. `requires` is already validated metadata; using it to build
the Scanner projection, and attaching stable fragment/range planning, remain
roadmap work. Arrow-supported sources still avoid `to_pylist()` in the RelFlow
pipeline. A factory-produced reader remains the advanced escape hatch and must
contain every active source dependency.

### PolarsDataModule

`PolarsDataModule` accepts `pl.DataFrame` splits, converts each one exactly once,
and delegates to `ArrowDataModule`:

```python
datamodule = rf.PolarsDataModule(
    model=model,
    train=train_frame,
    validate=validate_frame,
)
```

The retained values are Arrow tables. RelFlow does not call Polars again during
iteration, and `PolarsDataModule` has no dataset, loader, sampling, or sharding
implementation of its own. A `LazyFrame` must be collected explicitly before
construction so the materialization boundary stays visible. Its typed
constructor mirrors `ArrowDataModule`; internally it replaces each non-null
frame with `frame.to_arrow()` and calls `super()` once.

### CustomDataModule

`CustomDataModule` retains its current conceptual role: each named split is a
restartable PyTorch `IterableDataset` that yields one Python mapping per
observation.

```python
datamodule = rf.CustomDataModule(
    model=model,
    train=events,
    arrow_schema=event_schema,
)
```

Its only implementation is bounded ingress adaptation:

```text
IterableDataset mappings
  → bounded mapping batch
  → pa.RecordBatch.from_pylist(...)
  → shared Arrow loader
```

Both Python-facing adapters normalize to one internal restartable source:

```python
MappingSource = Callable[[], Iterator[Mapping[str, Any]]]
```

`CustomDataModule` obtains it from `dataset.__iter__`;
`SyntheticDataModule` receives it directly. One shared `adapt` operation turns
that source into a restartable Arrow factory. The adapter follows these rules:

- every yielded observation is a mapping; for an `IterableDataset` that yields
  Arrow units, pass `dataset.__iter__` as the `ArrowDataModule` source instead;
- `ingress_rows` bounds the temporary Python list used for each conversion;
- `arrow_schema` may be one schema or a per-stratum mapping;
- without `arrow_schema`, RelFlow infers once per configured split from its
  first non-empty ingress batch, persists the schema across epochs and in any
  cache manifest, and validates every later batch exactly;
- an empty source, ambiguous all-null field, late field, or incompatible type
  fails with an instruction to provide `arrow_schema`;
- missing mapping keys and explicit `None` both become Arrow null;
- the dataset must return a fresh iterator and stable order on every pass.

The shared Arrow engine enforces the source capabilities. A custom source is
currently single-reader. Coordinated finite caching for multiple workers or
ranks is roadmap work. Infinite training uses the single-reader stream-cutoff
contract; validation, test, and prediction must terminate.

`CustomDataModule` never creates a second DataLoader and never inspects rank or
worker state. It does not sample, shuffle, form model batches, query, mask, or
encode. After `from_pylist`, its output follows exactly the `ArrowDataModule`
path.

### SyntheticDataModule

`SyntheticDataModule` accepts restartable zero-argument generator functions
that yield Python mappings:

```python
def generate():
    for index in range(10_000):
        yield {"id": index, "value": float(index)}

datamodule = rf.SyntheticDataModule(
    model=model,
    train=generate,
    arrow_schema=synthetic_schema,
)
```

Each configured function is called once for every new source iterator and
passed to the same `adapt` operation as the custom source.
`SyntheticDataModule` owns no loader, conversion, scheduling, or encoding
implementation of its own.

The generator owns the reproducibility of generated values. An uncached
generator must produce the same schema, values, and canonical order on every
restart. The module's `seed` controls downstream sampling and shuffling; it does
not seed the generator or Torch masking. A stochastic generator must seed
itself explicitly. A coordinated finite snapshot for multiple workers or ranks
is roadmap work; an infinite training generator requires `epoch_size` and
remains single-reader. `arrow_schema` and `ingress_rows` have the same meaning
as in `CustomDataModule`.

### Constructor Contract

All four modules expose named `train`, `validate`, `test`, and `predict`
arguments, and at least one must be present. Each module applies one input type
to every named split:

| Module | Split type |
| --- | --- |
| `ArrowDataModule` | `ArrowSource` |
| `PolarsDataModule` | `pl.DataFrame` |
| `CustomDataModule` | `IterableDataset[Mapping[str, Any]]` |
| `SyntheticDataModule` | `Callable[[], Iterator[Mapping[str, Any]]]` |

There is no second source-map alias such as `datasets=` or `dataframes=` and no
behavior that silently installs one source into every stratum. Per-stratum
option mappings remain valid where documented below.

The canonical shape is:

```python
rf.ArrowDataModule(
    model,
    *,
    train=None,
    validate=None,
    test=None,
    predict=None,
    preprocessor=None,
    retain=(),
    seed=0,
    shuffle=None,
    sample=1.0,
    replacement=False,
    epoch_size=None,
    shuffle_rows=None,
    drop_last=False,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)
```

Shared configuration uses the same names and semantics for all four modules:

| Option | Meaning |
| --- | --- |
| `preprocessor` | Arrow `Batch` transform; scalar or per-stratum mapping |
| `retain` | Top-level processed columns retained for output postprocessing |
| `seed` | Root seed for operation-keyed deterministic randomness |
| `shuffle` | Defaults to true for train and false otherwise |
| `sample` | Logical-observation Bernoulli probability |
| `replacement` | Reserved; `True` currently fails explicitly |
| `epoch_size` | Maximum logical observations emitted per pass; callable sources admit the first N eligible rows before bounded shuffle |
| `shuffle_rows` | Streaming shuffle output quantum; the mixer may hold twice this many rows |
| `drop_last` | Drop an incomplete final model batch |
| `num_workers` | Reserved; currently must be zero |
| `persistent_workers` | Reserved; currently must be false |
| `pin_memory` | Accepted for loader compatibility; recursive encoded-carrier pinning is deferred |

Scalar values apply to every stratum. A mapping overrides named strata. The
statistical options live in the shared Arrow engine, never in a source adapter.
`model.batch_size` remains the single model-batch authority.

`retain` accepts a tuple of unique top-level column names, `"*"`, or, on a data
module, a per-stratum mapping of either form. Tuple order becomes Arrow field
order. `"*"` means every post-preprocessor column in schema order and forces a
full source scan unless declared preprocessor outputs prove that a column is
derived. A mapping must contain every configured split exactly once after
stratum normalization; missing or extra keys are errors. Direct
`Model.predict(...)` and deployment run only in predict strata and accept the
scalar forms, not a mapping.

The default is `()`: output does not duplicate modeled source data unless
requested. A retained name is projected from the source unless a preprocessor
declares it in `produces`. Every explicit name is then bound against the
post-preprocessor schema; a missing name fails with stratum, preprocessor, and
column context. Retained values travel in the processed `Batch` and appear
under the output `inputs` struct. A callback-attached postprocessor cannot
retroactively request a column that its data module did not retain.

`CustomDataModule` and `SyntheticDataModule` additionally expose only
`arrow_schema=None` and `ingress_rows=4096`. `arrow_schema` accepts one
`pa.Schema`, a per-stratum mapping of schemas, or `None`; `ingress_rows` is one
positive integer for the module. These configure adaptation and are not loader
or scheduling options.

Filesystem, format, source filters, and partitioning remain with the supplied
Dataset. RelFlow chooses ordinary scanner batch size and readahead defaults; an
advanced source can control them by returning a configured reader from its
factory. RelFlow exposes no ordinary `sharding`, `chunk_batch_size`,
`file_buffer_size`, or rank/worker identity arguments. It derives logical
scheduling from Lightning and stable identities; physical-read tuning may
affect I/O cost but never statistical membership or order.

**Roadmap:** when caching is required, one coordinator materializes one source snapshot and
publishes its manifest atomically before a barrier releases readers. Local
ranks may share a managed local cache; multi-node execution requires a cache
location visible to every rank. If no shared location is available, RelFlow
fails instead of materializing once per rank. `cache=False` always fails when a
requested operation requires an index; `cache=None` permits automatic caching
only for a source with known finite cardinality. An unknown-length custom or
synthetic source may run under the uncached single-reader contract, but any
operation or topology requiring an index needs an explicitly requested,
finite `cache=True` snapshot. Setting `cache=True` asserts that the source
terminates.

### Removed Dataset APIs

RelFlow retains four public module names and their distinct use cases over one
Arrow loader. Custom and synthetic constructor details become the focused
contracts above; mapping observations are converted in bounded ingress batches,
with `arrow_schema` available when inference is ambiguous. Among public data
module classes, only `StreamingDataModule` is removed, without an alias:

| Previous use | Replacement |
| --- | --- |
| Local or S3 files | Build a `pyarrow.dataset.Dataset`; pass it to `ArrowDataModule` |
| Custom Python `IterableDataset` | Use `CustomDataModule` |
| Synthetic mapping generator | Use `SyntheticDataModule` |
| Custom Arrow factory | Use `ArrowDataModule` |
| In-memory Arrow | Pass the table or record batch directly |
| In-memory Polars | Use `PolarsDataModule` |

The public `BatchDataset`, `CustomBatchDataset`, `PolarsBatchDataset`, loader
functions, `DatasetMap`, `DataFrameMap`, and file `Suffix` configuration are
removed as well. The current private synthetic dataset wrapper is replaced by
direct adaptation. `ShardingStrategy` is removed because physical planning is
no longer a statistical user option. One unexported `ArrowDataset` feeds
Lightning's `DataLoader`. The old dataset `base.py` and row `Pipeline`
disappear after genuinely shared Arrow types move to `data/arrow.py`.
`RawObservation`, `ProcessedObservation`, and `EncodedBatch` disappear with the
row carrier; `Batch` and the post-plugin tensor types replace them.

## Pipeline Order

The order is semantic and fixed:

```text
source manifest
  → assign stable source identities
  → plan deterministic physical reads
  → scan Arrow record batches
  → align compatible source schemas
  → run and materialize dataset-scoped preprocessors, when configured
  → run partition-scoped preprocessors
  → derive logical output identities
  → materialize an index when the requested schedule requires one
  → sample logical observations
  → inherit physical ownership or assign an indexed logical schedule
  → shuffle selected observations
  → form model batches
  → coalesce node queries, branch geometry, and leaf state
  → run plugin codecs
  → apply legacy Torch mask and prune policies
```

That final ordering records the audit-baseline implementation. The unified
masking spec replaces it with Arrow-side selection and split input/target
projection before plugin conversion.

Consequences:

- sampling treats each preprocessor output as an independently eligible
  logical observation;
- discarded observations consume no shuffle-buffer memory;
- default randomization reduces the risk that batching preserves contiguous
  source runs;
- queries see preprocessed data but do not change dataset membership or order;
- clipped values cannot update a vocabulary, counter, tokenizer, or normalizer.

## Arrow-Native Query Language

### Decision

Remove the JMESPath dependency and retain the user-facing name `query`.

`query` is a structural path, not a general expression language. It compiles to
Arrow operations and never materializes Python objects. Direct and explicit
bindings are plans in the same query tree.

### Node-Relative Resolution

`query` is allowed on the generated root, `Branch`, and leaf nodes.

- `Model(query=None)` selects the source row.
- `Model(query="payload")` selects a common source struct while retaining the
  generated singleton root.
- A child with `query=None` selects its own schema name from its parent value.
- An explicit child query replaces that same-name selection.
- Every explicit query is relative to the value selected by its parent.
- A repeated `Branch` query selects its collection exactly once. Every
  descendant resolves against the same elements and offsets.

For example:

```python
model = rf.Model(
    query="payload",
    d_model=64,
    n_layers=2,
    n_heads=4,
    line_items=rf.Branch(
        query="items[-32:]",
        length=32,
        sku=rf.Category(query="product.sku", size=2048),
        quantity=rf.Number(query="qty"),
    ),
    returned=rf.Category(query="outcome", mask=True, size=2),
)
```

The branch owns the shared slice. `sku` and `quantity` cannot independently
filter, compact, reorder, or shift the selected items.

### Grammar

The first version uses this grammar:

```ebnf
path       ::= first tail*
first      ::= identifier | quoted
tail       ::= "." identifier | bracket
bracket    ::= "[*]"
             | "[" [integer] ":" [integer] "]"
             | "[" literal "]"
identifier ::= letter { letter | digit | "_" }
quoted     ::= "[" JSON-string "]"
literal    ::= JSON-string | integer | boolean
```

Queries are observation-relative. They never begin with `$` or a batch/root
`[*]` selector.

| Syntax | Structural operation | Result |
| --- | --- | --- |
| `customer.name` | Struct member selection | Select nested fields |
| `["gross amount"]` | Quoted struct member | Select a non-identifier field name |
| `items[*].sku` | List traversal | Retain list offsets and select from each element |
| `legs[0].origin` | List index | Remove one list axis |
| `events[-10:]` | List slice | Retain a bounded list axis |
| `attributes["country"]` | Map lookup | Select one literal map key |
| `flags[true]` | Map lookup | Select one boolean map key |

The parser emits one neutral bracket-literal operation. Its meaning is resolved
from the bound Arrow type:

Here, list means Arrow `list`, `large_list`, or `fixed_size_list`.

- on a list, `[n]` is an index and `[start:stop]` is a slice;
- on a map, `[literal]` is a key lookup;
- on a struct, `["name"]` is a quoted member;
- `[*]` traverses exactly one list axis without flattening or compacting it.

Slices use Python's half-open, clamped semantics independently for every list:
omitted bounds mean the beginning or end, negative bounds count from the end,
and `start >= stop` produces an empty list. Slice step is deliberately omitted
in the first version. An out-of-range list index produces structural absence
rather than an exception.

Map lookup is literal-only. Binding accepts a literal only when it converts
losslessly to the declared Arrow key type: booleans are not integers, negative
values do not bind to unsigned keys, overflow is an error, and strings bind
only to Arrow string key types. Duplicate selected map keys are rejected rather
than silently choosing one occurrence.

### Excluded Syntax

The compiler rejects:

- filters and predicates;
- functions and pipes;
- arithmetic, comparisons, and boolean expressions;
- recursive descent;
- object wildcards;
- flattening;
- sorting;
- joins;
- fallback expressions;
- multiselects and field stacking.

Each error names the model address, query text, and unsupported segment. A
preprocessor owns any operation outside the structural grammar.

### Structural Contracts

- A generated-root query produces one struct or null per source row.
- A repeated `Branch` query produces a list of structs or null per parent.
- The generated root is the only singleton branch. Every child `Branch`
  selects a list, large list, or fixed-size list of structs.
- A leaf query produces the rank expected at its schema location and a type
  accepted by its plugin.
- A child query cannot change an ancestor branch's coordinate domain.
- Descendant selection preserves every ancestor list offset and position.
- Branch overflow is decided once and shared by all descendants.

At a child `Branch` boundary, RelFlow implicitly lifts each descendant selector
over that branch's element type. In the example, `product.sku` begins at each
selected item struct, while the `items[-32:]` list wrapper and its coordinates
remain owned by the branch. Users do not repeat `[*]` merely to enter a modeled
branch.

This makes alignment a property of the tree rather than a convention among
independent leaf expressions.

### Query State

The query executor returns an Arrow value projection and Arrow presence
information. It distinguishes:

| Condition | Resulting state |
| --- | --- |
| Valid selected leaf | `valued` |
| Selected leaf is null while every ancestor exists | `null` |
| Parent struct or list is null | `padded` |
| List index is out of range | `padded` |
| Map key is absent | `padded` |
| Map key exists with a null value | `null` |
| Capacity adds a position | `padded` |
| Required field is absent from the bound Arrow schema | Query binding error |

Arrow has a fixed struct schema and cannot preserve a per-row distinction
between an omitted mapping key and an explicit null after ordinary
columnarization. Both appear as a nullable field and therefore lower to
`null`. RelFlow does not maintain a second provenance sidecar for this
distinction.

### Compilation And Execution

Queries are parsed when the model is built and bound when the post-preprocessor
Arrow type is available. The current implementation caches a bound plan by
complete expression, type, and model address.

The executor uses Arrow structural kernels such as struct selection, list
element, list slice, and map lookup. List traversal transforms child buffers
and reuses parent offsets and validity. It never loops over Python values.

Direct and explicit paths form one schema tree of branch layouts:

```text
payload
└── items[-32:]
    ├── product
    │   └── sku
    └── qty
```

Each active branch is evaluated once per model batch. Its retained records,
projection/overflow result, shape, and dense placement are reused by every
active descendant. Leaves still own their final selector and value/null state.
Identical text under different branches cannot share a result because its
coordinate domain differs.

### Strata Plans

Binding produces separate input and target projections for each stratum.

- Train, validation, and test include target projections required by their
  active losses and metrics.
- Predict excludes only queryless leaves with unconditional `reconstruct=True`
  from source binding and scan pushdown. Such a reconstruction-only source
  field may therefore be absent without a query error.
- A query-backed reconstruction still requires its selector and modeled source
  columns so the Arrow/plugin type can bind independently of one batch's flags.
  A repeated ancestor also remains required to establish output cardinality.
- A schema policy with unconditional `reconstruct=True` creates its requested
  model coordinate from compiled geometry; it does not make an absent
  prediction label a required source input.
- A missing field in the active input plan remains a binding error.

The shipped bound-plan cache is keyed by exact expression, Arrow type, and
address. A richer stratum/input-role cache and Scanner projection over active
plans, declared preprocessor requirements, and `retain` columns are roadmap
work.

### Migration From JMESPath

Common structural expressions become shorter:

| Previous expression | Arrow-native query |
| --- | --- |
| `[*].payload.amount` | Root identity, then `amount=rf.Number(query="payload.amount")` |
| `[*].payload.items[*].product.sku` | Root identity + Branch `query="payload.items"`, or root `query="payload"` + Branch `query="items"`; then leaf `query="product.sku"` |
| `[*].legs[0].origin` | `legs[0].origin` |
| `[*].events[-10:].amount` | Branch `query="events[-10:]"`, then `amount=rf.Number` |
| `[*]."job code"` | `["job code"]` |

JMESPath filters and `map(...)` become one coordinated Arrow preprocessor that
creates the desired collection. Children then bind directly to that collection.
The old JMESPath strings are not interpreted by the new parser.

## Arrow Preprocessors

### Contract

A preprocessor receives and returns the Arrow-backed `Batch` carrier:

```python
@rf.preprocess
def prepare(
    batch: rf.Batch,
    *,
    strata,
    schema,
    encoding_context,
) -> rf.Batch | Iterable[rf.Batch] | None:
    ...
```

`Batch.data` is a `pa.Table`; all aligned dynamic metadata is Arrow as well.
Returning `None` or zero rows discards input observations. Returning more rows
or multiple batches supports explosion and one-to-many preprocessing while
bounding memory.

A preprocessor may:

- select, rename, or derive fields;
- filter or reorder observations;
- slice or sort nested collections;
- join Arrow lookup tables;
- explode or regroup observations;
- normalize datatypes;
- consume immutable fitted state or retain deterministic, partition-local
  streaming state;
- use PyArrow, Polars, DuckDB, native extensions, or arbitrary Python
  internally.

Its boundary remains Arrow. RelFlow never implicitly calls `to_pylist()` for a
preprocessor. A user may explicitly materialize rows inside their own function
when no columnar implementation is practical; that cost stays visible and
local to the function, and the result must be rebuilt as a `Batch`.

### Scope

Every preprocessor declares one execution scope:

- `partition` is the default. It runs independently over deterministic physical
  source partitions and may filter, reorder, derive, join against a fixed
  lookup, or expand rows using only that input. It cannot implement a global
  sort, aggregation, or regrouping whose result depends on another partition.
- `dataset` runs once against the logical split before ownership. It supports
  global sort, aggregation, regrouping, and cross-fragment operations. Its
  result is materialized as an indexable Arrow dataset before sharding.

Stateful partition preprocessors receive stable source order and explicit
physical partition boundaries. A transform whose result depends on arbitrary
scanner batch boundaries is invalid. If distributed scheduling needs exact
post-preprocessor counts, partition outputs are indexed or cached before rank
ownership is assigned.

Mutable partition state may carry a deterministic transformation across chunks
inside one physical partition. It may not learn global fitted statistics during
ordinary iteration or communicate implicitly across workers, ranks, epochs, or
strata.

### Schema

Preprocessor output has one exact structural Arrow schema per stratum before
model rebatching. Field names, nesting, nullability, and physical Arrow types
must match. Source alignment or the preprocessor performs any promotion first;
plugin casts happen later within an already stable selected leaf column.
Incompatible output fails with the preprocessor name and expected and actual
schemas.

An empty or all-null output must still carry its declared Arrow schema. Query
plans bind after preprocessing and are cached against that schema.

### Lineage

Framework row identity travels in `Batch.identity` and is transformed with
`Batch` operations:

- `batch.replace(data)` replaces columns while preserving same-length,
  same-order identity;
- `batch.filter(mask)` and `batch.take(indices)` select data and identity while
  preserving their existing processed order;
- `batch.order(indices)` deliberately reorders rows, preserves `logical`, and
  assigns stable new `order` values;
- `batch.expand(data, parents, ordinals)` accepts one parent row index and one
  explicit per-parent output ordinal per output, so identity is stable across
  separately yielded batches;
- `batch.explode(column, name=...)` is the common list-column expansion: it
  repeats all other columns, places each list item in `name`, and derives
  lineage from the parent row and item ordinal;
- `batch.group(data, parents)` accepts a list of parent row indices per output
  and derives a stable group identity from the ordered parent identities and an
  optional explicit group key.

Partition-scoped order includes the stable physical-partition prefix and output
ordinal. Dataset-scoped output is assigned one global order when its cache is
committed. These rules make order independent of scanner and returned-batch
boundaries while honoring deliberate preprocessor reordering.

Returning a bare table, changing cardinality through `replace`, or dropping
lineage is an error. The methods are columnar Arrow operations; none introduces
a Python observation carrier.

### Projection Pushdown

Query dependencies are statically known from the first path member. Direct
bindings are known from schema addresses. A preprocessor may declare the source
columns it reads with `requires=(...)` and the output columns it creates or
replaces with `produces=(...)`.

The decorator owns both declarations:

```python
@rf.preprocess(
    scope="partition",  # "partition" or "dataset"
    requires=("source_column",),
    produces=("output_column",),
)
```

`scope="partition"` is the default. `requires` is the complete set of
top-level source columns the function inspects. `produces` is the set of
top-level output columns it creates or replaces. Active model and query roots
not named in `produces` are treated as passthrough columns and projected
automatically; users do not enumerate them in `requires`. A same-named
transformation, such as normalizing `amount` into `amount`, declares that name
in both sets.

Omitting `requires` is valid. Names within each declaration must be unique. A
required name absent from a stratum fails with the preprocessor and column
names. The declaration is validated metadata today; it does not yet drive
projection pushdown.

The defaults are `requires=None` and `produces=()`. An explicit
`requires=()` means the function reads no source column beyond automatic
passthrough; it does not mean the same thing as `None`.

The roadmap projection rules are:

- with no preprocessor, scan only modeled/query roots plus `retain` columns;
- with declared requirements, scan `requires`, active roots not in `produces`,
  and retained roots not in `produces`; and
- with an undeclared arbitrary preprocessor, scan and pass all source columns
  into that preprocessor.

Today the Dataset Scanner reads its supplied schema. Derived output roots must
still be declared in `produces`, and only the resolved `retain` projection
enters output `inputs`.

No optimizer guesses which fields arbitrary Python code might inspect.

### User-Defined Preprocessors

These examples use the shipped Arrow preprocessor API. A processor receives one
`rf.Batch` and returns a lineage-safe `rf.Batch`, an iterable of batches, or
`None`; there is no `rf.Observation` row wrapper.

A preprocessor never constructs `Batch` directly and never returns a bare
Arrow table. It returns the result of the lineage operation matching its
change:

| Change | Return |
| --- | --- |
| Same rows in the same order | `batch.replace(data)` |
| Select rows | `batch.filter(mask)` or `batch.take(indices)` |
| Reorder rows | `batch.order(indices)` |
| Explode one list column | `batch.explode(column, name=...)` |
| One input to many outputs | `batch.expand(data, parents, ordinals)` |
| Many inputs to one output | `batch.group(data, parents)` |

The examples use these imports:

```python
import math

import awkward as ak
import pyarrow as pa
import pyarrow.compute as pc
import relflow as rf
```

#### Derive Columns

This transform casts source values and adds one model-facing column without
changing row count or order:

```python
@rf.preprocess(
    requires=("subtotal", "tax"),
    produces=("total",),
)
def total(batch: rf.Batch) -> rf.Batch:
    subtotal = pc.cast(batch.data["subtotal"], pa.float32())
    tax = pc.cast(batch.data["tax"], pa.float32())
    tax = pc.fill_null(tax, 0.0)
    data = batch.data.append_column("total", pc.add(subtotal, tax))
    return batch.replace(data)
```

`replace` validates equal length and preserves the identity sidecar. Source
columns may remain as unmodeled metadata or be removed explicitly.

#### Filter Observations

Filtering uses one Arrow boolean expression and applies it to data and identity
together:

```python
@rf.preprocess(requires=("status", "amount"))
def valid(batch: rf.Batch) -> rf.Batch:
    keep = pc.and_kleene(
        pc.equal(batch.data["status"], "complete"),
        pc.greater(batch.data["amount"], 0),
    )
    return batch.filter(pc.fill_null(keep, False))
```

Null predicates are made explicitly false. Sampling and shuffling operate on
the surviving logical observations after this function returns.

#### Bind A Lookup

User parameters remain keyword-only and are frozen with `partial`. For a
unique-key lookup, `index_in` plus `take` is simpler and more order-safe than a
relational join:

```python
merchant_lookup = pa.table({
    "merchant_id": ["market", "pharmacy"],
    "risk": [0.2, 0.1],
})

if pc.count_distinct(merchant_lookup["merchant_id"]).as_py() != len(merchant_lookup):
    raise ValueError("merchant lookup keys must be unique and non-null")

@rf.preprocess(
    requires=("merchant_id",),
    produces=("merchant_risk",),
)
def enrich(batch: rf.Batch, *, lookup: pa.Table) -> rf.Batch:
    positions = pc.index_in(
        batch.data["merchant_id"],
        value_set=lookup["merchant_id"],
    )
    risk = pc.take(lookup["risk"], positions)
    data = batch.data.append_column("merchant_risk", risk)
    return batch.replace(data)

prepare = enrich.partial(lookup=merchant_lookup)
```

Lookup keys must be unique. A missing key produces a null value. If a join can
duplicate source rows, it is an expansion and must use `batch.expand` instead.

#### Transform A Nested Collection

Awkward is useful inside a preprocessor for nested filtering, variable slicing,
sorting, and sibling-safe mutation. Here `events` is a list of structs:

```python
@rf.preprocess(
    requires=("events",),
    produces=("events",),
)
def logins(batch: rf.Batch) -> rf.Batch:
    index = batch.data.schema.get_field_index("events")
    field = batch.data.schema.field(index)
    events = ak.from_arrow(batch.data.column(index))
    keep = ak.fill_none(events["type"] == "login", False)
    events = events[keep][:, -32:]
    events = ak.to_arrow(events, extensionarray=False)
    events = pc.cast(events, field.type)
    data = batch.data.set_column(index, field, events)
    return batch.replace(data)
```

The predicate selects whole event structs, so timestamps, identifiers, and all
other siblings remain aligned. Awkward is transient; the function returns an
ordinary Arrow table with the declared schema. Null and empty collections stay
distinct.

#### Expand A Nested Collection

The common list-to-observation transform is deliberately one line. Empty or
null item lists emit no rows:

```python
@rf.preprocess(
    requires=("line_items",),
    produces=("line_item",),
)
def explode(batch: rf.Batch) -> rf.Batch:
    return batch.explode("line_items", name="line_item")
```

`Batch.explode` supports list, large-list, and fixed-size-list columns. It owns
the Arrow flattening, repeats passthrough columns, and derives identity from the
parent and stable item ordinal. `batch.expand(...)` remains available for an
arbitrary one-to-many transform that cannot be expressed as one list explode.

#### Order A Split

A global operation declares dataset scope. Sorting each physical partition
would not produce one globally chronological split:

```python
@rf.preprocess(
    scope="dataset",
    requires=("event_time",),
)
def chronological(batch: rf.Batch) -> rf.Batch:
    indices = pc.sort_indices(
        batch.data,
        sort_keys=[("event_time", "ascending")],
    )
    return batch.order(indices)
```

Dataset scope materializes the result before distributed ownership and is
intentionally more expensive than partition scope. Many-to-one aggregation is
the advanced form: build one output table plus the ordered parent-index list for
each row, then return `batch.group(data, parents)`.

#### Bind Fitted State

Preprocessor execution transforms data; it does not implicitly fit statistics.
Fit once from training data and bind immutable values so validation,
prediction, workers, and ranks all use the same state:

```python
values = pc.cast(train_table["amount"], pa.float64())
center = pc.mean(values).as_py()
spread = pc.stddev(values).as_py()

if (
    center is None
    or spread is None
    or not math.isfinite(center)
    or not math.isfinite(spread)
):
    raise ValueError("amount statistics must be finite")

spread = spread or 1.0

@rf.preprocess(
    requires=("amount",),
    produces=("amount",),
)
def standardize(
    batch: rf.Batch,
    *,
    center: float,
    spread: float,
) -> rf.Batch:
    values = pc.cast(batch.data["amount"], pa.float64())
    values = pc.divide(pc.subtract(values, center), spread)
    index = batch.data.schema.get_field_index("amount")
    field = batch.data.schema.field(index).with_type(pa.float64())
    data = batch.data.set_column(index, field, values)
    return batch.replace(data)

prepare = standardize.partial(center=center, spread=spread)
```

A preprocessor must not update fitted statistics during ordinary iteration.
Automatic fitting would require a separate coordinator-owned fit phase,
serialized frozen state, and restoration before any non-training stratum runs.
Preprocessor code and bound values remain external to `model.save(...)` and
must be versioned with the model artifact.

The same configured preprocessor is accepted by all four data modules:

```python
datamodule = rf.ArrowDataModule(
    model=model,
    train=train_table,
    validate=validate_table,
    preprocessor=prepare,
)

predictions = model.predict(records, preprocess=prepare)
```

The data-module argument remains `preprocessor`; direct model methods retain
`preprocess`. Both enter the same Arrow transform after their respective input
adapters.

## Sampling, Shuffling, And Sharding

### Statistical Unit

The unit is always one logical observation emitted by preprocessing. It is
never a file, fragment, row group, scanner batch, or nested branch item.

A `query` changes field projection only. It never changes observation count,
sampling membership, or order.

The **eligible set** is the post-preprocessor logical observations retained by
`sample`. The **epoch set** is the eligible subset actually scheduled after an
optional epoch limit and distributed tail handling. Rank outputs cover the
epoch set, which may deliberately be smaller than the eligible set.

### Defaults

- Training shuffles by default and samples without replacement by default.
- Validation, test, and prediction preserve processed source order by default.
- `shuffle=False` is an explicit opt-out for temporal, curriculum, or otherwise
  ordered training.
- Streaming replacement does not turn on implicitly.
- `replacement=True` requires an explicit global `epoch_size`.
- A one-observation shuffle window is removed as a default and treated as no
  shuffle.

`sample` is a Bernoulli probability in `(0, 1]`, not an exact fraction or row
count. Training includes epoch in the sampling key, so membership may change
between epochs. Evaluation uses a fixed sampling epoch. An exact
without-replacement subset of a finite population is a separate option and
requires an indexed or cached eligible set.

An infinite, uncached, single-reader training factory has one explicit
exception: `epoch_size=N` is a stream cutoff. RelFlow consumes until it has
selected `N` eligible logical observations, bounded-shuffles them, and emits
exactly `N`. This is not a uniform sample from an infinite population and cannot
use replacement, caching, multiple workers, or distributed execution.

A concise configuration remains possible:

```python
datamodule = rf.PolarsDataModule(
    model=model,
    train=train_frame,
    seed=42,
    sample=1.0,
    shuffle=True,
    replacement=False,
)
```

The same values may be configured per stratum.

### Deterministic Randomness

The shipped sampling and shuffle decisions do not use Python's process-global
`random` state. They derive from:

```text
(seed, stratum, epoch, operation, operation identity)
```

Sampling and shuffling use separate operation keys, so configuring one does not
silently consume the other's random stream. With the same source, seed, and
epoch, they reproduce the same logical sequence; training advances the epoch,
while evaluation and prediction use a stable epoch key.

Current identity roles are:

| Operation | Identity key |
| --- | --- |
| Without-replacement sampling | `logical` |
| Shuffle | `instance` |
| Ordered merge | `(order, instance)` |

`Batch.order()` therefore changes presentation order without changing sampling
membership. The current single-process guarantees are:

- the same seed, epoch, stratum, source snapshot, and topology reproduce the
  same logical sequence;
- with shuffling enabled, a new training epoch produces a new permutation;
- evaluation and prediction remain stable unless explicitly randomized;
- sampling membership is derived from logical identity rather than row values.

Legacy Torch model masking, pruning, branch masking, and datatype-specific
unavailable simulation are separate. They currently use Torch's random stream,
are not keyed by `Batch.identity`, and may change with model batching or source
chunking. Seed Lightning/Torch for run-level repeatability, but do not infer a
per-identity masking guarantee from the data-module seed.

This paragraph describes the shipped audit baseline. The unified masking spec
replaces it with identity-keyed selection and defines how seed, epoch, and
stratum reach coalescing.

Replacement draws, physical worker/rank ownership, stable distributed merge,
topology-invariant scheduling, and persistent-worker epoch replay are roadmap
work. Their intended identity keys and guarantees are described in the
following sections, but the current loader rejects those configurations.

Python `hash()` is never used because it is process-salted.

### Sampling Without Replacement

Bernoulli sampling uses a deterministic identity-based threshold:

```text
keep when random(seed, epoch, "sample", identity) < sample
```

This makes membership independent of worker scheduling and scanner chunking.
Duplicate payloads remain distinct observations because identity, not value
equality, defines membership.

Sampling occurs before the shuffle buffer, so rejected rows consume no buffer
memory.

### Full In-Memory Shuffle

Indexable post-preprocessor Arrow data uses a complete permutation of logical
row identities for every training epoch. Model batches call Arrow `take` with
successive index windows. RelFlow does not reorder the entire nested table
eagerly and does not materialize rows.

Only the permutation is held separately. Nested buffers are gathered in native
code when each model batch is formed.

An in-memory source does not by itself make arbitrary stateful, filtering, or
one-to-many output indexable. Such output uses bounded streaming shuffle unless
the user enables an Arrow cache. Dataset-scoped preprocessing always produces
that cache.

### Bounded Streaming Shuffle

Streaming shuffle combines three levels:

1. assign deterministic random priorities to owned fragments and row groups;
2. reassemble scanner output into fixed internal row quanta, then permute each
   quantum by stable observation identity;
3. mix observations across record-batch, row-group, and file boundaries in a
   bounded Arrow buffer.

Fragment shuffling alone is insufficient because one large file may itself be
ordered by target, entity, or time.

Internal quanta are defined over stable post-preprocessor logical order and
cross scanner-batch and preprocessor-output boundaries. Parallel output is
merged by deterministic partition priority and `identity.order` before it
reaches the shuffler. Buffer choices use identity-derived priorities, not
arrival timing or record-batch number. Rechunking the same logical output
therefore does not change order.

The buffer is memory-aware. `shuffle=True` chooses a meaningful default with a
hard byte ceiling and an advisory target measured in model batches.
Configuration may lower the row target or byte ceiling explicitly; both limits
apply, and bytes win. If the realized buffer holds fewer than several model
batches, RelFlow warns that source locality may remain high. One observation
larger than the byte ceiling is accepted alone rather than dropped and is
reported as an oversize observation.

The buffer contains Arrow batches and Arrow indices. It never contains Python
records. Emission uses Arrow `take`; batch boundaries are formed after mixing.

### Replacement

Replacement is uniformly logical-observation-based for every source adapter.

- `replacement=True` requires `epoch_size`.
- Global draw identities derive from epoch and draw index.
- Draw identities are partitioned across ranks and workers without overlap.
- Repeated source identities are expected and valid.
- Files are weighted by eligible observation count, not selected uniformly as
  files.

`epoch_size` is the requested global number of draws. In distributed training,
it must be divisible by `world_size × batch_size`; with explicit
`drop_last=True`, RelFlow instead rounds it down once to form the epoch set and
reports the dropped draw count. Each rank then receives the same number of draw
indices.

Uniform replacement over a cardinality-changing preprocessor requires an index
of its logical outputs. RelFlow may build an Arrow cache, including a
disk-backed cache. If a stream cannot be indexed or cached, exact replacement
fails with an actionable error instead of silently changing the sampling unit.

### Scheduling And Physical Reads

RelFlow exposes no user-facing sharding mode. Logical ownership is a
correctness decision; physical read grouping is an internal optimization.

There are two execution cases within the same semantic pipeline:

- A non-indexed stream assigns stable physical partitions to ranks and workers
  before scanning. Post-preprocessor logical outputs inherit that owner and are
  sampled and shuffled locally; RelFlow never assigns them a second owner.
- An indexed schedule samples the materialized logical identities first and
  assigns their authoritative rank positions. That schedule then drives grouped
  physical reads; fragment locality cannot override it.

For ordinary non-indexed streaming, physical ownership is hierarchical:

```text
rank owner   = assign(stable unit identity, world size)
worker owner = assign(stable unit identity, workers on owning rank)
```

Rank assignment is independent of local worker count. The planner automatically
uses the coarsest stable fragment, row group, or fixed source range that known
counts can balance. With `world_size=1`, it may fall back to one reader. Under
distributed execution, an unpartitionable source must be cached and indexed or
is rejected; assigning data only to rank zero would deadlock collective work.

Scanner batch numbers are not stable chunk identities. Changing scanner batch
size must not change ownership.

Physical balance estimates use source row counts. They say nothing about
logical-output balance after a cardinality-changing preprocessor unless its
output is cached and counted. RelFlow reports that limitation rather than
presenting an I/O grouping choice as a statistical guarantee.

For indexed without-replacement DDP training, RelFlow orders eligible
identities by `(epoch shuffle key, logical identity)` when `shuffle=True`, with
identity breaking priority collisions. With `shuffle=False`, it orders by
`(identity.order, logical identity)`. It applies any exact epoch limit, then
drops or rejects one deterministic tail so the epoch-set size is a multiple of
`world_size × batch_size`. Global position `i` belongs to rank
`i % world_size`; workers partition only their rank's positions. Rank and
worker outputs are pairwise disjoint and their union is exactly the epoch set.
No observation is repeated to balance a rank.

That logical schedule is authoritative. File and chunk grouping may optimize
reads, but cannot determine rank ownership or change sequence. If the source
cannot read scheduled rows efficiently and RelFlow cannot redistribute them,
equal-step DDP training is rejected.

Coarse file/chunk ownership alone cannot guarantee equal rank counts. DDP
training therefore requires an indexed logical-output schedule, or replacement
with a valid global `epoch_size`. A cardinality-changing partition preprocessor
must be promoted to dataset scope and cached for this purpose. The loader fails
before training when it cannot prove equal complete-batch counts.

Prediction preserves source order within each rank. Its writer persists stable
sequence identity so a consumer can reconstruct global distributed order with
an Arrow sort.

### Collapse Diagnostics

The loader reports enough information to diagnose poor mixing:

- configured and realized shuffle rows and bytes;
- source fragments represented in each buffer and batch;
- contiguous source-run lengths;
- sampled, dropped, emitted, and repeated observation counts;
- per-rank observation and batch counts.

RelFlow warns when training is unshuffled or when the effective streaming
window is too small. It does not inspect targets or promise that shuffling alone
prevents every form of model collapse.

## State And Unified Masks

Arrow validity and RelFlow state have distinct jobs:

- validity says whether an Arrow value exists;
- query presence says whether the structural coordinate exists;
- branch geometry adds capacity padding;
- the plugin converts retained content;
- the shipped baseline modifies model visibility after tensorization.

The query executor derives Arrow presence alongside selected values and carries
both into coalescing. It does not infer datatype-specific meaning.

The reserved data literal `"<MASK>"` is removed. It cannot be represented
consistently in typed numeric, binary, struct, and extension columns, especially
when modeled fields are selected only after preprocessing. The string
`"<MASK>"` is therefore ordinary string content.

The unified Mask contract now lives entirely in the masking spec. In brief,
nodes declare one `mask` argument, normalized to an immutable policy tuple;
dynamic selectors are ordinary Boolean Arrow fields that may be created by a
preprocessor; branch selection is resolved once against its shared layout; and
coalescing builds separate input and target projections. There is no parallel
identity-bearing `Selection` carrier.

Skipping is not an Arrow state token. Arrow chooses and projects coordinates;
Torch presence omits skipped coordinates from embedding, attention, and pooling.

## Coalescing And Plugins

`coalesce(batch, schema, strata)` accepts the logical Arrow model batch and
performs structural work only.

It performs these phases:

1. build the root layout and bind each node-relative query;
2. select and regularize every active repeated branch once;
3. carry retained records, declared geometry, and dense placement to children;
4. project each leaf and combine selected validity, structural presence, and
   capacity padding into final state;
5. emit one raw Arrow-backed `RaggedField` per encoded address.

Sibling leaves reuse the same materialized branch geometry. Overflow is applied
before descendant leaf queries, so discarded records do not enter a datatype
or trigger query errors below that branch.

The extension boundary becomes:

```python
@dataclass(frozen=True, slots=True)
class RaggedField:
    values: pa.Array | pa.ChunkedArray
    state: pa.Int8Array
    placement: pa.Int64Array
    shape: tuple[int, ...]
```

`state` and `placement` are flat row-major Arrow arrays. `shape` defines the
dense model geometry. NumPy or Torch conversion occurs once inside the codec.

The shared engine knows Arrow structure, validity, offsets, presence, and
placement. It does not name Number, Text, Category, or any other datatype.

After coalescing, each plugin is invoked once with its whole retained leaf
column. It owns Arrow compatibility and preparation through whole-column hooks
conceptually equivalent to:

```python
plugin.accepts(dtype: pa.DataType) -> bool
plugin.prepare(
    values: pa.Array | pa.ChunkedArray,
    *,
    address,
) -> pa.Array | pa.ChunkedArray
```

The final names follow RelFlow's short, single-word style. The contract, not
the shared ragged engine, decides whether to cast, dictionary-encode, tokenize,
parse, hash, or reject a leaf type. `Plugin(types=...)` supplies standard
Python-equivalent Arrow matchers; a third-party atom supplies its own physical
predicate with `Plugin(..., arrow={Type: matcher})`. The predicate sees a
logical Arrow extension type before RelFlow tries its storage terminals. No
shared switch names or dispatches on built-in tensorfield types.

Awkward may implement difficult list-offset and regularization kernels behind
`coalesce`. Any resulting valued leaf is returned to Arrow before crossing the
plugin boundary.

## Arrow-Native Prediction Output

### One Output Contract

The prediction path is the same for interactive use, Lightning callbacks, and
deployment:

```text
Torch predictions
  → plugin write: typed Arrow coordinate arrays
  → shared state, inferred, embedding, and shape assembly
  → Model.write: identity-aligned Batch
  → postprocessor: Batch → Batch
  → Parquet writer or explicit transport serialization
```

The loader's internal model item is a pair of the encoded `TensorDict` and its
source `Batch`. Lightning's transfer hook moves only the tensors; the Arrow
batch stays on CPU. `Model.write` receives both the internal tensor predictions
and that source batch:

```python
written = model.write(predictions, source=source)
```

The pair is an internal handoff rather than another public carrier. Training
may discard the source after encoding; prediction retains it through
`predict_step` so output identity and retained columns never need to be rebuilt
from tensors.

### Output Plan

Each prediction run compiles one output plan after checkpoint restoration,
distributed vocabulary synchronization, and any queued `Model.update(...)`.
Normalized schema `mask` tuples determine its reconstruction addresses; per-batch
selector values never change its shape or fields. In the unified vocabulary,
it contains separate `forward` and `writes` address plans. `forward` validates
decoder or embedding execution even for a plugin such as Text that deliberately
exposes no decoded value; `writes` contains only roles that contribute public
Arrow data.
The plan also contains:

- every address reached by a schema mask with `reconstruct=True` and no
  remaining random rate, whether its selection is unconditional or
  query-backed;
- every active `embed=True` address;
- the exact plugin and shared Arrow fields and compiled model axes for each
  address;
- the ordered `retain` projection used to construct `inputs`.

The plan is frozen for the run. Any later mutation affecting schema, geometry,
vocabulary, output configuration, or active addresses invalidates it and
forces recompilation before another prediction run. Output types are never
learned from prediction values.

Every `forward` address must be returned by the forward pass with its
compiled coordinate count. A missing address is an error with model and plugin
context; it is not silently replaced by nulls. A query-backed mask still
returns the expected address for the whole batch, with unselected coordinates
represented inside its typed output and `inferred` false. A forward value that
is absent from `forward` is an error. A valid forward value absent only from
`writes` is validated and then omitted without
calling a plugin writer.

`written.data` has one row per processed observation and this canonical schema
before an optional postprocessor:

```text
inputs: struct<retained processed columns...>
predictions: struct<
    "record": struct<embedding: fixed_size_list<float32>[d_model]>,
    "record/fraud": struct<
        state: struct<
            valued: float32,
            null: float32,
            padded: float32,
            masked: float32,
            other: float32
        >,
        content: struct<
            value: string,
            probability: float32,
            topk: list<struct<value: string, probability: float32>>
        >,
        inferred: bool
    >
>
```

Address strings are Arrow struct field names. Their order follows compiled
schema traversal, never prediction arrival or dictionary insertion order.
`inputs` reuses the requested `retain` columns without Python conversion. A
preprocessor must preserve any retained request identifier or unmodeled source
value needed by an output consumer.

If the compiled output plan has no prediction addresses, `predictions` is a
typed null column of the correct row count. It is not `struct<>`, which Parquet
cannot persist. Likewise, an empty `retain` projection becomes a typed null
`inputs` column rather than `struct<>`.

### Output Shapes

Output shape is compiled from model geometry and never inferred by squeezing
runtime values.

- The generated root singleton is omitted by schema rule.
- Every repeated child `Branch` axis becomes `fixed_size_list`, including a
  configured length of one.
- Vector values and embeddings remain `fixed_size_list`, including width one.
- Variable candidate collections such as Category top-k and thresholded Set
  values use ordinary `list<struct<...>>`.
- Null, padded, masked, and inferred positions do not change physical shape.

The runtime builds one flat row-major coordinate array, combines its shared and
plugin-owned fields, then wraps repeated model axes from innermost to outermost.
This replaces recursive `squeeze` behavior and keeps batch-size-one output
schema-identical to larger batches.

For batch size `N`, a plugin writes exactly:

```text
N × product(configured lengths of repeated Branch ancestors)
```

flat coordinate structs. Vector width and embedding width are fields inside a
coordinate and do not multiply that count. For example, a repeated leaf uses
coordinate-first storage:

```text
"record/items/sku": fixed_size_list<
    item: struct<state: struct<...>, content: struct<...>, inferred: bool>
>[32]
```

It does not create a struct whose individual fields each invent their own list
shape. This gives third-party plugins one unambiguous flat-write contract.

### Plugin Output Contract

Plugins own their datatype-specific Arrow output fragment. The shared engine
does not contain a table keyed by Number, Category, Text, or another built-in.

A plugin declares its stable coordinate type and writes values of exactly that
type:

```python
@plugin.register
def output(module: Model, address: Address) -> pa.StructType | None:
    ...


@plugin.register
def write(
    module: Model,
    prediction: Prediction,
    datatype: pa.StructType | None,
) -> pa.StructArray | None:
    ...
```

`output` depends only on immutable model, request, and plugin state. It must be
valid before the first prediction batch, including for an empty vocabulary or
an all-null batch. Returning `None` means the datatype has no decoded public
output. A configured embedding may still make that address writable.

Any value type needed by `output` is bound while compiling or restoring the
model, not inferred from the first prediction values. A plugin with multiple
accepted input types either records the selected Arrow output type in its
state, requires an explicit request setting, or declares one canonical output
representation. The built-in vocabulary plugins use `large_string` labels and
bind their stringified Arrow vocabulary once per vocabulary revision. An empty
vocabulary and an all-null source therefore have the same type as populated
output.

`write` receives the exact `datatype` returned once by `output` during output
plan compilation. It must use that supplied type rather than calling `output`
again or inferring a schema from values. It returns one flat struct value per
decoded model coordinate and owns
only plugin-specific fields such as `content` or `cluster`; `state`,
`inferred`, and `embedding` are reserved shared fields. The runtime validates:

- exact declared `StructType` and field order;
- exact coordinate count from compiled model geometry;
- no filtering, expansion, reordering, or identity handling;
- no dictionary, list, object-array, or NumPy return value;
- stable schema across batch size, values, masking, and epochs.

A plugin fragment containing any reserved shared field name is rejected.
When `output` is not `None`, the runtime adds stable `state` and `inferred`
fields and adds `embedding` when requested. When `output` is `None`, it adds no
state or inferred output; an `embed=True` address contains only `embedding`.
An address with neither decoded output nor an embedding is absent from the
public output plan.

Third-party plugins may return nested or Arrow extension fields. A persistence
sink that cannot represent an extension type reports the plugin, address, and
type; it never falls back to Python rows.

The shared runtime owns the five-way state softmax, the `inferred` mask, and
normalized embeddings because those contracts do not vary by datatype. This
removes the same state-writing code from every built-in plugin.

Built-ins use stable Arrow forms:

| Plugin | Plugin-owned coordinate fields |
| --- | --- |
| Boolean | `content: struct<probability: float32>` |
| Number | `content: float64` |
| Vector | `content: fixed_size_list<float32>[size]` |
| Category | `content: struct<value: large_string, probability: float32, topk: list<struct<value: large_string, probability: float32>>>` |
| Set | `content: list<struct<value: large_string, probability: float32>>` |
| Cluster | `cluster: struct<value: int32, probability: float32>` plus Category-like `content` |
| Text, Hash, DateParts | No decoded output until their plugin defines one; configured embeddings still write |

Set output no longer creates one dynamic struct field per vocabulary label or
row-specific Python dictionary. Without a threshold, its candidate list
contains the whole active vocabulary. With a threshold, list lengths vary but
the Arrow schema does not. Category top-k uses the same candidate struct.

Category gathers labels from one plugin-owned Arrow vocabulary array with
`take`. Set flattens the selected candidate indices, builds list offsets from
per-coordinate counts, and gathers the same way. Neither implementation walks
labels, candidates, coordinates, or rows in Python during a prediction batch.
Stringifying a static vocabulary once when its revision changes is plan
compilation, not row processing.

### Torch-To-Arrow Bridge

PyArrow does not construct primitive arrays directly from Torch tensors. One
shared representation helper therefore performs:

```text
detach → contiguous → CPU → NumPy view → Arrow primitive buffer
       → fixed-size-list wrapping from compiled geometry
```

GPU-to-CPU transfer and Parquet encoding still copy. For a contiguous CPU
tensor, NumPy is a short-lived buffer view and Arrow can commonly reuse its
primitive memory. The helper never calls `tolist()`, creates an object-dtype
array, or walks rows. Softmax, top-k, thresholding, and other tensor work stay
in Torch before only the selected output is transferred.

Short loops over schema addresses, Arrow fields, tensor rank, or the five state
tokens are control-plane work. Loops over observations, nested values,
vocabulary candidates, or prediction rows are not part of the standard path.

### Arrow Postprocessors

A postprocessor receives and returns the output `Batch`:

```python
@rf.postprocess
def compact(batch: rf.Batch, *, threshold: float) -> rf.Batch:
    inputs = batch.data["inputs"]
    predictions = batch.data["predictions"]
    fraud = pc.struct_field(predictions, "record/fraud")
    content = pc.struct_field(fraud, "content")
    probability = pc.struct_field(content, "probability")

    data = pa.table({
        "request_id": pc.struct_field(inputs, "request_id"),
        "fraud": pc.struct_field(content, "value"),
        "fraud_probability": probability,
        "high_confidence": pc.greater_equal(probability, threshold),
    })
    return batch.replace(data)


warehouse = compact.partial(threshold=0.8)
```

Arrow is immutable, so postprocessors cannot mutate a mapping and return
`None`. They must return `batch.replace(data)` with the same row count, order,
and identity. They may project, rename, derive, nest, flatten, or redact
columns, but they may not filter, reorder, expand, or group rows. Those changes
belong in preprocessing when they should affect model execution, or in
explicit application code after prediction or persistence. Postprocessor
output must contain at least one column because a standalone zero-column Arrow
table cannot retain its logical row count after identity is removed.

`Batch.replace` enforces type and row count and retains the original identity.
It cannot prove that arbitrary user code did not manually reorder values before
constructing its replacement table; preserving row order is therefore part of
the postprocessor contract. Built-in examples use only column expressions over
the supplied batch, and tests cover the contract with order-sensitive values.

The primary batch already contains processed inputs and predictions. The old
`PostprocessorProvider` mechanism disappears, including `input`, `batch`,
`metadata`, `observations`, `request`, `batch_indices`, `batch_idx`, and
`dataloader_idx`. Values needed by a postprocessor must be retained as Arrow
input columns; user configuration remains keyword-only and is bound with
`partial`.

Postprocessors run once per model batch in every runtime. They do not have a
dataset scope, accumulate state across callback invocations, or own mutable
first-batch schema state. A `Writer` locks the schema of its first result. A
deployment validates against its configured response schema or model when one
is present. An interactive call has one result table. In every case the
consumer reports schema errors; the postprocessor remains a pure Arrow
transformation.

### Public Prediction API

`Model.predict(...)` returns a `pa.Table`, not an address-keyed Python mapping.
Without a postprocessor it returns the canonical `inputs` and `predictions`
columns. With one, it returns the postprocessor's table:

```python
result = model.predict(
    records,
    retain=("request_id",),
    postprocess=warehouse,
)
```

The leading batch dimension is always the Arrow row dimension. Nested model
axes remain Arrow list dimensions, so a one-row call has the same schema as a
large call. Users may call `to_pylist()` explicitly at their own application
boundary, but RelFlow does not do so for interactive prediction.

A typed zero-row Arrow input skips Torch forward and returns the exact planned
zero-row output schema. A nonempty sequence of mappings remains a convenience
adapter for small calls; an empty sequence is rejected because it cannot declare
an Arrow schema.

The lower-level `Model.write(...)` returns `rf.Batch` because callbacks and
deployment still need identity. Internal tensor `Prediction` objects remain;
their recursive `serialize`, `squeeze`, `denest`, and `unbatch` helpers are
removed.

The Lightning `predict_step` performs `Model.write` exactly once and returns
that `Batch`. The callback, other callbacks, and `return_predictions=True`
therefore share one conversion rather than reconstructing output independently
from raw tensor predictions.

### Writer Callback

`rf.Writer` consumes the written `Batch`, applies its optional Arrow
postprocessor once, and writes with `pyarrow.parquet.ParquetWriter` directly.
It never constructs a Polars frame.

The writer prepends one protected column to whatever columns are present in the
postprocessed `Batch.data`:

```text
identity: struct<logical, instance, order>
inputs: struct<...>
predictions: struct<...>
```

The name `identity` is reserved and a collision is rejected. Persisting
identity makes canonical or reshaped rank shards self-describing, joinable,
and globally orderable; it remains excluded from ordinary `Model.predict` and
JSON output.

The first emitted table locks the exact Arrow schema. Typed empty arrays are
valid. Later field additions, removals, reordering, or type drift fail with the
postprocessor or plugin address. The callback does not maintain its own
recursive shape compatibility rules or silently cast later batches.

One `rank-{global_rank}.parquet` shard remains the output. Distributed file
order is deliberately not a second ordering protocol: consumers reconstruct
global processed order by sorting the persisted `(identity.order,
identity.instance)` values with Arrow. A built-in coordinated merge can be
added later without changing the file schema, but is outside the first
implementation.

The writer closes on prediction completion and exception. Partial-file and
atomic-publication policy remain explicit writer concerns; they do not cause a
fallback to Python collections.

### Deployment

JSON requests necessarily begin as Python or Pydantic values. The deployment
validates them, converts the valid microbatch to Arrow once, and then uses the
same preprocess, encode, forward, write, and postprocess path as batch
prediction.

Writing and postprocessing happen once for the valid microbatch, not once per
request. Request ordinals and `Batch.identity` restore caller order around
per-request validation errors. Recursive Python prediction splitting and
tensor `Prediction.unbatch` are removed.

For the default JSON response, deployment performs one explicit
`Table.to_pylist()` at the final transport boundary. Without a postprocessor it
first projects each row to `{"predictions": ...}` so retained inputs are not
exposed accidentally. With a postprocessor it serializes each postprocessed
row. Identity is always removed. The deployment then scatters validation
errors, optionally validates response rows with Pydantic, and serializes them
with the selected JSON backend. This terminal conversion is not a pipeline
representation. It is the unavoidable cost of a JSON protocol.

Arrow IPC responses can later bypass that conversion for Arrow-native clients,
but content negotiation and a second wire protocol are outside the first
implementation. The internal output contract does not need to change to add
them.

### Metadata And Polars Boundary

Unmodeled columns may remain temporarily in Arrow when preprocessing requires
them, but only the explicit `retain` projection enters output `inputs`. Those
columns follow identity like every other column. RelFlow builds output schema
metadata deliberately and does not copy arbitrary source schema metadata into
predictions. Opaque Python objects must be serialized or represented by a
plugin-owned Arrow extension type before they enter the pipeline.

Polars remains supported solely through `PolarsDataModule`, which calls
`DataFrame.to_arrow()` once per split. No loader, query, preprocessor,
postprocessor, plugin writer, prediction API, callback, Parquet writer, or
deployment implementation uses Polars as a second backend. Making that adapter
an optional package dependency is part of this change: importing core RelFlow
does not import Polars, and constructing `PolarsDataModule` without the extra
raises one focused `relflow[polars]` installation error.

This restriction applies to RelFlow's implementation, not user code. A custom
preprocessor or postprocessor may explicitly convert to another library and
must return an Arrow-backed `Batch`; the conversion remains visible at that
user-owned boundary.

## Parallelism

Arrow scanners and compute kernels own ordinary I/O and CPU parallelism.
The shared loader used by all four data modules defaults to `num_workers=0`
rather than also spawning one PyTorch worker per CPU.

When DataLoader workers are enabled:

- each stable source unit has one owner;
- workers receive current epoch state;
- Arrow batches cross the worker boundary without row conversion;
- random streams are isolated by operation;
- the global shuffle budget is not divided into ineffective one-row buffers;
- Polars frames are not copied into every worker because they have already been
  converted to Arrow.

Benchmarking decides whether scanner threads, DataLoader workers, or both help
a particular remote source. The default does not oversubscribe them blindly.

## Proposed Module Boundaries

Keep implementation phases few and semantic:

| Module | Responsibility |
| --- | --- |
| `data/arrow.py` | Arrow batch identity, slicing, taking, rebatching, and schema alignment |
| `data/query.py` | Parser, bound query tree, and structural projection called by `coalesce` |
| `data/processors.py` | Arrow preprocessor and postprocessor contracts |
| `data/ragged.py` | Coalescing orchestration, branch geometry, state, and `RaggedField` |
| `data/datasets/arrow.py` | `ArrowDataModule`, internal `ArrowDataset`, and one `loader` function |
| `data/datasets/polars.py` | `PolarsDataModule` conversion and delegation only |
| `data/datasets/custom.py` | `CustomDataModule` and shared mapping-to-Arrow `adapt` operation |
| `data/datasets/synthetic.py` | `SyntheticDataModule` mapping-source normalization and shared `adapt` use only |
| `data/datasets/__init__.py` | Export the four public data modules only |
| `tensorfields/output.py` | Shared tensor-buffer, state, shape, and embedding Arrow builders |
| `architecture/runtime.py` | Compile plugin arrays into one identity-aligned output `Batch` |
| `inference/callback.py` | Apply an Arrow postprocessor and write Arrow directly to Parquet |
| `inference/deployment.py` | Batch Arrow output and perform only terminal JSON conversion |

Use short semantic functions such as `adapt`, `scan`, `align`, `sample`,
`shuffle`, `batch`, `query`, `coalesce`, `array`, `shape`, `state`, and `write`.
Inline one-use forwarding helpers. Do not prefix functions or classes with
underscores; explicit exports control the public surface.

## Removal Scope

The completed change deletes or removes from the canonical path:

- Polars `to_dicts()` and named row iteration;
- Arrow `to_pylist()` during dataset operation;
- Python observation sampling, shuffling, and rebatching;
- recursive Python value/state normalization in `ragged.py`;
- JMESPath imports, validation, compilation, execution, and dependency;
- one absolute query evaluation per leaf;
- per-value plugin preparation for Arrow-compatible columns;
- Python prediction metadata reconstruction;
- implicit file-level replacement sampling;
- the one-observation training shuffle default;
- `StreamingDataModule` and its file-specific configuration;
- source-specific batch-dataset classes and dataloader functions;
- RelFlow file discovery, suffix, regex-pattern, and reader dispatch;
- the public `Suffix` and `ShardingStrategy` dataset enums;
- the `MASK_LITERAL` value and `MaskLiteral` type alias;
- the row-oriented dataset `Pipeline` and its source-specific observe helpers;
- address-keyed Python prediction dictionaries and recursive prediction
  serialization, squeezing, denesting, and unbatching;
- plugin writers that return dictionaries, Python lists, object arrays, or
  NumPy arrays;
- per-value Category and Set output packing and datatype-specific copies of
  shared state writing;
- mapping-based `Predictions`, `PostprocessorResult`, and `Metadata` aliases;
- mutable postprocessors that return `None` and runtime-specific row-provider
  injection;
- Polars prediction-frame assembly, concatenation, shape inference, and
  Arrow conversion in the writer callback;
- deployment's recursive Python batch splitting and per-request invocation of
  model writing and postprocessing;
- implicit `.tolist()` and `.to_pylist()` calls before an explicitly requested
  caller conversion or the terminal JSON transport boundary;
- all package-owned Polars use outside the `PolarsDataModule` ingress adapter.

There is no permanent compatibility engine hidden behind source-type checks.
Custom, synthetic, Polars, high-level prediction, and explicit caller-owned
Python conversion all converge into Arrow before preprocessing; every later
phase is shared.

## Implementation Plan

### Phase 1: Arrow Carrier

1. Add the Arrow batch/identity carrier.
2. Add the canonical `ArrowDataModule` and one internal Arrow iterable.
3. Convert `PolarsDataModule` into a conversion-only subclass.
4. Convert `CustomDataModule` into a bounded mapping-to-Arrow adapter.
5. Convert `SyntheticDataModule` into generator-to-mapping-source adaptation.
6. Pass dataset scanner record batches without `to_pylist()`.
7. Validate restartable factories and reject one-shot sources.
8. Implement Arrow slicing, taking, and exact model rebatching.
9. Keep prediction identity and retained input columns in Arrow.
10. Remove `StreamingDataModule` and all source-specific loader machinery.
11. Move Polars from core dependencies to an optional ingress extra and lazily
    expose `PolarsDataModule`.

### Phase 2: Native Query

1. Add `query` to the generated root and `Branch`.
2. Implement the parser and schema binder.
3. Compile implicit and explicit paths into one shared tree.
4. Implement struct, quoted-member, list traversal, index, slice, and map
   lookup operations.
5. Preserve presence and branch coordinates.
6. Remove JMESPath parsing and runtime execution.

### Phase 3: Arrow Preprocessors

1. Change preprocessors to the `Batch` boundary.
2. Implement `replace`, `filter`, `take`, `order`, `explode`, `expand`, and
   `group` lineage.
3. Validate exact structural output schemas and execution scope.
4. Add declared source requirements, produced outputs, and scanner projection
   pushdown.
5. Remove automatic row-oriented processor normalization.

### Phase 4: Randomized Scheduling

1. Add stable source and logical output identities.
2. Replace process-global random state with operation-keyed randomness.
3. Implement full indexed shuffle and bounded cross-chunk streaming shuffle.
4. Standardize logical-observation sampling and replacement.
5. Make rank and worker ownership independent and disjoint.
6. Enforce equal distributed training batch counts.
7. Add shuffle diagnostics and safe defaults.

### Phase 5: Arrow Coalescing And Plugins

1. Compute branch geometry once and reuse it.
2. Make `RaggedField` Arrow-backed.
3. Add plugin-owned Arrow compatibility and preparation.
4. Remove scalar callbacks from Arrow-compatible built-ins.
5. Remove Awkward-to-Python-to-NumPy codec round trips.
6. Keep any Awkward work transient and Arrow-backed.

### Phase 6: Arrow Output

1. Add plugin-owned `output` declarations and Arrow `write` implementations.
2. Compile and freeze the output plan after model and vocabulary restoration;
   invalidate it after output-affecting mutations.
3. Add shared tensor-buffer, shape, state, inferred, and embedding builders.
4. Make `Model.write` return an identity-aligned `Batch` and `Model.predict`
   return a `pa.Table`.
5. Change postprocessors to the same-row `Batch.replace` boundary and remove
   mapping mutation and runtime-specific row providers.
6. Make `predict_step` write once and make the callback persist its
   identity-bearing Arrow batches directly through `ParquetWriter`.
7. Write and postprocess each deployment microbatch once, with Python
   materialization confined to its final JSON boundary.
8. Remove recursive prediction collection helpers and all callback Polars
   assembly.

### Phase 7: Documentation

1. Update query, binding, data-module, preprocessing, output, performance,
   extension, and troubleshooting documentation.
2. Replace every JMESPath example with a structural query or coordinated
   preprocessor.
3. Remove JMESPath from package dependencies.
4. Rewrite the data-module, batch-inference, performance, quickstart, public
   API, README, and agent-guide references around the four-module, one-loader,
   one-output surface.

## Required Tests

### Data Modules

1. The root public API exports `ArrowDataModule`, `PolarsDataModule`,
   `CustomDataModule`, `SyntheticDataModule`, and `Batch`; it does not export
   `StreamingDataModule`, source-specific datasets, `Suffix`, or
   `ShardingStrategy`.
2. `Batch`, `Table`, `RecordBatch`, Dataset, reader factory, and Arrow iterable
   factory inputs converge on identical tensors and metadata.
3. `ArrowDataModule` rejects bare readers, iterators, Scanner objects, paths,
   mappings, and Polars values with the appropriate replacement in the error.
4. An uncached source factory is invoked for every new source iterator and
   restarts identically for the same epoch; a cached factory is invoked only to
   populate its coordinated snapshot.
5. An uncached factory rejects multiple workers and ranks even when it supplies
   identity; caching enables one shared indexed snapshot without duplicate
   factory invocation.
6. Identified uncached factory batches reject non-monotonic `identity.order`.
7. `PolarsDataModule` calls `to_arrow()` exactly once per configured split and
   then uses the same loader, scheduling, and coalescing implementation.
8. Each module requires at least one named split and rejects a split with the
   wrong input type for that module.
9. No module accepts alternate source-map aliases or silently copies one split
   into another; documented per-stratum option mappings remain valid.
10. `model.batch_size` is the only model-batch size used by all four modules.
11. `CustomDataModule` accepts restartable `IterableDataset` splits yielding
    mappings, converts at most `ingress_rows` observations at once, and enters
    the canonical loader immediately after `RecordBatch.from_pylist()`.
12. Custom schema inference runs once per split and persists across epochs and
    cache manifests; empty input, late fields, incompatible types, and
    ambiguous all-null input fail with an `arrow_schema` remedy.
13. In uncached mode, `SyntheticDataModule` calls each configured generator for
    every new source iterator and delegates mapping conversion to the shared
    `adapt` operation; cached iteration does not reinvoke the generator after
    snapshot population.
14. An uncached synthetic generator must reproduce identical schema, values,
    and order; the module seed affects downstream randomness only.
15. Uncached custom and synthetic sources reject multiple workers and ranks;
    a coordinated finite cache permits both without duplicate source
    invocation.
16. Infinite custom and synthetic training sources obey the single-reader
    `epoch_size` cutoff, while their validation, test, and prediction sources
    must terminate.
17. Source adapters contain no independent sampling, shuffling, model batching,
    masking, encoding, rank ownership, worker ownership, or DataLoader logic.
18. Core RelFlow imports and every non-Polars data module work without Polars
    installed; `PolarsDataModule` reports the exact optional extra when used.
19. `retain` projects only its requested top-level columns, respects
    preprocessor `produces`, and supplies the same postprocessor inputs in
    direct prediction, callback, and deployment paths.

### Arrow Integrity

1. Polars, custom, synthetic, Arrow Table, RecordBatch, Dataset-backed scan,
   explicit caller-owned Python conversion, and high-level prediction ingress
   produce the same tensors for the same Arrow-representable records when
   omitted-key versus explicit-null provenance is not semantically required.
2. Custom and synthetic Python mappings are bounded by `ingress_rows` and are
   neither retained nor passed beyond adapter conversion.
3. Standard dataset, query, coalescing, and prediction tests fail if RelFlow
   calls `to_dicts()`, named `iter_rows()`, or `to_pylist()` before an explicit
   caller conversion or the deployment JSON boundary.
4. Rechunking the same logical Arrow table at arbitrary boundaries does not
   change tensors, state, queries, sampling membership, or order.
5. Buffer-sharing tests cover numeric, string, binary, list, and struct input.
6. Boolean and other necessarily copied representations document and bound
   their allocations.
7. Logical identities are non-null and unique; replacement preserves logical
   identity while assigning unique instance identities.
8. A zero-column projection retains its logical row count through
   `Batch.identity`; adding a data column later validates against that count.

### Query

1. Root, Branch, and leaf queries are parent-relative.
2. Every grammar operator covers empty lists, null lists, null structs, quoted
   names, negative indices, omitted and overflowed slice bounds, and exact-type
   map lookup.
3. Missing map keys and out-of-range indices become `padded`; selected nulls
   become `null`.
4. Descendant selection never compacts or shifts ancestor coordinates.
5. Multiple descendants evaluate one branch query and shared prefix once.
6. Query rank and Arrow type errors name the model address and failing segment.
7. Filters, functions, pipes, flattening, expressions, and recursive descent
   fail during query compilation.
8. An absent supervised target is valid in predict but fails in a stratum whose
   active objective plan requires it.

### Preprocessing

1. Preprocessors cover filtering, reorder, derivation, join, explode,
   regrouping, zero output, and multiple output batches.
2. Lineage remains stable through filter, reorder, and one-to-many expansion.
3. Sampling occurs after preprocessing and treats expanded outputs
   independently.
4. Incompatible output schemas fail with processor and schema context.
5. Undeclared preprocessors disable projection pushdown rather than losing
   columns.
6. Partition-scoped global operations fail, while dataset-scoped equivalents
   materialize stable indexed output before sharding.
7. Returning a bare table or changing rows through `replace` fails rather than
   silently losing lineage.
8. Explicit expansion ordinals make identity independent of preprocessor output
   chunking.
9. `take` preserves processed order while `order` deliberately changes ordered
   prediction output.
10. `scope` accepts only `partition` or `dataset`; `requires` and `produces`
    accept unique top-level names and drive exact scanner projection without
    making users declare passthrough model columns.
11. `requires=None` retains all source columns, while `requires=()` retains
    only automatic modeled passthrough and explicit `retain` columns.
12. Bound fitted values are identical across strata, workers, and ranks;
    ordinary iteration never mutates fitted statistics.
13. Nested Awkward transforms return plain Arrow with the declared schema and
    preserve null versus empty collections.
14. `Batch.explode` handles list, large-list, and fixed-size-list columns,
    emits no rows for null or empty lists, and derives stable child identity.

### Sampling And Distribution

1. The same seed, epoch, and topology produce the same identity sequence.
2. A new training epoch changes ordering; evaluation remains stable.
3. Without-replacement rank and worker partitions are pairwise disjoint and
   have complete coverage of the epoch set.
4. Changing local worker count preserves rank membership and the
   pre-scheduling ordered reader merge.
5. Shuffling preserves every identity and duplicate payload count.
6. A source sorted into long target runs produces mixed early batches with the
   training default.
7. The streaming buffer mixes observations across record-batch, row-group, and
   file boundaries.
8. Rechunking and parallel-reader completion order do not change a seeded
   streaming sequence.
9. Replacement produces exactly `epoch_size` draws, permits repeated source
   observations, and uses disjoint draw identities across ranks.
10. Replacement probability is proportional to logical observations, not
   files.
11. Every supported distributed training plan emits the same number of complete
    batches; an unprovable coarse plan fails before training.
12. Sorting distributed prediction shards by persisted `(identity.order,
    identity.instance)` reconstructs complete global processed order
    independent of rank or worker completion order.

### Plugins And Output

1. Each built-in receives a whole Arrow leaf column, not scalar preparation
   callbacks.
2. A third-party plugin registers Arrow compatibility without editing the
   shared engine.
3. Vocabulary and stateful plugin ordering follow shuffled logical observation
   order exactly.
4. Every writable built-in declares a stable `StructType` before prediction
   and returns that exact type with one value per decoded coordinate.
5. Output-plan compilation follows checkpoint restore, vocabulary sync,
   `Model.update`, normalized schema `mask` tuples, and `retain`; relevant
   mutations invalidate and recompile it for the next run.
6. A third-party plugin declares and writes an Arrow extension or nested type
   without a datatype case in the shared writer.
7. Plugin type or coordinate-count mismatches, missing planned addresses, and
   unplanned forward addresses fail with plugin and model
   address context.
8. Batch size one, branch length one, vector width one, all-null content,
   empty vocabulary, top-k, and thresholded candidates retain the compiled
   output schema and shape.
9. A repeated plugin receives `N × product(ancestor lengths)` coordinates and
   produces the documented coordinate-first nested layout.
10. Category top-k and Set candidates use `list<struct<value, probability>>`
   without recursive Python packing or value-dependent fields.
11. Shared state, inferred, and embedding assembly is identical across plugins
    and is not reimplemented in built-ins; `output=None` omits state and
    inferred while still permitting embedding-only output.
12. Plugin fragments using reserved shared names are rejected.
13. The standard output bridge returns Arrow without `.tolist()`,
    `.to_pylist()`, object-dtype arrays, Python row loops, or Polars frames.
14. `Model.write` returns an identity-aligned `Batch`; `Model.predict` returns
    the same values as a `pa.Table` without exposing internal identity.
15. Retained prediction inputs and identity remain Arrow until an explicit
    caller conversion or the deployment JSON boundary.
16. Postprocessors execute once per model batch in interactive, callback, and
    deployment runtimes and may change columns only through same-row,
    same-order `Batch.replace`.
17. Returning a mapping or `None`, changing row count or order, or producing a
    zero-column result fails with postprocessor context.
18. `predict_step` converts tensors once even with multiple callbacks or
    `return_predictions=True`.
19. The callback writes Arrow directly through `ParquetWriter`, round-trips
    nested output exactly, closes on completion and exception, and never
    imports or constructs Polars output.
20. The writer locks its first postprocessed schema and rejects later field,
    order, or type drift without compatibility casting.
21. Rank shards contain non-null, disjoint identity and can be globally sorted
    without relying on filenames or completion order.
22. Writer persistence includes protected identity, while ordinary
    `Model.predict` and JSON responses exclude it.
23. No-output prediction and empty-retain `inputs` use typed null columns and
    round-trip through Parquet without an empty struct.
24. Deployment invokes write and postprocess once per valid microbatch,
    restores request order around row errors, and performs a single terminal
    Arrow-to-Python conversion before optional Pydantic validation and JSON.
25. Default deployment JSON contains predictions but neither retained inputs
    nor identity; postprocessed deployment JSON contains exactly the
    postprocessor's row fields.
26. Query-backed mask selection rejects selector shape or owner-geometry drift.
27. `"<MASK>"` remains ordinary string content and the legacy literal symbols
    are absent from the public API.

## Benchmark Plan

Measure the complete loader and its phases separately:

- Polars, custom, synthetic, Arrow Table, Parquet, NDJSON, and explicit
  caller-owned Python conversion;
- direct binding and every supported query operator;
- no preprocessor, Arrow-native filtering, join, and one-to-many expansion;
- full indexed shuffle and several streaming byte budgets;
- batch sizes 1, 4, 32, and 256;
- 1, 4, and 16 fields;
- nested lengths 64, 1,000, 10,000, and 100,000;
- 0%, 10%, and 50% nulls;
- single process, scanner threads, and explicit DataLoader workers;
- single-rank and distributed ownership;
- Number, Vector, embedding, Category top-k, and thresholded Set output;
- canonical writing, Arrow postprocessing, and direct Parquet persistence;
- deployment microbatches of 1, 32, and 256 requests, with terminal JSON
  conversion reported separately.

Report:

- observations and retained leaf values per second;
- median and p95 model-batch latency;
- scan, preprocess, sample/shuffle, query, coalesce, plugin, and Torch time;
- tensor-to-CPU, CPU-buffer-to-Arrow, shape assembly, postprocess, Parquet, and
  terminal JSON time;
- peak and steady-state RSS;
- bytes copied versus shared;
- Python object and row allocation counts before the terminal JSON boundary;
- realized shuffle rows, bytes, and source span;
- worker startup and epoch transition time;
- GPU idle time during real training.

A 10,000-item benchmark must verify that Arrow-native cases allocate no Python
rows. Custom and synthetic cases must bound temporary mappings by
`ingress_rows` and retain none after ingress; wall-clock comparisons alone are
insufficient. Output benchmarks must use the same tensors and assert numerical
and label parity while comparing the current collection path with the Arrow
writer. Intentional structural changes such as stable singleton dimensions and
Set candidate lists use explicit golden schemas and values. Unchanged outputs
use semantic `Table.equals` after documented legacy-shape normalization;
equivalent chunking, dictionary encoding, or buffer placement need not be
byte-identical. Record the collection baseline before deleting it rather than
retaining a compatibility engine.

## Acceptance Criteria

The design is complete when:

- Arrow is the only persistent data representation after ingress;
- `ArrowDataModule` owns the only loader implementation;
- `PolarsDataModule`, `CustomDataModule`, and `SyntheticDataModule` retain
  focused public ingress contracts and delegate into that loader;
- `StreamingDataModule`, source-specific datasets, and standalone loader
  functions are no longer public;
- all three data adapters and the high-level Python-record prediction adapter
  are ingress paths, not alternate engines;
- no standard dataset stage materializes Python rows;
- `query` resolves supported nested structs, lists, fixed-size lists, and maps
  entirely through Arrow;
- branch-level queries guarantee descendant coordinate alignment;
- preprocessors remain capable of coordinated and cardinality-changing Arrow
  transformations;
- training is meaningfully shuffled by default;
- seeded membership and ordering are reproducible and distributed streams are
  disjoint;
- replacement has one logical-observation definition across every source;
- plugins own all datatype-specific Arrow behavior;
- coalescing shares branch geometry and each plugin receives its raw Arrow
  column once;
- every writable plugin declares a stable Arrow output type and writes a
  coordinate-aligned `StructArray`;
- `Model.write` returns `Batch` and `Model.predict` returns `pa.Table` without
  value-dependent squeezing;
- output plans are frozen per run and invalidated by output-affecting model
  mutations;
- `retain` explicitly controls the processed input columns available to every
  output runtime;
- postprocessors return same-row, same-order, identity-aware `Batch` values;
- the callback writes Parquet directly from Arrow without Polars;
- deployment writes and postprocesses once per microbatch and creates Python
  rows only at the terminal JSON boundary;
- retained prediction inputs, identities, and writer output remain
  Arrow-native;
- Polars is an optional dependency used only by the `PolarsDataModule` ingress
  adapter;
- parity, allocation, randomization, and large nested benchmarks pass.

## References

- [Polars `DataFrame.to_arrow`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_arrow.html)
- [Arrow tables and record batches](https://arrow.apache.org/docs/python/api/tables.html)
- [Arrow Dataset scanner](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Scanner.html)
- [Arrow compute functions](https://arrow.apache.org/docs/python/api/compute.html)
- [Awkward conversion to and from Arrow](https://awkward-array.org/doc/main/user-guide/how-to-convert-arrow.html)

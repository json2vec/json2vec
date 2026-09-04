# Unified Mask Design Spec

- Status: Implemented breaking contract
- Date: 2026-09-03
- Audit baseline: `16d9471`
- Scope: Schema policies, Arrow selection, RaggedField projection, tensorfield
  conversion, routing, objectives, prediction, and preprocessing
- Related design: [Arrow-Native Pipeline](arrow-data-pipeline-design.md)

This document supersedes the masking roadmap inside the Arrow pipeline design.
That document remains authoritative for the implemented Arrow foundation; this
one owns the current masking contract and its implementation rationale.

## Decision Summary

RelFlow has one masking concept and one node configuration surface:

```python
mask: (
    rf.Mask
    | float
    | bool
    | list[rf.Mask]
    | tuple[rf.Mask, ...]
) = False
```

Every leaf and branch accepts that argument. The generated root receives the
same configuration through the reserved `Model(mask=...)` argument; a modeled
field literally named `mask` must therefore be passed positionally with an
explicit name, such as
`rf.Model(rf.Category(name="mask", size=128), ...)`. The same rule applies to
children of every `Branch`. `masks` remains available as an ordinary modeled
field name. There is no separate `masks`, `skip`, `skips`, `p_mask`,
`p_prune`, or public `rf.Prune` configuration API.

An `rf.Mask` answers three independent questions:

1. **Selection:** which node coordinates are selected?
2. **Effect:** does selected input become a learned mask representation, or is
   it skipped by encoder input entirely?
3. **Purpose:** is the selection training dropout, reconstruction, or an
   ablation without its own objective?

The common forms are:

```python
model = rf.Model(
    d_model=64,
    n_layers=2,
    n_heads=4,

    # Uniform learned-mask dropout.
    merchant=rf.Category(mask=0.10),

    # Data-selected dropout that never reaches the tensorizer or embedder.
    amount=rf.Number(mask=rf.Mask(query="drop_amount", skip=True)),

    # Masked-value reconstruction.
    notes=rf.Text(mask=rf.Mask(rate=0.15, reconstruct=True)),

    # Always-skipped reconstruction target. `mask=True` is the concise form.
    label=rf.Category(mask=True),

    # A persistent data-defined ablation: skip without reconstructing.
    balance=rf.Number(
        mask=rf.Mask(
            query="remove_balance",
            skip=True,
            dropout=False,
        ),
    ),

    # One data-defined decision per event, shared by every event field.
    events=rf.Branch(
        length=128,
        mask=rf.Mask(query="selected", skip=True),
        sku=rf.Category(size=2048),
        quantity=rf.Number,
    ),
)
```

Arrow owns selection and fixed schema geometry. Awkward is an optional,
transient view for constructing nested Boolean selector fields inside a
preprocessor. Torch still owns the final device-side `present`, `trainable`,
and `inferred` bits required by embedding, attention, pooling, loss, and
prediction writing.

The mask engine never overwrites the post-preprocessor `rf.Batch.data` with
nulls or sentinel values. The coalescer resolves masks against pristine Arrow
data, then creates separate input and target projections before forward input
or target tensorization.

## Why The Current Design Must Change

The implementation at the audit baseline had four unrelated behaviors:

- leaf `p_mask` samples every dense tensor position and substitutes a mask;
- fractional leaf `p_prune` samples whole observations but still substitutes a
  mask and still calls the input embedder;
- exactly `p_prune=1` / `target=True` skips the input parcel;
- branch `Mask` selection is recomputed independently for every descendant
  leaf.

The final point meant a branch mask did not guarantee atomic sibling
selection. Padding is also embedded and included in attention today. A real
skip contract therefore requires a routing change, not merely an API rename.

The new Arrow coalescer already computes each branch `Layout` once and reuses it
for all descendants. That shared layout is the correct place to resolve a
branch mask once.

## Public Mask Model

The public model is frozen, strict, and rejects unknown fields:

```python
class Mask:
    # Selection. `query` supplies eligible coordinates; `rate` may sample them.
    query: str | None = None
    rate: float | None = None

    # Encoder effect.
    skip: bool = False

    # Purpose. `dropout` is normalized during validation.
    dropout: bool | None = None
    reconstruct: bool = False
```

`dropout` defaults to `not reconstruct`. These are the only valid normalized
purpose combinations:

| `dropout` | `reconstruct` | Meaning |
| ---: | ---: | --- |
| `True` | `False` | Train-only corruption; no direct loss |
| `False` | `True` | Reconstruction objective |
| `False` | `False` | Ablation; no direct loss |
| `True` | `True` | Invalid |

This keeps reconstruction distinct from training-only dropout without
pretending that every non-reconstruction mask is train-only. The explicit
`dropout=False, reconstruct=False` state is needed for validation or prediction
ablations.

`Mask.dropout` is a Boolean policy purpose. It is unrelated to the numeric
neural-layer `dropout` configuration on model and branch modules.
It controls when a policy is active, not how coordinates are selected;
`query` and `rate` own selection.

In this spec, `trainable` names a coordinate that contributes a direct decoder
target and loss; it does not mean PyTorch `requires_grad`. A dropout
learned-mask coordinate has `trainable=False`, so the hidden value is not
reconstructed. The shared learned mask representation may still receive
gradients indirectly from other objectives that consume the corrupted context,
just as an ordinary dropout network still learns its remaining parameters.

`skip` changes only the encoder effect:

| `skip` | Selected input behavior |
| ---: | --- |
| `False` | The coordinate remains present as the datatype's learned mask representation. |
| `True` | The coordinate is absent from forward input tensorization, embedding, attention, and pooling. |

`skip` composes with every selector. A query-driven skip is conditional
structural omission; a rate-driven skip is stochastic structural omission. For
`Mask(query="exclude", skip=True)`, coordinates where `exclude` is true never
reach the input tensorizer or embedder; false coordinates remain ordinary
inputs. `Mask(rate=0.15, skip=True)` applies the same routing effect to a
stateless random 15 percent of eligible owner coordinates. On a branch, each
selected branch coordinate omits every descendant together.

Neither form deletes Arrow source rows or changes fixed model geometry:
`present` carries the omission into routing. Without a query or rate,
`skip=True` selects every owner coordinate whenever that policy is active.
Because `dropout` defaults to true for a non-reconstructing policy, bare
`Mask(skip=True)` skips the whole node during training only. Persistent skipping
is `Mask(skip=True, dropout=False)`.

The four important training combinations are therefore:

| Policy | Encoder sees | Own reconstruction loss |
| --- | --- | ---: |
| `Mask()` | Learned mask representation | No |
| `Mask(reconstruct=True)` | Learned mask representation | Yes |
| `Mask(skip=True)` | No input representation | No |
| `Mask(skip=True, reconstruct=True)` | No input representation | Yes |

### Selection

`query` supplies eligibility for every policy:

- no `query` means every occupied coordinate owned by the node is eligible;
- `query="selected"` reads one Boolean per owned coordinate.

Every policy may additionally sample its eligible coordinates:

- no `rate` selects every eligible coordinate;
- `rate=x` selects eligible coordinates with uniform probability `x`;
- `rate=0` is a no-op and `rate=1` selects every eligible coordinate.

| Selection form | Declaration | Decision source |
| --- | --- | --- |
| All coordinates | `Mask()` | Schema literal |
| Uniform random | `Mask(rate=0.15)` | Stateless RelFlow sampler |
| Literal per-record flags | `Mask(query="selected")` | Boolean Arrow source field |
| Dynamically derived flags | `Mask(query="selected")` | Boolean field created by a preprocessor |
| Sample eligible flags | `Mask(query="eligible", rate=0.15)` | Preprocessor/source eligibility plus sampler |

Literal and derived flags deliberately share one runtime contract. Their
provenance does not matter after preprocessing; alignment comes from residing
in the same Arrow records as the values they select.

Normalization removes a `rate=0` policy and rewrites `rate=1` to `rate=None`,
preserving any query. After normalization, prediction can distinguish a
deterministic all/query selector from an inactive generated-random selector
without special-casing numeric endpoints.

The first release deliberately has no `count`, `window`, `start`, `offset`, or
`branch` controls. Awkward or Arrow preprocessing expresses those operations
more clearly, and a queried selector composes with a uniform rate when sampling
within a custom candidate set is useful.

A mask operates at its owner's natural coordinate:

- a root leaf owns one coordinate per observation;
- a leaf inside a repeated branch owns one coordinate per retained parent
  record;
- a branch owns each of its retained branch records;
- the generated root owns each observation.

There is no separate `unit` switch. A preprocessor can broadcast an
observation-level Boolean over a repeated axis when whole-observation selection
is required.

### Shorthands

Normalize booleans before numbers because `bool` subclasses `int`:

| Node argument | Canonical node value |
| --- | --- |
| `mask=False` | `()` |
| `mask=0.15` | `(Mask(rate=0.15),)` |
| `mask=True` | `(Mask(skip=True, dropout=False, reconstruct=True),)` |
| `mask=Mask(...)` | `(Mask(...),)` |
| `mask=[Mask(...), Mask(...)]` | `(Mask(...), Mask(...))` |
| `mask=(Mask(...), Mask(...))` | `(Mask(...), Mask(...))` |

The `True` spelling intentionally replaces node-level `target=True`: a
supervised answer is structurally withheld, not represented by a learned mask
token. Users who want a full learned-mask reconstruction write
`Mask(reconstruct=True)` explicitly.

The bool-first normalization is intentionally visible because `mask=True` and
`mask=1.0` mean very different things. `True` is a skipped reconstruction;
`1.0` is full learned-mask dropout. Documentation and validation errors must show
their canonical forms rather than treating them as interchangeable numeric
values.

After node construction, every accepted form becomes one immutable
`tuple[Mask, ...]` stored under `mask`: the default `False` becomes `()`, a
singleton becomes `(normalized,)`, and a list or tuple is defensively copied to
a tuple. `None` is invalid rather than a second spelling for the default. Each
policy is normalized before duplicate policies are collapsed.
Collection entries must be `Mask` objects; Boolean and numeric shorthands are
accepted only as the singleton form, and nested containers are invalid. An
empty collection is the explicit no-policy form. Canonical serialization
therefore stores only the normalized `mask` tuple, never the original container
or shorthand.

Normalization order is part of the contract:

1. classify the outer `mask` Boolean before numeric shorthands;
2. validate each raw `Mask` field strictly;
3. replace `dropout=None` with `not reconstruct` and reject dropout plus
   reconstruction;
4. remove `rate=0` policies and rewrite `rate=1` to `rate=None`;
5. collapse exact canonical duplicates while preserving first declaration
   order;
6. only then compile queries, capabilities, objectives, and output plans.

Explicit `rate=None` is equivalent to omission. Endpoint normalization happens
before any query, capability, objective, or output-plan work.

## Branch Atomicity

A mask attached to a branch is a subtree policy.

RelFlow resolves or samples it exactly once for each retained coordinate of
that branch. The resulting decision is then broadcast to every active
descendant leaf. It is never resampled per leaf.

For a nested subtree:

```text
record
└── sessions[]             <- one decision for session 1
    ├── device                  same decision
    ├── duration                same decision
    └── events[]
        ├── kind                same decision for every child event
        └── amount              same decision for every child event
```

Each descendant intersects the broadcast decision with its own owner geometry.
An absent or empty nested branch does not acquire fabricated records, and
padding never becomes a reconstruction target. Within an existing owner record,
a missing leaf path can become a requested prediction coordinate under the
explicit reconstruction rules below; during fitting structural absence simply
contributes no target.

The first release applies a branch policy to the whole active subtree. There is
no `exclude` mini-language. Attach the mask to a narrower branch or to a leaf
when a narrower ownership boundary is required. A reconstructing branch mask
requires every active descendant leaf to support reconstruction; schema
validation fails otherwise.

A root mask follows the same rule. Skipping every input for one observation is
valid and exercises the defined zero-context decoder behavior. There is no
root-only exception.

## Masks From Preprocessing

Preprocessors define **where** a dynamic mask applies. The schema defines what
that selection means: learned mask versus skip, dropout versus reconstruction.

This separation is important. A batch must not invent a new reconstruction
role or change the model's output contract at runtime.

The simplest carrier is an ordinary, unmodeled Arrow Boolean field selected by
`Mask.query`. It automatically stays aligned through filtering, sampling,
shuffling, and rebatching. It does not require a second identity-bearing mask
container or new behavior on every `Batch` lineage method.

For a scalar field:

```python
import pyarrow.compute as pc
import relflow as rf


@rf.preprocess(
    requires=("amount",),
    produces=("mask_amount",),
)
def prepare(batch: rf.Batch) -> rf.Batch:
    selected = pc.greater(batch.data["amount"], 1_000)
    selected = pc.fill_null(selected, False)
    data = batch.data.append_column("mask_amount", selected)
    return batch.replace(data)


model = rf.Model(
    d_model=32,
    n_layers=1,
    n_heads=4,
    amount=rf.Number(
        mask=rf.Mask(query="mask_amount", skip=True),
    ),
)
```

The same field may arrive literally in source data instead of being derived:

```python
records = [
    {"amount": 20.0, "mask_amount": False},
    {"amount": 40.0, "mask_amount": True},
]
```

For repeated data, Awkward makes the coordinate calculation concise:

```python
import awkward as ak
import relflow as rf


@rf.preprocess(requires=("events",), produces=("events",))
def prepare(batch: rf.Batch) -> rf.Batch:
    index = batch.data.schema.get_field_index("events")
    events = ak.from_arrow(batch.data["events"])

    selected = ak.fill_none(events["amount"] < 0, False)
    events = ak.with_field(events, selected, "selected")

    values = ak.to_arrow(events, extensionarray=False)
    data = batch.data.set_column(index, "events", values)
    return batch.replace(data)


model = rf.Model(
    d_model=32,
    n_layers=1,
    n_heads=4,
    events=rf.Branch(
        length=128,
        mask=rf.Mask(query="selected", skip=True),
        kind=rf.Category(size=128),
        amount=rf.Number,
    ),
)
```

That one nested Boolean hides `kind` and `amount` together for each selected
event during training. Set `dropout=False` for an evaluation/prediction
ablation, or `reconstruct=True` to reconstruct the selected descendants. A
production preprocessor should cast the result to a declared Arrow type so
empty partitions cannot change its schema.

### Selector Query Contract

Mask queries use RelFlow's existing Arrow-native query grammar. They do not use
JMESPath and do not add predicate syntax.

- A leaf mask query is evaluated against the same containing records used to
  project that leaf. It may traverse nested structs or maps with the existing
  path grammar, but it must resolve one scalar Boolean for each owner record.
- A branch mask query is evaluated against that branch's already-retained flat
  `Layout.records`, after the branch query and overflow policy. Its selector
  therefore resolves inside each selected child record, as in the Awkward
  example above—not from a parallel list beside the branch column.
- The query projection must have `len(values) == len(Layout.records)`, Boolean
  value type, every structural `present` bit true, and no null values. The
  coalescer then scatters it through the owner's existing `Layout.placement`;
  it does not independently reconstruct nested offsets.
- Structural absence or null selector values are errors. The preprocessor must
  choose explicitly, normally with `fill_null(..., False)`.
- Length or structural-presence drift reports the policy query and owner
  address. A parallel Boolean list outside the owner records is intentionally
  unsupported.
- Selector fields are unmodeled unless the schema also declares them as leaves;
  they never reach a tensorfield plugin merely because a mask reads them.
- An active mask query contributes its source root to scan requirements unless
  a declared preprocessor `produces` it. Projection pushdown may not discard an
  unmodeled selector needed later by coalescing.

An optional `rate` samples only the `True` query results:

```python
mask=rf.Mask(query="eligible", rate=0.25)
```

Custom windows, exact counts, stratified decisions, and correlated policies
belong in the preprocessor. This avoids rebuilding a second array language
inside `Mask`.

Preprocessor selector fields should be deterministic eligibility or final
selection. Use `Mask.rate` for epoch-varying randomness, including stochastic
skip. Random values created by a preprocessor rerun at that processor's
iteration boundary and do not inherit the mask sampler's identity, epoch, or
rebatching guarantees. Dataset-scoped preprocessing currently materializes and
reruns for each iteration; it is not a persisted selector cache.

An ablation policy (`dropout=False, reconstruct=False`) is active in every
stratum. If it should be prediction-only, the preprocessor can request its
provided `strata` argument and emit `False` outside `rf.Strata.predict`; this
keeps stratum routing in data rather than adding another Mask option.

The selector pattern is visible to the model through either a learned mask token
or structural absence. A selector derived from the value being hidden can
therefore leak information. For example, masking exactly negative amounts
reveals their sign. Documentation calls out that data-dependent masking is
an intervention, not an automatic information barrier.

## Execution Boundary

Masking now resolves before datatype tensorization, but it cannot live entirely
in Arrow:

```text
Arrow Batch
  -> preprocessor (Awkward may be a temporary view)
  -> query + shared branch Layout + overflow + pristine leaf projection
  -> resolve each Mask once
  -> plugin Arrow canonicalization + observation against the pristine leaf
  -> project visible inputs and selected targets
  -> TensorField.new tensorizes both projections
  -> TensorField(state, content, targets, trainable, present, inferred)
  -> model-side learn from carried observations (train only)
  -> input embedding + present-aware routing
  -> decoder and loss
```

Arrow is the right plane for selection because it still has pristine values,
nested offsets, branch identity, and shared sibling geometry. Torch remains
necessary because learned mask representations, accelerator-side embedding,
attention key exclusion, pooling denominators, and loss selection are tensor
operations.

Awkward remains a user-facing compute tool, not a persistent runtime carrier.
The core does not convert every model batch to Awkward merely to apply a
Boolean mask. The Arrow/NumPy coalescer uses the vectorized kernels that keep
the implementation simplest.

## Arrow Projection Contract

Mask resolution leaves the post-preprocessor source table unchanged. Selection
creates projections; it does not replace modeled values in that table.

For each leaf, coalescing distinguishes structural and value geometry. Every
union below considers only policies active in the current stratum:

```text
owner     = coordinates backed by an actual root or retained branch record
observed  = owner coordinates whose leaf state is valued or explicit-null
requested = owner & union(selected policy where reconstruct=True)
modeled   = observed | (predict & requested)

skipped   = modeled & union(selected policy where skip=True)
masked    = modeled & ~skipped & union(selected policy where skip=False)
trainable = non-predict & observed & requested
inferred  = predict & requested
present   = modeled & ~skipped
visible   = observed & ~masked & ~skipped
```

During train, validation, and test, reconstruction target selection intersects
`observed`. Structural absence therefore creates neither a fabricated target
nor an error for one sparse coordinate; a wholly absent target column still
fails binding.
An explicit Arrow null is observed and remains a valid null-state target.
Omitted keys in Python mappings normally become Arrow nulls at ingress and have
the same semantics—RelFlow cannot recover key-presence information Arrow no
longer contains.

In prediction, a deterministic reconstruction mask may promote a structurally absent
leaf within an existing owner record into `modeled`. That is how a missing map
member or structurally absent field receives a learned mask representation or
skipped-output request
without an in-band sentinel. A queryless leaf with unconditional reconstruction
may be absent from prediction input. A query-backed reconstruction still requires both
its selector and modeled source columns in the first release, even if every
selector value is true, because binding cannot depend on one batch's flags.
Repeated targets still require the containing branch records to define their output
coordinates; RelFlow does not invent unknown repeated cardinality. Mixed rows
must provide valid typed input anywhere the field remains visible.

When an unconditional prediction reconstruction has no source column, its value carrier
is the one generic vacancy marker: a zero-length Arrow NullArray with empty
placement. It represents “no raw values supplied,” not a null-valued target.
The shared engine never guesses one of the plugin's accepted physical types.
The request or restored plugin state must already define every input/output
contract needed without source data; otherwise prediction-plan compilation
fails before reading a batch.

Generated-root owner geometry always comes from `len(Batch)`, not
`Batch.data.num_rows`. This matters when all prediction fields are omittable:
Arrow cannot represent a positive-row, zero-column table by itself, while the
Batch identity still has `N` rows. Coalescing uses a length-`N` empty-struct
carrier in that case, and mapping ingress preserves the count of inputs such as
`[{}, {}]` before Arrow conversion.

Composition is deliberately monotonic:

- any skip wins the encoder effect;
- any reconstruction wins objective eligibility;
- overlapping policies are independent of declaration order;
- dropout never erases another policy's reconstruction target;
- geometry outside `owner` is never selected, embedded, or trainable.

This is simpler than assigning one winning policy. For example, overlap between
a dropout skip and a reconstruction mask produces a skipped
reconstruction target: the strongest effect and strongest objective both win.

Coalescing produces two `RaggedField` projections with the original fixed
`shape`:

```text
input.state              = pristine state
input.state[masked]      = Tokens.masked
input.state[skipped]     = Tokens.padded
input.values/placement   = pristine valued entries where visible

target.state             = Tokens.padded everywhere
target.state[trainable]  = pristine state[trainable]
target.values/placement  = pristine valued entries where trainable
```

Each projection independently satisfies the existing `RaggedField` invariant:
`placement` names every and only valued state position. A selected explicit-null
target carries pristine null state but no raw value. `present` and the resolved
policy plan distinguish an intentional skip from original padding; there is no
new state token.

The forward input tensorizer receives only `input.values`. The separate
plugin-owned observation phase may inspect pristine values before this split,
as described below. Consequences:

- a dropout Mask does not waste forward input tensorization on hidden
  content;
- a dropout skip never exposes the selected raw value to the forward
  input tensorizer or input embedder;
- a reconstructing skip may use a separate target tensorizer;
- a Text reconstruction target may run its frozen encoder for the target
  representation, but that result never enters encoder context;
- changing skipped bytes while selection and estimator state are fixed cannot
  change the forward result.

Plugin estimation may observe pristine training exposure once before input
projection. Skipping is therefore an encoder ablation, not a data-secrecy
boundary. Use a preprocessor or `active=False` when the value must not be
inspected by any estimator.

### Plugin Lifecycle

Plugins receive five ordered phases for each leaf:

1. The shared engine builds the pristine `RaggedField`, then resolves and
   validates every applicable selector against its owner Layout. Invalid
   selection cannot mutate plugin state.
2. Existing `plugin.prepare(pristine.values, address=...)` validates and
   canonicalizes the complete Arrow value array once. It does not learn. The
   canonical array replaces `pristine.values` before either projection. This
   step is skipped for the generic source-less vacancy marker.
3. A registered, default-no-op
   `observe(field, *, address, schema, state, learn)` component sees the whole
   canonical pristine RaggedField—state, values, placement, and shape—and may
   return plugin-owned sufficient statistics. It is also skipped for the
   source-less vacancy marker.
4. The shared engine filters that canonical array into separate input and
   target RaggedFields without invoking datatype logic.
5. `TensorField.new(...)` tensorizes both projections against one frozen
   conversion context.

The global `Plugin` registry never stores learned state. `state` is the
address-specific entry in the model's synchronized, checkpointed encoding
state. `observe` must not mutate an authoritative model module
inside a DataLoader worker. Worker-side mutation is limited to that explicit
interprocess state. The same call may also return a bounded,
batch-transferable statistic for a distinct model-side resource. The plugin's
model-side `learn` component reduces that statistic across ranks and applies it
exactly once for each consumed training batch. No resource may be updated by
both paths. With `learn=False`, neither path may mutate state. Stateless plugins
use the default no-op. The shared engine never inspects datatype names.

The plugin component contract is explicit:

```python
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None: ...


def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None: ...
```

`Component.observe` and `Component.learn` are optional registrations with
validated signatures and default no-ops. Returning an observation without a
registered learner is an error. When a learner is registered and `learn=True`,
its observer must return the plugin's fixed-schema observation even when the
local leaf has no valued rows. The empty form carries zero counts rather than
using `None`. Every rank invokes every configured learner exactly once, in
deterministic schema-address order; only after its collectives may a learner
treat a globally empty observation as a no-op. `None` is valid only when no
learner is registered or learning is disabled. `Encoded.observations` carries
the immutable `Mapping[Address, TensorDict]` beside tensors/source/retain and
follows ordinary device transfer. At training-batch consumption, the runtime
invokes every learner once before forward and then discards that batch's observations. A
direct `Model.encode(strata="train")` call learns synchronously before returning
its TensorDict; validation, test, and prediction return no learnable
observations. This state mutation means train-stratum direct encoding is not
covered by the model's inference-immutability guard.

The split must not learn twice. Category- or Set-like plugins may reserve
vocabulary entries through address state and return counter observations for a
separate model-side resource in the same call. A Number-like plugin can return
sufficient statistics without mutating interprocess state. Its learner
all-reduces and updates running statistics before forward; the input embedder
and target loss then read that same authoritative post-update snapshot without
updating it again. Batch `N` therefore uses statistics through batch `N`,
matching the baseline online-normalizer order. The observer never rewrites the
pristine RaggedField. A reconstruction-only Text-like plugin may run a frozen
target encoder after the split. These are examples of
plugin-owned behavior, not datatype cases in the shared engine.

All data-derived plugin statistics move to this pristine observation path,
including state/content counters. A counter must not infer pristine exposure by
preferring sparse `targets` after projection. Observation resource creation,
synchronization, and checkpoint restoration are part of the plugin contract.

One immutable batch conversion context is created per encoded Batch and passed to
both projections at every address. Batch-scoped resources such as a hash salt
are therefore shared across input and target conversion and across compatible
fields.

The constructor receives enough shared state to preserve distinctions erased
inside Ragged state:

```python
TensorField.new(
    input=input,
    target=target,
    present=present,
    trainable=trainable,
    inferred=inferred,
    address=address,
    schema=schema,
    strata=strata,
    state=encoding[address],
    batch=conversion,
)
```

The old `TensorField.empty` special path is removed. Source-less unconditional
prediction targets and mock inputs use the generic zero-length NullArray
vacancy, empty placement, explicit shared routing bits, and the same
constructor. Every plugin must accept that carrier when it contains no placed
values; no plugin may interpret it as a raw null atom.

This lifecycle deliberately permits estimator state to depend on a skipped
training value. The guarantee is narrower and testable: once estimator state is
fixed, the value cannot reach the forward input tensorizer or embedder, and
changing it cannot change that batch's encoder result.

## Tensor And Routing Contract

Every tensorfield carries the minimal shared runtime state:

```python
state: torch.Tensor
content: torch.Tensor | TensorDict
targets: TensorDict
trainable: torch.BoolTensor
present: torch.BoolTensor
inferred: torch.BoolTensor
```

There is no `dropped` tensor. It is derivable from the resolved policy plan
and is useful only for diagnostics.

| Coordinate | `present` | Working state/content | `trainable` | `inferred` |
| --- | ---: | --- | ---: | ---: |
| Visible value or null | `True` | Original | `False` | `False` |
| Padding / absent owner | `False` | Safe filler | `False` | `False` |
| Learned Mask dropout | `True` | Masked state / safe content | `False` | `False` |
| Learned Mask reconstruction during fitting | `True` | Masked state / safe content | `True` | `False` |
| Learned Mask reconstruction during prediction | `True` | Masked state / safe content | `False` | `True` |
| Skip dropout or ablation | `False` | Safe filler | `False` | `False` |
| Skip reconstruction during fitting | `False` | Safe filler | `True` | `False` |
| Skip reconstruction during prediction | `False` | Safe filler | `False` | `True` |

`Tokens.masked` remains the learned-mask state. There is no skip token:
`present=False` carries that fact.

Every `Parcel` also carries `present` with
`present.shape == payload.shape[:-1]`. Flattening, concatenation, attention,
and pooling transform payload and presence together. False entries:

- cannot be attention keys or values;
- do not enter a mean-pool numerator or denominator;
- never shift a later coordinate into a different rotary position.

Fixed schema slots remain in place. Skipping changes participation, not
position numbering.

The shared embed path flattens coordinate axes, gathers indices where `present`
is true, calls the embedder only for those coordinates, then scatters the
resulting `[K, d_model]` payload into fixed schema slots. It gathers `present`,
not merely `visible`: explicit nulls and learned mask states are real model
inputs. If `K == 0`, the embedder is skipped.

Query- and rate-selected skip therefore create a dynamic `K`, but not a ragged
tensor contract for the rest of the model. Variability is contained inside the
shared gather/embed/scatter wrapper; TensorField geometry, Parcel geometry, and
branch positions remain fixed, with `present` carrying holes. Standard masked
dense attention is sufficient for correctness, though it does not reduce the
dense attention FLOP count. Packed variable-length attention is a separate
optimization, not a prerequisite for dynamic skip.

The shared carrier makes target non-exposure structural:

```python
class TensorInput:
    state: torch.Tensor
    content: torch.Tensor | TensorDict
    batch_size: int
```

`TensorFieldBase.take(indices) -> TensorInput` performs the semantic gather once
for every plugin. It includes every content Tensor/TensorDict leaf whose prefix
equals `state.shape`, flattening and gathering exactly that row-major coordinate
prefix while preserving arbitrary trailing feature axes. It cannot carry
`targets`, `trainable`, or `inferred` into the embedder. Plugins do not
reimplement `take`; the first release rejects content that violates the shared
prefix invariant.

An embedder accepts the compact input view and returns a `Parcel` whose payload
is exactly `[K, d_model]`. It must not inspect reconstruction targets; target
conversion stays behind the decoder/loss path. The shared wrapper scatters the
payload and restores fixed geometry, batch size, origin, destination, and
presence. This is required for position-level skipping and makes both the
no-input-embedding and no-target-leak guarantees testable for third-party
plugins as well as built-ins.

A branch is present at its parent only when it has at least one present child.
`DecoderBase.forward` gains explicit `batch_size` and `device` keyword
arguments rather than deriving both from `parcels[0]`. If its parcel list is
empty, the base bypasses pooling and calls `decode` with a zero tensor shaped
`[batch_size, product(target_shape), d_model]`. This produces the decoder's
unconditional prediction without fabricating an input parcel or invoking
attention over an all-masked memory.

The same rule applies row-wise in a mixed batch. Presence-aware pooling gathers
rows with at least one context coordinate, runs attention/pooling only for that
subset, and scatters the result beside zero target-shaped contexts for empty
rows. An all-skipped row therefore cannot create an empty mean or all-masked
attention NaN merely because another row in the batch has context.

Conditional tensorizer, embedder, branch, or decoder bypass must also be safe
under distributed training. Each rank all-reduces `local_has_objective` with
MAX. If the result is false, every rank takes the same no-update path. If it is
true, every rank returns a gradient-bearing loss; any trainable module bypassed
locally but used by a peer is covered by the configured unused-parameter
strategy or that module's zero-valued parameter anchor. No rank independently
skips backward because of local tensor contents or branch shape.

This changed the audit-baseline semantics, where padded slots reached embedding
and attention. It is an intentional part of the breaking masking contract and
requires model-quality regression tests, not only unit tests.

## Strata And Output Semantics

After normalization, policy activation is exact:

```text
train                = active
validate / test      = not dropout
predict              = not dropout and rate is None
```

`skip` changes the encoder effect only; it never changes these stratum rules.

| Purpose | Train | Validate/Test | Predict |
| --- | ---: | ---: | ---: |
| Dropout | Apply | Ignore | Ignore |
| Reconstruction | Apply + loss | Apply + loss | Apply only for deterministic all/query selection |
| Ablation | Apply | Apply | Apply only for deterministic all/query selection |

After endpoint normalization, any policy that still has a `rate` is generated
randomness and is inactive during ordinary prediction. This applies even when
the policy also has a query: the query defines eligibility, but the remaining
rate makes final selection stochastic. Query-only selection is deterministic
input data and may run in prediction. A reconstructing query requests decoder
execution for its selected owner leaves and contributes a public field only
when the plugin declares an Arrow output. A query-backed skip with
`dropout=False, reconstruct=False` performs an inference ablation without adding
output.

An always-selected `Mask(skip=True, reconstruct=True)` is a schema
reconstruction objective. It never enters encoder context in any stratum and
is always decoded. This is the canonical form produced by `mask=True`.

A stochastic or query-backed reconstruction is still a static model objective.
Runtime loss dispatch occurs only when `trainable.any()`. An empty selection
skips that leaf's loss; it never computes an empty mean or produces NaN.

### Compiled Plans

The schema must not overload one internal `schema.target` collection with
several meanings. Compilation derives four immutable relations with explicit
decoder/output/embedding roles:

- **objectives:** every leaf reached by any reconstructing leaf, branch, or root
  policy, after removing `rate=0` policies;
- **decodes:** the subset of objectives reached by at least one reconstruction
  policy with no remaining random rate;
- **forward:** objectives plus every `embed=True` address in train, validation,
  and test; decodes plus those embedding addresses in predict;
- **writes:** output-capable `decodes` plus every configured embedding address.

Every role returned by a stratum's forward plan is shape-validated. This keeps
stochastic reconstruction in fitting forward/loss while excluding it from
ordinary prediction decoder work and output.

Root and branch policies expand to their effective descendant leaves because
losses and decoded outputs live at leaf addresses. `where("reconstruct")`
selects those effective objective leaves, not the policy-owning branch.
Ordinary node predicates can select policy owners when mutation needs that
distinction.

Input withholding and source omission remain derived per-stratum geometry, not
additional named plans. This correctly handles an unconditional
non-reconstructing ablation, which is always absent from forward input, and a
default `Mask(skip=True)`, which is absent only during train. A queryless, rateless
prediction reconstruction may omit its source leaf; a repeated ancestor remains
required to establish cardinality.

Reconstruction capability and public output capability are deliberately
different. A Text-like plugin may support a training objective while declaring
`output=None`; a deterministic reconstruction remains in forward validation but
is not written publicly unless it also has an embedding role.

The prediction output plan is static for the run. A query-backed deterministic
reconstruction contributes its expected decoder result even when the current
batch selects no coordinates. It contributes a public field only when its plugin
declares output. Each decoded public field preserves compiled geometry and
carries `inferred=True` only at selected prediction coordinates. A writer may
populate every fixed coordinate; `inferred=False` is the sole signal that
consumers must ignore an unrequested decoded value. An independent `embed=True`
role adds an embedding to the static output plan; output-capable leaf plugins
may also write their ordinary decoded state/content with `inferred=False`.
A batch's selector values can therefore change contents, never its Arrow
schema.

A reconstruction policy with a remaining random rate belongs to objectives but
not `decodes`, because it is inactive in prediction. It creates no pointless
decoded field or decoder work. An independent `embed=True` role still belongs
to forward and writes. An ablation without `reconstruct=True` likewise does
not emit the skipped input itself.

## Randomness

Random masks must not change merely because available memory changes model
batch size.

Each random score is a stateless hash of:

```text
"relflow-mask-v1" + seed + stratum + train epoch
    + Batch.identity.instance + owner address
    + normalized selection digest + nested slot tuple
```

The nested slot tuple contains only ordinals below the observation, recovered
from `Layout.placement`; it never contains the row's current position in a
batch. The selection digest covers normalized `(query, rate)` only. Changing
`skip`, `dropout`, or `reconstruct` therefore preserves sampled coordinates,
which makes learned-mask versus skip comparisons meaningful. Identical
selection specifications on the same owner intentionally share one decision and
compose their effects. Users who need independently structured decisions can
materialize separate selector fields in preprocessing.

The fixed namespace domain-separates mask scores from dataset sampling and
shuffling. Do not use global Torch call order as policy identity.

The loader passes `seed`, `epoch`, and `stratum` through encoding into
coalescing. Direct `Model.encode(...)` exposes optional `seed` and `epoch`
arguments with deterministic defaults of zero. Its current `mask: bool`
execution switch is removed; strata and normalized policies completely define
whether masks apply. Rank and worker number are not hash inputs, so moving the
same identity between workers or ranks does not change its selection.

Rebatching stability applies when the same `rf.Batch.identity` is retained.
Raw Arrow units passed in separate direct calls receive fresh positional
identity; callers that need cross-call equivalence must construct and preserve
an `rf.Batch`.

This provides:

- the same selection after Arrow rechunking or model rebatching;
- one shared branch decision across every descendant;
- independence from schema traversal order;
- reproducible validation and test selection;
- random train masks that change by epoch.

Validation and test omit the train epoch from the hash so repeated evaluation
is stable. Duplicate canonical policies are collapsed during schema
normalization because applying the same union twice has no effect.

## Tensorfield Extension Contract

The shared engine owns selection, branch propagation, pristine/input/target
partitioning, and routing presence. Tensorfield plugins continue to own every
datatype-specific operation.

The new plugin boundary must support separate input and target projections:

- existing `prepare` validates and canonicalizes pristine Arrow values once;
- registered `observe` sees the complete pristine RaggedField and owns any
  data-derived codec state or statistics in an address-specific context;
- registered model-side `learn` applies transferable observations exactly once;
- the input projection contains only visible raw values;
- the target projection contains only pristine selected values needed by a
  reconstruction objective;
- the plugin supplies safe content for learned-mask and absent fixed slots;
- the plugin declares whether reconstruction is supported;
- shared `TensorFieldBase.take` gathers input state/content across arbitrary
  trailing axes into one compact coordinate dimension without targets;
- the embedder accepts that view and returns a compact `Parcel`;
- the shared engine never branches on built-in datatype names.

Reconstruction support is mechanical, not a duplicated Boolean flag: a plugin
supports it only when both `Decoder` and `loss` are registered. Those
components become optional for input-only plugins, and `NodeModule` instantiates
them only when the compiled schema needs them. A reconstructing policy requires
both. `embed=True` on a leaf requires a Decoder but not a loss; branch embedding
uses the branch encoder. A public `output`/`write` pair never substitutes for a
missing Decoder.

The old per-plugin `mask`, `target`, and `hide` methods are removed. They
currently duplicate policy mechanics across every built-in and make it
impossible for the shared layer to guarantee that selected raw values were
never converted.

Extension tests must spy on both forward input tensorizer and embedder inputs.
Arrow canonicalization, pristine observation, and a target tensorizer may see a
skipped raw value under their explicit contracts; the encoder path may not.

## Inference Literals

`"<MASK>"` has no special meaning. String-compatible plugins treat it as
ordinary data; every other plugin applies its normal Arrow type validation.
Reintroducing an in-band sentinel would force heterogeneous Arrow unions for
numeric, binary, struct, list, and extension values.

Literal masks are Boolean selector fields:

```python
model = rf.Model(
    d_model=32,
    n_layers=1,
    n_heads=4,
    amount=rf.Number(
        mask=rf.Mask(query="infer_amount", reconstruct=True),
    ),
)

records = [
    {"amount": 10.0, "infer_amount": False},
    {"amount": None, "infer_amount": True},
]
```

Deployment request signatures may expose those fields, or a deployment
preprocessor may derive them. Selector fields stay Arrow-native and are omitted
from output by default. `retain`, including `retain="*"`, makes them available
to the output `Batch`; a postprocessor may then preserve, reshape, or remove
them.

An out-of-band `Model.predict(..., mask=...)` convenience can be added later
by materializing the same Boolean selector columns after Arrow ingress and
before coalescing. It must not create a second execution model.

## Serialization And Mutation

Canonical schema serialization stores one ordered `mask` tuple per node. Each
entry stores normalized `query`, `rate`, `skip`, `dropout`, and `reconstruct`.

- `mask=True`, `mask=False`, and numeric rates are parse-time shorthands only.
- `p_mask`, `p_prune`, node-level `target`, and the old branch Mask geometry are
  rejected.
- `rf.Prune` does not exist.
- `where("reconstruct")` is a derived semantic selector for effective objective
  leaves; it is not stored node state.
- construction and schema mutation accept the same `mask` forms and normalize
  them immediately; all later phases see only the canonical tuple.
- `Model.encode(mask=...)` is removed; optional `seed` and `epoch` control
  deterministic generated selection instead.

Checkpoint loading does not translate the old schema. This is an intentional
breaking change. In particular, an old branch `Mask(rate=x)` created
reconstruction targets in fitting strata. The new `Mask(rate=x)` is train-only
dropout and creates no reconstruction objective unless `reconstruct=True` is
explicit.

Old fractional leaf `p_prune=x` most closely maps to
`Mask(rate=x, skip=True, reconstruct=True)`: select stochastically, omit the
selected input, and retain it as a reconstruction target. This is intentionally
not bit-for-bit compatibility. The old implementation still substituted a mask
representation and sampled one decision across the observation's remaining
dimensions; the new policy performs a real skip and samples its node's natural
owner coordinates. Attach it to the generated root or an appropriate ancestor
for whole-observation selection. Old `p_prune=1` or node-level `target=True`
normally migrates to `mask=True`, whose canonical policy is
`Mask(skip=True, dropout=False, reconstruct=True)`.

## Validation

Schema and runtime validation reject:

- non-finite rates, Boolean rates, and rates outside `[0, 1]`;
- `dropout=True` together with `reconstruct=True`;
- `mask=None`;
- a `mask` value outside the accepted singleton/list/tuple forms;
- non-`Mask` entries in a `mask` collection;
- nested `mask` collections;
- mask queries that do not compile;
- missing query results in a stratum where the policy is active;
- non-Boolean or null query results;
- selector geometry that differs from owner geometry after query and overflow;
- reconstructing branch masks containing a descendant without reconstruction
  support;
- a wholly absent objective column during train, validation, or test;
- plugin output that marks a false-present coordinate as routed context.

No-context rows are valid. A model with no possible objective remains valid for
embedding and inference, while `Trainer.fit` fails clearly before the loop.

## Implementation Map

This shipped as one public breaking change across five internal phases:

1. Replaced schema fields with the normalized `Mask` model and derived
   reconstruction/objective plans.
2. Threaded seed/epoch context into Arrow coalescing, built pristine leaves,
   resolved each selector once, and projected separate input and target
   RaggedFields.
3. Added address-owned observation resources, synchronization, checkpointing,
   and the model-side `learn` phase; then replaced plugin mutation methods with the
   split constructor and shared compact `take` contract.
4. Added `present`, `trainable`, and `inferred` to TensorField and presence to
   Parcel routing, attention, and pooling.
5. Updated prediction planning, mutation, every built-in plugin, examples,
   guides, serialization tests, and benchmarks.

There is one execution path. Do not retain a legacy Torch sampler or a compact
routing compatibility mode.

## Verification Contract

The following is the normative acceptance surface for this design. Focused
tests should retain these contracts as the implementation evolves; the list is
not a claim that every permutation has its own end-to-end test:

1. omitted `mask`, explicit `False`, `[]`, and `()` all normalizing to `()`;
   singleton/list/tuple normalization and list-copy immutability; `None`
   failing during construction, mutation, and schema loading; and canonical
   round-trip serialization;
2. the complete dropout/reconstruction purpose matrix and learned Mask/skip
   effects across literal, query, rate, and query-plus-rate selection;
3. one branch selection shared exactly by all descendants, including nested
   branches;
4. leaf policies remaining independent when attached separately;
5. Arrow query selectors produced by both Arrow and Awkward preprocessors;
6. selector alignment after filtering, shuffling, overflow, and rebatching;
7. random selections remaining identical across batch sizes and chunk layouts
   when Batch identity is retained, with `skip`/`dropout`/`reconstruct` changes
   preserving the selected coordinates;
8. explicit/ingress-normalized null, structurally absent query results,
   wholly absent columns, empty lists, and padding following their distinct
   rules without fabricated fitting targets;
9. both input and target projections satisfying the RaggedField state/placement
   invariant after every selection combination;
10. overlap composition where skip wins visibility and reconstruction wins
    objective eligibility;
11. spies proving skipped raw values enter neither the forward input tensorizer
    nor embedder;
12. pristine observation learning exactly once, synchronizing/checkpointing its
    address-owned resources, counting pristine rather than sparse projected
    values, and remaining frozen outside training;
13. target-side Text conversion remaining isolated from encoder context, with
    target tensor changes unable to change the produced input Parcel;
14. a third-party TensorField with TensorDict content and multiple trailing
    axes gathering, embedding, and scattering correctly through the shared
    input-only `take`;
15. presence-aware attention and pooling being invariant to filler bytes;
16. mixed, all-skipped, and zero-context rows without NaN;
17. empty stochastic reconstruction selections skipping loss safely;
18. the full stratum activation matrix for dropout, stochastic reconstruction,
    deterministic/query reconstruction, persistent ablation, and query-driven
    skip;
19. static query-backed prediction schemas and correct `inferred` bits for
    empty and partial selection, while random-only reconstruction adds no
    prediction field or decoder work;
20. branch reconstruction capability validation for custom plugins;
21. a positive-row Batch with zero data columns, including interactive
    prediction from `[{}, {}]`, preserving root identity and output cardinality;
22. source omission for queryless unconditional reconstruction—including
    normalized `rate=1`—while query-backed, genuinely stochastic, or repeated
    reconstruction retains its required source/layout; plus a third-party
    plugin accepting multiple unrelated Arrow families handling source-less
    reconstruction without shared type inference;
23. one conversion context preserving Hash-like equality across input, target,
    and sibling fields;
24. lower-level model protocol tests for one DDP rank with all-skipped input or
    an empty local objective while a peer uses those parameters, plus a globally
    empty objective step;
25. construction and schema mutation normalizing `mask` before selectors or
    plans inspect its canonical tuple;
26. clear rejection of every removed field, `Model.encode(mask=...)`, and the
    legacy sentinel assumption.

End-to-end worker/rank loader tests remain gated on the distributed Arrow data
plane, which the current data module rejects. The implementation establishes
the model-level observation, selection, and backward protocols without
pretending that distributed scheduling already ships.

## Performance Expectations

The implementation removes repeated per-leaf branch sampling and can avoid
expensive conversion for skipped Text, Set, Category, and custom values. The
expected gain scales with branch width, descendant count, and selected
fraction.

It is not guaranteed to accelerate cheap scalar fields at batch size one.
Arrow selection, hashing, gathering, and scattering have fixed overhead.
Benchmarks must report preprocessing, coalescing, codec, embedding, and forward
time separately. Use representative bounded scenarios across these dimensions,
with explicit total-coordinate and total-item budgets; do not construct the
full Cartesian product or materialize impossible worst-case batches:

- batch sizes `1`, `8`, `64`, and `256`;
- `1`, `8`, and `32` descendant leaves;
- branch lengths `1`, `32`, `1_000`, and `10_000`;
- selection rates `0`, `0.1`, `0.5`, and `1`;
- learned Mask and skip effects, both rate-selected and compared with one
  materialized query selector;
- Number and Text plugins;
- source-provided, preprocessor-derived, and generated-random selectors.

The primary acceptance criterion is simpler, consistent semantics. Speed is a
secondary measured benefit.

## Explicitly Deferred

- an out-of-band identity-bearing mask carrier;
- `Model.predict(..., mask=...)` convenience;
- exact-count sampling inside `Mask`;
- a branch descendant include/exclude language;
- filters or expressions in the query grammar;
- causal-importance claims for ablation results.

These can be added without changing the selection/effect/purpose model above.

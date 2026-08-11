# Reference Context

Status: Proposed

Date: 2026-08-08

Scope: Schema references, pooling, reduction, and runtime context routing

## Summary

RelFlow should expose `rf.Reference` as an exact, serializable pointer from a
consuming `Branch` to one other schema node. The referenced node contributes
its unpooled encoder memory to that Branch's context:

- a Reference can be declared only by a `Branch`;
- one Reference owns exactly one absolute `Address`;
- `Branch.reference` accepts one Reference or an ordered tuple of References;
- a leaf source contributes its full embedding;
- a branch source contributes its post-attention, pre-pool memory;
- the consuming Branch contextualizes native and referenced tokens together;
- the enriched Branch memory is also available to its descendant decoders;
- `graft=True` removes the source's ordinary route to its direct parent but
  never calls `Tensor.detach()`.

The visible AnyTree remains a schema tree. Binding compiles the structural and
Reference edges into a directed acyclic execution graph. This is necessary for
one-way cross-branch dependencies and for structural grafting, which can
leave an otherwise empty parent Branch compiled-inactive without deactivating
the grafted child.

`rf.Mean`, `rf.Attention`, and `rf.Convolution` are frozen pooling
configurations shared by Branch summaries, leaf decoders, and Reference
reduction. A pooling configuration owns its output `width`; there is no
separate `Branch.width`. `rf.Reduce` uses named schema axes and `einops.reduce`
to compress selected dimensions before Reference memory is flattened.

The companion [`einops-refactor-spec.md`](einops-refactor-spec.md) defines the
behavior-preserving tensor-layout cleanup used to implement these boundaries.

## Motivation

The current runtime is a strict tree. Leaves send embeddings to their parent.
Each Branch contextualizes its local child sequence, compresses it to one
token, and routes that summary upward. A target decoder receives summaries
along its heritage.

That loses item-level correspondence before sibling branches interact. The
experiment in `benchmarks/hash_transfer.py` demonstrates the failure mode:

1. branch A contains shuffled `(Hash ID, visible symbol)` pairs;
2. branch B contains the same shuffled IDs and predicts the corresponding
   symbols;
3. equal IDs receive equal hash encodings within the observation;
4. ordinary sibling summaries cannot preserve a large lookup table;
5. the target remains near chance even though observation-wide controls work.

Routing A's existing pooled summary to B would preserve the same bottleneck.
The useful representation is A's complete contextualized memory before pool.

## Goals

- Let Branch encoders consume exact-address context from leaves or Branches.
- Preserve the full pre-pool memory of a referenced Branch.
- Let one Branch declare several ordered, independently configured References.
- Reuse one cached source activation across fan-out and duplicate references.
- Let a Reference optionally block-reduce named schema dimensions.
- Let a Reference structurally replace, rather than duplicate, a source's
  ordinary parent route.
- Make reference dependencies deterministic, inspectable, and cycle-safe.
- Let descendant decoders see their reference-bearing ancestors' enriched
  memory without owning References themselves.
- Unify Branch and leaf pooling configuration and make it own output width.
- Preserve masking, target isolation, observation boundaries, and deterministic
  inference.
- Keep the runtime tensor operations direct and use einops where it makes axis
  intent clearer.

## Non-goals

- A Reference is not a relational join, foreign-key constraint, or value
  comparison. Its exact schema address must resolve, but no runtime key match is
  performed.
- A Reference is not a query. `rf.where(...)`, predicates, callables, wildcards,
  lists of addresses, and relative addresses are not accepted by `Reference`.
- References do not move information between observations, batches, workers,
  or distributed ranks.
- Tensorfields cannot own References, and decoded predictions cannot be
  Reference sources.
- V1 does not infer broadcasting, Cartesian products, or joins from equal axis
  lengths.
- V1 does not reduce a Branch source's heterogeneous local memory-token axis.
- Optimizer and scheduler reconciliation after parameter-changing mutation is
  outside this proposal.
- Backward-compatible migration from earlier schema or checkpoint layouts is
  outside this proposal.

## Public API

### Exact References

`Reference` is a frozen value object, not a tree node or tensorfield. Its
conceptual public shape is:

```python
class Reference:
    address: Address
    graft: StrictBool = False
    reduce: Reduce | None = None

    def __init__(
        self,
        address: str | Address,
        /,
        *,
        graft: StrictBool = False,
        reduce: Reduce | None = None,
    ): ...
```

The positional input is coerced to a plain absolute `Address` value. It is
literal: there is no root insertion, slash cleanup, case normalization,
wildcard expansion, name lookup, or relative resolution. Lookup occurs only
after the complete tree is bound, so forward declaration is valid.

```python
rf.Reference("record/a")
rf.Reference(rf.Address("record", "a"))
rf.Reference(
    "record/a/id",
    graft=True,
    reduce=rf.Reduce(a=1),
)
```

These are intentionally invalid:

```python
rf.Reference(rf.where(address="record/a"))
rf.Reference(rf.where("type") == "hash")
rf.Reference(["record/a", "record/b"])
```

`Branch.reference` uses the singular field name while accepting scalar or
tuple cardinality exactly as requested:

```python
reference: Reference | tuple[Reference, ...] = ()
```

The empty tuple means none. A scalar is declaration index `0`; tuple order is
semantic context order and is never sorted. Several sources require several
declarations:

```python
b = rf.Branch(
    reference=(
        rf.Reference("record/a"),
        rf.Reference("record/c/id", graft=True),
        rf.Reference(
            rf.Address("record", "d", "id"),
            reduce=rf.Reduce(d=True),
        ),
    ),
    id=rf.Hash(n_hashes=4),
    symbol=rf.Category(size=16, target=True),
)
```

Duplicate addresses are legal. They provide multiple ordered views of the same
cached source, for example one raw view and one reduced view. Each declaration
owns its reducer configuration and, when trainable, its own reducer parameters.

The generated root is a Branch, so `Model(reference=...)`, `Model.from_tree`,
and `Schema.from_tree` expose the same scalar-or-tuple root configuration. A
root Reference enriches context inherited by every decoded leaf and should be
used deliberately.

The outer Branch field does not accept a bare string or `Address`; users must
construct `rf.Reference(...)`. Tensorfields reserve and reject both `reference`
and the stale plural spelling `references` rather than retaining them as plugin
extras.

### Hash-transfer example

```python
import relflow as rf

model = rf.Model(
    d_model=64,
    n_layers=2,
    n_heads=4,
    a=rf.Branch(
        length=64,
        id=rf.Hash(n_hashes=4),
        symbol=rf.Category(size=16),
    ),
    b=rf.Branch(
        length=64,
        reference=rf.Reference("record/a"),
        id=rf.Hash(n_hashes=4),
        symbol=rf.Category(
            size=16,
            target=True,
            pooling=rf.Attention(n_layers=2),
        ),
    ),
)
```

This adds the encoder dependency `record/a -> record/b`. B's self-attention
sees native B IDs followed by A's contextualized ID/symbol memory. B's target
decoder later receives the resulting enriched B memory. B's pool width controls
how much of that context also routes structurally upward; it does not alter the
memory used by References or descendant decoders.

### General selector sugar

References use exact addresses only. General schema selection and mutation
remain queryable. `rf.where` gains an address-equality convenience overload:

```python
@overload
def where(attribute: str, /) -> NodeAttribute: ...

@overload
def where(*, address: str | Address) -> NodePredicate: ...
```

These transient predicates are equivalent for selection and share the same
cache key:

```python
rf.where(address="record/a")
rf.where("address") == "record/a"
```

The keyword form means exact equality only. It performs no path normalization
or implicit `is_in`. The implementation coerces `str | Address` to one plain
string before building equality so both spellings have the same cache key.
Supplying both overload forms, neither form, or anything other than
`str | Address` is an error.

Every general selector entry point also accepts a direct string or `Address`
as positional exact-address sugar:

```python
model.select("record/a/symbol")
model.update("record/a/symbol", weight=2.0)
model.extend("record/b", rf.Number("risk_score"))
model.delete("record/b/legacy")
model.reset("record/a")

with model.override("record/a/symbol", target=True):
    ...
```

This applies consistently to Model, Schema, SchemaEditor, `Deployment.update`,
and `Mask.exclude`. Each operation retains its existing root, cardinality, and
empty-selection behavior. Multiple positional selectors on Model/Schema
operations retain AND semantics; they are not an implicit list of addresses.
`Mask.exclude` retains its existing sequence/`any(...)` semantics.

Mutation selectors are positional. `model.update(address=...)` is not selector
sugar, and `Node.address` is derived and read-only, so it also is not a valid
attribute edit. Mutable-attribute validation must reject it explicitly before
an extra-allow tensorfield could retain it. Rename a node through `name=`:

```python
model.update("record/a", name="renamed")
```

Unlike a transient mutation selector, the address stored by a Reference is
persistent schema state. Renaming or deleting its source leaves it dangling and
must fail candidate validation unless the same schema transaction also updates
or removes the consuming Reference.

## Pooling Configuration

Branches and tensorfield decoders use the same object-based `pooling` union.
Reference reduction additionally accepts deterministic built-in reductions:

```python
class Mean:
    type: Literal["mean"] = "mean"
    width: StrictPositiveInt | None = None

class Attention:
    type: Literal["attention"] = "attention"
    width: StrictPositiveInt | None = None
    n_heads: StrictPositiveInt | None = None
    n_layers: StrictPositiveInt = 1
    dropout: Rate | None = None

class Convolution:
    type: Literal["convolution"] = "convolution"
    width: StrictPositiveInt | None = None
    kernel_size: StrictPositiveInt = 3
    n_layers: StrictPositiveInt = 1
    dropout: Rate | None = None
```

The tagged `PoolingConfig` union is `Mean | Attention | Convolution`.
Configurations are frozen schema values, never live modules. Reusing one value
does not share parameters; each owner constructs its own concrete module.

`width=None` is a serialized auto policy. Binding computes the effective width:

| Owner | Auto width |
| --- | ---: |
| Branch summary | `1` |
| Leaf decoder | `product(schema.shapes[leaf])` |
| Reference Reduce block | `1` |

An explicit Attention or Convolution width may be any positive integer for a
Branch. Branch Mean is fixed to width `1`; wider means would only duplicate one
summary. A leaf width must equal its flattened schema output size, because the
decoder has no implicit width-to-target projection. A reducer width must be
`None` or `1`, because a larger result would introduce an unnamed axis.

Every ordinary pool obeys:

```text
[B, L, d_model] -> [B, Q, d_model]
```

- `Mean` computes a global arithmetic mean. A leaf expands that value to its
  required `Q` output slots.
- `Attention` owns `Q` learned queries; every query attends the full memory for
  each configured query/FFN layer, followed by final normalization.
- `Convolution` applies distinct same-length residual Conv1d blocks along the
  ordered memory axis, then adaptive-average-pools it to `Q` bins.

Convolution uses full-channel, `groups=1`, biased convolutions, positive odd
kernels, same zero padding, GELU, dropout, and a residual connection. `Q > L`
is legal and follows PyTorch adaptive-pooling overlap/repetition semantics.
Ordinary Branch and leaf convolution treats the assembled memory as one
continuous sequence; kernels may cross child or Reference segment boundaries.

Omitted pool-local heads and dropout inherit the owning node's `n_heads` and
`dropout`. Explicit values override only the pool. `Branch.attention` and
`Branch.n_layers` continue to control the Branch encoder's self-attention mode
and depth; `pooling=rf.Attention(n_layers=...)` controls the separate learned
query pool.

`n_linear` is removed from `Branch`, `Leaf`, `Model`, `Model.from_tree`, and
`Schema.from_tree`. `Attention.n_layers` is the sole pool-depth setting. The
name is reserved and rejected before Branch/Model unknown-key-to-child routing,
in raw Branch validation, and on extra-allow tensorfields, so it can become
neither a child nor ignored plugin metadata. There is no compatibility alias or
old-checkpoint migration in this proposal. Public `CrossAttention` is likewise
removed; `rf.Attention` is the only public/schema name.

Branch encoders and leaf decoders match directly on the pooling value.
Attention and Convolution install their concrete module directly as `.pool`;
Mean runs inline. Do not add a factory or one-use wrapper where a local match is
clearer. An internal learned-query implementation should use `n_layers`
terminology consistently even if its class name is more descriptive than the
public configuration.

### Branch summary shapes

There is no separate `Branch.width` or root `Model(width=...)`. Let `W` be the
Branch pool's effective width. If encoded memory is
`[N, *outer, L_memory, C]`, pooling produces `[N, *outer, W, C]`. The routed
summary is:

```text
outer is empty:   [N, W, C]
outer is present: [N, *outer[:-1], outer[-1] * W, C]
```

The final outer coordinate is major and width is minor. Width `1` exactly
preserves the current routing shape. A Branch source always exposes its full
pre-pool memory to References regardless of `W`.

`Schema.shapes` continues to describe data/decoder coordinates. A derived
`summary_shapes` map exposes Branch latent routing shapes separately.

## Reference Reduction

`rf.Reduce` resizes genuine schema coordinates before Reference memory is
flattened. Its default algorithm is `rf.Mean()`:

```python
rf.Reduce(b1=True)
rf.Reduce(rf.Mean(), b1=1)
rf.Reduce(rf.Attention(n_heads=2), b1=4)
rf.Reduce(rf.Convolution(kernel_size=3), b1=1)
rf.Reduce("sum", b1=1)
rf.Reduce(axes={rf.Address("root/b1"): 1})
```

The reduction-only names are `"sum"`, `"min"`, `"max"`, and `"prod"`;
`"mean"` is accepted as constructor shorthand and normalizes to `rf.Mean()`.

Axis target input normalizes in strict type order:

- `True` becomes target size `1`;
- `False` is removed before name/address resolution;
- a positive non-boolean integer is retained;
- zero, negative integers, floats, strings, and coercive truthy values fail.

Thus `False` means omission, not size zero. If every entry is disabled, the
Reduce value normalizes away and the Reference stores `reduce=None`.

Short keyword names resolve to exactly one eligible Branch coordinate in the
source's axis signature. Full `Address` keys disambiguate repeated names or
names that cannot be Python keywords. Canonical state stores only full Branch
addresses and positive integer sizes; kwargs, booleans, and generated einops
patterns are construction/compilation details.

Leaf memory below `root/b1/b2` has named signature
`(root/b1, root/b1/b2)`. Branch `root/b1/b2` memory has named outer signature
`(root/b1)` plus one unnamed, heterogeneous local-memory axis. A selected
Branch's own local axis is not its `Branch.length` coordinate and cannot be
named by Reduce in v1. Consequently, reducing `b1` on a selected `b2` Branch
preserves all of `b2`'s local memory:

```text
[N, B1, L_b2_memory, C] -> [N, K, L_b2_memory, C]
```

For each active axis with physical extent `L` and target size `K`, compilation
requires `K <= L` and `L % K == 0`, then splits the axis as `(K, R)` with
`R = L / K`. Several axes split independently; their `R` factors form the
reduction domain and their `K` factors remain in schema order. There is no
padding, truncation, overlap, or adaptive resizing in this block operation.

Parameter-free modes lower directly to `einops.reduce`. Trainable modes fold
all retained coordinates into an effective batch and all reduced factors into
one ordered token axis:

```text
[N, *retained, *coarse, *reduced, C]
    -> [B_effective, R_total, C]
    -> [B_effective, 1, C]
    -> [N, *retained, *coarse, C]
```

Attention uses one learned query shared across blocks. Convolution runs within
each independent block, so kernels never cross source or retained-coordinate
boundaries. Convolution may resize exactly one named axis in v1; flattening
several axes would invent adjacency at row boundaries.

Because each Reference has exactly one source, every active Reduce axis must
apply to that source. There is no multi-source pass-through rule or implicit
source-identity reduction. The reduction plan and any trainable reducer module
are owned by that Reference declaration.

## Representation And Routing Semantics

### Memory and summary

Each encoded node has two logical views:

- leaf `memory` is the full embedder output; its ordinary `summary` is the same
  tensor;
- Branch `memory` is the post-attention sequence before structural pooling;
- Branch `summary` is the configured pooled and routed representation sent to
  its structural parent.

Forward activations live only in per-forward address-keyed maps. They are not
stored on Schema, Model, or the compiled graph. Fan-out reuses the same tensor,
dropout realization, and autograd graph.

An **encoder-eligible leaf** is schema-active and non-target. Eligibility is
static and does not depend on whether this batch's slots are valued, null,
padded, or masked; masked source memory must still be embedded and routed.

### Branch consumer

A live consuming Branch assembles context in this exact order:

1. structurally attached encoder-eligible direct leaves in schema order;
2. structurally attached direct child-Branch summaries in schema order;
3. one source-memory segment per Reference declaration in tuple order, after
   that declaration's optional Reduce.

The combined sequence passes through the consumer's ordinary Branch encoder.
Each duplicate Reference occurrence contributes another segment, but its raw
source activation is still computed once.

### Descendant decoders

Tensorfields cannot declare References. For every active leaf, binding compiles
one possible decoder context in this order:

1. available ordinary heritage summaries, root to leaf, including the leaf's
   own embedding when it exists;
2. the enriched memory of each live Reference-bearing ancestor Branch, root to
   parent.

An ancestor memory appears once regardless of how many References enriched it.
The decoder does not receive the original source again or rerun reducers. Every
non-batch axis of each context tensor is packed into a single token dimension,
then contexts are concatenated as `[N, L_decoder, C]`.

Decoder execution remains batch-dependent. `TensorField.trainable` is a
per-slot loss mask, not autograd `requires_grad`:

```python
if Strata.normalize(strata) == Strata.predict:
    should_decode = field.state.eq(Tokens.masked).any() or leaf.embed
    inferred = field.state.eq(Tokens.masked)
else:
    should_decode = field.trainable.any()
    inferred = field.trainable

if not should_decode:
    continue
```

Targets are trainable outside prediction by the existing tensorization
contract. During prediction, target placeholders and explicit mask literals
are masked but intentionally not trainable, so the predict rule is required.
If any slot qualifies, the decoder computes all `Q` slots once and the
`trainable`/`inferred` mask identifies relevant outputs. Context must therefore
be compiled and statically validated for every active leaf, not only leaves
decoded by one example batch.

### Coordinate compatibility

A descendant decoder may flatten arbitrary source coordinates after reduction;
only batch and `d_model` must agree.

A Branch consumer must preserve its own outer coordinate domain. Let `P` be
the consumer's named outer-coordinate signature and `S` the reduced source's
named signature. `P` must be an exact address-and-extent prefix of `S`.
Trailing source coordinates and the unnamed local-memory axis are flattened
into the Reference token dimension. There is no implicit broadcasting,
permutation, or Cartesian expansion, and equal numeric extents under different
addresses are not compatible.

For the common top-level case, `P` is empty and observation-local source memory
can be flattened freely. For a nested consumer, pooling away or resizing an
axis required by `P` is invalid unless the retained extent remains exactly the
consumer extent.

## Structural Grafting And Liveness

`Reference(..., graft=True)` means structural grafting, not autograd
detachment. Grafting never calls `Tensor.detach()`. After all References
resolve, compilation forms:

```text
grafted_sources = {ref.address for every ref with graft=True}
```

For each grafted source, its one native route to its direct schema parent is
suppressed:

- a grafted leaf is omitted from its parent's native leaf inputs;
- a grafted Branch's summary is omitted from its parent's native child inputs.

The source remains in the AnyTree, retains its address, is encoded when needed,
and remains available to every Reference. Its tensor is never cloned or passed
through `Tensor.detach()`. Consumer losses therefore backpropagate through the
Reference path into the source.

Grafting is global for that source's structural route. If duplicate or fan-out
References disagree, any `graft=True` cuts the route; `graft=False` does not
veto another declaration. Removing or changing the final grafting declaration
restores it. A root source cannot be grafted because it has no
parent.

Suppression can leave the original parent with no effective input. Branch
liveness is compiled recursively over attached native inputs and Reference
dependencies:

```text
live(branch) :=
    any attached encoder-eligible direct leaf
    or any live attached child Branch
    or any encoder-eligible exact leaf Reference
    or any live exact Branch Reference
```

An empty Branch is compiled-inactive and skipped; its absent summary can make
its parent inactive in turn. This is derived execution state, not mutation of
a schema `active` flag. A grafted child remains live when a Reference needs it,
even if its former parent and ancestors become inactive. Every live Branch
always computes and caches both memory and summary; grafting only controls
whether the parent gathers that summary.

The existing tree invariant that every retained Branch must have request
descendants must be replaced by this effective-input/liveness rule. Otherwise
Reference-only and graft-created inactive Branches could not be represented.

## Compiled Execution Graph

Binding derives one immutable `CompiledExecutionGraph`. It is runtime
architecture, not public schema or checkpoint state.

References are normalized inline:

```python
references = (
    (branch.reference,)
    if isinstance(branch.reference, Reference)
    else branch.reference
)
```

Each declaration receives a derived
`ReferenceId = (consumer_address, declaration_index)`. The ID is used for plans,
diagnostics, and module keys and is not serialized.

Encoder vertices are Branch addresses, including the generated root. After
exact address resolution and structural cuts, compilation adds:

- effective structural edge `child_branch -> parent_branch` for every attached
  non-root Branch;
- Reference edge `source_branch -> consumer_branch` for every Branch source.

A leaf source needs no scheduling vertex because required encoder-eligible
leaves embed before Branch encoding. A reducer is not a graph vertex. Duplicate
source-to-consumer dependency pairs count once for indegree while retaining all
Reference IDs as edge provenance.

The stable topological order uses Kahn's algorithm with priority:

```text
(-structural_depth, schema_preorder_index)
```

With no References it exactly matches the current deepest-to-root schedule.
Reference tuple order affects context order, not ready-node tie-breaking.
Execution completion order never determines tensor concatenation order; the
compiled input plan does.

The runtime keeps every Branch vertex in `encoder_order` and skips addresses
outside `active_branches`. Forward steps are:

1. embed every encoder-eligible leaf required by an effective route once;
2. visit Branches in `encoder_order`, gather compiled native and Reference
   inputs, apply each Reference reducer, and cache memory plus summary;
3. evaluate active leaf decoder predicates and gather their compiled contexts;
4. decode or skip according to the batch-local rule above.

Cycle detection runs on the effective graph after structural cuts. Self
references, mutual sibling references, ancestor/descendant feedback, and longer
mixed cycles fail. Diagnostics report a complete deterministic cycle with edge
kind, source address, consumer address, and `reference[index]` provenance.
Grafting can remove a structural edge, but it never removes the Reference
dependency itself.

The compiled artifact contains at least:

- structural depth and preorder ranks;
- `branch_references[consumer] -> tuple[ReferenceId, ...]`;
- `reference_source[ReferenceId] -> Address`;
- `reference_consumers[source] -> tuple[ReferenceId, ...]`;
- `grafted_by[source] -> tuple[ReferenceId, ...]`;
- effective native and ordered Reference input entries;
- one Reduce plan and effective reducer specification per ReferenceId;
- unique dependency edges with complete provenance;
- stable `encoder_order` and `active_branches`;
- memory availability and Branch summary shapes;
- `decoder_contexts[leaf] -> tuple[(view, address), ...]` for every active leaf.

All tensors remain forward-local. The compiled graph stores only immutable
addresses, shapes, patterns, and module specifications.

## Masking, Leakage, And Gradients

References consume only post-masking encoder representations from the current
observation.

- `target=True` leaves are never embedded and cannot be sources.
- inactive leaves cannot be sources.
- a referenced Branch omits target descendants exactly as normal encoding does.
- a masked source contributes its masked representation, never cached truth.
- decoded predictions are never sources.
- a descendant target may receive its parent's enriched memory, but that memory
  contains no target embedder payload.
- tensor layout operations never mix observations.

Reference routing is differentiable by default and under structural grafting.
One source activation fans out to every consumer; gradients from
every occurrence accumulate. Each trainable reducer owns an independent module
per Reference declaration. A consuming Branch's enriched memory is computed
once and reused by its pool and all descendant decoders.

Temporal or causal leakage remains the user's responsibility. Put a Reference
on the narrowest Branch whose descendants legitimately need the additional
context.

## Validation

Binding rejects:

- Reference placement on a Leaf/tensorfield;
- a `Branch.reference` value other than a Reference or tuple of References;
- a bare string/Address in the outer Branch field;
- a Reference input other than one string or Address;
- an empty or missing exact source address;
- a self Reference;
- grafting the generated root;
- an inactive or target leaf source;
- a Branch source that cannot produce memory in the compiled liveness graph;
- an attached direct-child leaf Reference that would duplicate identical native
  memory without a non-identity reduction;
- an empty compiled decoder context for any active leaf;
- a Reduce axis that is duplicated, unresolved, ambiguous, non-Branch, absent
  from the exact source signature, or names the generated root;
- a Reduce target that is boolean after normalization, non-integral, nonpositive,
  larger than its extent, or not an exact divisor;
- multi-axis Convolution reduction;
- Branch coordinate incompatibility after reduction;
- an invalid pool width or a width inconsistent with its owner;
- invalid Attention head geometry or Convolution kernel configuration;
- an encoder dependency cycle.

`graft` accepts only a strict boolean. Pool widths, heads, layer counts,
kernels, and active integer Reduce targets are positive non-boolean integers.
Attention heads must divide `d_model` and leave head width at least two.

An empty Reference tuple is valid. Duplicate exact declarations are valid and
intentional. Validation errors name the consumer, `reference[index]`, exact
source address, graft policy, and relevant axis/shape/cycle provenance.

Direct-child duplication is defined narrowly. A pass-through Reference to an
attached direct leaf duplicates its native tensor and fails. `graft=True`
removes the native copy and is valid. A shape-changing or trainable Reduce is a
distinct view and is valid. A direct child Branch is valid because its native
view is the pooled summary while its Reference view is pre-pool memory.

`validate=False` on general mutations may bypass existing per-node value
reconstruction only. It never bypasses Reference resolution, reduction/shape
checks, liveness, or DAG compilation.

## Schema, Serialization, And Introspection

`Reference`, `Reduce`, `AxisName`, `AxisResize`, `Mean`, `Attention`, and
`Convolution` are frozen Pydantic value objects with `extra="forbid"`.
Conceptual canonical Reference data is:

```json
{
  "address": "record/a",
  "graft": false,
  "reduce": null
}
```

A Branch `reference` field serializes as:

- `[]` for none;
- one object for one Reference;
- an ordered array for two or more References.

A Branch field validator and serializer make the cardinality canonical in both
Python and JSON modes. Before validation, `[]` or `()` becomes `()`; a
one-element list or tuple becomes its scalar Reference; and length two or more
becomes an ordered tuple. Serialization emits `[]`, one object, or an array,
respectively. Raw schema/checkpoint validation therefore accepts canonical
Reference mappings and arrays, including a one-object array which re-dumps as
one object. The public positional constructor may reject mappings and lists,
but that ergonomic overload must not break
`Reference.model_validate({"address": ...})`; use an optional sentinel plus raw
field data, or an equivalent Pydantic construction mechanism.

Canonical Reduce state contains one tagged reducer configuration and a tuple of
full-address/integer axis entries. `True` is stored as `1`, `False` is omitted,
and an all-disabled Reduce is stored as `null`. Pool `width=None` remains `null`
so auto-width provenance survives mutation and checkpoint round trips.

No resolved Reference IDs, runtime graph, liveness set, generated einops
pattern, module, tensor, selector AST, or source cache is serialized. Loading
rebinds the tree and recompiles every derived artifact. Old `selector`-based
Reference payloads, `CrossAttention`, and `n_linear` fail clearly; compatibility
is deliberately out of scope.

The Branch, not the Reference value, owns trainable reducer modules:

```text
nodes.<consumer>.reference_reducers.<declaration_index>.*
```

Parameter-free declaration indices create no placeholder module. Indexing uses
the original tuple position, so state ownership and context order agree.

Introspection should distinguish:

- Reference declarations;
- exact source occurrences, including duplicates;
- unique DAG dependency pairs;
- trainable reducer instances;
- structurally grafted source addresses;
- live versus compiled-inactive Branches.

Rich rendering displays the consumer, declaration index, exact source,
`graft`, Reduce summary, resolved axis geometry, and effective pool widths.

## Mutation Semantics

General mutation target selectors remain transient and may use direct exact
addresses, `rf.where`, or the existing predicate forms. References do not use
that selector system and never expand or contract dynamically.

Every schema-changing candidate follows this order:

1. apply the requested edit to an off-side candidate;
2. bind the complete tree and resolve every exact Reference address;
3. derive structural cuts and effective edges;
4. cycle-check and topologically sort the candidate DAG;
5. derive liveness, Reduce plans, shape compatibility, and decoder contexts;
6. commit the complete schema, compiled plan, and every schema-derived runtime
   artifact only if every step succeeds.

A successful plan-only commit atomically refreshes schema selection caches,
example input when the input signature changes, contract generation/scheduler
state, and any other schema-derived runtime cache while reusing the installed
modules. A failed candidate leaves all of those live artifacts untouched.
Exact Reference addresses are never rewritten automatically. Rename through
`name=` or deletion of a referenced source therefore fails unless a coordinated
transaction also updates/removes the Reference. A future multi-edit transaction
API may make that coordination ergonomic.

This proposal does not reconcile optimizer parameter groups, optimizer moments,
or schedulers. On a built Model, every mutation covered by this specification
may commit only when every registered module, Parameter, buffer, and state-dict
key can be reused unchanged. A candidate requiring module creation, removal,
replacement, or re-keying fails with an instruction to construct a new Model
from the mutated Schema. This includes pooling, effective-width, and reducer
changes, not only Reference routing. Standalone Schema mutation remains valid
before model construction. Existing explicit parameter-reset behavior is
outside this feature's guarantees and must not claim optimizer safety.

Changing only an exact source or `graft` flag may be plan-only when reducer
ownership and all tensor geometry remain compatible. Reordering, inserting, or
removing tuple entries with trainable reducers changes ordinal ownership and
requires a new Model. Parameter-changing temporary overrides are likewise
unsupported in v1.

## Performance

For a consumer Branch:

```text
L_total = L_native + sum(L_reduced(reference_i))
```

Duplicate addresses count once per declaration in `L_total`, even though their
upstream source memory is cached once. Branch self-attention remains quadratic
in `L_total` per layer. Attention pooling costs roughly `Q * L_total` per
query/FFN layer. Convolution is linear in memory length plus its output resize.

Structural grafting may reduce the original parent's input length but does
not promise activation or autograd-memory savings. The source remains live
until its final Reference consumer, and the same enriched Branch memory may be
reused by several descendant decoders.

Pool width is an explicit capacity/compute tradeoff. Widening a Branch summary
preserves more latent capacity along ordinary structural routes; a Reference
supplies context that the Branch did not possess. The two features are
complementary.

## Acceptance And Test Plan

### Public schema

- `Reference(str)` and `Reference(Address)` normalize to identical canonical
  data and resolve sources declared before or after the consumer.
- `Reference(where(...))`, predicates, callables, lists, tuples of addresses,
  relative paths, and missing addresses fail clearly.
- `Branch.reference` accepts a scalar, empty tuple, and ordered tuple; rejects
  a bare address and every tensorfield placement.
- Duplicate exact addresses preserve tuple order and independent reducer state.
- Root Reference configuration round-trips.
- Scalar/empty/multi Reference serialization round-trips canonically through
  both Python and JSON model dumps; a one-object raw array re-dumps as an object.
- `rf.where(address=...)` and expanded address equality share selection results
  and cache keys, but neither is accepted by Reference.
- Direct string/Address selectors work for select, update, extend, delete,
  reset, override, Deployment update, and Mask exclusion with unchanged
  cardinality/root rules.
- `update(address=...)` fails as an attempted edit of a derived attribute;
  `update("old/path", name=...)` performs the rename.
- `n_linear` and `CrossAttention` are absent from every public signature and
  rejected by raw schema validation.

### Pooling and reduction

- Default Branch Attention width is one; default Leaf width tracks the target
  schema shape through standalone-Schema/new-Model construction and checkpoint
  reload. A shape-changing built-Model mutation fails when it would replace
  learned queries.
- Attention, Convolution, and Mean produce exact Branch and flat Leaf shapes at
  root, top-level, and nested locations.
- Branch width ordering is coordinate-major and width-minor.
- `rf.Mean()` and `"mean"` normalize identically; mean, sum, min, max, and prod
  match manual block reductions.
- `True` equals integer `1`; `False` equals omission; disabled unknown names do
  not resolve; `0` remains invalid.
- Multi-axis mean/Attention preserves retained coordinates and row-major order;
  multi-axis Convolution fails.
- A Branch source's synthetic local-memory axis cannot be named as its own
  Branch coordinate.
- Trainable reducers register at the declaration-index key and receive
  gradients; parameter-free reducers create no module.
- Two identical trainable declarations own distinct parameters; duplicate raw
  source memory is computed once.

### Runtime and graph

- Leaf References contribute full embedder memory; Branch References contribute
  post-attention/pre-pool memory.
- Native context precedes Reference tuple order regardless of topological
  execution order.
- A descendant decoder receives each reference-bearing ancestor's enriched
  memory once, not once per Reference declaration.
- Decoder contexts compile for every active leaf; all-false non-predict fields
  skip decoding, while predict masks and predict embeddings decode.
- Target truth and inactive leaves never enter Reference memory.
- Fan-out computes each source once and accumulates gradients from every
  Reference occurrence.
- No-Reference models with default Attention pools retain the original
  deepest-to-root execution order and width-one shapes.
- Direct, mutual, structural, and longer mixed cycles fail with deterministic,
  index-bearing diagnostics.

### Structural grafting

- Grafting a leaf removes only its native parent input and preserves its
  Reference memory and gradients.
- Grafting a Branch suppresses only its summary-to-parent route.
- A true and false Reference to one source obey global true-wins semantics;
  removing the final true restores the route.
- Direct-child leaf replacement is valid under grafting; attached identity
  duplication fails.
- Empty-parent cascades update compiled liveness without deactivating referenced
  descendants.
- Ancestor/descendant grafting and Reference dependencies are evaluated on
  the final effective graph.
- Checkpoint loading recomputes cuts, liveness, decoder contexts, and encoder
  order from declarations.

### Hash transfer

Hold branch lengths, B's default width-one pool, decoder depth, optimizer
settings, seeds, and parameter budget controls constant across:

1. no Reference;
2. correct exact `record/a -> record/b` Reference;
3. shuffled/unlinked IDs;
4. optional structurally grafted A.

The linked condition must materially exceed no-Reference and unlinked/chance
controls as branch length and class count grow. Unlinked IDs must remain at
chance. A widened B-pool ablation may then measure the separate structural-
routing capacity tradeoff without changing the Reference's pre-pool input
semantics.

## Implementation Style

Install `einops` as a direct dependency and use `pack`, `unpack`, `rearrange`,
`reduce`, and `einsum` where the named pattern replaces nontrivial
reshape/transpose arithmetic. Keep simple local `mean`, `sum`, `squeeze`, and
broadcast operations in PyTorch when they are already clearer.

Prefer direct local pattern matching over factories and forwarding wrappers:

- Branch and leaf owners instantiate Attention or Convolution directly at
  `.pool`; Mean executes inline.
- A Branch owns trainable reducers directly in `.reference_reducers` by tuple
  index; built-ins execute inline.
- Runtime gathers compiled addresses from maps; it does not clone Parcels or
  add a generic routing abstraction.
- Memory and summary should be returned directly by Branch encoding rather
  than reconstructed through helper layers.

The companion einops spec is normative for axis names, row-major ordering,
root-singleton handling, odd-channel sinusoidal layouts, and parity tests.

## Alternatives Considered

### Query-defined References

Rejected for v1. `rf.where` source sets require a serializable predicate AST,
dynamic match-set mutation behavior, inner source ordering, reducer-sharing
rules, and substantially more graph diagnostics. The motivating hash-transfer
edge is exact. Several exact sources are clearer as several tuple declarations,
each with its own graft and Reduce policy.

### Referencing pooled summaries

Rejected because a widened or one-token Branch summary has already discarded
the item-level correspondence this feature exists to preserve.

### Increasing pool width alone

Useful but insufficient. Width can preserve more information a Branch already
has; it cannot supply missing sibling context.

### Flattening the complete model globally

Rejected because it discards schema locality and makes compute scale with the
entire observation even when only one subtree needs cross-context.

### Decoded-output References

Rejected because they introduce target leakage, decoder ordering, and
autoregressive semantics outside this feature.

## Deferred Questions

- Relative Reference addresses and explicit address rebasing during coordinated
  rename operations.
- Query-defined Reference source sets if exact tuples prove insufficient.
- Explicit reduction of a Branch's heterogeneous local memory-token axis.
- Segment/source embeddings before Reference concatenation.
- Broadcast, keyed alignment, and Cartesian expansion modes for nested Branch
  coordinates.
- Optimizer-aware parameter mutation and safe temporary architecture overrides.
- A first-class multi-edit schema transaction API.

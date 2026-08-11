# Einops Tensor-Shape Refactor

Status: Proposed

Date: 2026-08-08

Scope: Tensor-shape notation and behavior-preserving architecture cleanup

## Summary

RelFlow should use `einops` as the common notation for tensor operations whose
meaning depends on named axes. The refactor accompanies the Reference Context
work because references, widened pooling, structural grafting, and compiled
decoder contexts all require explicit transformations between schema
coordinates, memory tokens, pooling width, attention heads, and channels.

The objective is not to replace every `reshape`, `unsqueeze`, or reduction.
`einops` should replace shape arithmetic when its pattern expresses an
architectural contract more clearly than positional dimensions do. Simple
broadcasts and local PyTorch operations remain native. Several tensorfield
embedders should become simpler by removing flattening altogether, since
`nn.Embedding`, `nn.Linear`, broadcasting, and matrix multiplication already
support arbitrary leading dimensions.

This is an implementation-style companion to
[`reference-context-spec.md`](reference-context-spec.md). That specification
owns public Reference, pooling, reduction, graph, and grafting behavior. This
document owns the notation, refactor boundaries, and parity requirements used
to implement those behaviors legibly.

## Goals

- Make semantic tensor axes visible at each architectural boundary.
- Remove manual products, paired flatten/restore reshapes, and `-1` inference
  when the intended axis relationship can be stated directly.
- Use one consistent vocabulary for batch, schema coordinates, memory tokens,
  pooling width, heads, channels, hashes, frequencies, and output classes.
- Keep `einops` expressions inline when they are used once.
- Preserve tensor values, row-major order, gradients, masks, dtype, device, and
  batching behavior unless the Reference Context specification explicitly
  changes the architecture.
- Make root, top-level, nested, widened, referenced, and structurally grafted
  tensor layouts auditable from source code.
- Retain native PyTorch when it is shorter or when view, aliasing, or
  zero-stride behavior is part of the implementation contract.

## Non-goals

- A mechanical repository-wide replacement of every shape operation.
- A public tensor-shape DSL or user-authored einops patterns.
- Storing einops patterns in checkpoints or schema objects, except canonical
  Reference reduction configuration from which patterns are compiled.
- Changing ragged preprocessing, query execution, overflow handling, or CPU
  tensorization into dense array code.
- Adding wrapper functions whose only behavior is forwarding to one einops
  operation.
- Promising that every einops operation is a view or allocation-free.
- Backward-compatible migration for earlier checkpoint layouts.

## Dependency

`einops>=0.8.2` is a required runtime dependency, not an optional extra. RelFlow
imports the operations it uses directly and provides no fallback implementation.
The project metadata and lockfile must move with the first implementation slice.

## Audit Snapshot

At the time of this proposal, `src/relflow/architecture` and
`src/relflow/tensorfields` contain 91 calls to `.reshape(...)` and 127 calls
across the broader reshape/view/flatten/transpose/unsqueeze/squeeze/expand
family. These counts are evidence of repeated shape choreography, not cleanup
targets. Many calls are already the clearest expression and should remain.

The high-value clusters are:

1. attention head splitting, head/batch folding, and head merging;
2. Branch coordinate packing and pooled-summary restoration;
3. decoder Parcel flattening and concatenation;
4. tensorfield flatten/process/restore boilerplate;
5. Hash frequency/trigonometric interleaving and hash aggregation;
6. loss-boundary flattening that must preserve a `classes`, `feature`, `hash`,
   or `bucket` axis.

No useful dense-tensor refactor was found in the ragged data pipeline.

## Axis Vocabulary

Patterns use complete semantic names rather than single-letter aliases:

| Name | Meaning |
| --- | --- |
| `batch` | observations; may join coordinates only as an effective module batch |
| `coordinate` / `outer` | schema-derived repeated Branch coordinates |
| `token` | an encoder or decoder memory position |
| `width` | the output slots owned by a pooling configuration |
| `channel` | `d_model` or another embedding channel axis |
| `head` | attention heads |
| `head_channel` | channels within one attention head |
| `query` / `key` | attention query and key/value sequence positions |
| `classes` | categorical output classes, including state tokens |
| `feature` | a vector-valued tensorfield's final data axis |
| `hash` | independent Hash functions |
| `bucket` | one Hash decoder's categorical buckets |
| `frequency` | sinusoidal/Fourier frequencies |
| `trig` | sine/cosine pair axis |

`...` means an unknown number of preserved schema axes. `*` is reserved for
`pack`/`unpack`, where an unknown number of axes is intentionally collapsed and
later restored. Compiled Reference reduction patterns use safe aliases such as
`d0`, `d0_out`, and `d0_reduce`; raw node names and addresses never enter a
pattern string.

Batch may join coordinates into an effective row axis for independent module
calls or terminal loss evaluation. That transformation must preserve one row
per observation-coordinate pair; it never joins batch with memory tokens.

Patterns preserve ordinary row-major ordering. When two axes are merged, their
major/minor order must be visible in the expression. For example,
`(frequency trig)` preserves the Hash encoding's frequency-major,
trigonometric-minor interleave, while `(trig frequency)` would be a behavioral
change.

## Operation Selection

### `rearrange`

Use `rearrange` when axes are split, merged, reordered, or when a semantic final
axis must survive flattening:

```python
rearrange(
    projected,
    "batch token (head head_channel) -> batch head token head_channel",
    head=n_heads,
)

rearrange(logits, "... classes -> (...) classes")
```

Do not use it as a longer spelling of a local `.squeeze(-1)` or `.flatten()`
whose meaning is already obvious and does not preserve a semantic axis.

### `pack` and `unpack`

Use `pack` when an arbitrary number of structural axes must become one runtime
batch or memory axis. Use the returned shape metadata for the exact inverse:

```python
encoded, leading_shape = pack([context], "* token channel")
[restored] = unpack(encoded, leading_shape, "* token channel")
```

`pack` may also flatten and concatenate heterogeneous Parcel ranks while
preserving batch and channel:

```python
memory, _ = pack(
    [parcel.payload for parcel in parcels],
    "batch * channel",
)
```

Do not hide `pack`/`unpack` in generic schema-shape helpers. The local pattern
is the useful documentation.

### `reduce`

Use `reduce` when the pattern materially clarifies which axes disappear or are
retained. Compiled Reference block reduction is its primary use. Simple local
operations such as `losses.mean()`, `mask.sum(dim=-1)`, or a clearly named
one-axis normalization may stay native.

### `einsum`

Use named `einsum` for a contraction or outer product whose semantic axes would
otherwise require flattening or positional comments:

```python
einsum(
    normalized,
    frequencies,
    "... hash, frequency -> ... hash frequency",
)
```

Keep an ordinary `matmul` when it already states the linear algebra clearly,
such as Set multi-hot values multiplied by an embedding table.

### `repeat`

Use `repeat` only when materialized repetition is intentional and the named
pattern is clearer. Do not replace zero-stride `unsqueeze(...).expand(...)`
operations whose allocation-free aliasing behavior is deliberate. In
particular, learned pooling queries and repeated mean summaries may retain
`expand`.

## Import And Style Contract

Modules import only the operations they use:

```python
from einops import pack, rearrange, reduce, unpack
```

Patterns remain adjacent to the operation they describe. Do not introduce pool
factories, tensor adapters, `split_heads` wrappers, flatten helpers, or one-use
forwarding methods when one local expression is sufficient. Shared validation
and compiled Reference-pattern construction may use helpers because they own
real reusable policy rather than one tensor operation.

Pattern keyword arguments use the same semantic name as the pattern axis.
Avoid intermediate variables named only `N`, `D`, `L`, or `C` when the value is
not immediately obvious. PyTorch shape inspection remains appropriate for
validation and allocation sizes.

## Architecture Refactor

### Attention and rotary embeddings

Current head manipulation in `src/relflow/architecture/attention.py` combines
`reshape`, `transpose`, and helper methods. Inline named transformations:

```python
query = rearrange(
    self.q_proj(query),
    "batch query (head head_channel) -> batch head query head_channel",
    head=self.nhead,
)
key = rearrange(
    self.k_proj(key),
    "batch key (head head_channel) -> batch head key head_channel",
    head=self.n_kv_heads,
)
value = rearrange(
    self.v_proj(value),
    "batch key (head head_channel) -> batch head key head_channel",
    head=self.n_kv_heads,
)
```

`RotaryEmbedding.forward` should accept `[..., token, head_channel]` and rely on
broadcasting of `[token, rotary_frequency]` sine/cosine values. Attention can
then apply rotary directly to the four-dimensional query and key tensors,
removing both `splitheads` and `rotate`.

The SDPA result returns to channel-last form with:

```python
context = rearrange(
    context,
    "batch head query head_channel -> batch query (head head_channel)",
)
```

Rotary sine/cosine interleaving may use:

```python
rotated = rearrange(
    torch.stack((rotated_even, rotated_odd), dim=-1),
    "... rotary_frequency pair -> ... (rotary_frequency pair)",
)
```

The order is pair-minor and must remain exact. Odd head widths retain their
existing passthrough channel.

### Branch encoder

`src/relflow/architecture/encoder.py` should state the complete structural
transition explicitly:

```python
context = torch.cat(payloads, dim=-2)
encoded, leading_shape = pack([context], "* token channel")

for layer in self.encoder:
    encoded = layer(encoded)

[memory] = unpack(encoded, leading_shape, "* token channel")
pooled_flat = (
    encoded.mean(dim=-2, keepdim=True)
    if isinstance(self.pooling, Mean)
    else self.pool(encoded)
)
[pooled] = unpack(pooled_flat, leading_shape, "* width channel")
```

Here `*` captures `(batch, *outer)` and `leading_shape` restores those axes
exactly; it does not include the memory-token axis.

Every live Branch exposes its restored `memory`. It also computes `pooled`, then
merges only the final outer coordinate with pooling width to form the routed
`summary`:

```python
summary = rearrange(
    pooled,
    "batch ... coordinate width channel -> "
    "batch ... (coordinate width) channel",
)
```

That final outer coordinate belongs to the destination/parent domain; the
source Branch's own local instance/token axis has already been pooled. The
generated root/no-outer case keeps `[batch, width, channel]` directly.
Coordinate is major and width is minor. The compiled execution graph, not
arrival order, gathers attached native summaries and Reference memories.
Structurally grafted sources remain in the forward-local maps and are simply
absent from their original parent's compiled native-input list.

### Decoder context

`src/relflow/tensorfields/base.py` currently reshapes each Parcel and then
concatenates. Replace that boundary with one `pack` call:

```python
memory, _ = pack(
    [parcel.payload for parcel in parcels],
    "batch * channel",
)
pooled = (
    memory.mean(dim=-2, keepdim=True).expand(-1, self.pool_width, -1)
    if isinstance(self.pooling, Mean)
    else self.pool(memory)
)
```

This is valid for mixed `[batch, channel]`, `[batch, token, channel]`, and
`[batch, *coordinates, token, channel]` inputs and retains Parcel order. The
compiled `decoder_contexts` plan remains the source of that order and prevents
an empty list from reaching the decoder.

### Pooling and convolution

Attention and Convolution modules own output width as defined by the Reference
Context specification. Convolution uses named channel-order changes:

```python
channels_first = rearrange(memory, "batch token channel -> batch channel token")
memory = rearrange(channels_first, "batch channel token -> batch token channel")
```

Mean may stay as native `mean(..., keepdim=True)` plus `expand`; learned query
expansion may likewise remain native to preserve explicit zero-stride behavior.
Reference Attention and Convolution reducers use the same concrete modules with
the compiler's role-specific pack/reduce/restore layout kept inline.

### Reference block reduction

The Reference compiler emits one static pattern per `ReferenceId` from its
exact source address and resolved schema-axis addresses. A parameter-free
reduction is a direct `einops.reduce` call. For
example, reducing `b1` to four coarse blocks while preserving `b2` compiles to
the equivalent of:

```python
reduced = reduce(
    memory,
    "batch (d0_out d0_reduce) d1 channel -> "
    "batch d0_out d1 channel",
    reduction="mean",
    d0_out=4,
)
```

Multiple selected dimensions get independent `*_out` and `*_reduce` factors
in schema order. The pattern omits only the reduction factors; retained and
coarse axes remain ordered and visible. The compiler validates divisibility
before storing the pattern.

For Attention or Convolution reduction, retained and coarse axes become an
effective batch and the reduction factors become one token axis. The consuming
Branch calls `.reference_reducers[str(reference_index)]` on that tensor,
squeezes the required width-one result, and restores the retained/coarse axes
inline. Each declaration has exactly one source and owns its indexed module.
Separate declarations never share reducer parameters, even when their source
addresses and configurations are identical. Duplicate-address declarations
reuse source memory but run and concatenate each occurrence independently. The
public configuration, exact axis semantics, declaration order, and validation
rules remain owned by the Reference Context specification.

## Tensorfield Refactor

### Remove unnecessary flattening

Category, Set, Number, Vector, and DateParts embedders do not need to flatten
schema coordinates before applying standard PyTorch modules. These operations
already act on the final axis or preserve arbitrary leading axes.

For example, the Vector embedder can become:

```python
projected = self.linear(inputs.content)
embeddings = self.embeddings(inputs.state)

return Parcel(
    payload=projected + embeddings,
    origin=self.origin,
    destination=self.destination,
    batch_size=inputs.state.shape[0],
)
```

Category can embed state/content tensors in their existing shape. Set can keep
its clear `content.matmul(weights)` contraction. DateParts linears can consume
each existing final size-two axis. This removes `N, *dims`,
`D = math.prod(...)`, and paired restore reshapes without introducing einops.

Number can broadcast its frequency weights over `inputs.content[..., None]`,
but its feature order is distinct from Hash. It remains trigonometric-major:
all sine frequencies, followed by all cosine frequencies. The exact operation
is `torch.cat((weighted.sin(), weighted.cos()), dim=-1)`; using Hash's
`(frequency trig)` interleave would change the learned linear input contract.

```python
weighted = content[..., None] * self.weights
fourier = torch.cat((weighted.sin(), weighted.cos()), dim=-1)
```

Number's training jitter is the exception to removing its flattening entirely.
Pack normalized content to one canonical row-major slot axis before the two
`rand_like` calls, then unpack it immediately afterward. This preserves the
current random draw shape and fixed-seed behavior while the embedding, Fourier,
and linear operations retain their schema coordinates.

```python
flat_content, slot_shape = pack([content], "*")
flat_content = jitter(flat_content, jitter_amount=self.jitter)
[content] = unpack(flat_content, slot_shape, "*")
```

### Text

Text is the genuine flatten/restore case because the external encoder consumes
`[document, token]`. Use `pack` metadata across early returns and chunked model
execution:

```python
flat_ids, documents = pack([content[INPUT_IDS]], "* token")
flat_mask, _ = pack([content[ATTENTION_MASK]], "* token")
flat_state, _ = pack([state], "*")

# Encode valued documents in chunks.

[embeddings] = unpack(flat_embeddings, documents, "* channel")
```

The dynamic `torch.cat(encoded_chunks, dim=0)` remains native. Masked mean
pooling may remain native because its numerator and denominator are already
clear.

### Hash

Hash encoding benefits from retaining all schema coordinates and naming its
special axes:

```python
normalized = inputs.content.to(self.weights.dtype).div(_HASH_NORMALIZER)
weighted = einsum(
    normalized,
    self.weights,
    "... hash, frequency -> ... hash frequency",
)
sinusoidal = rearrange(
    torch.stack((weighted.sin(), weighted.cos()), dim=-1),
    "... hash frequency trig -> ... hash (frequency trig)",
)[..., : self.d_model]
content = reduce(sinusoidal, "... hash channel -> ... channel", "sum")

payload = torch.where(
    inputs.state.eq(Tokens.valued)[..., None],
    content,
    self.state_embeddings(inputs.state),
)
```

The frequency buffer is changed to `[frequency]` so the named `einsum` contract
is exact; earlier checkpoint buffer shapes do not constrain this refactor.
Tests must preserve frequency-major,
trig-minor interleaving, summation across Hash functions, odd `d_model`
truncation, and raw integer normalization.

Hash decoder losses should replace repeated `total_slots * n_hashes` arithmetic
with named layouts:

```python
logits = rearrange(
    prediction.payload[TensorKey.content],
    "... (hash bucket) -> (... hash) bucket",
    hash=n_hashes,
    bucket=n_buckets,
)
targets = rearrange(targets, "... hash -> (... hash)")
per_slot = rearrange(per_hash, "(slot hash) -> slot hash", hash=n_hashes)
```

### Loss boundaries

Tensorfield losses may standardize semantic flattening after embedder changes:

```python
state_logits = rearrange(state_logits, "... classes -> (...) classes")
state_targets = rearrange(state_targets, "... -> (...)")
trainable = rearrange(trainable, "... -> (...)")

vector_values = rearrange(vector_values, "... feature -> (...) feature")
```

Category, Set, Number, DateParts, Text, Vector, Boolean, and Hash must all use
the same row-major slot order for logits, targets, and masks. This is a
mechanical second phase after embedder parity is established, not a prerequisite
for the Reference runtime.

## Native PyTorch Boundaries

Keep these operations native unless surrounding architecture changes give them
new semantic axes:

- attention padding-mask broadcast such as `mask[:, None, None, :]`;
- local scalar `squeeze`/`unsqueeze`, obvious one-axis means and sums, and
  `torch.stack(losses)`;
- zero-stride query or mean-result expansion;
- masking with `unsqueeze(-1).expand_as(...)` where allocation behavior matters;
- text encoder chunk concatenation;
- recursive `Prediction.squeeze` behavior for tensors, NumPy values, and Python
  containers;
- branch-mask flattening in `tensorfields/base.py` whose in-place writes rely on
  view aliasing;
- ragged traversal in `data/nested.py`, querypath iteration, overflow handling,
  and dataset streaming.

`rearrange` may return a view or a copy depending on layout. Code must not rely
on aliasing through an einops result. Any path that mutates a flattened view
must remain native or be redesigned around an explicit canonical tensor and
tested separately.

## Compilation And Runtime Constraints

- Patterns in ordinary modules are static string literals.
- Reference reduction patterns are compiled once from safe aliases after schema
  binding and stored only in the immutable execution plan.
- Reference patterns and trainable reducer calls are keyed by
  `ReferenceId = (consumer_address, reference_index)`; the plan stores that
  declaration's one exact source address. Dependency scheduling may deduplicate
  a source-to-consumer pair, but context occurrences never are.
- User node names, addresses, and arbitrary strings never enter executable
  patterns.
- Divisibility, axis applicability, coordinate-prefix compatibility, and
  effective output shapes fail during schema compilation rather than first
  forward execution.
- Device, dtype, autocast, train/eval dropout, and distributed behavior remain
  owned by PyTorch tensors and modules; einops introduces no alternate state.
- Non-contiguous inputs are supported when the equivalent PyTorch operation
  supported them. Tests should not assert whether a result happens to be a view.
- No live shape metadata, packed-shape list, callable reducer, or tensor is
  serialized in the schema or checkpoint.

## Migration Sequence

### Slice 1: architecture boundaries

Refactor `architecture/attention.py`, `architecture/rotary.py`,
`architecture/encoder.py`, pooling implementations, and
`tensorfields/base.py`. Implement Reference memory/summary routing, pooling
width, compiled grafting, and decoder contexts against these named layouts.

This slice establishes the conventions needed by the Reference feature and is
included in that refactor.

### Slice 2: embedders

Remove unnecessary flattening from Category, Set, Number, Vector, and DateParts.
Convert Text to explicit `pack`/`unpack`. Convert Hash to named frequency,
trigonometric, hash, and channel axes.

### Slice 3: losses

Normalize loss-boundary flattening across tensorfields with protected semantic
last axes. Keep each datatype's objective, masking, metrics, and edge-case
behavior unchanged.

Each slice should be a distinct reviewable change with its focused tests passing
before the next slice begins. The final branch may contain all slices, but
behavioral failures must be attributable to one boundary.

## Acceptance Tests

### Architecture

- Attention fixed-weight outputs and input/parameter gradients match the
  pre-refactor implementation for query/KV head counts `1`, `2`, and `4`.
- Grouped-query attention preserves distinct query and KV head counts, unequal
  query/key lengths, padding-mask behavior, and SDPA dropout behavior.
- Rotary embeddings preserve even/odd pairing and odd-width passthrough for
  three- and four-dimensional inputs.
- Root, top-level, nested, multiply widened, referenced, and structurally
  grafted Branches preserve exact memory and summary shapes plus
  coordinate-major/width-minor ordering.
- Scalar and tuple-valued Branch references preserve declaration order; two
  declarations naming one cached source run their reducers separately and
  contribute two context segments without changing topological order.
- `pack`/`unpack` Branch transitions preserve values and gradients for
  contiguous and non-contiguous inputs.
- Decoder `pack("batch * channel")` preserves declared Parcel order, supports
  mixed ranks and unequal memory lengths, and propagates gradients to every
  input.
- Compiled-inactive dangling Branches execute no encoder/pool while their
  grafted sources retain values and gradients through Reference consumers.

### Tensorfields

- Category, Set, Number, Vector, and DateParts embedder outputs and gradients
  match fixed-seed pre-refactor results across root and nested shapes.
- Text preserves valued-document selection, no-valued early return, encoder
  batching, all pooling modes, target-embedding caching, device placement, and
  restored structural shape.
- Hash fixed-weight tests prove exact integer normalization,
  frequency-major/trig-minor ordering, per-hash summation, state fallback, odd
  `d_model`, and nested shapes.
- Every loss aligns logits, targets, and trainable/valued masks in the same
  row-major order and preserves objective and metric values.

### Runtime qualities

- Tests cover CPU and supported accelerator dtype/device placement, autocast,
  train/eval dropout, and backward passes.
- No refactored path mixes batch observations or changes schema-coordinate
  order.
- No one-use tensor-shape wrapper or factory is introduced.
- `git diff --check`, focused architecture/tensorfield tests, the public API
  suite, and static type checks pass after each slice.

## Documentation And Review

This is an unpublished implementation specification. It should not be added to
the Quarto navigation until the refactor is implemented. Public examples
continue to describe schema behavior rather than expose internal einops
patterns.

Code review should evaluate each pattern as executable shape documentation:

1. Are every preserved and collapsed axis visible?
2. Is the major/minor order of merged axes correct?
3. Would native PyTorch be shorter without losing semantic information?
4. Does the code rely on view, aliasing, or zero-stride behavior?
5. Do fixed-value and gradient tests cover the boundary?

## Deferred Questions

- Whether internal tensor annotations should adopt a shape-aware typing tool in
  addition to einops patterns.
- Whether repeated loss-boundary layouts justify shared constants; v1 keeps
  patterns inline to avoid indirection.
- Whether future custom tensorfields need a documented axis vocabulary beyond
  the conventions in this file.
- Whether `torch.compile` should become a supported/tested contract; this
  refactor introduces no such guarantee by itself.

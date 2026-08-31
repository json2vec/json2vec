# Unified Masking And Pruning Design Spec

- Status: Draft; proposed behavior, not the current public API
- Date: 2026-08-29
- Audit baseline: `e7cf72d`
- Scope: Schema, tensorization, tensorfields, routing, attention, pooling, loss,
  inference, and checkpoint serialization

## Decision Summary

RelFlow will model two different encoder interventions with two immutable public
policy types:

- `rf.Mask(...)` replaces selected input with the learned mask representation.
- `rf.Prune(...)` removes selected input from encoder context. A pruned value is
  never passed through its input embedder and never contributes an attention or
  pooling entry.

Both policies use the same selection vocabulary and one `target` flag:

- `target=False` means train-only regularization. The selected value is not
  trainable and creates no loss of its own.
- `target=True` means reconstruction. The pristine value is retained as a
  target, marked trainable, and scored during train, validation, and test.

This makes the requested supervised form concise and literal:

```python
label=rf.Category(prune=rf.Prune(target=True))
```

The label has no input/context embedding. Its datatype may still construct a
target-side representation used only by its loss. For example, a pruned Text
target may run the frozen text encoder to define its target embedding; that
representation must never enter encoder context.

The first public release supports whole-node `Prune` policies. Arbitrary
position-level pruning remains gated until tensorfield extensions can embed a
selected ragged subset without ever embedding the pruned values. `Mask`
supports both position and whole-node selection in the first release.

## Motivation

The current API combines several different ideas:

- `p_mask` samples positions, substitutes a mask representation, and
  reconstructs them;
- fractional `p_prune` samples a whole leaf, but also substitutes a mask
  representation and reconstructs it;
- exactly `p_prune=1` / `target=True` skips the leaf parcel and reconstructs it;
- branch `Mask` policies select coordinates and reconstruct them;
- `active=False` removes a node from the active runtime contract;
- `"<MASK>"` requests a prediction-time masked value.

The discontinuity at `p_prune=1` makes *prune* misleading. It also conflates
three independent questions:

1. What is selected?
2. Does the encoder see a learned mask token or no input at all?
3. Does the selected value create a reconstruction objective?

The new API answers those questions with selection geometry, the concrete
`Mask`/`Prune` type, and `target` respectively.

## Goals

- Give leaves and branches the same policy grammar.
- Make feature-dropout regularization concise.
- Give pruning one strict structural meaning.
- Let structural omission optionally create a reconstruction target.
- Define one canonical model without legacy execution modes.
- Keep policy composition deterministic and independent of declaration order.
- Centralize structural traversal so datatypes only own content conversion and
  corruption.

## Non-Goals

- Replacing ordinary neural-network dropout.
- Claiming causal feature importance from an ablation.
- Applying stochastic configured policies during ordinary prediction.
- Using a new `Tokens.pruned` or `Tokens.ablated` value.
- Making Awkward Array part of the masking runtime; masking remains in Torch.
- Supporting policies on the generated root in the first release.

## Terms And Invariants

### Mask

The selected input remains present in encoder context. Its original state and
content are replaced by the learned masked-state representation. The model can
observe that a value was withheld, but cannot observe the original value.

### Prune

The selected input is structurally absent from encoder context:

- the original value does not enter its input embedder;
- it contributes no parcel payload, attention key/value, or pooling member;
- changing the original bytes while keeping the prune decision fixed cannot
  change encoder output;
- its absence and any target-only loss representation may still change model
  output and gradients.

`Prune` is a representation-level intervention. Tensorization, vocabulary
reservation, preprocessing, and explicitly separated estimator observation may
still inspect the raw value. Removing a field from the entire data-estimation
pipeline is the separate `active=False` operation.

### Regularize

A policy with `target=False` limits input information only during `train`. Its
selected coordinates are not marked `trainable`, do not activate their own
decoder, and create no direct loss. They affect learning only through another
objective that uses the corrupted context.

### Reconstruct

A policy with `target=True` retains pristine state/content and trains the
corresponding leaf decoder. Reconstruction policies run in train, validation,
and test. A batch in which a stochastic policy selects nothing simply skips
that field's loss; it must not reduce an empty tensor or produce NaN.

### Deactivate

`active=False` persistently excludes a node from querying, tensorization,
forward routing, output, counters, and objectives. It is not a stochastic
policy or a request-scoped ablation.

### Explicit Inference Mask

`"<MASK>"` remains the prediction-time imputation protocol. It deliberately
creates a learned mask representation and requests a decoded output. It is not
pruning.

## Semantic Matrix

| Policy | `target` | Input/context behavior | Own target/loss | Strata |
| --- | ---: | --- | --- | --- |
| `Mask` | `False` | Learned mask representation | No | Train only |
| `Mask` | `True` | Learned mask representation | Yes | Train, validation, test |
| `Prune` | `False` | No input/context embedding | No | Train only |
| `Prune` | `True` | No input/context embedding | Yes | Train, validation, test |

Stochastic/configured policy sampling is bypassed in predict. An effective
deterministic whole-node `Prune(target=True)` is the exception: it defines a
schema target and remains structurally absent in every stratum, including
predict. In particular, even `Mask(rate=1, unit="node", target=True)` is a
reconstruction policy rather than an automatic prediction role: normal
prediction leaves its input visible and does not emit it unless the caller uses
`"<MASK>"` or another explicit output request.

## Public Policy Models

The public types are frozen and reject unknown fields:

```python
class SelectionPolicy:
    name: str | None = None

    # Exactly one after normalization. Omitting both means rate=1.
    rate: float | None = None
    count: int | None = None

    unit: Literal["position", "node"]
    target: bool = False

    # Branch-position selection geometry.
    window: int | None = None
    branch: bool = False
    start: bool = False
    offset: int = 0
    exclude: tuple[str, ...] = ()

    @property
    def regularize(self) -> bool:
        return not self.target


class Mask(SelectionPolicy):
    unit: Literal["position", "node"] = "position"


class Prune(SelectionPolicy):
    unit: Literal["node"] = "node"  # First public release.
```

The concrete class is the encoder effect. There is no `effect="mask"` string
and no conflicting pair of stored `regularize` and `target` booleans.
`policy.regularize` is an immutable derived property: it is exactly
`not policy.target`. This exposes the useful user-facing concept while
`rf.Prune(target=True)` remains the unambiguous reconstruction spelling.

### Selection Amount

- `rate` is a Bernoulli probability in `[0, 1]`.
- `count` chooses exactly that many eligible positions per candidate group,
  capped by the number available.
- Passing both is an error.
- Omitting both canonicalizes to `rate=1`.
- `count` is valid only for `unit="position"` and therefore is initially a
  `Mask` feature.
- `rate=0` is a no-op and does not count as a training objective.

### Selection Unit

`unit="position"` means:

- a leaf `Mask` samples configured leaf positions independently;
- a branch `Mask` samples retained branch coordinates atomically across every
  included active descendant leaf;
- a nested branch samples separately for each parent coordinate.

`unit="node"` makes one decision per top-level observation and broadcasts it
through every configured position owned by the leaf or branch. On a branch,
the decision covers every included active descendant.

Branch `window`, `start`, `offset`, and `branch` alter position candidates only.
`exclude` removes exact descendant leaf addresses from that policy only.

Eligibility has one canonical rule: padding is structural absence and is never
an input, Mask candidate, or regularization candidate.

- a leaf position `Mask` can select valued and explicit-null positions;
- a branch position `Mask` selects occupied branch coordinates and broadcasts
  through occupied descendant positions;
- a node policy makes one owner-level decision, but its Mask effect intersects
  occupied positions;
- Prune removes the selected owner/subtree from routing; positions that were
  already padded remain absent.

This aligns padding and Prune at the attention boundary without conflating their
provenance: padding comes from data shape, while Prune comes from a policy.

### Node Arguments

Leaves and non-root branches accept:

```python
mask: Mask | float | bool | None = None
masks: tuple[Mask, ...] = ()
prune: Prune | float | bool | None = None
prunes: tuple[Prune, ...] = ()
```

The singular and plural spelling for one effect cannot be combined. Plural
arguments accept only policy collections. A `Mask` is rejected by `prune`, and
a `Prune` is rejected by `mask`.

Normalize booleans before numbers because `bool` subclasses `int`:

| Input | Canonical value |
| --- | --- |
| `mask=None` / `False` | No local Mask policy |
| `mask=0.15` | `Mask(rate=.15, target=False)` |
| `mask=True` | `Mask(rate=1, unit="node", target=False)` |
| `mask=Mask(...)` | That policy |
| `prune=None` / `False` | No local Prune policy |
| `prune=0.15` | `Prune(rate=.15, target=False)` |
| `prune=True` | `Prune(rate=1, target=False)` |
| `prune=Prune(...)` | That policy |

Examples:

```python
model = rf.Model(
    # Per-value learned-mask dropout.
    merchant=rf.Category(mask=0.10),

    # Whole-field structural dropout.
    amount=rf.Number(prune=0.10),

    # Masked-value modeling.
    notes=rf.Text(mask=rf.Mask(rate=0.15, target=True)),

    # Structurally withheld supervised target.
    label=rf.Category(prune=rf.Prune(target=True)),
)
```

### Branch Policies

Branches have no datatype decoder. A target-bearing branch policy expands to
its included active descendant leaves:

- one branch-coordinate decision is shared atomically by those descendants;
- each descendant intersects the broadcast decision with its own occupied
  (non-padded) mask, so deeper padding is never made trainable or pruned;
- selected positions become trainable for each descendant leaf decoder;
- output and metric addresses remain the leaf addresses;
- leaf weights and datatype losses continue to apply normally;
- `exclude` removes a leaf from both the effect and the objective.

A deterministic whole-node `Prune(target=True)` on a branch makes all included
descendant leaves effective schema targets. Descendants already targeted by
their own or another effective policy remain targets and are never turned into
context. The generated root rejects policies in the first release; root
semantics are deferred until this branch contract has shipped and been
profiled.

### Automatic Schema Targets

There is no separate node-level `target=True` argument or `target` property.
`Schema.targets` is derived from deterministic whole-node
`Prune(target=True)` policies:

- on a leaf, that leaf is a schema target;
- on a branch, every included active descendant leaf is a schema target;
- an ancestor exclusion removes the descendant from that target role.

Canonical serialization contains only `masks` and `prunes`, never singular
shorthands. Schema mutation replaces or edits named policies directly rather
than translating through a second target API.

## Policy Identity, Sampling, And Composition

All policies select from one pristine pre-policy snapshot. Selection is
centralized in one policy engine after tensorization and estimator observation,
but before input embedding.

When a node has multiple policies, each must have a unique non-empty `name`.
The policy engine derives a generator stream from the data-loader mask seed,
epoch, schema address, and policy name. This makes exact selections stable when
policies are reordered. A single unnamed policy receives a stable reserved ID.
Validation/test reconstruction policies resample according to the seeded
data-loader lifecycle, matching normal stochastic-evaluation behavior; a fixed
seed reproduces a run.

For already-sampled selections define:

```text
occupied = pristine_state != Tokens.padded

P_target = union(Prune selections where target=True)
P_reg    = union(Prune selections where target=False)
M_target = union(Mask selections where target=True)
M_reg    = union(Mask selections where target=False)

pruned      = occupied & (P_target | P_reg)
present     = occupied & ~pruned
masked      = present & (M_target | M_reg)
trainable   = occupied & (P_target | (M_target & ~pruned))
regularized = occupied & (P_reg | (M_reg & ~pruned)) & ~trainable
```

The effect and purpose of the winning policy stay together:

- any Prune wins over a Mask at the same coordinate;
- a target Prune wins over a regularizing Prune;
- a regularizing Prune suppresses an overlapping Mask reconstruction rather
  than accidentally synthesizing a target-bearing Prune;
- within Masks, reconstruction wins over regularization.

Selections are order-independent. `mask=False` and `prune=False` remove local
policies only; they do not cancel inherited policies. An ancestor policy's
`exclude` is the way to opt a descendant out.

## TensorField Contract

The canonical working batch contains value state, pristine reconstruction
targets, and policy/routing provenance:

```python
state: Tensor
content: Tensor | TensorDict
targets: TensorDict
trainable: BoolTensor
regularized: BoolTensor
present: BoolTensor
```

All three boolean tensors have the state shape, bool dtype, and state device.
`targets` is a pristine, schema-shaped snapshot only when a target-bearing
selection needs it; its presence does not itself imply a loss. Runtime loss
dispatch is gated by `trainable.any()`.

| Role | `present` | Working state/content | `trainable` | `regularized` |
| --- | ---: | --- | ---: | ---: |
| Visible | `True` | Original | `False` | `False` |
| Padding | `False` | Padded / safe content | `False` | `False` |
| `Mask(target=False)` | `True` | Masked / zeroed | `False` | `True` |
| `Mask(target=True)` | `True` | Masked / zeroed | `True` | `False` |
| `Prune(target=False)` | `False` | Sanitized masked / zeroed | `False` | `True` |
| `Prune(target=True)` | `False` | Sanitized masked / zeroed | `True` | `False` |

The masked/zeroed storage behind `present=False` is hygiene, not a mask-token
encoder effect: it must never reach the input embedder. Target-side loss code
reads pristine `targets` instead.

Presence and objective eligibility are separate. `present` starts as
`state != Tokens.padded`; Prune can only turn additional positions false. A
whole-node Prune omits the complete occupied input geometry for that
observation. `trainable` is resolved independently and always intersects
occupied positions. Padding is storage geometry, never a reconstruction
target. If sequence extent should be learned, it must be represented by an
explicit structural or numeric objective. A false-present occupied coordinate
may be trainable against pristine targets without being an encoder input.

Required invariants:

- `trainable` and `regularized` are disjoint;
- `regularized` is empty outside train;
- present masked positions have `state == Tokens.masked`;
- false-present positions never reach input embedding or encoder routing;
- every trainable position has pristine state/content in `targets`;
- prediction `"<MASK>"` may be masked with neither provenance bit;
- `Tokens` gains no prune/ablate state.

The policy engine invokes one datatype-owned `sanitize(selection)` operation
after it snapshots targets. Every tensorfield implements this contract;
`regularized` and `present` are required members.

## Skipping Input Embedding

Whole-node Prune is implementable without changing a datatype's schema-shaped
embedding contract:

1. Split the TensorField on the outer batch dimension into present and pruned
   observations.
2. Pass only present observations, with their normal inner schema shape, to the
   input embedder.
3. Scatter returned payloads into fixed schema slots.
4. Mark every scattered slot for pruned observations `present=False`.

If every observation is pruned, skip the embedder entirely. The same
rank-conditional issue applies to branch encoders, pools, and decoders when a
rank has no present context or selected target. RelFlow therefore uses one
global rule: every conditionally bypassed trainable module contributes a
zero-valued autograd anchor over its parameters to that rank's loss. The anchor
computes no value representation but keeps the DDP parameter graph consistent.

Before skipping an optimizer step, ranks all-reduce a `has_objective` flag. If
any rank has work, ranks without local work backpropagate the anchored zero
loss; only a globally empty batch may skip together. An alternative
unused-parameter DDP strategy is acceptable only if it is the configured global
strategy and passes the same multi-rank tests.

Position-level Prune cannot use whole-observation subset embedding for arbitrary nested
coordinates when embedders consume and return the full schema geometry. It is
therefore rejected by validation in the first release.
A later release may add an explicit extension protocol such as
`embed_selected(field, selection) -> (payload, scatter_index)`. It becomes
public only after every built-in, nested branch, Text, and custom-extension
implementation satisfies the no-input-embedding invariant.

## Parcel, Attention, And Pooling Contract

Every `Parcel` carries validity metadata:

```python
class Parcel:
    payload: Tensor
    present: BoolTensor
    origin: Address
    destination: Address | None
```

The exact contract is:

- `present.shape == payload.shape[:-1]`;
- it is bool and on the payload device;
- flattening and concatenation apply identically to payload and presence;
- attention excludes false keys/values;
- mean pooling excludes false entries from both numerator and denominator;
- a branch output is present for a parent row iff at least one local child
  parcel is present.

`present` is false for padding and Prune. Fixed slots preserve rotary
coordinates; a false slot is never compacted into a different position. A row
with no valid context bypasses attention/pooling and supplies a zero `d_model`
context directly to the decoder. The decoder therefore defines the
unconditional prediction from zero context; no fabricated field embedding is
introduced. An
`embed=True` output for an internally empty branch is the same zero vector;
routing presence remains internal rather than adding an unspecified public
output key. Runtime inference pruning rejects selected `embed=True` addresses
as described below.

Fixed-slot routing is the only layout. There is no compact compatibility mode.

## Observation, Counts, And Normalization

Public counts and normalization statistics describe pristine training
exposures, not random augmentation outcomes. There are two lifecycle points:

```python
# Before TensorField construction: raw ordered values and worker-safe resources.
vocabulary.reserve(ragged_field.values, learn=True)

# After TensorField construction, before policies: numeric/tensor observations.
embedder.observe(pristine_field)

# After policies: input/context representation only.
parcel = embedder(present_subset)
```

Vocabulary reservation remains in tensorization because a TensorField no longer
contains raw labels and `InterprocessEncodingContext` owns worker-safe ordering.
The post-tensorization observation hook is the only place counters and
normalization update, so each training exposure is observed exactly once.
Number separates normalizer update from normalization apply.

This is why `Prune` is specifically a neural representation intervention, not
a guarantee that raw bytes have no future estimator effect. A full
estimator/data ablation uses persistent schema deactivation or a preprocessing
change.

## Decoder, Loss, And Output Rules

- A leaf decoder/loss runs only when it is a schema target or that batch has at
  least one `trainable` position. `embed=True` may request output but never
  causes loss against an empty target.
- If no stochastic target position is selected, the field loss is skipped.
- If a rank has no selected objective while another rank does, it backpropagates
  the anchored zero loss. Only when the all-reduced objective flag is empty may
  every rank skip the step together; never compute an empty mean.
- A custom tensorfield advertises an independent objective through
  `Request.has_objective() -> bool` (default `False`). This capability is static
  and must not depend on seeing a batch.
- A model with no possible supervised, reconstruction, or declared extension
  objective remains valid for embedding/inference. `Trainer.fit` raises a clear
  no-training-objective error at loop start instead of rejecting construction.
- Non-predict output reports `inferred = trainable`; regularized or merely
  pruned positions are not inferred outputs.

## Runtime Inference Pruning

Request-scoped deterministic ablation avoids graph rebuilds:

```python
baseline = model.predict(records)
without_amount = model.predict(records, prune="record/amount")
without_history = model.predict(
    records,
    prune=rf.where("address").matches(r"^record/history(?:/|$)"),
)
```

The same argument is accepted by `Model.encode` only with `strata="predict"`.
It accepts addresses, address collections, or selectors and initially operates
on whole addresses for every observation. A branch expands to active context
descendants; effective targets are already absent and are ignored. An explicit
target address is an error. If expansion finds a non-target `embed=True`
descendant, reject the whole request rather than silently changing the public
embedding-output set.

Resolution occurs before querying/tensorization but after preprocessing. A
fully request-pruned field may be absent from the processed observation. It may
be absent from the raw record only when the configured preprocessor itself can
run without that key; otherwise the caller must still supply it. If present,
the processed field is ignored by tensorization and its input embedder. This
remains a model-input ablation rather than a preprocessor ablation.

Explicitly selecting any effective target or selecting/expanding to an
`embed=True` output is rejected so the task prediction/output set stays
identical between baseline and ablation. Runtime prune never decodes the pruned
input itself. It is distinct from:

- `"<MASK>"`, which retains a mask parcel and requests inference;
- `active=False`, which persistently excludes a node from the active contract;
- configured Prune policies, which are part of training schema.

## Serialization

Canonical schema serialization writes only the new model:

- nodes contain immutable `masks` and `prunes` tuples;
- every policy dictionary includes `name`, `rate` or `count`, `unit`, `target`,
  and its non-default geometry fields;
- branch position selection is always atomic across included descendants;
- routing always uses fixed slots and required presence masks;
- `p_mask`, `p_prune`, a node-level `target`, legacy Branch Mask payloads, and
  compact routing are invalid input.

Checkpoint `version` remains product provenance. Loading an incompatible old
schema fails clearly; RelFlow does not rewrite or emulate it. Dump/load/dump of
the new schema is stable and exact.

## Validation

Schema validation rejects:

- both `rate` and `count`, invalid bounds, and `count` on node selection;
- position-level `Prune` until the selected-embedding protocol ships;
- branch-coordinate controls on node policies;
- exclusions outside the owning subtree;
- multiple policies on one node without unique names;
- `Mask` values passed through `prune` and vice versa;
- singular and plural forms of the same effect together;
- removed fields such as `p_mask`, `p_prune`, or node-level `target`;
- policies on the generated root in the first release.

No-context rows are valid and use the zero-context decoder rule. A no-objective
error occurs at fit start, not Model construction.

## Delivery

Ship the contract as one breaking change. The launch includes immutable policy
parsing, seeded policy identity, train-only regularization, pristine
observation, required TensorField/Parcel presence, node-subset embedding,
fixed-slot attention and pooling, empty context, branch target expansion,
runtime whole-address pruning, and the new extension contract. The same change
removes `p_mask`, `p_prune`, node-level `target`, and the old Branch Mask schema,
and updates every user-facing guide, example, API table, README section, and
whitepaper passage.

Position-level Prune and root policies are deferred features, not alternate
execution modes. Position-level Prune requires one selected-embedding protocol
implemented by every tensorfield before it becomes public.

## Acceptance Tests

The breaking launch must cover:

1. all four policy/target combinations across train, validation, test, and
   predict;
2. leaf, repeated branch, nested branch, null, and padded coordinates;
3. rate boundaries, exact Mask counts, windows, offsets, overflow, exclusions,
   and atomic sibling selection;
4. seeded policy reorder producing identical selections;
5. every overlap in the composition table, including a regularizing Prune
   suppressing a Mask target;
6. pristine targets, counts, and normalization under mixed policies;
7. a spy proving pruned input values never enter the input embedder while
   target-side Text representation remains possible;
8. attention/pooling invariance when sanitized bytes behind `present=False`
   change;
9. mixed rows, all-pruned rows, empty nested branches, and empty decoder
   heritage without NaN;
10. target-bearing branch expansion, exclusions, weights, metrics, and output
    addresses;
11. stochastic empty selections and `embed=True` without empty-target loss;
12. runtime prune with omitted raw keys, selector expansion, target/embed
    rejection, and no graph/Parameter rebuild;
13. DDP ranks conditionally bypassing embedders, branch pools, and decoders,
    including a rank-local empty objective, without a hang;
14. static `active=False`, runtime prune, `"<MASK>"`, and target semantics
    remaining distinct;
15. canonical schema dump/load/dump stability and clear rejection of removed
    schema fields;
16. custom tensorfields satisfying the required presence/sanitize contract.

The later position-Prune feature adds arbitrary selected-coordinate, nested
scatter, and extension conformance tests. Root policies require a separate
root-specific proposal.

## Documentation Rollout

This Markdown file is intentionally excluded from the rendered Quarto site.
Publish the API and update the dynamic masking, embeddings, data-types, branch,
data-module, evaluation, field-importance, mutation, custom-tensorfield, and
public-API pages atomically with the implementation, together with the README,
examples, and whitepaper.

## Remaining Release Decisions

1. Should position Prune become part of `rf.Prune` or a distinct
   advanced policy with a visibly stronger extension requirement?
2. Should root policies reuse the zero-context rule in a later release, or stay
   forbidden because pruning an entire observation is rarely useful?

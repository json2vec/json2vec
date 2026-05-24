# Dynamic Node Mutations

This spec covers post-construction schema-tree mutations and runtime node-state
mutations.

## Goal

Allow users to evolve an existing model schema with explicit in-place
mutations:

- `model.extend(...)` adds new schema nodes under exactly one selected `array`
  node.
- `model.delete(...)` permanently removes selected schema nodes from the tree.
- `model.reset(...)` reinitializes selected node modules while preserving schema
  hyperparameters.

Schema-changing operations rebuild compatible runtime modules so the same model
instance can keep using existing weights where shapes still match. Reset is more
targeted: it intentionally discards selected runtime module state.

## Public API

Extension:

```python
model.extend(
    *args,
    include_root=True,
    use_cache=True,
    validate=True,
)
```

Deletion:

```python
model.delete(
    *predicates,
    include_root=False,
    use_cache=True,
    validate=True,
)
```

Reset:

```python
model.reset(
    *predicates,
    include_root=True,
    use_cache=True,
    descendants=False,
)
```

All model methods are in-place mutators and return `None`. For symmetry with
`Model.update(...)`, `Hyperparameters.extend(...)` and
`Hyperparameters.delete(...)` should provide the schema-only implementations.
`Model.extend(...)` and `Model.delete(...)` should delegate to those methods
before rebuilding modules. `Model.reset(...)` is runtime-only and should not
mutate `Hyperparameters`.

### Extend Arguments

`extend` positional arguments are parsed as:

1. Zero or more selection predicates, using the same predicate objects accepted
   by `model.select(...)`.
2. One or more extension fields, each a tensorfield request or nested `Array`.

The first schema field marks the boundary between predicates and extension
fields. After that boundary, every remaining positional argument must also be a
schema field.

```python
model.extend(
    j2v.where("address") == "record/transactions",
    j2v.Number("risk_score"),
    j2v.Category("device_id", max_vocab_size=4096),
)
```

### Delete Arguments

`delete` positional arguments are all selection predicates. The selected nodes
are removed permanently.

```python
model.delete(j2v.where("address") == "record/transactions/risk_score")

model.delete(
    (j2v.where("type") == "number")
    & j2v.where("address").matches("^record/transactions/")
)
```

### Reset Arguments

`reset` positional arguments are all selection predicates. The selected runtime
node modules are reinitialized from their existing hyperparameters.

```python
model.reset(j2v.where("address") == "record/transactions/risk_score")
model.reset(j2v.where("type") == "number")
model.reset()
```

With no predicates, `reset()` resets every runtime node, including the root
array. By default, selecting an array resets only that array node's encoder. Set
`descendants=True` to reset the selected array and its full subtree.

## Selection Semantics

`Model.select(...)` and `Hyperparameters.select(...)` remain inspect-only and
both return `list[Node]`. Mutation is performed through `update(...)`,
`extend(...)`, `delete(...)`, and `reset(...)`.

### Boolean Attribute Predicates

Selecting boolean attributes should not require explicit equality checks. A
bare `where(...)` attribute should be accepted anywhere a predicate is accepted
and should match nodes where the attribute value is `True`.

```python
model.select(j2v.where("active"))
model.select(j2v.where("embed"))
model.update(j2v.where("active"), p_mask=0.10)
model.delete(j2v.where("active") & (j2v.where("type") == "number"))
```

Missing attributes should not match. For example, arrays do not define
`active`, so `j2v.where("active")` selects active leaf nodes only. Explicit
comparisons remain valid when users need exact values:

```python
model.select(j2v.where("active") == False)
model.select(j2v.where("embed") == True)
```

Negation should use `~`, not Python's `not`. Python `not` eagerly evaluates to
a `bool`, so `model.select(not j2v.where("active"))` cannot produce a
predicate.

```python
model.select(~j2v.where("active"))
```

`~where("flag")` should mean "not truthy" and should match nodes where the
attribute is `False`, `None`, or missing. Use an explicit comparison when the
attribute must exist and be exactly `False`:

```python
model.select(j2v.where("active") == False)
```

Implementation requirements:

- Make `NodeAttribute` a valid predicate by teaching `_normalize_predicate(...)`
  to convert `where("flag")` into `where("flag") == True`.
- Add `NodeAttribute.__invert__` so `~j2v.where("active")` returns a
  `NodePredicate` equivalent to `not bool(where("active").get(node))`.

## Temporary Overrides

Add a model-level context manager for temporary schema changes:

```python
with model.override(j2v.where("address") == leaf.address, active=False):
    trainer.test(model, datamodule=datamodule)
```

This replaces the current lower-level pattern:

```python
with model.hyperparameters.override(j2v.where("address") == leaf.address, active=False):
    ...
```

`model.override(...)` should be a thin runtime wrapper around
`Hyperparameters.override(...)`, but it must keep the live module graph in sync:

1. Enter by applying the hyperparameter override.
2. Call `model._rebuild()` so inactive/deactivated nodes, targets, embeds, and
   changed parameters are reflected in `model.nodes` and `example_input_array`.
3. Yield the underlying `MutationResult`.
4. On exit, let `Hyperparameters.override(...)` restore the schema.
5. Call `model._rebuild()` again in `finally`, including when the context body
   raises.

The context manager should return no chainable model object. It should be used
for reversible experiments such as ablation and evaluation-time masking, not
for permanent schema evolution. Permanent deletion remains `model.delete(...)`.

Example:

```python
for leaf in model.select(j2v.where("type") == "number"):
    with model.override(j2v.where("address") == leaf.address, active=False):
        trainer.test(model, datamodule=datamodule)
```

Acceptance tests:

- `model.override(..., active=False)` removes the selected active leaf from
  `active_requests`, `model.nodes` runtime usage, and example inputs inside the
  context.
- The schema and modules are restored after the context exits.
- The schema and modules are restored when the context body raises.
- The yielded value is the apply-side `MutationResult`.
- `model.hyperparameters.override(...)` remains available as a schema-only
  primitive, but docs should prefer `model.override(...)` for runtime use.

### Extend Selection

Extension must use `select(...)` internally. The implementation selects matching
nodes, filters them to `type == "array"`, and requires exactly one array:

```python
selected = self.select(*predicates, include_root=include_root, use_cache=use_cache)
arrays = [node for node in selected if node.type == "array"]
```

If no predicates are supplied, `select()` is still used. This means
`model.extend(j2v.Number("score"))` works only when the model contains exactly
one array. A nested model must disambiguate the parent array.

Errors:

- no extension fields: `ValueError("extend requires at least one field")`
- zero matching arrays: `ValueError("extend requires exactly one array node, found 0")`
- multiple matching arrays: include the count and matched addresses
- non-field argument after the first field: `TypeError("schema fields must follow all predicates")`

### Delete Selection

Deletion must use `select(...)` internally and should default to
`include_root=False`. Deleting the root array is not supported.

The implementation accepts one or more selected nodes. If an array is selected,
the full subtree under that array is deleted. If both an array and one of its
descendants are selected, the descendant is deduplicated because deleting the
ancestor already deletes it.

Errors:

- no predicates: `ValueError("delete requires at least one predicate")`
- zero matching nodes: `ValueError("delete matched no nodes")`
- root selected: `ValueError("delete cannot remove the root array")`

### Reset Selection

Reset must use `select(...)` internally. It accepts zero or more selected nodes:

- no predicates means reset every node in `model.nodes`
- `include_root=True` means the root array can be reset
- `descendants=False` means array selections reset only the array node module
- `descendants=True` expands selected arrays to include descendant arrays and
  requests

If a selected array and one of its descendants are both selected while
`descendants=True`, the descendant is deduplicated because resetting the
ancestor subtree already resets it.

Errors:

- zero matching nodes when predicates are supplied:
  `ValueError("reset matched no nodes")`
- selected node address is not present in `model.nodes`:
  `ValueError("reset selected a node without runtime state: <address>")`

## Node Normalization

Extension fields should be normalized with the same rules as
`Model.from_schema(...)`:

- raw `Leaf` instances become concrete tensorfield request models
- names are sanitized with `sanitize_node_name(...)`
- inferred queries use `_schema_query(...)`
- explicit `query=...` values are preserved
- nested `Array` nodes are normalized recursively

The inferred query path is based on the selected parent array. The root array is
not included in request queries.

Examples:

```python
model.extend(j2v.Number("merchant_risk"))
# address: record/merchant_risk
# query:   [*].merchant_risk

model.extend(
    j2v.where("address") == "record/transactions",
    j2v.Number("risk_score"),
)
# address: record/transactions/risk_score
# query:   [*].transactions[*].risk_score

model.extend(
    j2v.where("address") == "record/customers",
    j2v.Array(j2v.Number("amount"), name="orders", max_length=32),
)
# address: record/customers/orders/amount
# query:   [*].customers[*].orders[*].amount
```

Implementation helper:

```python
def _array_path_for_child_queries(parent: Array) -> tuple[str, ...]:
    return tuple(node.name for node in parent.path[2:] if node.type == "array")
```

`parent.path[2:]` drops the `Hyperparameters` node and root array. For the root
array it returns `()`, matching the existing top-level query convention.

## Validation And Atomicity

### Extend Validation

Extension must validate the complete candidate child list before mutating the
live tree.

Validation must reject:

- duplicate child names under the selected parent after name sanitization
- invalid tensorfield request parameters
- invalid array parameters
- invalid inferred or explicit JMESPath queries

The live model tree must remain unchanged if any candidate fails validation.

Recommended flow:

1. Parse predicates and field args.
2. Select exactly one parent `Array`.
3. Normalize every new field with the selected parent query path.
4. Validate an `Array` copy containing existing children plus new children.
5. Append normalized children to `parent.fields`.
6. Bind each new child's `parent` to the selected array.
7. Clear tree caches and refresh selection caches.
8. Record an extend mutation.

### Delete Validation

Deletion must validate the resulting tree before mutating the live tree.
Deleting a node is not the same as setting `active=False`:

- `active=False` applies only to leaf nodes and keeps the node in the schema so
  it can be reactivated later.
- `delete(...)` removes the node from its parent array. Deleted nodes disappear
  from `arrays`, `requests`, `active_requests`, `shapes`, `target`, `embed`,
  plots, data-module tensorization, and `model.nodes` after rebuild.
- A deleted address may be reused later by `extend(...)`; a deactivated address
  remains occupied by the inactive node.

Validation must reject:

- deleting the root array
- deleting all request leaves from the schema
- leaving any non-root array with no descendant request leaves
- resulting duplicate child names, if deletion and future validation helpers are
  shared with extension

The live model tree must remain unchanged if any candidate fails validation.

Recommended flow:

1. Select matching nodes with `include_root=False` by default.
2. Reject an empty selection.
3. Reject the root array if it is selected explicitly.
4. Deduplicate selected descendants whose ancestor is already selected.
5. Build and validate a candidate tree with those nodes removed.
6. Remove each selected top-level node from its parent array's `fields`.
7. Detach each removed node from the tree.
8. Clear tree caches and refresh selection caches.
9. Record a delete mutation.

### Reset Semantics

Reset does not validate or mutate the schema tree. It keeps every selected
node's hyperparameters unchanged and replaces only runtime state.

For each selected address, reset should:

- create a fresh `NodeModule` from the current `Hyperparameters`, address, and
  `batch_size`
- replace `model.nodes[address]` with that fresh module
- discard all previous parameters, buffers, online vocabulary state, counters,
  normalization buffers, decoder state, and other module-local state owned by
  that runtime node
- refresh `example_input_array` because tensorfield interprocess contexts and
  active request maps may have changed indirectly through runtime replacement

Resetting a request resets both its embedder and decoder. Resetting an array
resets its encoder. Resetting an array with `descendants=False` does not reset
its child requests or child arrays. Resetting with `descendants=True` resets the
selected subtree.

External optimizer state is not schema hyperparameter state, but it can still
reference old parameter objects. The implementation should either clear/update
optimizer state for reset parameters when the active optimizer is accessible,
or mark optimizers stale and raise a clear error before the next training loop
unless optimizers are recreated. Docs should recommend optimizer factories over
long-lived optimizer instances when using reset.

## Runtime Rebuild

`Model.extend(...)` should call `self.hyperparameters.extend(...)`, then
`self._rebuild()`. `Model.delete(...)` should call
`self.hyperparameters.delete(...)`, then `self._rebuild()`.

Existing compatible weights should be restored through the current
`_rebuild()` behavior. Newly added node modules are initialized from scratch.
Existing parent array modules may be reinitialized when their parameter shapes
change because the child set changed.

`Model.reset(...)` should not call `_rebuild()` for the entire model unless it
is resetting every node. For targeted resets, replacing selected `NodeModule`
instances avoids disturbing compatible state in unrelated nodes.

## Data Module Refresh

Data modules created with `PolarsDataModule.from_model(...)` or
`StreamingDataModule.from_model(...)` should not need to be recreated after a
schema mutation or reset. To make that true, they must observe both the model's
live `Hyperparameters` object and the model's current interprocess encoding
context when constructing future dataloaders. A data module that stores only the
one-time `model.interprocess_encoding_context` dictionary can become stale after
`extend(...)`, `delete(...)`, `update(...)`, `override(...)`, or `reset(...)`
because those operations may rebuild embedders and replace vocabulary/counter
context objects.

Existing dataloader iterators and worker processes cannot be reliably refreshed
in place. A mutation should be treated as a boundary:

- safe: mutate the model, then call `datamodule.train_dataloader()` /
  `val_dataloader()` / `test_dataloader()` / `predict_dataloader()` again
- unsafe: mutate the model while a dataloader iterator is active
- unsafe: rely on already-started persistent dataloader workers to notice the
  mutation
- unsafe: reuse encoded batches created before the mutation

Implementation should prefer automatic freshness for future dataloader calls:

- keep data modules bound to the live `Hyperparameters` object rather than a
  serialized snapshot
- keep data modules bound to a model or context provider so each new dataloader
  receives the current `model.interprocess_encoding_context`
- add a mutation revision counter that changes for schema mutations and runtime
  resets
- have dataloader datasets capture the revision at iterator start
- raise a clear error if the revision changes during iteration
- document that persistent worker dataloaders must be recreated after mutation

Optional: model-created data modules may register weak references with the
model so mutations can notify them to clear any local dataloader cache. This
should be best-effort only; it cannot update active iterators or worker-local
dataset copies.

## Mutation Lock

Schema and runtime-state mutations must only run while the model is outside
active training, validation, testing, prediction, and inference loops. Mutating
the schema or resetting runtime nodes while the model or dataloader is reading
the old state can desynchronize `hyperparameters`, `model.nodes`, encoded
batches, callbacks, and worker-local dataset copies.

Add a small model-level mutation guard and a Lightning callback that owns loop
locks:

```python
model.update(...)
model.extend(...)
model.delete(...)
model.reset(...)
with model.override(...):
    ...
```

Each mutator should call a private guard before touching `Hyperparameters` or
runtime modules:

```python
def _assert_mutation_allowed(self, operation: str) -> None:
    if self._mutation_locks:
        reasons = ", ".join(sorted(self._mutation_locks))
        raise RuntimeError(f"cannot {operation} model while active: {reasons}")
```

Use a counted lock rather than a single boolean because validation can run
inside fitting and inference can be nested through serving helpers:

```python
@contextmanager
def _mutation_lock(self, reason: str):
    self._mutation_locks[reason] += 1
    try:
        yield
    finally:
        self._mutation_locks[reason] -= 1
        if self._mutation_locks[reason] == 0:
            del self._mutation_locks[reason]
```

Loop-level coverage should live in a `MutationLockCallback`, returned by
`Model.configure_callbacks()`:

```python
class MutationLockCallback(Callback):
    def _on_loop_start(self, trainer, pl_module, strata: Strata):
        pl_module._enter_mutation_lock(strata)

    def _on_loop_end(self, trainer, pl_module, strata: Strata):
        pl_module._exit_mutation_lock(strata)

    on_train_start = partialmethod(_on_loop_start, strata=Strata.train)
    on_train_end = partialmethod(_on_loop_end, strata=Strata.train)
```

The callback should cover:

- train start/end: lock `Strata.train`
- validation start/end: lock `Strata.validate`
- test start/end: lock `Strata.test`
- predict start/end: lock `Strata.predict`
- exception cleanup: release loop locks owned by the callback

The model should still own non-Lightning locks:

- `evaluate(...)`, `predict(...)`, and `embed(...)`: lock `inference` for the
  full encode-forward-write call
- `forward(...)`: use a short `forward` lock as a last line of defense for
  direct calls such as deployment request handling

The loop-level callback is more important than a per-batch lock. A per-batch
lock would still allow mutation between batches while callbacks, dataloaders,
and trainer state are active.

Deployment updates queued with `Deployment.update(...)` are allowed during
`API.setup(...)` before serving begins. Runtime request handling should be
protected by the model's `forward` lock, and serving docs should treat schema
mutation after server startup as unsupported.

`model.override(...)` and `model.reset(...)` must obey the same lock. They are
intended for explicit setup/evaluation blocks outside a trainer loop. Users
should enter an override or apply a reset and then start `trainer.test(...)`,
not attempt to call them from a callback inside an already-running loop.

Mutation lock errors should be actionable:

- name the requested operation
- list active lock reasons
- suggest moving the mutation before `trainer.fit(...)`,
  `trainer.validate(...)`, `trainer.test(...)`, `trainer.predict(...)`, or
  outside request handling
- for ablation, suggest wrapping the whole trainer call in `model.override(...)`

## Mutation History

Mutation bookkeeping must support extension, deletion, and reset operations:

```python
MutationChange.action: Literal["update", "restore", "extend", "delete", "reset"]
MutationResult.action: Literal["update", "restore", "extend", "delete", "reset"]
```

An extend result should have:

- `matched=1`
- `updated=<number of top-level children appended>`
- `skipped=0`
- one `MutationChange` per appended top-level child

Each extend change should use the appended node address and enough serialized
node data to audit what was added:

```python
MutationChange(
    node="record/transactions/risk_score",
    field="node",
    old=None,
    new=child.model_dump(mode="python", round_trip=True),
    action="extend",
)
```

A delete result should have:

- `matched=<number of nodes selected before ancestor deduplication>`
- `updated=<number of top-level nodes removed after ancestor deduplication>`
- `skipped=<number of selected descendants skipped because an ancestor was already removed>`
- one `MutationChange` per removed top-level node

Each delete change should use the removed node address and enough serialized
node data to audit what was removed. For arrays, `old` should include the full
subtree.

```python
MutationChange(
    node="record/transactions/risk_score",
    field="node",
    old=child.model_dump(mode="python", round_trip=True),
    new=None,
    action="delete",
)
```

A reset result should have:

- `matched=<number of nodes selected before descendant deduplication>`
- `updated=<number of runtime node modules reinitialized>`
- `skipped=<number of selected descendants skipped because an ancestor subtree was already reset>`
- one `MutationChange` per reset runtime node

Reset changes should not store tensor values. They should record enough
metadata to audit what was reset without serializing weights or sensitive
runtime state:

```python
MutationChange(
    node="record/transactions/risk_score",
    field="runtime_state",
    old={"state_keys": [...], "parameter_count": 12345},
    new={"reinitialized": True},
    action="reset",
)
```

Reset can reuse the `MutationResult` / `MutationChange` shape for consistency,
but it is a model-runtime mutation. Recording reset history must not alter
schema hyperparameters.

## Expected UX Issues

These API changes make schema and runtime mutation more powerful, but several
user-facing edges need explicit handling.

- `extend` may not be an obvious verb for single-field addition. Examples and
  error messages should consistently say "extend the schema under an array" and
  show both one-field and multi-field usage.
- `extend` positional parsing is unusual because predicates and fields share
  one `*args` channel. The first schema field must clearly end predicate
  parsing, and errors should point at the first argument that appears out of
  order.
- `extend` requires exactly one selected array. Users will often select a leaf
  by address or accidentally match multiple arrays. Errors should include
  matched addresses and suggest narrowing with `where("address") == ...`.
- Inferred queries for nested extension are convenient but hidden. `extend`
  should make it easy to inspect the added node afterward, and docs should show
  the inferred address and query.
- `Model.select(...)` and `Hyperparameters.select(...)` should have matching
  list-returning behavior. Any internal cache object should stay private so
  users do not have to learn two selection return types.
- Bare boolean predicates are ergonomic, but `~where("active")` matching missing
  attributes may surprise users. Docs should contrast `~where("active")` with
  `where("active") == False`.
- Python `not where("active")` cannot work. If possible, `NodeAttribute` should
  raise a helpful `TypeError` from `__bool__` telling users to use `~`.
- `delete` is destructive and permanent. The name should stay strong, errors
  should avoid euphemisms, and docs should compare it directly against
  reversible `active=False`.
- Deleting arrays deletes whole subtrees. Deletion summaries should include the
  descendant count or removed addresses so users do not miss the blast radius.
- Ancestor/descendant delete deduplication can make `matched`, `updated`, and
  `skipped` feel non-obvious. Mutation history should preserve enough detail to
  explain why a selected descendant was skipped.
- Deleting all leaves or leaving an empty non-root array fails by design. The
  error should identify the array or schema state that would become invalid.
- Deleted addresses can be re-added later, but prior weights for deleted modules
  are not retained unless still present in a compatible state dict. Docs should
  avoid implying delete is undoable.
- `_rebuild()` preserves only compatible state entries. Parent contexts may be
  reinitialized when child structure changes. This can affect training
  continuity even when most weights survive.
- Data modules built from a model should stay fresh through their live
  hyperparameter reference, but active dataloader iterators, persistent workers,
  and encoded batches cannot be refreshed in place. User-facing docs should say
  to request new dataloaders after schema or runtime-state mutations.
- `model.override(...)` is temporary but still rebuilds modules on entry and
  exit. Users should not use it inside hot prediction paths unless they accept
  rebuild overhead.
- `model.override(...)` and `model.delete(...)` are easy to confuse during
  ablation workflows. Ablation docs should use `override(..., active=False)`;
  schema-pruning docs should use `delete(...)`.
- Context manager restoration must happen after exceptions. If restoration
  fails, the error should make clear that the model may be left in a partially
  overridden state.
- Mutating a model while a trainer, deployment, or concurrent request is using
  it is unsafe. Docs should frame these APIs as setup/evaluation-time schema
  operations, not request-time operations.
- Deployment has a builder-style `Deployment.update(...)` API that remains
  chainable, while `Model.update(...)` is in-place and returns `None`. Docs and
  type hints should keep that distinction visible.
- `reset` is destructive to learned runtime state while leaving schema
  hyperparameters unchanged. Docs should present it as "reinitialize this node"
  rather than "undo schema changes."
- Resetting categorical or set nodes also resets online vocabularies and other
  interprocess encoding context. Existing dataloaders can silently encode with
  stale vocabularies unless future dataloader construction pulls fresh context
  from the model.
- Resetting a node after optimizers have been created can leave optimizer state
  pointing at old parameter objects. The implementation should either update
  optimizer state or require optimizer/trainer recreation before the next fit.
- Reset subtree behavior is easy to misunderstand. Docs should be explicit that
  `descendants=False` resets only selected node modules, while
  `descendants=True` resets the selected branch.
- Mutation history grows as users experiment. Long-running notebooks may need a
  way to inspect or summarize recent mutations without dumping large deleted
  subtrees.
- `delete` mutation history stores serialized removed subtrees. This is useful
  for auditability but may expose sensitive schema metadata or become large for
  wide branches.
- `reset` mutation history must not store raw tensors, learned vocab contents,
  or optimizer state unless the user explicitly asks for a checkpoint.

## Non-Goals

This spec does not add moving, renaming, schema diffing, rollback for added or
deleted nodes, reset undo, optimizer checkpoint surgery, or automatic data
preprocessing. New fields read from their configured or inferred queries;
missing source values continue to be handled by each tensorfield's existing
missing/null behavior.

## Acceptance Tests

### Extend

- Extending a flat model with one root request adds the request, infers
  `[*].name`, rebuilds modules, and records an extend mutation.
- Extending a selected nested array with multiple requests appends both fields,
  infers nested queries, updates `requests`, `active_requests`, `shapes`, and
  `model.nodes`.
- Extending an existing array with a nested `Array` recursively infers child
  queries and addresses.
- Calling `model.extend(...)` with no predicates on a model with multiple arrays
  raises the exact-single-array error.
- A predicate that selects only leaf nodes raises the zero-array error.
- Duplicate child names fail before mutating the live tree.
- Invalid request parameters fail before mutating the live tree.
- Selection caches populated before extend are refreshed after extend.
- Existing compatible module state survives `_rebuild()`.

### Delete

- Deleting one leaf removes it from its parent `fields`, `requests`,
  `active_requests`, `shapes`, `model.nodes`, and generated example input.
- Deleting an array removes the entire subtree from `arrays`, `requests`,
  `active_requests`, `shapes`, `depthwise`, `target`, `embed`, and `model.nodes`.
- Deleting an array and a descendant in the same call records one top-level
  delete change and counts the descendant as skipped.
- Calling `model.delete(...)` with no predicates raises the no-predicate error.
- Calling `model.delete(...)` with a predicate that matches no nodes raises the
  no-match error.
- Attempting to delete the root array raises the root-delete error.
- Attempting to delete every request leaf fails before mutating the live tree.
- Attempting to leave a non-root array with no descendant request leaves fails
  before mutating the live tree.
- `active=False` remains reversible through `model.update(..., active=True)`;
  `delete(...)` is permanent and the deleted address is absent until re-added
  with `extend(...)`.
- Selection caches populated before delete are refreshed after delete.
- Existing compatible module state survives `_rebuild()`.

### Reset

- Resetting one leaf replaces that leaf's runtime `NodeModule` while preserving
  its schema hyperparameters, address, query, target/embed flags, and shape.
- Resetting one request resets both its embedder and decoder state.
- Resetting one array resets that array encoder without resetting child nodes
  when `descendants=False`.
- Resetting one array with `descendants=True` resets the array and all
  descendant array/request runtime modules.
- Calling `model.reset()` with no predicates resets every runtime node while
  preserving the full schema.
- Calling `model.reset(...)` with a predicate that matches no nodes raises the
  no-match error.
- Resetting does not alter `arrays`, `requests`, `active_requests`, `shapes`,
  `target`, `embed`, or `depthwise`.
- Resetting refreshes `example_input_array` and future interprocess encoding
  context.
- Reset mutation history records reset metadata without serializing tensor
  values or learned runtime state.
- Existing compatible module state outside selected nodes survives targeted
  reset.

### Mutation Lock

- `model.update(...)`, `model.extend(...)`, `model.delete(...)`,
  `model.reset(...)`, and `model.override(...)` raise while the model is locked
  for training, validation, testing, prediction, inference, or forward
  execution.
- The lock is counted: nested locks with different reasons preserve all active
  reasons and only clear after the matching exits.
- Lightning loop hooks lock the full loop, not only individual batch steps.
- `model.evaluate(...)`, `model.predict(...)`, and `model.embed(...)` lock for
  the full encode-forward-write call.
- A direct `model(inputs)` call locks during `forward(...)`.
- The lock is released when a loop, forward call, or inference helper raises.
- `Deployment.update(...)` queued mutations still apply during `API.setup(...)`
  before request serving begins.
- Error messages include the requested mutation operation and active lock
  reasons.

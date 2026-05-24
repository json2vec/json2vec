# Mutation History Removal Spec

## Summary

Remove persistent mutation history, revision counters, and detailed changelog
objects from the dynamic node mutation API.

The current implementation records every schema/runtime mutation as
`MutationResult` plus nested `MutationChange` records, stores those results on
`Hyperparameters`, exposes `last_mutation`, `mutation_history`, and
`mutation_revision`, and uses those records in a few tests, docs, and deployment
bookkeeping paths.

The proposed direction is to make mutations simple in-place operations:

```python
model.update(j2v.where("active"), p_mask=0.10)
model.extend(j2v.Number("risk_score"))
model.delete(j2v.where("name") == "legacy_field")
model.reset(j2v.where("address") == "record/amount")

with model.override(j2v.where("address") == "record/amount", active=False):
    ...
```

Users who want to inspect what will be affected should call `select(...)`
before mutating.

## Motivation

The mutation history API adds significant internal weight:

- Every mutation needs to allocate operation IDs, count matched/updated/skipped
  nodes, serialize before/after values, and store changelog records.
- Runtime-only mutations like `reset(...)` need fake schema mutation records even
  though no hyperparameter changed.
- Temporary mutations like `override(...)` need paired update/restore records,
  which makes a context manager feel audit-log driven instead of operational.
- Public return/context types make future API cleanup harder because callers can
  reasonably depend on exact changelog shape.
- The persistent `_mutation_history` list grows with every mutation and is not
  clearly useful after the immediate debugging moment has passed.

The value is mostly notebook/debug convenience. That can be replaced by the
simpler and more explicit workflow:

```python
nodes = model.select(j2v.where("type") == "number")
model.update(j2v.where("type") == "number", p_mask=0.10)
```

## Proposed Public API

Remove these public exports and properties:

- `MutationChange`
- `MutationResult`
- `Hyperparameters.last_mutation`
- `Hyperparameters.mutation_history`
- `Hyperparameters.mutation_revision`

Keep mutation methods in-place:

- `Model.update(...) -> None`
- `Model.extend(...) -> None`
- `Model.delete(...) -> None`
- `Model.reset(...) -> None`
- `Hyperparameters.update(...) -> None`
- `Hyperparameters.extend(...) -> None`
- `Hyperparameters.delete(...) -> None`

Change override context managers to yield nothing:

```python
with model.override(j2v.where("active"), active=False):
    ...

with hyperparameters.override(j2v.where("active"), active=False):
    ...
```

Do not support `as result` for `override(...)` after the removal.

## Internal Design

Mutation methods should perform validation, apply the mutation, clear structural
caches, refresh selection caches, and return.

Remove:

- `MutationAction`
- `_mutation_history`
- `_mutation_revision`
- `_record_mutation(...)`
- construction of `MutationChange`
- construction of `MutationResult`
- operation IDs and parent operation IDs

Keep:

- validation errors for invalid selectors, invalid values, duplicate extension
  fields, and unsafe deletes
- cache invalidation after permanent schema mutations
- model rebuilds after runtime-affecting mutations
- mutation lock checks on `Model`
- `override(...)` snapshot/restore behavior

Selection cache refresh does not need `mutation_revision`; mutation methods
already call cache clearing/refreshing at the point where the tree changes.

## Implementation Impact

### `src/json2vec/structs/experiment.py`

Delete mutation result models and stored history state.

Update `Hyperparameters.update(...)` to:

- normalize `target=...`
- select matching nodes
- validate applicable values
- set attributes
- clear tree caches
- refresh selection cache
- return `None`

Update `Hyperparameters.extend(...)` and `delete(...)` to stop building change
records and return `None`.

Update `Hyperparameters.override(...)` to:

- take a snapshot of selected values
- call `update(...)`
- yield without a value
- restore the snapshot in `finally`
- clear/refresh caches after restore

### `src/json2vec/architecture/root.py`

Remove `MutationChange` / `MutationResult` imports.

Update `Model.reset(...)` to reinitialize matched modules without building or
recording a runtime mutation result.

Update `Model.override(...)` to return `Iterator[None]` and yield without a
result.

No change is needed for the mutation lock callback or rebuild behavior.

### `src/json2vec/inference/deployment.py`

Remove `applied_update_operations` if its only source is
`hyperparameters.last_mutation`.

If deployment still needs observability, record a simple count or store the
caller-provided `update_operations` configuration as submitted. Do not synthesize
changelogs from applied mutations.

### `src/json2vec/__init__.py`

Remove public exports:

- `MutationChange`
- `MutationResult`

### Docs And Tests

Update docs that currently recommend `last_mutation`:

- `docs/reference/api.md`
- `docs/guides/model-update.ipynb`

Replace examples like:

```python
pprint(model.hyperparameters.last_mutation)
```

with:

```python
nodes = model.select(j2v.where("type") == "number")
```

Update tests that assert mutation history:

- check direct schema/runtime effects instead of `last_mutation.action`
- check `select(...)` results before/after mutation when selector behavior is
  the behavior under test
- check `override(...)` restoration directly, not the restore changelog

## User Experience Impact

Benefits:

- mutation methods are easier to explain: they mutate in place or raise
- `override(...)` becomes a normal context manager
- no user-facing operation IDs or changelog record shapes to document
- less memory growth from repeated notebook mutations
- less coupling between runtime-only operations and schema audit structures

Costs:

- users lose a built-in "what just changed" object
- broad selectors require explicit preflight inspection with `select(...)`
- notebooks cannot print `last_mutation` as a quick sanity check
- deployment no longer reports applied update changelogs

The replacement UX should be documented as:

```python
nodes = model.select(j2v.where("type") == "number")
assert [str(node.address) for node in nodes] == ["record/amount"]
model.update(j2v.where("type") == "number", p_mask=0.10)
```

## Migration

This is a breaking API change and should be released as direct removal. Do not
keep compatibility shims.

Required migration for the next release:

1. Remove the APIs before the dynamic mutation surface is treated as stable.
2. Update notebooks and docs in the same PR.
3. Add release notes that say mutation methods no longer record or return
   changelog objects; use `select(...)` before mutation for inspection.

## Recommendation

Remove history, revision, and changelog objects.

The implementation complexity and public API weight are larger than the value of
retaining a persistent audit log. The core mutation API becomes clearer if
selection is used for inspection and mutation methods are strictly in-place
operations.

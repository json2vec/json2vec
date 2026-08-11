"""Compile tree structure and exact References into a deterministic runtime plan."""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from anytree import PreOrderIter

from relflow.structs.pooling import Mean
from relflow.structs.reference import AxisResize, Reference
from relflow.structs.structure import Branch
from relflow.structs.tree import Address, Leaf

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema

ReferenceId = tuple[Address, int]
View = Literal["memory", "summary"]


@dataclass(frozen=True)
class InputPlan:
    """One address-keyed activation gathered for a Branch or decoder."""

    view: View
    address: Address
    reference_id: ReferenceId | None = None


@dataclass(frozen=True)
class ReductionAxis:
    """One source-memory dimension resized by a compiled Reference."""

    address: Address
    position: int
    extent: int
    size: int


@dataclass(frozen=True)
class ReferencePlan:
    """One exact Reference declaration owned by a consuming Branch."""

    id: ReferenceId
    consumer: Address
    source: Address
    declaration: Reference
    axes: tuple[ReductionAxis, ...]


@dataclass(frozen=True)
class CompiledExecutionGraph:
    """Immutable, tensor-free execution metadata derived from a bound Schema."""

    encoder_order: tuple[Address, ...]
    active_branches: frozenset[Address]
    branch_inputs: dict[Address, tuple[InputPlan, ...]]
    branch_references: dict[Address, tuple[ReferenceId, ...]]
    references: dict[ReferenceId, ReferencePlan]
    reference_source: dict[ReferenceId, Address]
    reference_consumers: dict[Address, tuple[ReferenceId, ...]]
    grafted_sources: frozenset[Address]
    grafted_by: dict[Address, tuple[ReferenceId, ...]]
    decoder_contexts: dict[Address, tuple[InputPlan, ...]]


def _declarations(branch: Branch) -> tuple[Reference, ...]:
    reference = branch.reference
    if isinstance(reference, Reference):
        return (reference,)
    return reference


def _resolve_reduction_axes(
    *,
    schema: Schema,
    consumer: Address,
    index: int,
    source: Address,
) -> tuple[ReductionAxis, ...]:
    declaration = _declarations(schema.branches[consumer])[index]
    source_node = (schema.branches | schema.requests)[source]
    signature = [ancestor for ancestor in source_node.ancestors if isinstance(ancestor, Branch)]
    eligible = [axis for axis in signature if axis is not schema.fields]
    resolved: list[ReductionAxis] = []
    seen: set[Address] = set()

    for resize in declaration.reduce.axes if declaration.reduce is not None else ():
        raw = resize.address
        if type(raw) is Address:
            matches = [axis for axis in eligible if axis.address == raw]
        else:
            matches = [axis for axis in eligible if axis.name == str(raw)]
        if len(matches) != 1:
            raise ValueError(
                f"branch '{consumer}' reference[{index}] Reduce axis '{raw}' "
                f"resolved to {len(matches)} dimensions on source '{source}'"
            )

        axis = matches[0]
        address = Address(str(axis.address))
        if address in seen:
            raise ValueError(f"branch '{consumer}' reference[{index}] repeats Reduce axis '{address}'")
        seen.add(address)
        if resize.size > axis.length or axis.length % resize.size:
            raise ValueError(
                f"branch '{consumer}' reference[{index}] cannot resize axis '{address}' "
                f"from {axis.length} to {resize.size}"
            )

        resolved.append(
            ReductionAxis(
                address=address,
                position=next(position for position, candidate in enumerate(signature) if candidate is axis),
                extent=axis.length,
                size=resize.size,
            )
        )

    resolved.sort(key=lambda item: item.position)

    consumer_signature = [ancestor for ancestor in schema.branches[consumer].ancestors if isinstance(ancestor, Branch)]
    source_addresses = tuple(axis.address for axis in signature)
    consumer_addresses = tuple(axis.address for axis in consumer_signature)
    if source_addresses[: len(consumer_addresses)] != consumer_addresses:
        raise ValueError(
            f"branch '{consumer}' reference[{index}] source '{source}' has coordinate "
            f"signature {source_addresses}, which is not prefixed by consumer coordinates "
            f"{consumer_addresses}"
        )

    effective_sizes = {axis.address: axis.length for axis in signature}
    effective_sizes.update({axis.address: axis.size for axis in resolved})
    for axis in consumer_signature:
        if effective_sizes[axis.address] != axis.length:
            raise ValueError(
                f"branch '{consumer}' reference[{index}] reduces required outer axis "
                f"'{axis.address}' from {axis.length} to {effective_sizes[axis.address]}"
            )

    return tuple(resolved)


def compile_execution_graph(schema: Schema) -> CompiledExecutionGraph:
    """Resolve exact References, validate dependencies, and compile gather order."""

    branches = schema.branches
    requests = schema.requests
    nodes = {**branches, **requests}
    preorder_nodes = tuple(PreOrderIter(schema.fields))
    preorder = {node.address: index for index, node in enumerate(preorder_nodes)}

    references: dict[ReferenceId, ReferencePlan] = {}
    branch_references: dict[Address, tuple[ReferenceId, ...]] = {}
    reference_consumers: dict[Address, list[ReferenceId]] = defaultdict(list)
    grafted_by: dict[Address, list[ReferenceId]] = defaultdict(list)

    for consumer, branch in branches.items():
        ids: list[ReferenceId] = []
        for index, declaration in enumerate(_declarations(branch)):
            reference_id = (consumer, index)
            source = Address(str(declaration.address))
            if not source:
                raise ValueError(f"branch '{consumer}' reference[{index}] requires a non-empty address")
            if source not in nodes:
                raise ValueError(f"branch '{consumer}' reference[{index}] points to missing source '{source}'")
            if source == consumer:
                raise ValueError(f"branch '{consumer}' reference[{index}] cannot reference itself")

            source_node = nodes[source]
            if isinstance(source_node, Leaf) and (not source_node.active or source_node.target):
                role = "inactive" if not source_node.active else "target"
                raise ValueError(f"branch '{consumer}' reference[{index}] cannot use {role} leaf '{source}'")
            if declaration.graft and source == schema.fields.address:
                raise ValueError(f"branch '{consumer}' reference[{index}] cannot graft root '{source}'")

            plan = ReferencePlan(
                id=reference_id,
                consumer=consumer,
                source=source,
                declaration=declaration,
                axes=(),
            )
            references[reference_id] = plan
            ids.append(reference_id)
            reference_consumers[source].append(reference_id)
            if declaration.graft:
                grafted_by[source].append(reference_id)

        branch_references[consumer] = tuple(ids)

    references = {
        reference_id: ReferencePlan(
            id=plan.id,
            consumer=plan.consumer,
            source=plan.source,
            declaration=plan.declaration,
            axes=_resolve_reduction_axes(
                schema=schema,
                consumer=plan.consumer,
                index=reference_id[1],
                source=plan.source,
            ),
        )
        for reference_id, plan in references.items()
    }

    # Short constructor names are only an unbound convenience. Once the full
    # tree exists, keep stable full addresses in schema/checkpoint state.
    for reference_id, plan in tuple(references.items()):
        reduction = plan.declaration.reduce
        if reduction is None:
            continue
        canonical_axes = tuple(AxisResize(axis.address, axis.size) for axis in plan.axes)
        canonical_reduction = reduction.model_copy(update={"axes": canonical_axes})
        canonical_declaration = plan.declaration.model_copy(update={"reduce": canonical_reduction})
        references[reference_id] = replace(plan, declaration=canonical_declaration)

    for consumer, reference_ids in branch_references.items():
        declarations = tuple(references[reference_id].declaration for reference_id in reference_ids)
        branches[consumer].reference = (
            () if not declarations else declarations[0] if len(declarations) == 1 else declarations
        )

    grafted_sources = frozenset(grafted_by)

    for plan in references.values():
        source_node = nodes[plan.source]
        reducer = plan.declaration.reduce.reducer if plan.declaration.reduce is not None else None
        identity_reduction = False
        if plan.declaration.reduce is not None and isinstance(reducer, (Mean, str)):
            identity_reduction = all(axis.size == axis.extent for axis in plan.axes)
        if (
            isinstance(source_node, Leaf)
            and isinstance(source_node.parent, Branch)
            and source_node.parent.address == plan.consumer
            and plan.source not in grafted_sources
            and (plan.declaration.reduce is None or identity_reduction)
        ):
            raise ValueError(
                f"branch '{plan.consumer}' reference[{plan.id[1]}] duplicates attached "
                f"direct leaf '{plan.source}'; use graft=True or a non-identity Reduce"
            )

    # Edges point from a dependency to the Branch that consumes it. Duplicate
    # structural/reference reasons share one indegree edge while the Reference
    # declarations above retain complete provenance.
    outgoing: dict[Address, set[Address]] = {address: set() for address in branches}
    dependencies: dict[Address, set[Address]] = {address: set() for address in branches}

    for source, branch in branches.items():
        parent = branch.parent
        if source in grafted_sources or not isinstance(parent, Branch):
            continue
        destination = parent.address
        dependencies[destination].add(source)
        outgoing[source].add(destination)

    for plan in references.values():
        if plan.source not in branches:
            continue
        dependencies[plan.consumer].add(plan.source)
        outgoing[plan.source].add(plan.consumer)

    indegree = {address: len(sources) for address, sources in dependencies.items()}
    ready: list[tuple[int, int, Address]] = []
    for address, degree in indegree.items():
        if degree == 0:
            node = branches[address]
            heapq.heappush(ready, (-node.depth, preorder[address], address))

    encoder_order: list[Address] = []
    while ready:
        _, _, source = heapq.heappop(ready)
        encoder_order.append(source)
        for destination in sorted(outgoing[source], key=lambda item: preorder[item]):
            indegree[destination] -= 1
            if indegree[destination] == 0:
                node = branches[destination]
                heapq.heappush(ready, (-node.depth, preorder[destination], destination))

    if len(encoder_order) != len(branches):
        remaining = {address for address, degree in indegree.items() if degree > 0}
        state: dict[Address, int] = {}
        stack: list[Address] = []
        positions: dict[Address, int] = {}
        cycle: tuple[Address, ...] | None = None

        def visit(source: Address) -> None:
            nonlocal cycle
            state[source] = 1
            positions[source] = len(stack)
            stack.append(source)
            for destination in sorted(
                outgoing[source] & remaining,
                key=lambda item: (-branches[item].depth, preorder[item]),
            ):
                if cycle is not None:
                    return
                if state.get(destination, 0) == 0:
                    visit(destination)
                elif state[destination] == 1:
                    cycle = tuple((*stack[positions[destination] :], destination))
                    return
            stack.pop()
            positions.pop(source)
            state[source] = 2

        for source in sorted(
            remaining,
            key=lambda item: (-branches[item].depth, preorder[item]),
        ):
            if state.get(source, 0) == 0:
                visit(source)
            if cycle is not None:
                break

        if cycle is None:  # pragma: no cover - Kahn guarantees a cyclic remainder
            cycle = tuple(sorted(remaining, key=lambda item: preorder[item]))
        rendered = " -> ".join(map(str, cycle))
        raise ValueError(f"reference cycle detected: {rendered}")

    active: set[Address] = set()
    branch_inputs: dict[Address, tuple[InputPlan, ...]] = {}

    for address in encoder_order:
        branch = branches[address]
        inputs: list[InputPlan] = []

        for child in branch.fields:
            child_address = child.address
            if isinstance(child, Leaf) and child_address not in grafted_sources and child.active and not child.target:
                inputs.append(InputPlan(view="summary", address=child_address))

        for child in branch.fields:
            child_address = child.address
            if isinstance(child, Branch) and child_address not in grafted_sources and child_address in active:
                inputs.append(InputPlan(view="summary", address=child_address))

        for reference_id in branch_references[address]:
            plan = references[reference_id]
            source_node = nodes[plan.source]
            if isinstance(source_node, Branch) and plan.source not in active:
                raise ValueError(
                    f"branch '{address}' reference[{reference_id[1]}] source '{plan.source}' does not produce memory"
                )
            inputs.append(
                InputPlan(
                    view="memory",
                    address=plan.source,
                    reference_id=reference_id,
                )
            )

        if inputs:
            active.add(address)
            branch_inputs[address] = tuple(inputs)
        else:
            branch_inputs[address] = ()

    decoder_contexts: dict[Address, tuple[InputPlan, ...]] = {}
    for address, request in schema.active_requests.items():
        contexts: list[InputPlan] = []
        for heritage_address in request.heritage:
            if heritage_address in branches and heritage_address in active:
                contexts.append(InputPlan(view="summary", address=heritage_address))
            elif heritage_address == address and request.active and not request.target:
                contexts.append(InputPlan(view="summary", address=address))

        for ancestor in request.ancestors:
            if not isinstance(ancestor, Branch):
                continue
            ancestor_address = ancestor.address
            if ancestor_address in active and branch_references.get(ancestor_address):
                contexts.append(InputPlan(view="memory", address=ancestor_address))

        if not contexts:
            raise ValueError(f"request '{address}' has no available decoder context")
        decoder_contexts[address] = tuple(contexts)

    return CompiledExecutionGraph(
        encoder_order=tuple(encoder_order),
        active_branches=frozenset(active),
        branch_inputs=branch_inputs,
        branch_references=branch_references,
        references=references,
        reference_source={reference_id: plan.source for reference_id, plan in references.items()},
        reference_consumers={address: tuple(ids) for address, ids in reference_consumers.items()},
        grafted_sources=grafted_sources,
        grafted_by={address: tuple(ids) for address, ids in grafted_by.items()},
        decoder_contexts=decoder_contexts,
    )

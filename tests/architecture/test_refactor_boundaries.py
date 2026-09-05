from __future__ import annotations

import ast
from logging.handlers import BufferingHandler
from pathlib import Path

import torch
from lightning.pytorch.utilities.model_summary.model_summary import summarize

import relflow as rf
from relflow.architecture.checkpoint import CheckpointState
from relflow.architecture.graph import ModelGraph
from relflow.architecture.mutations import SchemaEditor
from relflow.logging import logger
from relflow.logging.config import CONTEXT
from relflow.structs import experiment, selectors

FRAMEWORK_PROTOCOL_METHODS = frozenset(
    {
        "_load_from_state_dict",
        "_mime_",
        "_missing_",
        "_repr_html_",
        "_repr_mimebundle_",
        "_save_to_state_dict",
    }
)


def _model() -> rf.Model:
    return rf.Model(
        rf.Number(name="amount"),
        rf.Category(name="label", mask=True, size=4),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )


def test_model_uses_mutation_facade() -> None:
    model = _model()

    assert isinstance(model.schema, rf.Schema)
    assert isinstance(model._schema_editor, SchemaEditor)
    assert model.schema.select(rf.where("name") == "amount") == model.select(rf.where("name") == "amount")


def test_model_mutations_emit_structured_logs() -> None:
    model = _model()
    records = BufferingHandler(capacity=100)
    logger.logger.addHandler(records)

    try:
        model.update(rf.where("name") == "amount", weight=2.0)
        model.update(rf.where("name") == "amount", benchmark="schema_api", allow_extra=True)
        model.update(rf.where("name") == "amount", mask=True)
        model.extend(rf.where("name") == "record", rf.Category(name="extra", size=4))
        model.reset(rf.where("name") == "amount")
        with model.override(rf.where("name") == "amount", weight=3.0):
            pass
        model.delete(rf.where("name") == "extra")
    finally:
        logger.logger.removeHandler(records)

    events = [getattr(record, CONTEXT) for record in records.buffer]
    messages = [record.getMessage() for record in records.buffer]

    mutation_events = [event for event in events if event.get("component") == "schema_mutation"]
    actions = {event["action"] for event in mutation_events}

    assert {"update", "extend", "reset", "override", "override_restore", "delete"} <= actions
    assert any(
        event.get("attribute") == "weight" and event.get("definition_attribute") is True for event in mutation_events
    )
    assert any(
        event.get("attribute") == "benchmark" and event.get("definition_attribute") is False
        for event in mutation_events
    )
    assert any(
        event.get("attribute") == "mask" and event.get("definition_attribute") is True for event in mutation_events
    )
    assert any(
        event.get("address") == "record/amount"
        and event.get("attribute") == "weight"
        and event.get("previous_value") == "1.0"
        and event.get("value") == "2.0"
        and event.get("change") == "weight: 1.0 -> 2.0"
        for event in mutation_events
    )
    assert any("mutated record/amount: weight 1.0 -> 2.0" in message for message in messages)
    assert any("extended schema node record/extra under record" in message for message in messages)
    assert any("deleted schema node record/extra descendants=0" in message for message in messages)


def test_model_mutation_logs_include_previous_and_current_address_for_renames() -> None:
    model = _model()
    records = BufferingHandler(capacity=10)
    logger.logger.addHandler(records)

    try:
        model.update(rf.where("name") == "amount", name="total")
    finally:
        logger.logger.removeHandler(records)

    events = [getattr(record, CONTEXT) for record in records.buffer]
    messages = [record.getMessage() for record in records.buffer]

    mutation_events = [event for event in events if event.get("component") == "schema_mutation"]

    assert any(
        event.get("action") == "update"
        and event.get("attribute") == "name"
        and event.get("previous_address") == "record/amount"
        and event.get("address") == "record/total"
        and event.get("previous_node_name") == "amount"
        and event.get("node_name") == "total"
        and event.get("change") == "name: 'amount' -> 'total'"
        for event in mutation_events
    )
    assert any("mutated record/amount -> record/total: name 'amount' -> 'total'" in message for message in messages)


def test_model_graph_rebuild_preserves_compatible_state() -> None:
    model = _model()
    name, before = next(iter(model.state_dict().items()))

    ModelGraph.rebuild(model)

    assert torch.equal(model.state_dict()[name], before)


def test_model_summary_uses_forward_kwargs_example_input() -> None:
    model = _model()

    summary = summarize(model, max_depth=1)

    assert summary.total_parameters > 0


def test_checkpoint_state_round_trip(tmp_path) -> None:
    model = _model()
    path = tmp_path / "model.ckpt"

    CheckpointState.save(model, path)
    restored = CheckpointState.load(rf.Model, path)

    assert restored.schema.model_dump(mode="python") == model.schema.model_dump(mode="python")
    assert restored.batch_size == model.batch_size


def test_experiment_reexports_selector_api() -> None:
    assert experiment.where is selectors.where
    assert experiment.NodePredicate is selectors.NodePredicate
    assert experiment.NodeAttribute is selectors.NodeAttribute


def test_source_functions_and_classes_do_not_use_private_names() -> None:
    source = Path(__file__).parents[2] / "src" / "relflow"
    violations: list[str] = []

    for path in sorted(source.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
                continue
            if not node.name.startswith("_") or node.name.startswith("__"):
                continue
            if node.name in FRAMEWORK_PROTOCOL_METHODS:
                continue
            violations.append(f"{path.relative_to(source)}:{node.lineno}: {node.name}")

    assert not violations, "private function/class names:\n" + "\n".join(violations)

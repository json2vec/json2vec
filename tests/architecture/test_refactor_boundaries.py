from __future__ import annotations

import torch
from lightning.pytorch.utilities.model_summary.model_summary import summarize
from loguru import logger

import json2vec as j2v
from json2vec.architecture.checkpoint import CheckpointState
from json2vec.architecture.graph import ModelGraph
from json2vec.architecture.mutations import SchemaEditor
from json2vec.architecture.runtime import EvaluationResult
from json2vec.structs import experiment, selectors


def _model() -> j2v.Model:
    return j2v.Model.from_schema(
        j2v.Number(name="amount"),
        j2v.Category(name="label", target=True, max_vocab_size=4),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )


def test_model_uses_mutation_facade() -> None:
    model = _model()

    assert isinstance(model.schema, SchemaEditor)
    assert model.schema.select(j2v.where("name") == "amount") == model.select(j2v.where("name") == "amount")


def test_model_mutations_emit_structured_logs() -> None:
    model = _model()
    events: list[dict[str, object]] = []
    sink_id = logger.add(lambda message: events.append(dict(message.record["extra"])))

    try:
        model.update(j2v.where("name") == "amount", weight=2.0)
        model.update(j2v.where("name") == "amount", benchmark="schema_api", allow_extra=True)
        model.update(j2v.where("name") == "amount", target=True)
        model.extend(j2v.where("name") == "record", j2v.Category(name="extra", max_vocab_size=4))
        model.reset(j2v.where("name") == "amount")
        with model.override(j2v.where("name") == "amount", weight=3.0):
            pass
        model.delete(j2v.where("name") == "extra")
    finally:
        logger.remove(sink_id)

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
        event.get("attribute") == "target" and event.get("definition_attribute") is False for event in mutation_events
    )


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
    restored = CheckpointState.load(j2v.Model, path)

    assert restored.hyperparameters.model_dump(mode="python") == model.hyperparameters.model_dump(mode="python")
    assert restored.batch_size == model.batch_size


def test_runtime_evaluation_result_keeps_tuple_shape() -> None:
    result = EvaluationResult(predictions={"record/label": {"value": [1]}}, embeddings={})

    assert result.as_tuple() == (result.predictions, result.embeddings)


def test_experiment_reexports_selector_api() -> None:
    assert experiment.where is selectors.where
    assert experiment.NodePredicate is selectors.NodePredicate
    assert experiment.NodeAttribute is selectors.NodeAttribute

from __future__ import annotations

import torch

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


def test_model_graph_rebuild_preserves_compatible_state() -> None:
    model = _model()
    name, before = next(iter(model.state_dict().items()))

    ModelGraph.rebuild(model)

    assert torch.equal(model.state_dict()[name], before)


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

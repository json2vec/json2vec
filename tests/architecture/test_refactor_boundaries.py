from __future__ import annotations

import torch
from lightning.pytorch.utilities.model_summary.model_summary import summarize

import relflow as rf
from relflow.architecture.checkpoint import CheckpointState
from relflow.architecture.graph import ModelGraph
from relflow.architecture.mutations import SchemaEditor
from relflow.rich import console, set_verbose
from relflow.structs import experiment, selectors


def _model() -> rf.Model:
    return rf.Model(
        rf.Number(name="amount"),
        rf.Category(name="label", target=True, size=4),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )


def test_model_uses_mutation_facade() -> None:
    model = _model()

    assert isinstance(model.schema, rf.Schema)
    assert isinstance(model._schema_editor, SchemaEditor)
    assert model.schema.select(rf.where("name") == "amount") == model.select(rf.where("name") == "amount")


def test_model_mutations_do_not_emit_console_chatter() -> None:
    model = _model()
    with console.capture() as captured:
        model.update(rf.where("name") == "amount", weight=2.0)
        model.update(rf.where("name") == "amount", benchmark="schema_api", allow_extra=True)
        model.update(rf.where("name") == "amount", target=True)
        model.extend(rf.where("name") == "record", rf.Category(name="extra", size=4))
        model.reset(rf.where("name") == "amount")
        with model.override(rf.where("name") == "amount", weight=3.0):
            pass
        model.delete(rf.where("name") == "extra")
    assert captured.get() == ""
    assert model.schema.requests[rf.Address("record/amount")].weight == 2.0
    assert model.schema.requests[rf.Address("record/amount")].target is True


def test_model_mutation_rename_updates_address_without_console_chatter() -> None:
    model = _model()
    with console.capture() as captured:
        model.update(rf.where("name") == "amount", name="total")
    assert captured.get() == ""
    assert rf.Address("record/amount") not in model.schema.requests
    assert rf.Address("record/total") in model.schema.requests


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
    with console.capture() as default_output:
        restored = CheckpointState.load(rf.Model, path)

    assert default_output.get() == ""
    assert restored.schema.model_dump(mode="python") == model.schema.model_dump(mode="python")
    assert restored.batch_size == model.batch_size

    try:
        set_verbose(True)
        with console.capture() as verbose_output:
            CheckpointState.load(rf.Model, path)
        assert "loading model checkpoint" in verbose_output.get()
        assert path.name in verbose_output.get()
    finally:
        set_verbose(False)


def test_experiment_reexports_selector_api() -> None:
    assert experiment.where is selectors.where
    assert experiment.NodePredicate is selectors.NodePredicate
    assert experiment.NodeAttribute is selectors.NodeAttribute

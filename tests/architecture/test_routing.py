import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.structs.enums import Strata, TensorKey
from relflow.tensorfields.base import TensorFieldBase, TensorInput


class Field(TensorFieldBase):
    @classmethod
    def new(cls, input, target, present, trainable, inferred, address, schema, strata, context):
        raise NotImplementedError


def test_plain_input_node_allocates_only_an_embedder():
    model = rf.Model(value=rf.Number, d_model=8, n_layers=1, n_heads=2)

    node = model.nodes["record/value"]

    assert hasattr(node, "embedder")
    assert not hasattr(node, "decoder")


def test_tensorfield_take_gathers_every_content_prefix():
    field = Field()
    field.state = torch.arange(6).reshape(2, 3)
    field.content = TensorDict(
        {
            "scalar": torch.arange(6).reshape(2, 3),
            "vector": torch.arange(24).reshape(2, 3, 4),
        },
        batch_size=[2, 3],
    )

    compact = field.take(torch.tensor([1, 4]))

    assert isinstance(compact, TensorInput)
    assert compact.batch_size == torch.Size([2])
    assert not hasattr(compact, "targets")
    assert not hasattr(compact, "present")
    assert not hasattr(compact, "trainable")
    assert not hasattr(compact, "inferred")
    assert torch.equal(compact.state, torch.tensor([1, 4]))
    assert torch.equal(compact.content["scalar"], torch.tensor([1, 4]))
    assert torch.equal(
        compact.content["vector"],
        field.content["vector"].reshape(6, 4).index_select(0, torch.tensor([1, 4])),
    )


def test_tensorfield_take_rejects_an_invalid_content_prefix():
    field = Field()
    field.state = torch.arange(6).reshape(2, 3)
    field.content = torch.arange(6).reshape(3, 2)

    with pytest.raises(ValueError, match="content must start with state shape"):
        field.take(torch.tensor([1, 4]))


def test_embedder_compacts_present_coordinates_and_restores_fixed_geometry():
    model = rf.Model(
        value=rf.Number(mask=rf.Mask(query="skip", skip=True, dropout=False)),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    inputs = model.encode(
        pa.table({"value": [1.0, 2.0, 3.0], "skip": [False, True, False]}),
        strata=Strata.train,
    )
    field = inputs["record/value"]
    embedder = model.nodes["record/value"].embedder
    seen = []
    handle = embedder.register_forward_pre_hook(lambda module, args: seen.append(args[0]))

    try:
        parcel = embedder.embed(field)
    finally:
        handle.remove()

    assert len(seen) == 1
    assert isinstance(seen[0], TensorInput)
    assert seen[0].state.shape == (2,)
    assert parcel.payload.shape == (3, 1, 8)
    assert torch.equal(parcel.present, field.present)
    assert torch.equal(parcel.payload[1], torch.zeros_like(parcel.payload[1]))


def test_embedder_is_not_called_for_an_all_skipped_field():
    model = rf.Model(
        value=rf.Number(mask=rf.Mask(skip=True, dropout=False)),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    field = model.encode(pa.table({"value": [1.0, 2.0]}), strata=Strata.train)["record/value"]
    embedder = model.nodes["record/value"].embedder
    calls = 0

    def count(module, args):
        nonlocal calls
        calls += 1

    handle = embedder.register_forward_pre_hook(count)
    try:
        parcel = embedder.embed(field)
    finally:
        handle.remove()

    assert calls == 0
    assert parcel.payload.shape == (2, 1, 8)
    assert not parcel.present.any()
    assert torch.equal(parcel.payload, torch.zeros_like(parcel.payload))


def test_runtime_keeps_mixed_skip_rows_out_of_branch_context():
    model = rf.Model(
        value=rf.Number(mask=rf.Mask(query="skip", skip=True, dropout=False)),
        d_model=8,
        n_layers=1,
        n_heads=2,
        embed=True,
    )
    inputs = model.encode(
        pa.table({"value": [1.0, 2.0], "skip": [False, True]}),
        strata=Strata.train,
    )

    predictions = model(inputs, strata=Strata.train)
    root = next(prediction for prediction in predictions if prediction.address == "record")
    embedding = root.payload[TensorKey.embedding]

    assert torch.isfinite(embedding).all()
    assert not torch.equal(embedding[0], torch.zeros_like(embedding[0]))
    assert torch.equal(embedding[1], torch.zeros_like(embedding[1]))


def test_peer_objective_selection_runs_the_same_decoder_and_anchors_local_backward(monkeypatch):
    model = rf.Model(
        context=rf.Number,
        first=rf.Boolean(mask=rf.Mask(query="first_selected", dropout=False, reconstruct=True)),
        second=rf.Boolean(mask=rf.Mask(query="second_selected", dropout=False, reconstruct=True)),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    inputs = model.encode(
        pa.table(
            {
                "context": [1.0, 2.0],
                "first": [True, False],
                "second": [False, True],
                "first_selected": [False, False],
                "second_selected": [False, False],
            }
        ),
        strata=Strata.train,
    )
    calls = []

    def select_peer(local):
        calls.append(local.clone())
        return torch.tensor([0, 1], dtype=local.dtype, device=local.device)

    monkeypatch.setattr("relflow.architecture.runtime.all_reduce_max", select_peer)

    predictions = model(inputs, strata=Strata.train)
    output = model.training_step(inputs, batch_idx=0)

    assert len(calls) == 2
    assert all(torch.equal(call, torch.zeros(2, dtype=torch.uint8)) for call in calls)
    assert [prediction.address for prediction in predictions] == ["record/second"]
    assert output["loss"].requires_grad
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert all(parameter.grad is not None for parameter in model.nodes["record/second"].decoder.parameters())
    assert all(parameter.grad is None for parameter in model.nodes["record/first"].decoder.parameters())


@pytest.mark.parametrize(
    ("strata", "step"),
    [
        (Strata.validate, "validation_step"),
        (Strata.test, "test_step"),
    ],
)
def test_evaluation_loss_does_not_require_a_gradient_anchor(strata, step):
    model = rf.Model(
        context=rf.Number,
        label=rf.Boolean(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    inputs = model.encode(
        pa.table(
            {
                "context": [1.0, 2.0],
                "label": [True, False],
            }
        ),
        strata=strata,
    )

    with torch.inference_mode():
        output = getattr(model, step)(inputs, batch_idx=0)

    assert torch.isfinite(output["loss"])
    assert output["loss"].item() > 0.0
    assert not output["loss"].requires_grad


def test_evaluation_returns_zero_when_only_a_peer_rank_has_targets(monkeypatch):
    model = rf.Model(
        context=rf.Number,
        label=rf.Boolean(mask=rf.Mask(query="selected", dropout=False, reconstruct=True)),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    inputs = model.encode(
        pa.table(
            {
                "context": [1.0, 2.0],
                "label": [True, False],
                "selected": [False, False],
            }
        ),
        strata=Strata.validate,
    )
    monkeypatch.setattr(
        "relflow.architecture.runtime.all_reduce_max",
        lambda local: torch.ones_like(local),
    )

    with torch.inference_mode():
        output = model.validation_step(inputs, batch_idx=0)

    assert output["loss"].item() == 0.0
    assert not output["loss"].requires_grad


def test_globally_empty_objective_skips_the_training_update(monkeypatch):
    model = rf.Model(
        context=rf.Number,
        label=rf.Boolean(mask=rf.Mask(query="selected", dropout=False, reconstruct=True)),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    inputs = model.encode(
        pa.table(
            {
                "context": [1.0, 2.0],
                "label": [True, False],
                "selected": [False, False],
            }
        ),
        strata=Strata.train,
    )
    monkeypatch.setattr("relflow.architecture.runtime.all_reduce_max", lambda local: local)

    output = model.training_step(inputs, batch_idx=0)

    assert output is None

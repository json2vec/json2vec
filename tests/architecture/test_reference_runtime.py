from __future__ import annotations

import pytest
import torch

import relflow as rf
from relflow.architecture.encoder import BranchEncoding
from relflow.structs.enums import Strata, TensorKey
from relflow.structs.packages import Parcel

RECORDS = [
    {
        "a": [{"x": 1.0}, {"x": 2.0}],
        "b": [{"y": 3.0}, {"y": 4.0}],
    },
    {
        "a": [{"x": 5.0}],
        "b": [{"y": 6.0}],
    },
]


def _branch(*, length: int = 2, **fields) -> rf.Branch:
    return rf.Branch(
        length=length,
        attention="none",
        pooling=rf.Mean(),
        **fields,
    )


def _model(*, reference: rf.Reference | tuple[rf.Reference, ...], embed_b: bool = False) -> rf.Model:
    return rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
        attention="none",
        pooling=rf.Mean(),
        a=_branch(x=rf.Number),
        b=_branch(
            reference=reference,
            embed=embed_b,
            y=rf.Number(pooling=rf.Mean()),
        ),
    )


def test_sibling_branch_reference_routes_full_memory_and_enriches_decoder_context():
    model = _model(reference=rf.Reference("record/a"))
    encodings: dict[str, BranchEncoding] = {}
    decoder_context: list[torch.Size] = []

    handles = [
        model.nodes["record/a"].encoder.register_forward_hook(
            lambda _module, _args, output: encodings.__setitem__("a", output)
        ),
        model.nodes["record/b"].encoder.register_forward_hook(
            lambda _module, _args, output: encodings.__setitem__("b", output)
        ),
        model.nodes["record/b/y"].decoder.register_forward_pre_hook(
            lambda _module, args, _kwargs: decoder_context.extend(parcel.payload.shape for parcel in args[0]),
            with_kwargs=True,
        ),
    ]

    try:
        inputs = model.encode(RECORDS, strata=Strata.train, mask=False)
        field = inputs["record/b/y"]
        field.hide(torch.ones_like(field.trainable))
        predictions = model(inputs, strata=Strata.train)
    finally:
        for handle in handles:
            handle.remove()

    plans = model.execution_graph.branch_inputs["record/b"]
    assert [(plan.view, str(plan.address), plan.reference_id) for plan in plans] == [
        ("summary", "record/b/y", None),
        ("memory", "record/a", ("record/b", 0)),
    ]
    assert encodings["a"].memory.shape == (2, 1, 2, 8)
    assert encodings["a"].summary.shape == (2, 1, 8)
    assert encodings["b"].memory.shape == (2, 1, 4, 8)
    assert encodings["b"].summary.shape == (2, 1, 8)
    assert decoder_context == [
        torch.Size((2, 1, 8)),
        torch.Size((2, 1, 8)),
        torch.Size((2, 1, 2, 8)),
        torch.Size((2, 1, 4, 8)),
    ]
    assert [str(prediction.address) for prediction in predictions] == ["record/b/y"]


def test_graft_cuts_structural_route_but_reference_path_remains_differentiable():
    torch.manual_seed(7)
    model = _model(
        reference=rf.Reference("record/a", graft=True),
        embed_b=True,
    )

    assert model.execution_graph.grafted_sources == {"record/a"}
    assert [str(plan.address) for plan in model.execution_graph.branch_inputs["record"]] == ["record/b"]
    assert "record/a" in model.execution_graph.active_branches

    inputs = model.encode(RECORDS, strata=Strata.train, mask=False)
    predictions = model(inputs, strata=Strata.train)
    embedding = next(
        prediction.payload[TensorKey.embedding] for prediction in predictions if prediction.address == "record/b"
    )
    weights = torch.arange(1, 9, device=embedding.device, dtype=embedding.dtype)
    embedding.mul(weights).sum().backward()

    source_parameters = tuple(model.nodes["record/a/x"].embedder.parameters())
    assert source_parameters
    assert all(parameter.grad is not None for parameter in source_parameters)
    assert sum(parameter.grad.abs().sum() for parameter in source_parameters).item() > 0


def test_grafted_leaf_keeps_reference_route_while_dangling_ancestors_are_inactive():
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        attention="none",
        pooling=rf.Mean(),
        staging=_branch(source=_branch(value=rf.Number)),
        sink=_branch(
            reference=rf.Reference("record/staging/source/value", graft=True),
            output=rf.Number,
        ),
    )

    assert model.execution_graph.grafted_sources == {"record/staging/source/value"}
    assert "record/staging/source" not in model.execution_graph.active_branches
    assert "record/staging" not in model.execution_graph.active_branches
    assert {"record/sink", "record"} <= model.execution_graph.active_branches
    assert model.execution_graph.branch_inputs["record/sink"][-1].address == ("record/staging/source/value")
    assert [plan.address for plan in model.execution_graph.branch_inputs["record"]] == ["record/sink"]


def test_any_grafting_reference_suppresses_the_single_structural_route():
    model = _model(
        reference=(
            rf.Reference("record/a"),
            rf.Reference("record/a", graft=True),
        )
    )

    assert model.execution_graph.grafted_sources == {"record/a"}
    assert [plan.address for plan in model.execution_graph.branch_inputs["record"]] == ["record/b"]
    references = [plan for plan in model.execution_graph.branch_inputs["record/b"] if plan.reference_id is not None]
    assert [plan.address for plan in references] == ["record/a", "record/a"]


def test_failed_source_rename_and_delete_leave_model_routing_intact():
    model = _model(reference=rf.Reference("record/a"))
    schema_before = model.schema.model_dump(mode="python", round_trip=True)
    graph_before = model.execution_graph
    node_keys_before = tuple(model.nodes)
    branch_keys_before = tuple(model.schema.branches)

    with pytest.raises(ValueError, match="missing source 'record/a'"):
        model.update("record/a", name="renamed")
    assert model.schema.model_dump(mode="python", round_trip=True) == schema_before
    assert model.execution_graph is graph_before
    assert tuple(model.nodes) == node_keys_before
    assert tuple(model.schema.branches) == branch_keys_before

    with pytest.raises(ValueError, match="missing source 'record/a'"):
        model.delete("record/a")
    assert model.schema.model_dump(mode="python", round_trip=True) == schema_before
    assert model.execution_graph is graph_before
    assert tuple(model.nodes) == node_keys_before
    assert tuple(model.schema.branches) == branch_keys_before


def test_delete_can_leave_a_reference_only_branch_live():
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        attention="none",
        pooling=rf.Mean(),
        a=_branch(value=rf.Number),
        b=_branch(reference=rf.Reference("record/a"), temporary=rf.Number),
    )

    model.delete("record/b/temporary")

    assert model.schema.branches["record/b"].fields == []
    assert "record/b" in model.execution_graph.active_branches
    assert [plan.address for plan in model.execution_graph.branch_inputs["record/b"]] == ["record/a"]


def test_duplicate_attention_references_own_distinct_modules_and_reuse_source_memory():
    references = (
        rf.Reference(
            "record/a/x",
            reduce=rf.Reduce(rf.Attention(n_heads=2), a=True),
        ),
        rf.Reference(
            "record/a/x",
            reduce=rf.Reduce(rf.Attention(n_heads=2), a=True),
        ),
    )
    model = _model(reference=references, embed_b=True)
    reducers = model.nodes["record/b"].reference_reducers

    assert list(reducers) == ["0", "1"]
    assert reducers["0"] is not reducers["1"]
    assert next(reducers["0"].parameters()).data_ptr() != next(reducers["1"].parameters()).data_ptr()

    source_calls = 0
    branch_inputs: list[torch.Size] = []

    def count_source(_module, _args, _output: Parcel) -> None:
        nonlocal source_calls
        source_calls += 1

    handles = [
        model.nodes["record/a/x"].embedder.register_forward_hook(count_source),
        model.nodes["record/b"].encoder.register_forward_pre_hook(
            lambda _module, args: branch_inputs.extend(payload.shape for payload in args[0])
        ),
    ]
    try:
        inputs = model.encode(RECORDS, strata=Strata.train, mask=False)
        model(inputs, strata=Strata.train)
    finally:
        for handle in handles:
            handle.remove()

    assert source_calls == 1
    assert branch_inputs == [
        torch.Size((2, 1, 2, 8)),
        torch.Size((2, 1, 1, 8)),
        torch.Size((2, 1, 1, 8)),
    ]
    reference_plans = model.execution_graph.branch_inputs["record/b"][1:]
    assert [plan.reference_id for plan in reference_plans] == [
        ("record/b", 0),
        ("record/b", 1),
    ]


def test_reference_graph_and_indexed_reducers_round_trip_through_checkpoint(tmp_path):
    references = (
        rf.Reference(
            "record/a/x",
            reduce=rf.Reduce(rf.Attention(n_heads=2), a=True),
        ),
        rf.Reference(
            "record/a/x",
            graft=True,
            reduce=rf.Reduce(rf.Attention(n_heads=2), a=True),
        ),
    )
    model = _model(reference=references)
    path = tmp_path / "reference.ckpt"

    model.save(path)
    restored = rf.Model.load(path)

    assert restored.schema.branches["record/b"].reference == model.schema.branches["record/b"].reference
    assert restored.execution_graph.encoder_order == model.execution_graph.encoder_order
    assert restored.execution_graph.grafted_sources == model.execution_graph.grafted_sources
    assert tuple(restored.nodes["record/b"].reference_reducers) == ("0", "1")
    assert tuple(restored.state_dict()) == tuple(model.state_dict())
    for name, value in model.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)

    bound = restored.schema.branches["record/b"].reference
    assert isinstance(bound, tuple)
    assert all(reference.reduce.axes[0].address == "record/a" for reference in bound)
    assert all(isinstance(reference.reduce.axes[0].address, rf.Address) for reference in bound)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {
                "b": _branch(
                    reference=rf.Reference("record/missing"),
                    value=rf.Number,
                )
            },
            "points to missing source 'record/missing'",
        ),
        (
            {
                "a": rf.Number(target=True),
                "b": _branch(
                    reference=rf.Reference("record/a"),
                    value=rf.Number,
                ),
            },
            "cannot use target leaf 'record/a'",
        ),
        (
            {
                "a": _branch(
                    reference=rf.Reference("record/b"),
                    value=rf.Number,
                ),
                "b": _branch(
                    reference=rf.Reference("record/a"),
                    value=rf.Number,
                ),
            },
            "reference cycle detected",
        ),
        (
            {
                "x": _branch(
                    b=_branch(
                        reference=rf.Reference("record/y/value"),
                        value=rf.Number,
                    )
                ),
                "y": _branch(value=rf.Number),
            },
            "not prefixed by consumer coordinates",
        ),
        (
            {"only": rf.Number(target=True)},
            "has no available decoder context",
        ),
    ],
)
def test_reference_graph_rejects_invalid_sources_and_cycles(fields, message):
    with pytest.raises(ValueError, match=message):
        rf.Schema.from_tree(
            d_model=8,
            n_layers=1,
            n_heads=2,
            **fields,
        )


def test_mean_reduce_block_resizes_named_axis_before_reference_routing():
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
        attention="none",
        pooling=rf.Mean(),
        a=_branch(length=4, x=rf.Number),
        b=_branch(
            reference=rf.Reference(
                "record/a/x",
                reduce=rf.Reduce(a=2),
            ),
            y=rf.Number,
        ),
    )
    records = [
        {
            "a": [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}],
            "b": [{"y": 5.0}, {"y": 6.0}],
        },
        {
            "a": [{"x": 7.0}, {"x": 8.0}],
            "b": [{"y": 9.0}],
        },
    ]
    captured: dict[str, torch.Tensor | list[torch.Tensor]] = {}
    handles = [
        model.nodes["record/a/x"].embedder.register_forward_hook(
            lambda _module, _args, output: captured.__setitem__("source", output.payload)
        ),
        model.nodes["record/b"].encoder.register_forward_pre_hook(
            lambda _module, args: captured.__setitem__("inputs", args[0])
        ),
    ]
    try:
        inputs = model.encode(records, strata=Strata.train, mask=False)
        model(inputs, strata=Strata.train)
    finally:
        for handle in handles:
            handle.remove()

    source = captured["source"]
    routed = captured["inputs"][1]
    assert isinstance(source, torch.Tensor)
    assert isinstance(routed, torch.Tensor)
    expected = source.reshape(2, 1, 2, 2, 8).mean(dim=3)
    assert routed.shape == (2, 1, 2, 8)
    assert torch.equal(routed, expected)

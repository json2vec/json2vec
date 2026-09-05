from logging.handlers import BufferingHandler
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pydantic
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.helpers import Jitter
from relflow.logging import logger
from relflow.logging.config import CONTEXT
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.structs.tree import Mask
from relflow.tensorfields.base import TENSORFIELDS, TensorInput
from relflow.tensorfields.extensions.number import (
    Decoder,
    Embedder,
    GlobalOnlineNormalizer,
    TensorField,
    loss,
    moments,
    write,
)
from relflow.tensorfields.extensions.number import (
    output as output_type,
)
from tests.arrow import batch as arrow_batch
from tests.tensorfields.helpers import tensorize

ADDRESS = "root/items/amount"


def structure_payload(*, mask: bool | Mask = False, jitter: Jitter | dict[str, object] | None = None) -> dict:
    field: dict = {
        "name": "amount",
        "type": "number",
        "mask": mask,
    }
    if jitter is not None:
        field["jitter"] = jitter
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "dropout": 0.1,
            "fields": [
                {
                    "name": "items",
                    "type": "branch",
                    "length": 2,
                    "fields": [field],
                }
            ],
        },
    }


def tensorfield(rows: list[list[Any]], *, schema: Schema, strata: Strata) -> TensorField:
    batch = arrow_batch([{"items": [{"amount": value} for value in row]} for row in rows])
    projection = coalesce(batch, schema=schema, strata=strata)[ADDRESS]
    return tensorize(
        TensorField,
        projection,
        TENSORFIELDS["number"],
        address=ADDRESS,
        schema=schema,
        strata=strata,
    )


def test_number_request_hydrates_jitter_from_a_mapping():
    structure = Schema.model_validate(structure_payload(jitter={"add": 1.5, "multiply": 0.25, "normalize": False}))
    jitter = structure.requests[ADDRESS].jitter

    assert isinstance(jitter, Jitter)
    assert jitter == Jitter(add=1.5, multiply=0.25, normalize=False)


def test_number_jitter_round_trips_through_schema_serialization():
    structure = Schema.model_validate(structure_payload(jitter=Jitter(add=1.5, multiply=0.25, normalize=False)))

    restored = Schema.model_validate_json(structure.model_dump_json(round_trip=True))

    assert restored.requests[ADDRESS].jitter == Jitter(add=1.5, multiply=0.25, normalize=False)


def test_number_jitter_round_trips_through_model_checkpoint(tmp_path: Path):
    configured = Jitter(add=1.5, multiply=0.25, normalize=False)
    model = rf.Model(
        rf.Number("amount", jitter=configured),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    pathname = tmp_path / "jitter.ckpt"

    model.save(pathname)
    restored = rf.Model.load(pathname)

    request = restored.schema.requests[rf.Address("record", "amount")]
    embedder = restored.nodes[rf.Address("record", "amount")].embedder
    assert request.jitter == configured
    assert embedder.jitter == configured


@pytest.mark.parametrize("value", [None, 0.0, 0.2, 1, True])
def test_number_rejects_legacy_scalar_jitter(value: object):
    with pytest.raises(pydantic.ValidationError):
        rf.Number("amount", jitter=value)


def test_default_jitter_does_not_advance_rng_during_training():
    structure = Schema.model_validate(structure_payload())
    field = tensorfield(rows=[[2.0, 4.0]], schema=structure, strata=Strata.train)
    embedder = Embedder(schema=structure, address=ADDRESS)
    embedder.train()
    before = torch.random.get_rng_state().clone()

    embedder.embed(field)

    assert torch.equal(torch.random.get_rng_state(), before)


def test_number_jitters_only_finite_valued_coordinates(monkeypatch: pytest.MonkeyPatch):
    structure = Schema.model_validate(structure_payload(jitter=Jitter(add=0.5)))
    embedder = Embedder(schema=structure, address=ADDRESS)
    captured: list[torch.Tensor] = []

    def capture(
        self: Jitter,
        inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self == Jitter(add=0.5)
        captured.append(mask.clone())
        return inputs

    monkeypatch.setattr(Jitter, "apply", capture)
    inputs = TensorInput(
        state=torch.tensor([Tokens.valued, Tokens.null, Tokens.masked, Tokens.valued]),
        content=torch.tensor([2.0, 0.0, 0.0, float("inf")]),
        batch_size=[4],
    )

    embedder.train()
    embedder(inputs)

    assert len(captured) == 1
    assert captured[0].tolist() == [True, False, False, False]


def test_number_jitter_runs_only_in_training_mode(monkeypatch: pytest.MonkeyPatch):
    structure = Schema.model_validate(structure_payload(jitter=Jitter(add=0.5)))
    embedder = Embedder(schema=structure, address=ADDRESS)
    field = tensorfield(rows=[[2.0]], schema=structure, strata=Strata.train)
    calls: list[torch.Tensor] = []

    def capture(
        self: Jitter,
        inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self == Jitter(add=0.5)
        calls.append(inputs.clone())
        return inputs

    monkeypatch.setattr(Jitter, "apply", capture)

    embedder.train()
    embedder.embed(field)
    embedder.eval()
    embedder.embed(field)

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("normalize", "expected_perturb", "expected_clamp"),
    [
        (True, 2.0, 5.0),
        (False, 14.0, 3.5),
    ],
)
def test_number_jitter_respects_normalization_order(
    monkeypatch: pytest.MonkeyPatch,
    normalize: bool,
    expected_perturb: float,
    expected_clamp: float,
):
    structure = Schema.model_validate(structure_payload(jitter=Jitter(add=3.0, normalize=normalize)))
    embedder = Embedder(schema=structure, address=ADDRESS)
    embedder.normalizer.mean.fill_(10.0)
    embedder.normalizer.var.fill_(4.0)
    field = tensorfield(rows=[[14.0]], schema=structure, strata=Strata.train)
    observed: dict[str, torch.Tensor] = {}

    def add_three(
        self: Jitter,
        inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self == Jitter(add=3.0, normalize=normalize)
        observed["apply"] = inputs.clone()
        return inputs + mask.to(dtype=inputs.dtype).mul(3.0)

    def capture_clamp(self: Embedder, content: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        observed["clamp"] = content.clone()
        return content

    monkeypatch.setattr(Jitter, "apply", add_three)
    monkeypatch.setattr(Embedder, "clamp", capture_clamp)

    embedder.train()
    embedder.embed(field)

    assert observed["apply"].item() == pytest.approx(expected_perturb, abs=1e-4)
    assert observed["clamp"].item() == pytest.approx(expected_clamp, abs=1e-4)


def test_number_jitter_is_configuration_not_checkpoint_state():
    structure = Schema.model_validate(structure_payload(jitter=Jitter(add=1.5, multiply=0.25, normalize=False)))
    embedder = Embedder(schema=structure, address=ADDRESS)

    assert embedder.jitter == Jitter(add=1.5, multiply=0.25, normalize=False)
    assert not any("jitter" in name for name in embedder.state_dict())
    assert not any("jitter" in name for name, _ in embedder.named_buffers())


def test_number_jitter_mutation_rebuilds_runtime_configuration():
    model = rf.Model(
        rf.Number("amount", jitter=rf.Jitter(add=0.1)),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    address = rf.Address("record", "amount")

    model.update(
        lambda node: node.address == address,
        jitter=rf.Jitter(add=0.8, multiply=0.3, normalize=False),
    )

    request = model.schema.requests[address]
    embedder = model.nodes[address].embedder
    assert request.jitter == rf.Jitter(add=0.8, multiply=0.3, normalize=False)
    assert embedder.jitter == request.jitter


class TrackingModule:
    def __init__(self, schema: Schema, embedder: Embedder, decoder: Decoder):
        self.schema = schema
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder, decoder=decoder)}

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        return value


def test_number_loss_does_not_mutate_counter():
    structure = Schema.model_validate(structure_payload(mask=Mask(reconstruct=True)))
    schema = structure

    field = tensorfield(
        rows=[[1.0, None], [2.0]],
        schema=schema,
        strata=Strata.train,
    )
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = TrackingModule(schema=structure, embedder=embedder, decoder=decoder)

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(*field.content.shape, 1),
            },
            batch_size=field.batch_size,
        ),
    )

    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    expected_counts = torch.ones(len(Tokens), dtype=torch.int64)
    assert torch.equal(embedder.counter.counts, expected_counts)


def test_number_normalizer_ignores_nonfinite_values_when_updating():
    normalizer = GlobalOnlineNormalizer()
    normalizer.train()
    inputs = torch.tensor([1.0, float("inf"), float("-inf"), float("nan")])
    mask = torch.ones_like(inputs, dtype=torch.bool)

    output = normalizer(inputs=inputs, mask=mask)

    assert torch.isfinite(normalizer.mean).all()
    assert torch.isfinite(normalizer.var).all()
    assert normalizer.count.item() == 1
    assert torch.isfinite(output[0])
    assert torch.isinf(output[1])
    assert torch.isinf(output[2])
    assert torch.isnan(output[3])


def test_number_normalizer_learns_precomputed_finite_moments():
    normalizer = GlobalOnlineNormalizer()
    observation = moments(torch.tensor([1.0, 3.0, float("nan"), float("inf")]))

    normalizer.learn(observation)

    assert normalizer.count.item() == 2
    assert normalizer.mean.item() == pytest.approx(2.0)
    assert normalizer.var.item() == pytest.approx(1.0)


def test_number_embedder_clamps_unsafe_fourier_inputs_and_warns():
    structure = Schema.model_validate(structure_payload())
    embedder = Embedder(schema=structure, address=ADDRESS)
    bound = embedder.max_fourier_input.detach()
    content = torch.stack(
        [
            bound.mul(2),
            bound.mul(-3),
            torch.tensor(float("inf")),
            torch.tensor(float("nan")),
            bound.mul(4),
        ]
    )
    state = torch.tensor(
        [
            Tokens.valued,
            Tokens.valued,
            Tokens.valued,
            Tokens.valued,
            Tokens.padded,
        ],
        dtype=torch.int64,
    )
    records = BufferingHandler(capacity=10)
    records.setLevel("WARNING")
    logger.logger.addHandler(records)

    try:
        clamped = embedder.clamp(content=content, state=state)
    finally:
        logger.logger.removeHandler(records)

    events = [getattr(record, CONTEXT) for record in records.buffer]
    messages = [record.getMessage() for record in records.buffer]

    assert torch.isfinite(clamped).all()
    assert torch.allclose(clamped[:3], torch.stack([bound, -bound, bound]))
    assert clamped[3].item() == 0.0
    assert clamped[4].item() == bound.item()
    assert any("number Fourier inputs exceed safe range" in message for message in messages)
    assert any(
        event.get("component") == "tensorfield"
        and event.get("field_type") == "number"
        and event.get("address") == ADDRESS
        and event.get("count") == 5
        and event.get("valued_count") == 4
        and event.get("nonfinite_count") == 2
        for event in events
    )


def test_number_embedder_outputs_finite_payload_for_extreme_outliers():
    structure = Schema.model_validate(structure_payload())
    field = tensorfield(
        rows=[[1.0, 2.0]],
        schema=structure,
        strata=Strata.train,
    )
    field.content[0, 0, 0] = float("inf")

    embedder = Embedder(schema=structure, address=ADDRESS)
    embedder.train()
    parcel = embedder.embed(field)

    assert torch.isfinite(embedder.normalizer.mean).all()
    assert torch.isfinite(embedder.normalizer.var).all()
    assert torch.isfinite(parcel.payload).all()


def test_number_write_emits_flat_arrow_content():
    structure = Schema.model_validate(structure_payload())
    state_logits = torch.zeros(2, 1, len(Tokens))
    state_logits[0, 0, Tokens.valued.value] = 10.0
    state_logits[1, 0, Tokens.null.value] = 10.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: torch.tensor([[[1.5]], [[2.5]]]),
            },
            batch_size=[2],
        ),
    )

    module = SimpleNamespace(schema=structure)
    datatype = output_type(module, ADDRESS)
    output = write(module=module, prediction=prediction, datatype=datatype)

    assert output.type == datatype
    assert output.field(TensorKey.content.name).to_pylist() == [1.5, 2.5]

from datetime import datetime

import pyarrow as pa
import pydantic
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.helpers import Jitter
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.tensorfields.base import TensorInput
from relflow.tensorfields.extensions.dateparts import DatePart, Embedder

ADDRESS = rf.Address("record", "created")


def build(jitter: Jitter | dict[str, object] | None = None) -> rf.Model:
    options = {} if jitter is None else {"jitter": jitter}
    return rf.Model(
        rf.DateParts("created", dateparts=["day_of_week"], **options),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )


def test_dateparts_request_hydrates_jitter_from_a_mapping():
    model = build({"add": 0.2, "multiply": 0.1, "normalize": False})

    assert model.schema.requests[ADDRESS].jitter == Jitter(add=0.2, multiply=0.1, normalize=False)


@pytest.mark.parametrize("value", [None, 0.0, 0.2, 1, True])
def test_dateparts_rejects_non_jitter_configuration(value: object):
    with pytest.raises(pydantic.ValidationError):
        rf.DateParts("created", dateparts=["day_of_week"], jitter=value)


@pytest.mark.parametrize("normalize", [True, False])
def test_dateparts_jitters_its_only_continuous_boundary(
    monkeypatch: pytest.MonkeyPatch,
    normalize: bool,
):
    configured = Jitter(add=0.2, normalize=normalize)
    model = build(configured)
    embedder: Embedder = model.nodes[ADDRESS].embedder
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture(self: Jitter, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert self == configured
        captured.append((inputs.clone(), mask.clone()))
        return inputs

    monkeypatch.setattr(Jitter, "apply", capture)
    inputs = TensorInput(
        state=torch.tensor([Tokens.valued, Tokens.null, Tokens.masked, Tokens.valued]),
        content=TensorDict(
            {
                DatePart.day_of_week: torch.tensor(
                    [
                        [0.5, -0.5],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [float("inf"), 1.0],
                    ]
                )
            },
            batch_size=[4],
        ),
        batch_size=[4],
    )

    embedder.train()
    embedder(inputs)
    embedder.eval()
    embedder(inputs)

    assert len(captured) == 1
    content, eligible = captured[0]
    assert content.shape == (4, 2)
    assert eligible.tolist() == [
        [True, True],
        [False, False],
        [False, False],
        [False, True],
    ]


def test_dateparts_jitters_unit_circle_inputs_without_mutating_targets(monkeypatch: pytest.MonkeyPatch):
    configured = Jitter(add=0.2)
    model = build(configured)
    embedder: Embedder = model.nodes[ADDRESS].embedder
    field = model.encode(
        pa.table({"created": [datetime(2026, 5, 28, 14, 30, 45)]}),
        strata=Strata.train,
    )[ADDRESS]
    content = field.content[DatePart.day_of_week].clone()
    targets = field.targets[TensorKey.content][DatePart.day_of_week].clone()
    captured: list[torch.Tensor] = []

    def shift(self: Jitter, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        captured.append(inputs.clone())
        return inputs + mask.to(dtype=inputs.dtype).mul(0.25)

    monkeypatch.setattr(Jitter, "apply", shift)
    embedder.train()
    embedder.embed(field)

    assert len(captured) == 1
    assert torch.equal(captured[0], content.reshape(-1, 2))
    assert torch.equal(field.content[DatePart.day_of_week], content)
    assert torch.equal(field.targets[TensorKey.content][DatePart.day_of_week], targets)


def test_default_dateparts_jitter_does_not_advance_rng_during_training():
    model = build()
    embedder: Embedder = model.nodes[ADDRESS].embedder
    field = model.encode(
        pa.table({"created": [datetime(2026, 5, 28, 14, 30, 45)]}),
        strata=Strata.train,
    )[ADDRESS]
    before = torch.random.get_rng_state().clone()

    embedder.train()
    embedder.embed(field)

    assert torch.equal(torch.random.get_rng_state(), before)

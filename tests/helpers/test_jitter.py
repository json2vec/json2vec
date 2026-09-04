import pydantic
import pytest
import torch

import relflow as rf
from relflow.helpers import Jitter


def test_jitter_has_one_public_definition():
    assert Jitter.__module__ == "relflow.helpers.jitter"
    assert Jitter.apply.__module__ == "relflow.helpers.jitter"
    assert rf.Jitter is Jitter


def test_jitter_defaults_to_identity_configuration():
    jitter = Jitter()

    assert jitter.add == 0.0
    assert jitter.multiply == 0.0
    assert jitter.normalize is True


def test_jitter_amounts_are_nonnegative_scales_not_rates():
    jitter = Jitter(add=100.0, multiply=2.5)

    assert jitter.add == 100.0
    assert jitter.multiply == 2.5


def test_jitter_is_immutable():
    jitter = Jitter(add=0.1)

    with pytest.raises(pydantic.ValidationError, match="frozen"):
        jitter.add = 0.2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("add", -0.1),
        ("add", float("nan")),
        ("add", float("inf")),
        ("add", True),
        ("multiply", -0.1),
        ("multiply", float("nan")),
        ("multiply", float("inf")),
        ("multiply", False),
        ("normalize", 0),
        ("normalize", 1),
    ],
)
def test_jitter_rejects_invalid_configuration(field: str, value: object):
    with pytest.raises(pydantic.ValidationError):
        Jitter.model_validate({field: value})


def test_jitter_rejects_unknown_configuration():
    with pytest.raises(pydantic.ValidationError, match="extra"):
        Jitter.model_validate({"scale": 0.1})


def test_apply_multiplies_then_adds_with_independent_noise(monkeypatch: pytest.MonkeyPatch):
    inputs = torch.tensor([2.0, 4.0])
    draws = iter(
        (
            torch.full_like(inputs, 0.75),
            torch.full_like(inputs, 0.25),
            torch.full_like(inputs, 0.25),
            torch.full_like(inputs, 0.75),
        )
    )

    def draw(reference: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        return next(draws)

    monkeypatch.setattr(torch, "rand_like", draw)

    output = Jitter(add=2.0, multiply=0.5).apply(
        inputs,
        torch.ones_like(inputs, dtype=torch.bool),
    )

    assert torch.allclose(output, torch.tensor([1.5, 4.0]))
    with pytest.raises(StopIteration):
        next(draws)


def test_apply_changes_only_selected_values(monkeypatch: pytest.MonkeyPatch):
    inputs = torch.tensor([2.0, 4.0, 6.0])
    draws = iter((torch.ones(1), torch.zeros(1)))

    monkeypatch.setattr(torch, "rand_like", lambda values: next(draws).to(values))

    output = Jitter(add=0.5).apply(
        inputs,
        torch.tensor([True, False, False]),
    )

    assert output[1:].equal(inputs[1:])
    assert output[0] == 2.5


def test_identity_apply_does_not_advance_rng():
    inputs = torch.tensor([2.0, 4.0])
    before = torch.random.get_rng_state().clone()

    output = Jitter().apply(inputs, torch.ones_like(inputs, dtype=torch.bool))

    assert output is inputs
    assert torch.equal(torch.random.get_rng_state(), before)


def test_apply_leaves_normalization_placement_to_consumer():
    inputs = torch.tensor([2.0, 4.0])
    mask = torch.ones_like(inputs, dtype=torch.bool)

    torch.manual_seed(7)
    normalized = Jitter(add=0.5, multiply=0.25, normalize=True).apply(inputs, mask)
    torch.manual_seed(7)
    raw = Jitter(add=0.5, multiply=0.25, normalize=False).apply(inputs, mask)

    assert torch.equal(normalized, raw)

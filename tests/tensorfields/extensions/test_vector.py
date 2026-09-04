import pydantic
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.helpers import Jitter
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.structs.tree import Mask
from relflow.tensorfields.base import TENSORFIELDS, TensorInput
from relflow.tensorfields.extensions.vector import Decoder, Embedder, TensorField, loss, write
from relflow.tensorfields.extensions.vector import output as output_type
from tests.arrow import batch as arrow_batch
from tests.tensorfields.helpers import tensorize

ADDRESS = "root/items/embedding"


def structure_payload(
    *,
    n_dim: int = 3,
    objective: str = "l2",
    mask: bool | Mask = False,
    jitter: Jitter | dict[str, object] | None = None,
) -> dict:
    field: dict = {
        "name": "embedding",
        "type": "vector",
        "n_dim": n_dim,
        "objective": objective,
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


def values() -> list:
    return [
        [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
        [[[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]],
    ]


def tensorfield(*, values: list, schema: Schema, strata: Strata) -> TensorField:
    batch = arrow_batch([{"items": [{"embedding": value} for value in root]} for (root,) in values])
    projection = coalesce(batch, schema=schema, strata=strata)[ADDRESS]
    return tensorize(
        TensorField,
        projection,
        TENSORFIELDS["vector"],
        address=ADDRESS,
        schema=schema,
        strata=strata,
    )


def test_vector_request_is_available_in_structure():
    structure = Schema.model_validate(structure_payload())
    request = structure.requests[ADDRESS]
    assert request.type == "vector"
    assert request.n_dim == 3


def test_vector_request_rejects_non_positive_n_dim():
    with pytest.raises(ValueError, match="greater than 0"):
        Schema.model_validate(structure_payload(n_dim=0))


@pytest.mark.parametrize("normalize", [True, False])
def test_vector_request_hydrates_jitter_from_a_mapping(normalize: bool):
    structure = Schema.model_validate(structure_payload(jitter={"add": 0.2, "multiply": 0.1, "normalize": normalize}))

    assert structure.requests[ADDRESS].jitter == Jitter(add=0.2, multiply=0.1, normalize=normalize)


@pytest.mark.parametrize("value", [None, 0.0, 0.2, 1, True])
def test_vector_rejects_legacy_scalar_jitter(value: object):
    with pytest.raises(pydantic.ValidationError):
        rf.Vector("embedding", n_dim=3, jitter=value)


def test_vector_tensorfield_new_rejects_wrong_embedding_length():
    structure = Schema.model_validate(structure_payload(n_dim=3))
    schema = structure
    bad_values = [
        [[[0.1, 0.2], [0.3, 0.4, 0.5]]],
        [[[0.6, 0.7, 0.8], [0.9, 1.0, 1.1]]],
    ]

    with pytest.raises(ValueError, match="expects every value to have length 3"):
        tensorfield(
            values=bad_values,
            schema=schema,
            strata=Strata.train,
        )


def test_vector_embedder_and_decoder_shapes():
    structure = Schema.model_validate(structure_payload(n_dim=3))
    schema = structure

    field = tensorfield(
        values=values(),
        schema=schema,
        strata=Strata.train,
    )

    embedder = Embedder(schema=structure, address=ADDRESS)
    parcel = embedder.embed(field)
    assert parcel.payload.shape == (2, 1, 2, 16)

    decoder = Decoder(schema=structure, address=ADDRESS)
    prediction = decoder(
        [parcel],
        batch_size=field.state.shape[0],
        device=parcel.payload.device,
    )
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 3)


def test_vector_jitters_only_finite_valued_coordinates(monkeypatch: pytest.MonkeyPatch):
    structure = Schema.model_validate(structure_payload(jitter=Jitter(add=0.5)))
    embedder = Embedder(schema=structure, address=ADDRESS)
    captured: list[torch.Tensor] = []

    def capture(self: Jitter, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert self == Jitter(add=0.5)
        captured.append(mask.clone())
        return inputs

    monkeypatch.setattr(Jitter, "apply", capture)
    inputs = TensorInput(
        state=torch.tensor([Tokens.valued, Tokens.null, Tokens.masked, Tokens.valued]),
        content=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, float("inf"), float("nan")],
            ]
        ),
        batch_size=[4],
    )

    embedder.train()
    embedder(inputs)

    assert len(captured) == 1
    assert captured[0].tolist() == [
        [True, True, True],
        [False, False, False],
        [False, False, False],
        [True, False, False],
    ]


@pytest.mark.parametrize("normalize", [True, False])
def test_vector_jitter_runs_only_in_training_mode(monkeypatch: pytest.MonkeyPatch, normalize: bool):
    structure = Schema.model_validate(structure_payload(jitter=Jitter(add=0.5, normalize=normalize)))
    embedder = Embedder(schema=structure, address=ADDRESS)
    field = tensorfield(values=values(), schema=structure, strata=Strata.train)
    calls: list[torch.Tensor] = []

    def capture(self: Jitter, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        calls.append(inputs.clone())
        return inputs

    monkeypatch.setattr(Jitter, "apply", capture)

    embedder.train()
    embedder.embed(field)
    embedder.eval()
    embedder.embed(field)

    assert len(calls) == 1


def test_vector_jitter_preserves_content_and_reconstruction_targets(monkeypatch: pytest.MonkeyPatch):
    structure = Schema.model_validate(structure_payload(jitter=Jitter(add=0.5)))
    field = tensorfield(values=values(), schema=structure, strata=Strata.train)
    embedder = Embedder(schema=structure, address=ADDRESS)
    content = field.content.clone()
    targets = field.targets[TensorKey.content].clone()

    def replace(self: Jitter, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return inputs.masked_fill(mask, 99.0)

    monkeypatch.setattr(Jitter, "apply", replace)

    embedder.train()
    embedder.embed(field)

    assert torch.equal(field.content, content)
    assert torch.equal(field.targets[TensorKey.content], targets)


def test_vector_jitter_is_configuration_not_checkpoint_state():
    configured = Jitter(add=0.2, multiply=0.1, normalize=False)
    structure = Schema.model_validate(structure_payload(jitter=configured))
    embedder = Embedder(schema=structure, address=ADDRESS)

    assert embedder.jitter == configured
    assert not any("jitter" in name for name in embedder.state_dict())
    assert not any("jitter" in name for name, _ in embedder.named_buffers())


class DummyModule:
    def __init__(self, structure: Schema):
        self.schema = structure
        self.logged: list[tuple[tuple[str, ...], float]] = []

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        self.logged.append((names, float(value.detach().cpu())))
        return value


@pytest.mark.parametrize(("objective", "expected"), [("l1", 2.0), ("l2", 4.0)])
def test_vector_loss_uses_selected_objective(objective: str, expected: float):
    structure = Schema.model_validate(structure_payload(objective=objective, mask=Mask(reconstruct=True)))
    schema = structure

    field = tensorfield(
        values=values(),
        schema=schema,
        strata=Strata.train,
    )
    state_logits = torch.full((*field.state.shape, len(Tokens)), -50.0)
    state_logits.scatter_(-1, field.targets[TensorKey.state].unsqueeze(-1), 50.0)
    prediction_tensor = field.targets[TensorKey.content] + 2.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: prediction_tensor,
            },
            batch_size=[2],
        ),
    )

    module = DummyModule(structure)
    output = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
    assert torch.isclose(output, torch.tensor(expected, dtype=output.dtype))


def test_vector_write_returns_content_payload():
    structure = Schema.model_validate(structure_payload())
    state_logits = torch.full((2, 2, len(Tokens)), -50.0)
    state_logits[0, :, Tokens.valued.value] = 50.0
    state_logits[1, :, Tokens.null.value] = 50.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: torch.ones(2, 2, 3),
            },
            batch_size=[2],
        ),
    )

    module = DummyModule(structure)
    datatype = output_type(module, ADDRESS)
    output = write(module=module, prediction=prediction, datatype=datatype)
    assert output.type == datatype
    content = output.field(TensorKey.content.name)
    assert len(content) == 4
    assert content.to_pylist() == [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]

from types import SimpleNamespace

import torch
from loguru import logger
from tensordict import TensorDict

from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions.number import (
    Decoder,
    Embedder,
    GlobalOnlineNormalizer,
    TensorField,
    jitter,
    loss,
    write,
)

ADDRESS = "root/items/amount"


def _structure_payload() -> dict:
    field: dict = {
        "name": "amount",
        "type": "number",
        "query": "[*].items[*].amount",
    }
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


def test_number_request_allows_jitter_above_one():
    payload = _structure_payload()
    payload["fields"]["fields"][0]["fields"][0]["jitter"] = 1.5

    structure = Schema.model_validate(payload)

    assert structure.requests[ADDRESS].jitter == 1.5


class _TrackingModule:
    def __init__(self, schema: Schema, embedder: Embedder, decoder: Decoder):
        self.schema = schema
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder, decoder=decoder)}

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        return value


def test_number_loss_does_not_mutate_counter():
    structure = Schema.model_validate(_structure_payload())
    schema = structure

    field = TensorField.new(
        values=[[[1.0, None]], [[2.0]]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )
    field.mask(1.0)

    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _TrackingModule(schema=structure, embedder=embedder, decoder=decoder)

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


def test_number_embedder_clamps_unsafe_fourier_inputs_and_warns():
    structure = Schema.model_validate(_structure_payload())
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
    events: list[dict[str, object]] = []
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: (
            events.append(dict(message.record["extra"])),
            messages.append(message.record["message"]),
        ),
        level="WARNING",
    )

    try:
        clamped = embedder.clamp(content=content, state=state)
    finally:
        logger.remove(sink_id)

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
    structure = Schema.model_validate(_structure_payload())
    field = TensorField.new(
        values=[[[1.0, 2.0]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
    )
    field.content[0, 0, 0] = float("inf")

    embedder = Embedder(schema=structure, address=ADDRESS)
    embedder.train()
    parcel = embedder(field)

    assert torch.isfinite(embedder.normalizer.mean).all()
    assert torch.isfinite(embedder.normalizer.var).all()
    assert torch.isfinite(parcel.payload).all()


def test_number_embedder_preserves_flattened_jitter_draw_order():
    payload = _structure_payload()
    payload["fields"]["fields"][0]["fields"][0]["jitter"] = 0.25
    structure = Schema.model_validate(payload)
    field = TensorField.new(
        values=[[[1.0, 2.0]], [[3.0]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
    )
    embedder = Embedder(schema=structure, address=ADDRESS)
    embedder.train()
    embedder.normalizer.eval()

    flat_state = field.state.reshape(-1)
    flat_content = embedder.normalizer(
        inputs=field.content.reshape(-1),
        mask=flat_state.eq(Tokens.valued),
    )
    torch.manual_seed(123)
    flat_content = jitter(flat_content, jitter_amount=embedder.jitter)
    flat_content = embedder.clamp(content=flat_content, state=flat_state)
    weighted = flat_content.unsqueeze(-1).mul(embedder.weights)
    fourier = torch.cat((weighted.sin(), weighted.cos()), dim=-1)
    expected = embedder.embeddings(flat_state) + torch.nn.functional.gelu(embedder.linear(fourier))

    torch.manual_seed(123)
    output = embedder(field).payload

    assert torch.equal(output.reshape_as(expected), expected)


def test_number_write_emits_state_probability_map():
    structure = Schema.model_validate(_structure_payload())
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

    output = write(module=SimpleNamespace(schema=structure), prediction=prediction)
    state_payload = output[TensorKey.state.name]

    assert set(state_payload.keys()) == set(Tokens.__members__.keys())
    assert all(probabilities.shape == (2, 1) for probabilities in state_payload.values())
    assert state_payload[Tokens.valued.name][0, 0] > 0.99
    assert state_payload[Tokens.null.name][1, 0] > 0.99
    assert output[TensorKey.content.name].shape == (2, 1, 1)

from types import SimpleNamespace

import torch
from loguru import logger
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.experiment import Schema
from json2vec.structs.packages import Prediction
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TENSORFIELDS
from json2vec.tensorfields.extensions.number import (
    Decoder,
    Embedder,
    GlobalOnlineNormalizer,
    NormalizerSyncCallback,
    TensorField,
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


def test_normalizer_update_accumulates_pending_raw_stats():
    normalizer = GlobalOnlineNormalizer()
    normalizer.train()
    values = torch.tensor([1.0, 2.0, 3.0, 4.0])

    normalizer.update(values)

    assert normalizer._pending_count.item() == 4  # noqa: SLF001
    assert torch.isclose(normalizer._pending_sum, torch.tensor([10.0]))  # noqa: SLF001
    assert torch.isclose(normalizer._pending_sumsq, torch.tensor([30.0]))  # noqa: SLF001
    assert torch.isclose(normalizer.count, torch.tensor([4.0]))


def test_normalizer_forward_does_not_call_distributed_collective(monkeypatch):
    def fail_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        raise AssertionError("GlobalOnlineNormalizer.forward must not call all_reduce_sum")

    monkeypatch.setattr("json2vec.tensorfields.extensions.number.all_reduce_sum", fail_all_reduce)
    normalizer = GlobalOnlineNormalizer()
    normalizer.train()

    values = torch.tensor([0.5, 1.5])
    mask = torch.ones_like(values, dtype=torch.bool)
    normalizer(values, mask=mask)


def test_normalizer_sync_folds_others_contribution_via_chan(monkeypatch):
    # Local rank saw [1.0, 3.0]; the simulated peer saw [5.0, 7.0, 9.0]. Globals
    # are (n=5, sum=25, sumsq=165). `sync()` reduces count, sum, sumsq in order.
    normalizer = GlobalOnlineNormalizer()
    normalizer.train()
    normalizer.update(torch.tensor([1.0, 3.0]))

    globals_in_order = [torch.tensor([5.0]), torch.tensor([25.0]), torch.tensor([165.0])]
    reduced: list[torch.Tensor] = []

    def fake_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        reduced.append(tensor.clone())
        return globals_in_order[len(reduced) - 1].clone()

    monkeypatch.setattr("json2vec.tensorfields.extensions.number.all_reduce_sum", fake_all_reduce)

    normalizer.sync()

    assert len(reduced) == 3
    assert normalizer._pending_count.item() == 0  # noqa: SLF001
    assert normalizer._pending_sum.item() == 0  # noqa: SLF001
    assert normalizer._pending_sumsq.item() == 0  # noqa: SLF001

    expected_values = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    assert torch.isclose(normalizer.count, torch.tensor([5.0]))
    assert torch.isclose(normalizer.mean, expected_values.mean().unsqueeze(0))
    assert torch.isclose(normalizer.var, expected_values.var(unbiased=False).unsqueeze(0))


def test_normalizer_sync_is_noop_when_others_have_no_data(monkeypatch):
    def identity_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    monkeypatch.setattr("json2vec.tensorfields.extensions.number.all_reduce_sum", identity_all_reduce)
    normalizer = GlobalOnlineNormalizer()
    normalizer.train()
    normalizer.update(torch.tensor([1.0, 3.0, 5.0]))

    snapshot_mean = normalizer.mean.clone()
    snapshot_var = normalizer.var.clone()
    snapshot_count = normalizer.count.clone()

    normalizer.sync()

    assert torch.equal(normalizer.mean, snapshot_mean)
    assert torch.equal(normalizer.var, snapshot_var)
    assert torch.equal(normalizer.count, snapshot_count)
    assert normalizer._pending_count.item() == 0  # noqa: SLF001


def test_normalizer_sync_alpha_mode_applies_one_ema_step_for_others(monkeypatch):
    # Local saw [1.0, 3.0] (n=2, sum=4, sumsq=10). Simulated globals add a peer
    # rank with n=2, sum=8, sumsq=46 so others_mean=4 and others_var=7.
    globals_in_order = [torch.tensor([4.0]), torch.tensor([12.0]), torch.tensor([56.0])]
    reduced: list[torch.Tensor] = []

    def fake_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        reduced.append(tensor.clone())
        return globals_in_order[len(reduced) - 1].clone()

    monkeypatch.setattr("json2vec.tensorfields.extensions.number.all_reduce_sum", fake_all_reduce)

    normalizer = GlobalOnlineNormalizer(alpha=0.5)
    normalizer.train()
    pre_mean = normalizer.mean.clone()
    pre_var = normalizer.var.clone()
    normalizer.update(torch.tensor([1.0, 3.0]))

    after_local_mean = normalizer.mean.clone()
    after_local_var = normalizer.var.clone()
    assert torch.isclose(after_local_mean, 0.5 * pre_mean + 0.5 * torch.tensor([2.0]))
    assert torch.isclose(after_local_var, 0.5 * pre_var + 0.5 * torch.tensor([1.0]))

    normalizer.sync()

    expected_mean = 0.5 * after_local_mean + 0.5 * torch.tensor([4.0])
    expected_var = 0.5 * after_local_var + 0.5 * torch.tensor([7.0])

    assert torch.isclose(normalizer.mean, expected_mean)
    assert torch.isclose(normalizer.var, expected_var)
    assert normalizer.count.item() == 0


def test_normalizer_pending_buffers_are_not_persistent():
    normalizer = GlobalOnlineNormalizer()
    state = normalizer.state_dict()

    assert "mean" in state
    assert "var" in state
    assert "count" in state
    assert "_pending_count" not in state
    assert "_pending_sum" not in state
    assert "_pending_sumsq" not in state


def test_normalizer_sync_callback_syncs_in_deterministic_address_order(monkeypatch):
    calls: list[str] = []

    def record(self: GlobalOnlineNormalizer) -> None:
        calls.append(self._tag)  # type: ignore[attr-defined]  # noqa: SLF001

    def tagged(tag: str) -> GlobalOnlineNormalizer:
        normalizer = GlobalOnlineNormalizer()
        normalizer._tag = tag  # type: ignore[attr-defined]  # noqa: SLF001
        return normalizer

    module = SimpleNamespace(
        nodes={
            Address("root", "z"): SimpleNamespace(embedder=SimpleNamespace(normalizer=tagged("z"))),
            Address("root", "a"): SimpleNamespace(embedder=SimpleNamespace(normalizer=tagged("a"))),
            Address("root", "m"): SimpleNamespace(embedder=SimpleNamespace(normalizer=tagged("m"))),
            Address("root", "skip"): SimpleNamespace(embedder=SimpleNamespace()),  # no normalizer
            Address("root", "noembed"): SimpleNamespace(embedder=None),
        }
    )
    monkeypatch.setattr(GlobalOnlineNormalizer, "sync", record)

    NormalizerSyncCallback().on_train_epoch_end(trainer=None, pl_module=module)

    assert calls == ["a", "m", "z"]


def test_normalizer_sync_callback_is_registered_on_number_plugin():
    assert NormalizerSyncCallback in TENSORFIELDS["number"].callback_factories

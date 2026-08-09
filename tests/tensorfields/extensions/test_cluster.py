from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions.cluster import (
    ClusterReviveCallback,
    Decoder,
    Embedder,
    TensorField,
    loss,
    write,
)
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel

ADDRESS = "root/items/cluster"


def _structure_payload(
    *,
    capacity: int = 8,
    bounds: int | list[int] | tuple[int, int] = (2, 4),
    p_unavailable: float | None = None,
    sparsity_weight: float | None = None,
    gumbel_tau: float | None = None,
    balance_epsilon: float | None = None,
    revive_scale: float | None = None,
    revive_noise: float | None = None,
) -> dict:
    field: dict = {
        "name": "cluster",
        "type": "cluster",
        "query": "[*].items[*].label",
        "capacity": capacity,
        "bounds": bounds,
    }
    if p_unavailable is not None:
        field["p_unavailable"] = p_unavailable
    if sparsity_weight is not None:
        field["sparsity_weight"] = sparsity_weight
    if gumbel_tau is not None:
        field["gumbel_tau"] = gumbel_tau
    if balance_epsilon is not None:
        field["balance_epsilon"] = balance_epsilon
    if revive_scale is not None:
        field["revive_scale"] = revive_scale
    if revive_noise is not None:
        field["revive_noise"] = revive_noise

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


def _state(size: int = 8):
    return OnlineVocabularyModel(size=size).state


# ---------- TensorField ----------


def test_cluster_tensorfield_routes_oov_to_sentinel_at_validate():
    structure = Schema.model_validate(_structure_payload(p_unavailable=0.0))
    state = _state(size=structure.requests[ADDRESS].capacity)

    TensorField.new(
        values=[[["ALPHA"]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )
    field = TensorField.new(
        values=[[["OMEGA"]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.validate,
        interprocess_encoding_context=state,
    )

    sentinel = structure.requests[ADDRESS].capacity
    assert torch.equal(
        field.state,
        torch.tensor([[[Tokens.valued.value, Tokens.padded.value]]], dtype=torch.int64),
    )
    assert torch.equal(
        field.content,
        torch.tensor([[[sentinel, 0]]], dtype=torch.int64),
    )


def test_cluster_tensorfield_simulates_unavailable_during_training():
    structure = Schema.model_validate(_structure_payload(p_unavailable=1.0))
    state = _state(size=structure.requests[ADDRESS].capacity)

    field = TensorField.new(
        values=[[["ALPHA", None]], [["BETA"]]],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )

    sentinel = structure.requests[ADDRESS].capacity
    assert torch.equal(
        field.content,
        torch.tensor(
            [
                [[sentinel, 0]],
                [[sentinel, 0]],
            ],
            dtype=torch.int64,
        ),
    )


# ---------- Embedder ----------


def test_cluster_embedder_train_forward_is_stochastic():
    structure = Schema.model_validate(_structure_payload(bounds=4, capacity=8, p_unavailable=0.0))
    embedder = Embedder(schema=structure, address=ADDRESS)
    embedder.train()

    with torch.no_grad():
        # Flat logits so Gumbel noise dominates the argmax.
        embedder.embeddings[TensorKey.cluster.name].weight.zero_()

    field = TensorField(
        state=torch.tensor([[Tokens.valued.value]], dtype=torch.int64),
        content=torch.tensor([[3]], dtype=torch.int64),
        trainable=torch.zeros((1, 1), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )

    torch.manual_seed(1)
    a = embedder(field).payload.clone()
    torch.manual_seed(2)
    b = embedder(field).payload.clone()
    assert not torch.equal(a, b)


def test_cluster_embedder_routes_sentinel_row_for_oov_tokens():
    structure = Schema.model_validate(_structure_payload(bounds=4, capacity=8, p_unavailable=0.0))
    embedder = Embedder(schema=structure, address=ADDRESS)
    embedder.eval()
    sentinel = structure.requests[ADDRESS].capacity

    field = TensorField(
        state=torch.tensor([[Tokens.valued.value]], dtype=torch.int64),
        content=torch.tensor([[sentinel]], dtype=torch.int64),
        trainable=torch.zeros((1, 1), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )
    output = embedder(field).payload
    assert torch.isfinite(output).all()

    assign_logits = embedder.embeddings[TensorKey.cluster.name](torch.tensor([sentinel]))
    hard = torch.zeros_like(assign_logits)
    hard.scatter_(-1, assign_logits.argmax(dim=-1, keepdim=True), 1.0)
    expected_state = embedder.embeddings[TensorKey.state.name](field.state)
    expected_cluster = embedder.embeddings[TensorKey.content.name](hard)
    assert torch.allclose(output, expected_state + expected_cluster.reshape(1, 1, -1))


def test_cluster_embedder_zeros_cluster_contribution_for_non_valued_state():
    structure = Schema.model_validate(_structure_payload(bounds=4, capacity=8, p_unavailable=0.0))
    embedder = Embedder(schema=structure, address=ADDRESS)
    embedder.eval()

    field = TensorField(
        state=torch.tensor(
            [
                [
                    Tokens.valued.value,
                    Tokens.padded.value,
                    Tokens.null.value,
                    Tokens.masked.value,
                ]
            ],
            dtype=torch.int64,
        ),
        content=torch.tensor([[2, 0, 0, 0]], dtype=torch.int64),
        trainable=torch.zeros((1, 4), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )

    output = embedder(field).payload
    expected = embedder.embeddings[TensorKey.state.name](field.state)
    assign_logits = embedder.embeddings[TensorKey.cluster.name](torch.tensor([2]))
    hard = torch.zeros_like(assign_logits)
    hard.scatter_(-1, assign_logits.argmax(dim=-1, keepdim=True), 1.0)
    expected[:, 0] += embedder.embeddings[TensorKey.content.name](hard).reshape(-1)
    assert torch.allclose(output, expected)


def test_cluster_embedder_rejects_indices_beyond_capacity():
    structure = Schema.model_validate(_structure_payload(bounds=4, capacity=8, p_unavailable=0.0))
    embedder = Embedder(schema=structure, address=ADDRESS)

    field = TensorField(
        state=torch.tensor([[Tokens.valued.value]], dtype=torch.int64),
        content=torch.tensor([[9]], dtype=torch.int64),
        trainable=torch.zeros((1, 1), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )
    with pytest.raises(ValueError):
        embedder(field)


# ---------- write ----------


class _DummyVocab:
    def __init__(self, tokens: list[str]):
        self._tokens = list(tokens)

    def snapshot(self) -> list[str]:
        return list(self._tokens)


class _DummyEmbedder:
    def __init__(self, *, vocab_tokens: list[str], capacity: int, K: int, assign_weight: torch.Tensor | None = None):
        self.vocab = _DummyVocab(vocab_tokens)
        self.capacity = capacity
        if assign_weight is None:
            assign_weight = torch.zeros(capacity + 1, K)
            for i in range(capacity + 1):
                assign_weight[i, i % K] = 1.0
        self.embeddings = {TensorKey.cluster.name: SimpleNamespace(weight=assign_weight)}


class _DummyWriteModule:
    def __init__(self, embedder: _DummyEmbedder):
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder)}


def test_cluster_write_emits_state_cluster_and_content_payloads():
    K = 4
    capacity = 5
    embedder = _DummyEmbedder(
        vocab_tokens=["ALPHA", "BETA", "GAMMA", "DELTA", "EPS"],
        capacity=capacity,
        K=K,
    )
    module = _DummyWriteModule(embedder)

    state_logits = torch.zeros(2, 1, len(Tokens))
    state_logits[0, 0, Tokens.valued.value] = 10.0
    state_logits[1, 0, Tokens.padded.value] = 10.0

    cluster_logits = torch.zeros(2, 1, K)
    cluster_logits[0, 0, 1] = 10.0
    cluster_logits[1, 0, 2] = 10.0

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: state_logits, TensorKey.cluster: cluster_logits},
            batch_size=[2],
        ),
    )

    output = write(module=module, prediction=prediction)
    assert set(output.keys()) == {TensorKey.state.name, TensorKey.cluster.name, TensorKey.content.name}

    state_payload = output[TensorKey.state.name]
    assert set(state_payload.keys()) == set(Tokens.__members__.keys())
    for v in state_payload.values():
        assert v.shape == (2, 1)
    assert state_payload[Tokens.valued.name][0, 0] > 0.99
    assert state_payload[Tokens.padded.name][1, 0] > 0.99

    cluster_payload = output[TensorKey.cluster.name]
    assert cluster_payload[TensorKey.value.name].tolist() == [[1], [2]]
    assert cluster_payload[TensorKey.probability.name].shape == (2, 1)
    assert (cluster_payload[TensorKey.probability.name] > 0.99).all()

    # Assignment table row i is one-hot on cluster (i % K), so vocab argmax on cluster 1 -> BETA,
    # cluster 2 -> GAMMA.
    content_payload = output[TensorKey.content.name]
    assert content_payload[TensorKey.value.name].tolist() == [["BETA"], ["GAMMA"]]
    assert content_payload[TensorKey.probability.name].shape == (2, 1)


def test_cluster_write_drops_sentinel_column_from_content_prediction():
    K = 3
    capacity = 3
    assign_weight = torch.zeros(capacity + 1, K)
    assign_weight[0, 1] = 1.0
    assign_weight[1, 1] = 1.0
    assign_weight[2, 2] = 1.0
    # Sentinel row (index=capacity) is one-hot on cluster 0.
    assign_weight[3, 0] = 1.0
    embedder = _DummyEmbedder(
        vocab_tokens=["A", "B", "C"],
        capacity=capacity,
        K=K,
        assign_weight=assign_weight,
    )
    module = _DummyWriteModule(embedder)

    cluster_logits = torch.zeros(1, 1, K)
    cluster_logits[0, 0, 0] = 10.0
    state_logits = torch.zeros(1, 1, len(Tokens))

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: state_logits, TensorKey.cluster: cluster_logits},
            batch_size=[1],
        ),
    )
    output = write(module=module, prediction=prediction)

    # Sentinel dropped -> best real vocab wins. Vocab rows for A/B point at cluster 1 which has ~0
    # probability under our logits, so all vocab candidates are equal; argmax returns index 0 -> A.
    content_payload = output[TensorKey.content.name]
    assert content_payload[TensorKey.value.name].tolist() == [["A"]]


def test_cluster_write_returns_none_labels_when_vocab_is_empty():
    K = 4
    capacity = 5
    embedder = _DummyEmbedder(vocab_tokens=[], capacity=capacity, K=K)
    module = _DummyWriteModule(embedder)

    state_logits = torch.zeros(1, 1, len(Tokens))
    cluster_logits = torch.zeros(1, 1, K)

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: state_logits, TensorKey.cluster: cluster_logits},
            batch_size=[1],
        ),
    )
    output = write(module=module, prediction=prediction)
    content_payload = output[TensorKey.content.name]
    assert content_payload[TensorKey.value.name].tolist() == [[None]]
    assert content_payload[TensorKey.probability.name].tolist() == [[0.0]]


# ---------- loss ----------


class _TrackingModule:
    def __init__(self, schema: Schema, embedder: Embedder, decoder: Decoder):
        self.schema = schema
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder, decoder=decoder)}
        self.tracked: dict[tuple, torch.Tensor] = {}

    def track(self, names: tuple, value: torch.Tensor) -> torch.Tensor:
        self.tracked[names] = value
        return value


def _prepared_train_batch(structure: Schema, *, values: list) -> TensorField:
    state = _state(size=structure.requests[ADDRESS].capacity)
    field = TensorField.new(
        values=values,
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=state,
    )
    field.mask(1.0)
    return field


def test_cluster_loss_sinkhorn_pushes_gradient_toward_spread():
    # balance_epsilon must be commensurate with the logit gap for Sinkhorn to actually spread
    # mass; at default 0.05, a gap of 5.0 saturates the Boltzmann map to a one-hot Q.
    structure = Schema.model_validate(
        _structure_payload(bounds=(4, 4), capacity=8, p_unavailable=1.0, balance_epsilon=1.0)
    )
    field = _prepared_train_batch(structure, values=[[["ALPHA", "BETA"]]] * 4)
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _TrackingModule(structure, embedder, decoder)
    K = structure.requests[ADDRESS].size

    cluster_logits = torch.zeros(*field.state.shape, K)
    cluster_logits[..., 0] = 0.5
    cluster_logits = cluster_logits.detach().requires_grad_(True)

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.cluster: cluster_logits,
            },
            batch_size=field.batch_size,
        ),
    )
    total = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
    balance = module.tracked[(ADDRESS, Strata.train, "loss", TensorKey.cluster)]
    assert torch.isfinite(balance) and balance.item() > 0.0

    total.backward()
    grad = cluster_logits.grad
    assert grad is not None
    valued_mask = field.targets[TensorKey.state].eq(Tokens.valued.value)
    valued_grad = grad[valued_mask]
    # Collapsed column receives positive gradient (push probability down); empty columns receive
    # negative gradient (pull them up).
    assert valued_grad[:, 0].mean().item() > 0.0
    assert valued_grad[:, 1:].mean().item() < 0.0


def test_cluster_loss_sentinel_share_reflects_target_content():
    structure = Schema.model_validate(_structure_payload(bounds=(2, 4), capacity=8, p_unavailable=1.0))
    field = _prepared_train_batch(structure, values=[[["ALPHA", "BETA"]], [["BETA", "ALPHA"]]])
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _TrackingModule(structure, embedder, decoder)
    K = structure.requests[ADDRESS].size

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.cluster: torch.randn(*field.state.shape, K),
            },
            batch_size=field.batch_size,
        ),
    )
    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    share = module.tracked[(ADDRESS, Strata.train, "cluster", "sentinel_share")]
    assert share.item() == 1.0


def test_cluster_loss_usage_ema_updates_toward_batch_distribution():
    structure = Schema.model_validate(_structure_payload(bounds=(4, 4), capacity=8, p_unavailable=1.0))
    field = _prepared_train_batch(structure, values=[[["ALPHA", "BETA"]]] * 4)
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _TrackingModule(structure, embedder, decoder)
    K = structure.requests[ADDRESS].size

    prior = embedder.usage_ema.clone()
    cluster_logits = torch.zeros(*field.state.shape, K)
    cluster_logits[..., 0] = 10.0

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.cluster: cluster_logits,
            },
            batch_size=field.batch_size,
        ),
    )
    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    # Column 0 usage should increase, others should decrease (weighted by ema_decay).
    assert embedder.usage_ema[0].item() > prior[0].item()
    for k in range(1, K):
        assert embedder.usage_ema[k].item() < prior[k].item()
    # Distribution stays a probability vector.
    assert torch.isclose(embedder.usage_ema.sum(), torch.tensor(1.0), atol=1e-5)


# ---------- ClusterReviveCallback ----------


def _seed_uncommitted_batch(*, revive_scale: float | None = None, revive_noise: float | None = None):
    """Prepare a train batch + real embedder/decoder with a range-bounded cluster leaf.

    Marks the upper half of the committed buffer as uncommitted so the loss will accumulate
    ``adherence_ema`` mass, and biases the cluster logits toward those columns.
    """
    structure = Schema.model_validate(
        _structure_payload(
            bounds=(2, 4),
            capacity=8,
            p_unavailable=1.0,
            revive_scale=revive_scale,
            revive_noise=revive_noise,
        )
    )
    field = _prepared_train_batch(structure, values=[[["ALPHA", "BETA"]]] * 4)
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    K = structure.requests[ADDRESS].size
    lower = structure.requests[ADDRESS].n_clusters[0]
    embedder.committed[lower:] = False

    cluster_logits = torch.zeros(*field.state.shape, K)
    cluster_logits[..., lower:] = 5.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.cluster: cluster_logits,
            },
            batch_size=field.batch_size,
        ),
    )
    module = _TrackingModule(structure, embedder, decoder)
    return structure, field, module, embedder, decoder, prediction


def test_adherence_ema_updates_without_sparsity_weight():
    structure, field, module, embedder, _, prediction = _seed_uncommitted_batch()
    assert structure.requests[ADDRESS].sparsity_weight == 0.0
    assert embedder.adherence_ema.item() == 0.0

    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    # EMA received a positive uncommitted-mass update even with sparsity_weight == 0.
    assert embedder.adherence_ema.item() > 0.0
    # ...and the loss-side track was NOT emitted (sparsity_weight == 0 gate).
    assert (ADDRESS, Strata.train, "cluster", "adherence") not in module.tracked


class _FakeTrainer:
    def __init__(self, is_global_zero: bool = True):
        self.is_global_zero = is_global_zero


def test_revive_callback_noop_when_revive_scale_zero():
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch()
    embedder.adherence_ema.fill_(0.9)
    prior_assign = embedder.embeddings[TensorKey.cluster.name].weight.detach().clone()
    prior_usage = embedder.usage_ema.detach().clone()

    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(), module)

    # revive_scale defaults to 0 -> callback is a no-op, no weights touched, ema preserved.
    assert torch.equal(embedder.embeddings[TensorKey.cluster.name].weight, prior_assign)
    assert torch.equal(embedder.usage_ema, prior_usage)
    assert embedder.adherence_ema.item() == pytest.approx(0.9)


def test_revive_callback_noop_when_bounds_are_fixed():
    structure = Schema.model_validate(
        _structure_payload(bounds=(4, 4), capacity=8, p_unavailable=1.0, revive_scale=10.0)
    )
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    embedder.adherence_ema.fill_(0.9)
    prior_assign = embedder.embeddings[TensorKey.cluster.name].weight.detach().clone()

    module = _TrackingModule(structure, embedder, decoder)
    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(), module)

    # lower == upper -> no dead columns are eligible; weights untouched, adherence preserved.
    assert torch.equal(embedder.embeddings[TensorKey.cluster.name].weight, prior_assign)
    assert embedder.adherence_ema.item() == pytest.approx(0.9)


def test_revive_callback_noop_when_adherence_ema_is_zero():
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_scale=10.0)
    prior_assign = embedder.embeddings[TensorKey.cluster.name].weight.detach().clone()

    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(), module)

    assert torch.equal(embedder.embeddings[TensorKey.cluster.name].weight, prior_assign)
    assert embedder.adherence_ema.item() == 0.0


def test_revive_callback_splits_donor_into_dead_column_and_resets_state():
    torch.manual_seed(0)
    # bounds=(3, 4) yields exactly 1 uncommitted column so the donor is halved exactly once.
    structure = Schema.model_validate(
        _structure_payload(
            bounds=(3, 4), capacity=8, p_unavailable=1.0, revive_scale=100.0, revive_noise=0.0
        )
    )
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    lower = structure.requests[ADDRESS].n_clusters[0]
    embedder.committed[lower:] = False
    module = _TrackingModule(structure, embedder, decoder)

    # Force adherence high enough that the per-column trial passes deterministically.
    embedder.adherence_ema.fill_(1.0)

    # Concentrate all usage on column 0 so the donor is deterministic.
    embedder.usage_ema.zero_()
    embedder.usage_ema[0] = 1.0

    # Give distinct values to donor + the dead column to detect the copy.
    assign_w = embedder.embeddings[TensorKey.cluster.name].weight
    content_w = embedder.embeddings[TensorKey.content.name].weight
    cluster_w = decoder.linears[TensorKey.cluster.name].weight
    dead = int((~embedder.committed).nonzero(as_tuple=True)[0][0].item())

    donor_assign = torch.arange(assign_w.shape[0], dtype=assign_w.dtype)
    donor_content = torch.arange(content_w.shape[0], dtype=content_w.dtype)
    donor_cluster = torch.arange(cluster_w.shape[1], dtype=cluster_w.dtype)
    assign_w.data[:, 0] = donor_assign
    content_w.data[:, 0] = donor_content
    cluster_w.data[0, :] = donor_cluster
    assign_w.data[:, dead] = 0.0
    content_w.data[:, dead] = 0.0
    cluster_w.data[dead, :] = 0.0

    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(), module)

    # With revive_noise=0 and adherence_ema=1.0, dead column receives an exact copy of donor.
    assert torch.equal(assign_w.data[:, dead], donor_assign)
    assert torch.equal(content_w.data[:, dead], donor_content)
    assert torch.equal(cluster_w.data[dead, :], donor_cluster)
    # Donor usage halved; revived column inherits half of donor's usage.
    assert embedder.usage_ema[0].item() == pytest.approx(0.5)
    assert embedder.usage_ema[dead].item() == pytest.approx(0.5)
    # adherence_ema reset after revive.
    assert embedder.adherence_ema.item() == 0.0


def test_revive_callback_respects_non_global_zero_rank():
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_scale=100.0, revive_noise=0.0)
    embedder.adherence_ema.fill_(1.0)
    prior_assign = embedder.embeddings[TensorKey.cluster.name].weight.detach().clone()

    # No distributed backend is up -> broadcast_object is a passthrough, so a rank-non-zero
    # trainer produces an empty plans dict; no weights change.
    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(is_global_zero=False), module)

    assert torch.equal(embedder.embeddings[TensorKey.cluster.name].weight, prior_assign)
    assert embedder.adherence_ema.item() == pytest.approx(1.0)


from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions.cluster import (
    ClusterMergeCallback,
    ClusterReviveCallback,
    Decoder,
    Embedder,
    TensorField,
    loss,
    write,
)
from relflow.tensorfields.shared.counter import Counter
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel

ADDRESS = "root/items/cluster"


def _structure_payload(
    *,
    capacity: int = 8,
    bounds: int | list[int] | tuple[int, int] = (2, 4),
    p_unavailable: float | None = None,
    sparsity_weight: float | None = None,
    revive_temperature: float | None = None,
    ema_decay: float | None = None,
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
    if revive_temperature is not None:
        field["revive_temperature"] = revive_temperature
    if ema_decay is not None:
        field["ema_decay"] = ema_decay

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


def test_cluster_embedder_init_commits_only_lower_columns():
    # Init at ``lower`` keeps Sinkhorn from locking n_committed at ``upper`` from step 0.
    structure = Schema.model_validate(_structure_payload(bounds=(2, 6), capacity=8))
    embedder = Embedder(schema=structure, address=ADDRESS)

    assert embedder.committed.tolist() == [True, True, False, False, False, False]
    assert torch.allclose(
        embedder.usage_ema,
        torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
    )


def test_cluster_embedder_init_perplexity_matches_lower_bound():
    # perplexity(usage_ema) at init must equal ``lower`` so the loss's n_committed selection
    # starts small and only grows through revive.
    import math

    structure = Schema.model_validate(_structure_payload(bounds=(3, 8), capacity=16))
    embedder = Embedder(schema=structure, address=ADDRESS)

    normalized = embedder.usage_ema / embedder.usage_ema.sum().clamp_min(1e-12)
    safe = normalized.clamp_min(1e-12)
    entropy = -(safe * safe.log()).sum().item()
    perplexity = math.exp(entropy)
    assert round(perplexity) == 3


def test_cluster_embedder_init_matches_upper_when_bounds_fixed():
    # lower == upper is the "fixed K" case; init should commit all columns uniformly.
    structure = Schema.model_validate(_structure_payload(bounds=(4, 4), capacity=8))
    embedder = Embedder(schema=structure, address=ADDRESS)

    assert embedder.committed.tolist() == [True] * 4
    assert torch.allclose(embedder.usage_ema, torch.full((4,), 0.25))


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
    # Logits must be commensurate with ε=0.05 so Sinkhorn's Boltzmann map is not saturated.
    structure = Schema.model_validate(
        _structure_payload(bounds=(4, 4), capacity=8, p_unavailable=1.0)
    )
    field = _prepared_train_batch(structure, values=[[["ALPHA", "BETA"]]] * 4)
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _TrackingModule(structure, embedder, decoder)
    K = structure.requests[ADDRESS].size

    cluster_logits = torch.zeros(*field.state.shape, K)
    cluster_logits[..., 0] = 0.025
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


def test_cluster_loss_balance_penalizes_uncommitted_mass():
    # Balance loss uses a full-K log-softmax so uncommitted decoder rows receive gradient
    # through the normalizer. Raising uncommitted logits (as revive does when it copies a
    # donor row into a dead column) must INCREASE balance loss, which is exactly the signal
    # the model needs to eventually push those uncommitted rows back down.
    structure = Schema.model_validate(
        _structure_payload(bounds=(2, 4), capacity=8, p_unavailable=1.0)
    )
    field = _prepared_train_batch(structure, values=[[["ALPHA", "BETA"]]] * 4)
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    K = structure.requests[ADDRESS].size
    lower = structure.requests[ADDRESS].n_clusters[0]

    base_logits = torch.zeros(*field.state.shape, K)
    base_logits[..., 0] = 0.075
    base_logits[..., 1] = 0.025

    with torch.no_grad():
        embedder.usage_ema.zero_()
        embedder.usage_ema[0] = 1.0
        embedder.usage_ema[1] = 0.9

    def _balance(cluster_logits: torch.Tensor) -> float:
        module = _TrackingModule(structure, embedder, decoder)
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
        return module.tracked[(ADDRESS, Strata.train, "loss", TensorKey.cluster)].item()

    baseline = _balance(base_logits.clone())

    revived_logits = base_logits.clone()
    revived_logits[..., 2] = base_logits[..., 0]
    revived_logits[..., 3] = base_logits[..., 0]

    _, expected_committed = torch.tensor([1.0, 0.9, 0.0, 0.0]).topk(lower)
    assert set(expected_committed.tolist()) == {0, 1}

    revived = _balance(revived_logits)
    assert revived > baseline


def test_cluster_loss_balance_gradient_reaches_uncommitted_columns():
    # Under the full-K normalizer, backprop from balance loss must produce a *negative*-going
    # gradient on uncommitted logits (push them down) while committed logits still get pulled
    # up. This is the mechanism that lets ``adherence_ema`` decay once the true K is reached.
    structure = Schema.model_validate(
        _structure_payload(bounds=(2, 4), capacity=8, p_unavailable=1.0)
    )
    field = _prepared_train_batch(structure, values=[[["ALPHA", "BETA"]]] * 4)
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    module = _TrackingModule(structure, embedder, decoder)
    K = structure.requests[ADDRESS].size

    with torch.no_grad():
        embedder.usage_ema.zero_()
        embedder.usage_ema[0] = 1.0
        embedder.usage_ema[1] = 0.9

    cluster_logits = torch.zeros(*field.state.shape, K)
    cluster_logits[..., 0] = 0.05
    cluster_logits[..., 1] = 0.02
    # Uncommitted logits raised above baseline; a healthy loss should pull them back down.
    cluster_logits[..., 2] = 0.10
    cluster_logits[..., 3] = 0.10
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
    total.backward()

    grad = cluster_logits.grad
    assert grad is not None
    valued_mask = field.targets[TensorKey.state].eq(Tokens.valued.value)
    valued_grad = grad[valued_mask]
    # Uncommitted (columns 2, 3): positive gradient (loss increases with these logits -> SGD
    # will push them down).
    assert valued_grad[:, 2:].mean().item() > 0.0
    # Committed (columns 0, 1): non-positive net gradient (loss decreases as these logits rise).
    assert valued_grad[:, :2].mean().item() <= 0.0


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


def test_cluster_content_counter_rebalances_content_loss_by_inverse_frequency():
    # Content CE consumes the ``Counter(size=capacity + 1)`` weight identically to Category:
    # uniform counts leave the loss unchanged, skewed counts rebalance it toward rare tokens.
    structure = Schema.model_validate(_structure_payload(bounds=(4, 4), capacity=8, p_unavailable=0.0))
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    K = structure.requests[ADDRESS].size
    capacity = structure.requests[ADDRESS].capacity

    content_counter = embedder.counters[TensorKey.content.name]
    assert isinstance(content_counter, Counter)
    assert content_counter.size == capacity + 1

    # Encode the batch through the embedder's own vocab so the vocab-size gated accuracy
    # metric can evaluate under known content targets.
    field = TensorField.new(
        values=[[["ALPHA", "BETA"]]] * 4,
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
        interprocess_encoding_context=embedder.vocab.state,
    )
    field.mask(1.0)
    torch.manual_seed(0)
    cluster_logits = torch.randn(*field.state.shape, K)
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

    module_uniform = _TrackingModule(structure, embedder, decoder)
    loss(module=module_uniform, prediction=prediction, batch=field, strata=Strata.train)
    loss_uniform = module_uniform.tracked[(ADDRESS, Strata.train, "loss", TensorKey.content)].item()

    with torch.no_grad():
        content_counter.counts.fill_(1)
        content_counter.counts[0] = 1000

    module_skewed = _TrackingModule(structure, embedder, decoder)
    loss(module=module_skewed, prediction=prediction, batch=field, strata=Strata.train)
    loss_skewed = module_skewed.tracked[(ADDRESS, Strata.train, "loss", TensorKey.content)].item()

    assert loss_uniform != pytest.approx(loss_skewed, abs=1e-6)


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


def _seed_uncommitted_batch(*, revive_temperature: float | None = None):
    """Prepare a train batch + real embedder/decoder with a range-bounded cluster leaf.

    Seeds ``usage_ema`` to concentrate on the lower half so dynamic committed selection lands
    on ``[0, lower)``, marks the upper half of ``committed`` as dead (for callback consumers),
    and biases the cluster logits toward those dead columns so ``adherence_ema`` accumulates.
    """
    structure = Schema.model_validate(
        _structure_payload(
            bounds=(2, 4),
            capacity=8,
            p_unavailable=1.0,
            revive_temperature=revive_temperature,
        )
    )
    field = _prepared_train_batch(structure, values=[[["ALPHA", "BETA"]]] * 4)
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    K = structure.requests[ADDRESS].size
    lower = structure.requests[ADDRESS].n_clusters[0]
    with torch.no_grad():
        # Force perplexity ~ lower so dynamic n_committed lands at `lower` and topk picks [0, lower).
        embedder.usage_ema.zero_()
        embedder.usage_ema[:lower] = 1.0 / lower
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
    def __init__(self, is_global_zero: bool = True, current_epoch: int = 0):
        self.is_global_zero = is_global_zero
        self.current_epoch = current_epoch


def test_revive_callback_noop_when_revive_temperature_zero():
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_temperature=0.0)
    embedder.adherence_ema.fill_(0.9)
    prior_assign = embedder.embeddings[TensorKey.cluster.name].weight.detach().clone()
    prior_usage = embedder.usage_ema.detach().clone()

    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(), module)

    # revive_temperature == 0 -> callback is a no-op, no weights touched, ema preserved.
    assert torch.equal(embedder.embeddings[TensorKey.cluster.name].weight, prior_assign)
    assert torch.equal(embedder.usage_ema, prior_usage)
    assert embedder.adherence_ema.item() == pytest.approx(0.9)


def test_revive_callback_noop_when_bounds_are_fixed():
    structure = Schema.model_validate(
        _structure_payload(bounds=(4, 4), capacity=8, p_unavailable=1.0, revive_temperature=10.0)
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


def test_revive_callback_noop_when_all_signals_are_off():
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_temperature=10.0)
    # Past warmup, adherence == 0, committed usage concentrated on a single column so it is
    # not saturated. With every signal off the gate stays closed even under a live temperature.
    with torch.no_grad():
        embedder.usage_ema.zero_()
        embedder.usage_ema[0] = 1.0
    prior_assign = embedder.embeddings[TensorKey.cluster.name].weight.detach().clone()

    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(current_epoch=100), module)

    assert torch.equal(embedder.embeddings[TensorKey.cluster.name].weight, prior_assign)
    assert embedder.adherence_ema.item() == 0.0


def test_revive_callback_warmup_fires_without_adherence():
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_temperature=10.0)
    # During warmup the callback fires even with adherence == 0, seeding dead columns from the
    # top-usage committed donor(s) so the model can reach the upper bound without needing to
    # first accumulate mass on columns whose weights are still random.
    with torch.no_grad():
        embedder.usage_ema.zero_()
        embedder.usage_ema[0] = 1.0
    prior_assign = embedder.embeddings[TensorKey.cluster.name].weight.detach().clone()

    torch.manual_seed(0)
    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(current_epoch=0), module)

    assert not torch.equal(embedder.embeddings[TensorKey.cluster.name].weight, prior_assign)


def test_revive_callback_does_not_fire_on_saturated_committed_alone():
    # Saturation of the committed set (perfectly uniform usage on committed columns) is the
    # natural end state of Sinkhorn balance, not a signal that more clusters are needed. Past
    # warmup, with adherence == 0, saturation alone must NOT fire revive.
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_temperature=10.0)
    lower = module.schema.requests[ADDRESS].n_clusters[0]
    with torch.no_grad():
        embedder.usage_ema.zero_()
        embedder.usage_ema[:lower] = 1.0 / lower  # perfectly balanced -> "saturated"
        embedder.adherence_ema.zero_()
    prior_assign = embedder.embeddings[TensorKey.cluster.name].weight.detach().clone()

    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(current_epoch=100), module)

    assert torch.equal(embedder.embeddings[TensorKey.cluster.name].weight, prior_assign)


def test_revive_callback_splits_donor_into_dead_column_and_resets_state():
    torch.manual_seed(0)
    # bounds=(3, 4) yields exactly 1 uncommitted column so the donor is halved exactly once.
    structure = Schema.model_validate(
        _structure_payload(
            bounds=(3, 4), capacity=8, p_unavailable=1.0, revive_temperature=10.0
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

    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(current_epoch=0), module)

    # Symmetry-breaking noise is small, so the dead column is close to the donor.
    assert torch.allclose(assign_w.data[:, dead], donor_assign, atol=0.2)
    assert torch.allclose(content_w.data[:, dead], donor_content, atol=0.2)
    assert torch.allclose(cluster_w.data[dead, :], donor_cluster, atol=0.2)
    # Donor usage halved; revived column inherits half of donor's usage.
    assert embedder.usage_ema[0].item() == pytest.approx(0.5)
    assert embedder.usage_ema[dead].item() == pytest.approx(0.5)
    # adherence_ema reset after revive.
    assert embedder.adherence_ema.item() == 0.0


def test_revive_callback_respects_non_global_zero_rank():
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_temperature=10.0)
    embedder.adherence_ema.fill_(1.0)
    prior_assign = embedder.embeddings[TensorKey.cluster.name].weight.detach().clone()

    # No distributed backend is up -> broadcast_object is a passthrough, so a rank-non-zero
    # trainer produces an empty plans dict; no weights change.
    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(is_global_zero=False), module)

    assert torch.equal(embedder.embeddings[TensorKey.cluster.name].weight, prior_assign)
    assert embedder.adherence_ema.item() == pytest.approx(1.0)


def test_revive_callback_base_probability_decays_with_epoch():
    # At epoch >> revive_temperature the base probability decays to ~0 and revival never fires
    # even with saturated adherence.
    torch.manual_seed(0)
    structure = Schema.model_validate(
        _structure_payload(
            bounds=(3, 4), capacity=8, p_unavailable=1.0, revive_temperature=1.0
        )
    )
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    lower = structure.requests[ADDRESS].n_clusters[0]
    embedder.committed[lower:] = False
    embedder.adherence_ema.fill_(1.0)
    module = _TrackingModule(structure, embedder, decoder)
    prior_assign = embedder.embeddings[TensorKey.cluster.name].weight.detach().clone()

    # exp(-100/1) ≈ 3.7e-44, so the per-column Bernoulli trial cannot pass.
    ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(current_epoch=100), module)

    assert torch.equal(embedder.embeddings[TensorKey.cluster.name].weight, prior_assign)
    # Adherence stays intact because no revive occurred.
    assert embedder.adherence_ema.item() == pytest.approx(1.0)


def test_revive_callback_warmup_caps_expected_revivals_at_one_per_epoch():
    # During warmup the base probability is 1.0 and adherence is boosted to 1.0, but the
    # per-column cap is still ``1 / n_dead`` so the expected number of revivals in a single
    # epoch is ~1 (not "all dead columns"). This is what prevents ``n_committed`` from jumping
    # straight to the upper bound on the first epoch.
    structure = Schema.model_validate(
        _structure_payload(
            bounds=(1, 12), capacity=8, p_unavailable=1.0, revive_temperature=10.0
        )
    )
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    lower = structure.requests[ADDRESS].n_clusters[0]
    embedder.committed[:] = False
    embedder.committed[:lower] = True
    embedder.adherence_ema.zero_()  # warmup should drive the plan even without adherence

    n_dead = int((~embedder.committed).sum().item())
    assert n_dead == 11

    # ``_apply`` sets ``usage_ema[k] = usage_ema[donor] * 0.5`` for each revived dead column,
    # so counting non-zero entries after the callback is an exact proxy for revive count.
    revive_counts: list[int] = []
    module = _TrackingModule(structure, embedder, decoder)
    for seed in range(50):
        with torch.no_grad():
            embedder.usage_ema.zero_()
            embedder.usage_ema[0] = 1.0
        torch.manual_seed(seed)
        ClusterReviveCallback().on_train_epoch_end(_FakeTrainer(current_epoch=0), module)
        revived = int((embedder.usage_ema > 0).sum().item()) - 1  # minus the donor column
        revive_counts.append(revived)

    mean = sum(revive_counts) / len(revive_counts)
    # cap = 1/11, signal = 1 -> expected revivals per epoch = 1.
    assert 0.5 <= mean <= 2.0, f"expected mean ≈ 1, got {mean} (samples: {revive_counts[:10]})"
    # No single epoch should revive more than a small handful of columns.
    assert max(revive_counts) < 6


# ---------- ClusterMergeCallback ----------


def _seed_committed_pair(*, similarity: float):
    """Build a real embedder/decoder with two committed columns whose content_w columns and
    cluster_w rows share ``similarity`` cosine similarity (both weights, since the merge plan
    requires joint redundancy)."""
    structure = Schema.model_validate(
        _structure_payload(bounds=(2, 4), capacity=8, p_unavailable=1.0)
    )
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    lower = structure.requests[ADDRESS].n_clusters[0]
    K = structure.requests[ADDRESS].size

    # Commit only [0, 1, 2] so K==4 with one dead column; merge target is between 0 and 1.
    embedder.committed[:] = False
    embedder.committed[: lower + 1] = True
    assert int(embedder.committed.sum().item()) > lower

    torch.manual_seed(0)
    content_w = embedder.embeddings[TensorKey.content.name].weight
    cluster_w = decoder.linears[TensorKey.cluster.name].weight
    with torch.no_grad():
        content_w.data.normal_()
        cluster_w.data.normal_()
        # Blend column 1 (content) and row 1 (cluster) toward column/row 0.
        content_w.data[:, 1] = similarity * content_w.data[:, 0] + (
            1.0 - similarity
        ) * content_w.data[:, 1]
        cluster_w.data[1, :] = similarity * cluster_w.data[0, :] + (
            1.0 - similarity
        ) * cluster_w.data[1, :]
        # Winner is column 0 (higher usage) so the merge should decommit column 1.
        embedder.usage_ema.zero_()
        embedder.usage_ema[0] = 0.6
        embedder.usage_ema[1] = 0.3
        embedder.usage_ema[2] = 0.1

    module = _TrackingModule(structure, embedder, decoder)
    return structure, module, embedder, decoder, K


def test_merge_callback_decommits_duplicate_lower_usage_column():
    structure, module, embedder, decoder, _ = _seed_committed_pair(similarity=1.0)
    prior_usage_winner = float(embedder.usage_ema[0].item())
    prior_usage_loser = float(embedder.usage_ema[1].item())

    ClusterMergeCallback().on_train_epoch_end(_FakeTrainer(), module)

    # Column 1 (lower usage of the duplicated pair) is demoted; column 0 keeps its commit.
    assert not bool(embedder.committed[1].item())
    assert bool(embedder.committed[0].item())
    # Loser mass rolled into winner so perplexity can actually drop.
    assert embedder.usage_ema[1].item() == pytest.approx(0.0)
    assert embedder.usage_ema[0].item() == pytest.approx(
        prior_usage_winner + prior_usage_loser
    )


def test_merge_callback_zeros_loser_decoder_row():
    # The loser's decoder row is zeroed so the softmax cannot immediately reattract mass to it
    # via random noise; this lets ``n_committed`` actually shrink on the next epoch.
    _, module, embedder, decoder, _ = _seed_committed_pair(similarity=1.0)
    ClusterMergeCallback().on_train_epoch_end(_FakeTrainer(), module)

    loser_row = decoder.linears[TensorKey.cluster.name].weight.data[1, :]
    assert torch.equal(loser_row, torch.zeros_like(loser_row))


def test_merge_callback_noop_when_columns_are_distinct():
    _, module, embedder, decoder, _ = _seed_committed_pair(similarity=0.0)
    prior_committed = embedder.committed.detach().clone()
    prior_cluster_w = decoder.linears[TensorKey.cluster.name].weight.detach().clone()

    ClusterMergeCallback().on_train_epoch_end(_FakeTrainer(), module)

    # No pair exceeds the similarity threshold -> nothing changes.
    assert torch.equal(embedder.committed, prior_committed)
    assert torch.equal(
        decoder.linears[TensorKey.cluster.name].weight, prior_cluster_w
    )


def test_merge_callback_noop_when_bounds_are_fixed():
    # With a fixed range there is no "shrink room"; the merge plan must not fire even under
    # perfect redundancy so the model can rely on the full committed set.
    structure = Schema.model_validate(
        _structure_payload(bounds=(3, 3), capacity=8, p_unavailable=1.0)
    )
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    with torch.no_grad():
        content_w = embedder.embeddings[TensorKey.content.name].weight
        cluster_w = decoder.linears[TensorKey.cluster.name].weight
        content_w.data.normal_()
        cluster_w.data.normal_()
        content_w.data[:, 1] = content_w.data[:, 0]
        cluster_w.data[1, :] = cluster_w.data[0, :]
    prior_committed = embedder.committed.detach().clone()

    module = _TrackingModule(structure, embedder, decoder)
    ClusterMergeCallback().on_train_epoch_end(_FakeTrainer(), module)

    assert torch.equal(embedder.committed, prior_committed)


def test_merge_callback_noop_at_lower_bound():
    # ``n_committed == lower`` means the model already believes it's at the minimum plausible K
    # for this batch; refusing to shrink further prevents cannibalizing the last cluster.
    structure = Schema.model_validate(
        _structure_payload(bounds=(2, 4), capacity=8, p_unavailable=1.0)
    )
    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    lower = structure.requests[ADDRESS].n_clusters[0]
    # Only ``lower`` columns committed, and make them duplicates to prove the guard fires
    # before the similarity check.
    embedder.committed[:] = False
    embedder.committed[:lower] = True
    with torch.no_grad():
        content_w = embedder.embeddings[TensorKey.content.name].weight
        cluster_w = decoder.linears[TensorKey.cluster.name].weight
        content_w.data.normal_()
        cluster_w.data.normal_()
        content_w.data[:, 1] = content_w.data[:, 0]
        cluster_w.data[1, :] = cluster_w.data[0, :]
    prior_committed = embedder.committed.detach().clone()

    module = _TrackingModule(structure, embedder, decoder)
    ClusterMergeCallback().on_train_epoch_end(_FakeTrainer(), module)

    assert torch.equal(embedder.committed, prior_committed)


def test_merge_callback_respects_non_global_zero_rank():
    _, module, embedder, decoder, _ = _seed_committed_pair(similarity=1.0)
    prior_committed = embedder.committed.detach().clone()

    # Rank != 0 -> planning is skipped, no distributed backend means broadcast_object returns
    # the empty dict, and nothing is applied.
    ClusterMergeCallback().on_train_epoch_end(_FakeTrainer(is_global_zero=False), module)

    assert torch.equal(embedder.committed, prior_committed)


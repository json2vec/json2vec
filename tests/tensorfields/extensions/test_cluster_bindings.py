"""Tests for Cluster runtime state accessors."""

from __future__ import annotations

import pytest
import torch

import relflow as rf
from relflow.tensorfields.extensions.cluster import Embedder

ADDRESS = rf.Address("record/merchant_id")
CAPACITY = 6
N_CLUSTERS = 3


def _model() -> rf.Model:
    torch.manual_seed(0)
    return rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
        attention="none",
        merchant_id=rf.Cluster(
            capacity=CAPACITY,
            n_clusters=(2, N_CLUSTERS),
            p_mask=0.1,
            p_unavailable=0.0,
        ),
    )


def _embedder(model: rf.Model) -> Embedder:
    embedder = model.nodes[ADDRESS].embedder
    assert isinstance(embedder, Embedder)
    return embedder


def _learn(model: rf.Model, *tokens: str) -> None:
    model.encode(
        [{"merchant_id": token} for token in tokens],
        strata=rf.Strata.train,
        mask=False,
    )


def test_cluster_vocabulary_returns_immutable_model_snapshot() -> None:
    model = _model()
    _learn(model, "ALPHA", "BETA")

    snapshot = rf.Cluster.vocabulary(model, ADDRESS)

    assert snapshot == ("ALPHA", "BETA")
    assert isinstance(snapshot, tuple)

    _learn(model, "GAMMA")

    assert snapshot == ("ALPHA", "BETA")
    assert rf.Cluster.vocabulary(model, "record/merchant_id") == (
        "ALPHA",
        "BETA",
        "GAMMA",
    )


def test_cluster_vocabulary_reads_latest_encoding_context_snapshot() -> None:
    model = _model()
    encoding_context = model.interprocess_encoding_context

    _learn(model, "ALPHA", "BETA")

    assert rf.Cluster.vocabulary(encoding_context, ADDRESS) == ("ALPHA", "BETA")


def test_cluster_vocabulary_reads_same_length_context_replacement() -> None:
    model = _model()
    _learn(model, "ALPHA")
    encoding_context = model.interprocess_encoding_context

    _embedder(model).vocab.load_snapshot(["BETA"])

    assert rf.Cluster.vocabulary(encoding_context, ADDRESS) == ("BETA",)


def test_cluster_assignments_return_softmax_snapshots_without_sentinel() -> None:
    model = _model()
    # Cluster 2 starts uncommitted, but assignment probabilities and eval argmax
    # still span every configured cluster column.
    rf.Cluster.assign(model, ADDRESS, "ALPHA", [0.2, 0.1, 0.7])
    rf.Cluster.assign(model, ADDRESS, "BETA", [0.6, 0.1, 0.3])

    snapshot = rf.Cluster.assignments(model, ADDRESS)

    assert tuple(snapshot) == ("ALPHA", "BETA")
    assert snapshot["ALPHA"]["cluster"] == 2
    assert snapshot["ALPHA"]["probabilities"] == pytest.approx((0.2, 0.1, 0.7))
    assert snapshot["BETA"]["cluster"] == 0
    assert snapshot["BETA"]["probabilities"] == pytest.approx((0.6, 0.1, 0.3))
    assert isinstance(snapshot["ALPHA"]["probabilities"], tuple)

    rf.Cluster.assign(model, ADDRESS, "ALPHA", [0.8, 0.1, 0.1])

    assert snapshot["ALPHA"]["probabilities"] == pytest.approx((0.2, 0.1, 0.7))
    assert rf.Cluster.assignments(model, ADDRESS)["ALPHA"]["cluster"] == 0


def test_cluster_assignments_empty_vocabulary_is_empty() -> None:
    assert rf.Cluster.assignments(_model(), ADDRESS) == {}


def test_cluster_assignments_preserve_source_precision_for_eval_argmax() -> None:
    model = _model().double()
    _learn(model, "ALPHA")
    weight = _embedder(model).embeddings["cluster"].weight
    with torch.no_grad():
        weight[0].copy_(torch.tensor([1.0, 1.00000001, 0.0], dtype=torch.float64))

    snapshot = rf.Cluster.assignments(model, ADDRESS)["ALPHA"]

    assert snapshot["cluster"] == 1
    assert snapshot["probabilities"] == pytest.approx(tuple(torch.softmax(weight[0], dim=-1).tolist()))


def test_cluster_status_returns_detached_cpu_native_snapshot() -> None:
    model = _model()
    embedder = _embedder(model)
    with torch.no_grad():
        embedder.committed.copy_(torch.tensor([False, True, True]))
        embedder.usage_ema.copy_(torch.tensor([0.1, 0.2, 0.7]))

    snapshot = rf.Cluster.status(model, ADDRESS)

    assert snapshot["committed"] == (1, 2)
    assert snapshot["usage"] == pytest.approx((0.1, 0.2, 0.7))
    assert isinstance(snapshot["committed"], tuple)
    assert isinstance(snapshot["usage"], tuple)

    with torch.no_grad():
        embedder.committed.zero_()
        embedder.usage_ema.zero_()

    assert snapshot["committed"] == (1, 2)
    assert snapshot["usage"] == pytest.approx((0.1, 0.2, 0.7))


def test_cluster_status_preserves_usage_precision() -> None:
    model = _model().double()
    embedder = _embedder(model)
    expected = (0.1234567890123, 0.2345678901234, 0.6419753208643)
    with torch.no_grad():
        embedder.usage_ema.copy_(torch.tensor(expected, dtype=torch.float64))

    assert rf.Cluster.status(model, ADDRESS)["usage"] == expected


@pytest.mark.parametrize(
    "binding",
    [rf.Cluster.vocabulary, rf.Cluster.assignments, rf.Cluster.status],
    ids=["vocabulary", "assignments", "status"],
)
def test_cluster_model_bindings_reject_missing_address(binding) -> None:
    with pytest.raises(KeyError, match="missing"):
        binding(_model(), "record/missing")


@pytest.mark.parametrize(
    "binding",
    [rf.Cluster.vocabulary, rf.Cluster.assignments, rf.Cluster.status],
    ids=["vocabulary", "assignments", "status"],
)
def test_cluster_model_bindings_reject_non_cluster_field(binding) -> None:
    model = rf.Model(
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        amount=rf.Number,
    )

    with pytest.raises(TypeError, match="not a Cluster field"):
        binding(model, "record/amount")


def test_cluster_vocabulary_validates_context_resource_and_source() -> None:
    with pytest.raises(TypeError, match="VocabularyState"):
        rf.Cluster.vocabulary({ADDRESS: object()}, ADDRESS)

    with pytest.raises(TypeError, match="Model or InterprocessEncodingContext"):
        rf.Cluster.vocabulary(object(), ADDRESS)


@pytest.mark.parametrize("binding", [rf.Cluster.assignments, rf.Cluster.status])
def test_cluster_model_only_bindings_validate_source(binding) -> None:
    with pytest.raises(TypeError, match="must be a Model"):
        binding(object(), ADDRESS)

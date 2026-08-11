"""Tests for `Model.commit_cluster` and `Model.predict(cluster_hints=...)`.

These cover Path A (transient per-call hints) and Path B (persistent commits)
of the OOV cluster-assignment API.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.structs.enums import Tokens
from relflow.structs.tree import Address
from relflow.tensorfields.extensions.cluster import Embedder, TensorField

CAPACITY = 8
K = 4
ADDRESS = "record/merchant_id"


def _model() -> rf.Model:
    torch.manual_seed(0)
    return rf.Model(
        name="record",
        d_model=16,
        n_layers=1,
        n_heads=4,
        batch_size=4,
        merchant_id=rf.Cluster(capacity=CAPACITY, n_clusters=(K, K), p_unavailable=0.0),
    )


def _embedder(model: rf.Model) -> Embedder:
    embedder = model.nodes[Address(ADDRESS)].embedder
    assert isinstance(embedder, Embedder)
    return embedder


def _one_valued_token(index: int) -> TensorField:
    return TensorField(
        state=torch.tensor([[Tokens.valued.value]], dtype=torch.int64),
        content=torch.tensor([[index]], dtype=torch.int64),
        trainable=torch.zeros((1, 1), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )


# ---------- commit_cluster ----------


def test_commit_cluster_grows_vocab_and_writes_row_for_int_cluster():
    model = _model()
    embedder = _embedder(model)
    assert len(embedder.vocab.master) == 0

    index = model.commit_cluster(ADDRESS, "brand-new", cluster=2)

    assert index == 0
    assert list(embedder.vocab.master) == ["brand-new"]
    row = embedder.embeddings["cluster"].weight[index]
    assert int(row.argmax().item()) == 2


def test_commit_cluster_accepts_probability_vector():
    model = _model()
    embedder = _embedder(model)
    probs = [0.05, 0.05, 0.6, 0.3]

    model.commit_cluster(ADDRESS, "soft-token", probs=probs)

    row = embedder.embeddings["cluster"].weight[0]
    # A probability vector round-trips to logits via log(probs); argmax must land on 2.
    assert int(row.argmax().item()) == 2


def test_commit_cluster_overwrites_existing_token():
    model = _model()
    embedder = _embedder(model)
    idx = model.commit_cluster(ADDRESS, "shared-token", cluster=1)
    assert int(embedder.embeddings["cluster"].weight[idx].argmax().item()) == 1

    idx_again = model.commit_cluster(ADDRESS, "shared-token", cluster=3)

    assert idx_again == idx
    assert len(embedder.vocab.master) == 1
    assert int(embedder.embeddings["cluster"].weight[idx].argmax().item()) == 3


def test_commit_cluster_raises_when_capacity_full():
    model = _model()
    embedder = _embedder(model)
    for i in range(CAPACITY):
        model.commit_cluster(ADDRESS, f"tok-{i}", cluster=i % K)
    assert len(embedder.vocab.master) == CAPACITY

    with pytest.raises(ValueError, match="at capacity"):
        model.commit_cluster(ADDRESS, "one-too-many", cluster=0)


def test_commit_cluster_rejects_out_of_range_cluster():
    model = _model()

    with pytest.raises(ValueError, match=r"cluster must be in \[0, 4\)"):
        model.commit_cluster(ADDRESS, "tok", cluster=K)


def test_commit_cluster_rejects_conflicting_arguments():
    model = _model()

    with pytest.raises(ValueError, match="exactly one"):
        model.commit_cluster(ADDRESS, "tok", cluster=0, probs=[1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="exactly one"):
        model.commit_cluster(ADDRESS, "tok")


def test_commit_cluster_rejects_bad_probs_shape():
    model = _model()

    with pytest.raises(ValueError, match="shape"):
        model.commit_cluster(ADDRESS, "tok", probs=[1.0, 0.0])


def test_commit_cluster_rejects_probs_that_do_not_sum_to_one():
    model = _model()

    with pytest.raises(ValueError, match="sum to 1"):
        model.commit_cluster(ADDRESS, "tok", probs=[0.5, 0.3, 0.1, 0.0])


def test_commit_cluster_rejects_negative_probs():
    model = _model()

    with pytest.raises(ValueError, match="non-negative"):
        model.commit_cluster(ADDRESS, "tok", probs=[1.5, -0.5, 0.0, 0.0])


def test_commit_cluster_raises_for_unknown_address():
    model = _model()

    with pytest.raises(KeyError):
        model.commit_cluster("record/nonexistent", "tok", cluster=0)


def test_commit_cluster_raises_for_non_cluster_field():
    torch.manual_seed(0)
    model = rf.Model(
        name="record",
        d_model=16,
        n_layers=1,
        n_heads=4,
        batch_size=4,
        amount=rf.Number,
    )

    with pytest.raises(TypeError, match="not a Cluster field"):
        model.commit_cluster("record/amount", "tok", cluster=0)


# ---------- predict(cluster_hints=...) ----------


def test_predict_cluster_hints_apply_transiently():
    model = _model()
    embedder = _embedder(model)
    initial_vocab_size = len(embedder.vocab.master)
    initial_row_2 = embedder.embeddings["cluster"].weight[
        min(initial_vocab_size, CAPACITY)
    ].detach().clone()

    model.predict(
        [{"merchant_id": "hinted"}],
        cluster_hints={ADDRESS: {"hinted": 3}},
    )

    assert len(embedder.vocab.master) == initial_vocab_size
    # The row that got temporarily written must be reverted to its original values.
    reverted_row = embedder.embeddings["cluster"].weight[
        min(initial_vocab_size, CAPACITY)
    ]
    assert torch.equal(reverted_row, initial_row_2)


def test_predict_cluster_hints_override_encoder_row_for_oov_token():
    """The encoder must consume the hinted assign row, not the sentinel, for hinted OOV.

    Hints act on the encoder path; the write() cluster.value comes from the
    decoder (context-driven), so we probe the embedder's forward output directly.
    """
    model = _model()
    embedder = _embedder(model)
    embedder.eval()

    with torch.no_grad():
        embedder.embeddings["cluster"].weight[CAPACITY].zero_()  # dull sentinel
        embedder.embeddings["cluster"].weight[CAPACITY, 0] = 10.0  # sentinel argmax = 0

    with embedder.transient_commit({"hinted-oov": 2}):
        # During the hint, "hinted-oov" is at index 0 and encoded as valued/index 0.
        hinted_index = embedder.vocab.state.index["hinted-oov"]
        hinted_output = embedder(_one_valued_token(hinted_index)).payload.clone()

    # Reference: forward pass for a hand-built row that argmaxes to cluster 2.
    with torch.no_grad():
        reference_row = torch.full((K,), -10.0)
        reference_row[2] = 10.0
        embedder.embeddings["cluster"].weight[0].copy_(reference_row)
    reference_output = embedder(_one_valued_token(0)).payload.clone()

    # Both should route through cluster 2 in eval mode (argmax), so the cluster
    # embedding contribution must match. State embedding is identical either way.
    assert torch.allclose(hinted_output, reference_output)


def test_predict_cluster_hints_accept_probability_vector():
    model = _model()
    embedder = _embedder(model)
    embedder.eval()

    with embedder.transient_commit({"soft": [0.0, 0.1, 0.9, 0.0]}):
        index = embedder.vocab.state.index["soft"]
        # In eval mode, forward hard-argmaxes assign_logits. probs=[0,0.1,0.9,0]
        # produces logits log(probs); argmax is cluster 2.
        assign_logits = embedder.embeddings["cluster"](torch.tensor([index]))
        assert int(assign_logits.argmax(dim=-1).item()) == 2


def test_commit_cluster_persists_across_predict_calls():
    model = _model()
    embedder = _embedder(model)
    embedder.eval()

    model.commit_cluster(ADDRESS, "returning-token", cluster=3)
    committed_index = embedder.vocab.state.index["returning-token"]
    first_row = embedder.embeddings["cluster"].weight[committed_index].detach().clone()

    # Simulate a subsequent predict call by running the embedder again.
    _ = embedder(_one_valued_token(committed_index))
    second_row = embedder.embeddings["cluster"].weight[committed_index].detach().clone()

    assert torch.equal(first_row, second_row)
    assert int(first_row.argmax().item()) == 3

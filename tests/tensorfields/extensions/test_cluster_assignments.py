"""Tests for persistent Cluster assignments and scoped overrides."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.structs.enums import Strata, Tokens
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


# ---------- persistent assignments ----------


def test_assign_grows_vocab_and_writes_row_for_int_cluster():
    model = _model()
    embedder = _embedder(model)
    assert len(embedder.vocab.master) == 0

    index = rf.Cluster.assign(model, ADDRESS, "brand-new", 2)

    assert index == 0
    assert list(embedder.vocab.master) == ["brand-new"]
    row = embedder.embeddings["cluster"].weight[index]
    assert int(row.argmax().item()) == 2


def test_assign_accepts_probability_vector():
    model = _model()
    embedder = _embedder(model)
    probs = [0.05, 0.05, 0.6, 0.3]

    rf.Cluster.assign(model, ADDRESS, "soft-token", probs)

    row = embedder.embeddings["cluster"].weight[0]
    # A probability vector round-trips to logits via log(probs); argmax must land on 2.
    assert int(row.argmax().item()) == 2


def test_assign_overwrites_existing_token():
    model = _model()
    embedder = _embedder(model)
    idx = rf.Cluster.assign(model, ADDRESS, "shared-token", 1)
    assert int(embedder.embeddings["cluster"].weight[idx].argmax().item()) == 1

    idx_again = rf.Cluster.assign(model, ADDRESS, "shared-token", 3)

    assert idx_again == idx
    assert len(embedder.vocab.master) == 1
    assert int(embedder.embeddings["cluster"].weight[idx].argmax().item()) == 3


def test_assign_raises_when_capacity_full():
    model = _model()
    embedder = _embedder(model)
    for i in range(CAPACITY):
        rf.Cluster.assign(model, ADDRESS, f"tok-{i}", i % K)
    assert len(embedder.vocab.master) == CAPACITY

    with pytest.raises(ValueError, match="at capacity"):
        rf.Cluster.assign(model, ADDRESS, "one-too-many", 0)


def test_assign_rejects_out_of_range_cluster():
    model = _model()

    with pytest.raises(ValueError, match=r"cluster must be in \[0, 4\)"):
        rf.Cluster.assign(model, ADDRESS, "tok", K)


def test_assign_requires_exactly_one_assignment():
    model = _model()

    with pytest.raises(TypeError):
        rf.Cluster.assign(model, ADDRESS, "tok")
    with pytest.raises(TypeError):
        rf.Cluster.assign(model, ADDRESS, "tok", 0, [1.0, 0.0, 0.0, 0.0])


def test_assign_rejects_bad_probability_shape():
    model = _model()

    with pytest.raises(ValueError, match="shape"):
        rf.Cluster.assign(model, ADDRESS, "tok", [1.0, 0.0])


def test_assign_rejects_probabilities_that_do_not_sum_to_one():
    model = _model()

    with pytest.raises(ValueError, match="sum to 1"):
        rf.Cluster.assign(model, ADDRESS, "tok", [0.5, 0.3, 0.1, 0.0])


def test_assign_rejects_negative_probabilities():
    model = _model()

    with pytest.raises(ValueError, match="non-negative"):
        rf.Cluster.assign(model, ADDRESS, "tok", [1.5, -0.5, 0.0, 0.0])


def test_assign_raises_for_unknown_address():
    model = _model()

    with pytest.raises(KeyError):
        rf.Cluster.assign(model, "record/nonexistent", "tok", 0)


def test_assign_raises_for_non_cluster_field():
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
        rf.Cluster.assign(model, "record/amount", "tok", 0)


# ---------- scoped overrides ----------


def test_override_applies_only_within_prediction_scope():
    model = _model()
    embedder = _embedder(model)
    initial_vocab_size = len(embedder.vocab.master)
    initial_row_2 = embedder.embeddings["cluster"].weight[min(initial_vocab_size, CAPACITY)].detach().clone()

    with rf.Cluster.override(model, ADDRESS, {"overridden": 3}):
        model.predict([{"merchant_id": "overridden"}])

    assert len(embedder.vocab.master) == initial_vocab_size
    # The row that got temporarily written must be reverted to its original values.
    reverted_row = embedder.embeddings["cluster"].weight[min(initial_vocab_size, CAPACITY)]
    assert torch.equal(reverted_row, initial_row_2)


def test_override_is_used_by_tensorization():
    model = _model()

    with rf.Cluster.override(model, ADDRESS, {"overridden": 3}):
        overridden = model.encode([{"merchant_id": "overridden"}])[Address(ADDRESS)]
        assert int(overridden.content.item()) == 0
        assert int(overridden.state.item()) == Tokens.valued.value

    unavailable = model.encode([{"merchant_id": "overridden"}])[Address(ADDRESS)]
    assert int(unavailable.content.item()) == CAPACITY


def test_override_rolls_back_when_prediction_scope_raises():
    model = _model()
    embedder = _embedder(model)
    original = embedder.embeddings["cluster"].weight[0].detach().clone()

    with pytest.raises(RuntimeError, match="prediction failed"):
        with rf.Cluster.override(model, ADDRESS, {"overridden": 3}):
            assert embedder.vocab.state.index["overridden"] == 0
            raise RuntimeError("prediction failed")

    assert "overridden" not in embedder.vocab.state.index
    assert torch.equal(embedder.embeddings["cluster"].weight[0], original)


def test_override_rolls_back_when_entering_scope_raises():
    model = _model()
    embedder = _embedder(model)
    original = embedder.embeddings["cluster"].weight[0].detach().clone()

    with pytest.raises(ValueError, match="shape"):
        with rf.Cluster.override(model, ADDRESS, {"valid": 2, "invalid": [1.0, 0.0]}):
            pytest.fail("invalid overrides should fail before entering the scope")

    assert list(embedder.vocab.master) == []
    assert torch.equal(embedder.embeddings["cluster"].weight[0], original)


def test_override_restores_an_existing_assignment():
    model = _model()
    embedder = _embedder(model)
    index = rf.Cluster.assign(model, ADDRESS, "existing", 1)
    original = embedder.embeddings["cluster"].weight[index].detach().clone()

    with rf.Cluster.override(model, ADDRESS, {"existing": 3}):
        assert int(embedder.embeddings["cluster"].weight[index].argmax().item()) == 3

    assert list(embedder.vocab.master) == ["existing"]
    assert embedder.vocab.state.index["existing"] == index
    assert torch.equal(embedder.embeddings["cluster"].weight[index], original)


def test_override_blocks_checkpointing_and_model_mutation(tmp_path: Path):
    model = _model()

    with rf.Cluster.override(model, ADDRESS, {"temporary": 2}):
        with pytest.raises(RuntimeError, match="assignment overrides are active"):
            model.save(tmp_path / "overridden.rf")
        with pytest.raises(RuntimeError, match="active loop"):
            model.reset(ADDRESS)

    assert "temporary" not in _embedder(model).vocab.state.index


def test_runtime_assignments_reject_active_models_and_shared_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _model()
    embedder = _embedder(model)

    model.locks[Strata.train] += 1
    try:
        with pytest.raises(RuntimeError, match="model is active"):
            rf.Cluster.assign(model, ADDRESS, "active", 1)
    finally:
        model.locks.pop(Strata.train)

    monkeypatch.setattr(type(embedder.vocab), "is_shared", property(lambda self: True))
    with pytest.raises(RuntimeError, match="vocabulary state is shared"):
        with rf.Cluster.override(model, ADDRESS, {"shared": 1}):
            pytest.fail("shared vocabulary should fail before entering the scope")


def test_override_replaces_encoder_row_for_oov_token():
    """The encoder consumes the override row instead of the OOV sentinel.

    Overrides act on the encoder path; the write() cluster.value comes from the
    decoder, so probe the embedder's forward output directly.
    """
    model = _model()
    embedder = _embedder(model)
    embedder.eval()

    with torch.no_grad():
        embedder.embeddings["cluster"].weight[CAPACITY].zero_()  # dull sentinel
        embedder.embeddings["cluster"].weight[CAPACITY, 0] = 10.0  # sentinel argmax = 0

    with rf.Cluster.override(model, ADDRESS, {"overridden-oov": 2}):
        # During the override, this token is encoded as valued/index 0.
        overridden_index = embedder.vocab.state.index["overridden-oov"]
        overridden_output = embedder(_one_valued_token(overridden_index)).payload.clone()

    # Reference: forward pass for a hand-built row that argmaxes to cluster 2.
    with torch.no_grad():
        reference_row = torch.full((K,), -10.0)
        reference_row[2] = 10.0
        embedder.embeddings["cluster"].weight[0].copy_(reference_row)
    reference_output = embedder(_one_valued_token(0)).payload.clone()

    # Both should route through cluster 2 in eval mode (argmax), so the cluster
    # embedding contribution must match. State embedding is identical either way.
    assert torch.allclose(overridden_output, reference_output)


def test_override_accepts_probability_vector():
    model = _model()
    embedder = _embedder(model)
    embedder.eval()

    with rf.Cluster.override(model, ADDRESS, {"soft": [0.0, 0.1, 0.9, 0.0]}):
        index = embedder.vocab.state.index["soft"]
        # In eval mode, forward hard-argmaxes assignment logits. This distribution
        # produces log probabilities whose argmax is cluster 2.
        assign_logits = embedder.embeddings["cluster"](torch.tensor([index]))
        assert int(assign_logits.argmax(dim=-1).item()) == 2


def test_persistent_assignment_survives_predict_calls():
    model = _model()
    embedder = _embedder(model)
    embedder.eval()

    rf.Cluster.assign(model, ADDRESS, "returning-token", 3)
    assigned_index = embedder.vocab.state.index["returning-token"]
    first_row = embedder.embeddings["cluster"].weight[assigned_index].detach().clone()

    # Simulate a subsequent predict call by running the embedder again.
    _ = embedder(_one_valued_token(assigned_index))
    second_row = embedder.embeddings["cluster"].weight[assigned_index].detach().clone()

    assert torch.equal(first_row, second_row)
    assert int(first_row.argmax().item()) == 3


def test_persistent_assignment_round_trips_through_checkpoint(tmp_path: Path):
    model = _model()
    index = rf.Cluster.assign(model, ADDRESS, "persistent", 2)
    pathname = tmp_path / "cluster.rf"

    model.save(pathname)
    restored = rf.Model.load(pathname)
    embedder = _embedder(restored)

    assert embedder.vocab.state.index["persistent"] == index
    assert int(embedder.embeddings["cluster"].weight[index].argmax().item()) == 2


def test_assign_resolves_the_live_embedder_after_reset():
    model = _model()
    original = _embedder(model)

    model.reset(rf.where("address") == ADDRESS)
    rebuilt = _embedder(model)
    assert rebuilt is not original

    index = rf.Cluster.assign(model, ADDRESS, "after-reset", 2)
    assert rebuilt.vocab.master[index] == "after-reset"
    assert int(rebuilt.embeddings["cluster"].weight[index].argmax().item()) == 2


def test_cluster_api_does_not_leak_into_model_or_runtime_signatures():
    assert not hasattr(rf.Cluster, "bind")
    assert "cluster_hints" not in inspect.signature(rf.Model.predict).parameters
    assert not hasattr(rf.Model, "commit_cluster")
    assert not hasattr(rf.Model, "_cluster_embedder")

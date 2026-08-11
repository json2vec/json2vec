"""End-to-end reproducers for cluster K-selection.

Three scenarios that MUST hold for the dynamic-K mechanism to be considered
working:

1. Categorical downstream (``merchant_id``): repeated IDs share a label; the
   model discovers K=5 via label-driven pooling.
2. Regime downstream (``regime``): repeated IDs each belong to one of K
   hidden ``y = f_k(x)`` relations; the model discovers K=5 via functional
   pooling with no explicit label supervision.
3. Dormant field (plain input): loss is never called; K is frozen at init.
   This documents the actual root cause of the "anchored at init" symptom
   reported in benchmarks with unique-per-observation IDs.
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pytest
import torch

import relflow as rf
from relflow.tensorfields.extensions.cluster import Embedder

TRUE_K: int = 5
IDS_PER_CLUSTER: int = 10
OBS_PER_ID: int = 20
LOWER: int = 3
UPPER: int = 15


def _regime_records(
    *,
    num_ids: int = 100,
    obs_per_id: int = 30,
    seed: int = 42,
) -> pl.DataFrame:
    """Repeated IDs, each mapped to one of K hidden ``y = f_k(x)`` regimes.

    Unlike the naive "unique id per row" version that made the mechanism dormant,
    each ID here appears ``obs_per_id`` times so the cluster head can pool
    gradient across observations of the same ID.
    """
    rng = np.random.default_rng(seed)
    epsilon = 0.05
    x_low, x_high = 0.0, 10.0

    relations = [
        lambda x: np.sin(x) + 10.0,
        lambda x: np.cos(x) + 4.0,
        lambda x: np.zeros_like(x) - 2.0,
        lambda x: 0.5 * x - 8.0,
        lambda x: -0.3 * x - 14.0,
    ]
    assert len(relations) == TRUE_K

    id_to_cluster = rng.integers(0, TRUE_K, size=num_ids)

    n = num_ids * obs_per_id
    id_indices = np.repeat(np.arange(num_ids), obs_per_id)
    cluster_indices = id_to_cluster[id_indices]
    x = rng.uniform(x_low, x_high, size=n) + rng.normal(0.0, epsilon, size=n)
    y = np.empty(n)
    for k, f in enumerate(relations):
        mask = cluster_indices == k
        y[mask] = f(x[mask]) + rng.normal(0.0, epsilon, size=int(mask.sum()))

    perm = rng.permutation(n)
    return pl.DataFrame(
        {
            "id": [f"id-{i:04d}" for i in id_indices[perm].tolist()],
            "x": x[perm].tolist(),
            "y": y[perm].tolist(),
        }
    )


def _synthetic_records(seed: int = 0) -> pl.DataFrame:
    rng = torch.Generator().manual_seed(seed)
    rows: list[dict[str, object]] = []
    for cluster_idx in range(TRUE_K):
        for id_idx in range(IDS_PER_CLUSTER):
            merchant_id = f"c{cluster_idx}-id{id_idx}"
            for _ in range(OBS_PER_ID):
                rows.append(
                    {
                        "merchant_id": merchant_id,
                        "label": f"L{cluster_idx}",
                    }
                )
    indices = torch.randperm(len(rows), generator=rng).tolist()
    return pl.DataFrame([rows[i] for i in indices])


class _CommittedTrajectory(lit.Callback):
    def __init__(self, address: rf.Address) -> None:
        self.address = address
        self.trajectory: list[dict[str, float]] = []

    def on_train_epoch_end(self, trainer: lit.Trainer, pl_module) -> None:  # type: ignore[override]
        embedder = pl_module.nodes[self.address].embedder
        assert isinstance(embedder, Embedder)
        usage = embedder.usage_ema.detach()
        usage_norm = usage / usage.sum().clamp_min(1e-12)
        entropy = -(usage_norm.clamp_min(1e-12) * usage_norm.clamp_min(1e-12).log()).sum()
        self.trajectory.append(
            {
                "epoch": trainer.current_epoch,
                "n_committed": int(embedder.committed.sum().item()),
                "perplexity": float(torch.exp(entropy).item()),
                "adherence": float(embedder.adherence_ema.item()),
            }
        )


def _format(trajectory: list[dict[str, float]]) -> str:
    header = f"{'epoch':>5} {'n_committed':>12} {'perplexity':>11} {'adherence':>10}"
    rows = [
        f"{r['epoch']:>5.0f} {r['n_committed']:>12.0f} {r['perplexity']:>11.3f} {r['adherence']:>10.4f}"
        for r in trajectory
    ]
    return "\n".join([header, *rows])


@pytest.mark.skipif("not config.getoption('--run-slow')", reason="Only run when --run-slow is given")
def test_cluster_n_committed_converges_to_true_k_when_loss_engaged():
    """When the loss is engaged (via ``p_mask``), K must converge close to true K.

    The mechanism should reach K in [true_K - 1, true_K + 2] within 30 epochs,
    and MUST NOT stick at either boundary.
    """
    torch.manual_seed(0)
    records = _synthetic_records()

    model = rf.Model(
        name="event",
        d_model=32,
        n_layers=1,
        n_heads=4,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        merchant_id=rf.Cluster(capacity=64, n_clusters=(LOWER, UPPER), p_mask=0.5),
        label=rf.Category(target=True, size=TRUE_K, p_unavailable=0.0),
    )

    datamodule = rf.PolarsDataModule(
        model=model,
        train=records,
        validate=records,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    address = rf.Address("event", "merchant_id")
    trajectory = _CommittedTrajectory(address)

    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=30,
        callbacks=[trajectory],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=datamodule)

    tail = trajectory.trajectory[-5:]
    final_n = [row["n_committed"] for row in tail]
    final_ppl = [row["perplexity"] for row in tail]

    # Strict window: exactly the true K, with narrow tolerance for optimizer noise.
    assert all(TRUE_K - 1 <= n <= TRUE_K + 2 for n in final_n), (
        f"n_committed did not converge near true K={TRUE_K} over final 5 epochs: "
        f"{final_n}\nFull trajectory:\n{_format(trajectory.trajectory)}"
    )
    assert all(abs(p - TRUE_K) <= 1.5 for p in final_ppl), (
        f"usage perplexity did not settle near true K={TRUE_K}: {final_ppl}\n"
        f"Full trajectory:\n{_format(trajectory.trajectory)}"
    )
    # Explicit boundary checks — the failure modes that inspired this test.
    assert final_n[-1] != UPPER, (
        f"n_committed glued to upper bound: {final_n}\nFull trajectory:\n{_format(trajectory.trajectory)}"
    )
    assert final_n[-1] != LOWER or LOWER == TRUE_K, (
        f"n_committed glued to lower bound (mechanism dormant): {final_n}\n"
        f"Full trajectory:\n{_format(trajectory.trajectory)}"
    )


@pytest.mark.skipif("not config.getoption('--run-slow')", reason="Only run when --run-slow is given")
def test_cluster_loss_is_dormant_when_field_is_plain_input():
    """A plain-input ``Cluster`` never engages the loss, so ``usage_ema`` never moves.

    This is the actual root cause of the "n_committed anchored at init" symptom
    reported in benchmarks: if the user does not opt into training the cluster
    (via ``p_mask``, ``p_prune``, or ``target=True``), K is frozen at whatever
    the embedder was initialized to.
    """
    torch.manual_seed(0)
    records = _synthetic_records()

    model = rf.Model(
        name="event",
        d_model=32,
        n_layers=1,
        n_heads=4,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        merchant_id=rf.Cluster(capacity=64, n_clusters=(LOWER, UPPER)),
        label=rf.Category(target=True, size=TRUE_K, p_unavailable=0.0),
    )

    datamodule = rf.PolarsDataModule(
        model=model,
        train=records,
        validate=records,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    address = rf.Address("event", "merchant_id")
    trajectory = _CommittedTrajectory(address)

    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=5,
        callbacks=[trajectory],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=datamodule)

    n_series = [row["n_committed"] for row in trajectory.trajectory]
    adherence = [row["adherence"] for row in trajectory.trajectory]
    assert len(set(n_series)) == 1, f"expected frozen n_committed for dormant cluster, got {n_series}"
    assert all(a == 0.0 for a in adherence), f"expected zero adherence for dormant cluster, got {adherence}"


@pytest.mark.skipif("not config.getoption('--run-slow')", reason="Only run when --run-slow is given")
def test_cluster_n_committed_converges_via_regime_regression():
    """Repeated IDs, each tagged to one of 5 hidden ``y = f_k(x)`` regimes.

    The model sees no cluster label — only ``(id, x, y)``. It must discover
    that IDs partition into K=5 regimes purely from the pressure to predict
    ``y`` from ``x`` plus the masked-id reconstruction signal. This is the
    corrected form of the benchmark scenario that originally motivated the
    "anchored at upper bound" investigation.
    """
    torch.manual_seed(0)
    records = _regime_records(num_ids=100, obs_per_id=30)

    model = rf.Model(
        name="obs",
        d_model=32,
        n_layers=1,
        n_heads=4,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        id=rf.Cluster(capacity=128, n_clusters=(LOWER, UPPER), p_mask=0.5),
        x=rf.Number,
        y=rf.Number(target=True),
    )

    datamodule = rf.PolarsDataModule(
        model=model,
        train=records,
        validate=records,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    address = rf.Address("obs", "id")
    trajectory = _CommittedTrajectory(address)

    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=30,
        callbacks=[trajectory],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=datamodule)

    tail = trajectory.trajectory[-5:]
    final_n = [row["n_committed"] for row in tail]
    final_ppl = [row["perplexity"] for row in tail]

    assert all(TRUE_K - 1 <= n <= TRUE_K + 2 for n in final_n), (
        f"n_committed did not converge near true K={TRUE_K} over final 5 epochs: "
        f"{final_n}\nFull trajectory:\n{_format(trajectory.trajectory)}"
    )
    assert all(abs(p - TRUE_K) <= 1.5 for p in final_ppl), (
        f"usage perplexity did not settle near true K={TRUE_K}: {final_ppl}\n"
        f"Full trajectory:\n{_format(trajectory.trajectory)}"
    )
    assert final_n[-1] != UPPER, (
        f"n_committed glued to upper bound: {final_n}\nFull trajectory:\n{_format(trajectory.trajectory)}"
    )
    assert final_n[-1] != LOWER, (
        f"n_committed glued to lower bound: {final_n}\nFull trajectory:\n{_format(trajectory.trajectory)}"
    )

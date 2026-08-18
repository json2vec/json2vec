"""Integration test: Lightning must reset per-strata state metrics between epochs.

The `Plugin.state_tracking` refactor made state accuracy a stateful
`torchmetrics.Metric` accumulated inside `decoder.state_metrics`, replacing the
per-batch mean-of-hits computation. That change is only semantically correct if
Lightning actually calls `.reset()` on the tracker at each epoch boundary; if it
did not, epoch-level compute values would run away as history accumulated.

This test drives a real `lit.Trainer.fit()` for two epochs and asserts that at
the start of every epoch the tracker's `_update_count` is 0, confirming that
Lightning's auto-reset covers the nested `ModuleDict` layout that
`Plugin.build_state_metric_registry` produces.
"""

from __future__ import annotations

import lightning.pytorch as lit
import polars as pl
import torch

import relflow as rf


class _StateMetricSnapshotter(lit.Callback):
    """Capture `state_metrics[train]['accuracy']._update_count` at epoch start."""

    def __init__(self, address: rf.Address) -> None:
        self.address = address
        self.snapshots: list[tuple[int, int]] = []

    def on_train_epoch_start(self, trainer: lit.Trainer, pl_module) -> None:  # type: ignore[override]
        decoder = pl_module.nodes[self.address].decoder
        tracker = decoder.state_metrics["train_state_metrics"]["accuracy"]
        self.snapshots.append((trainer.current_epoch, int(tracker._update_count)))


def test_state_metrics_reset_between_epochs():
    torch.manual_seed(0)
    records = pl.DataFrame(
        {
            "amount": [float(i) for i in range(16)],
            "label": [bool(i % 2) for i in range(16)],
        }
    )

    model = rf.Model(
        name="event",
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=8,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-3),
        amount=rf.Number,
        label=rf.Boolean(target=True),
    )

    datamodule = rf.PolarsDataModule(
        model=model,
        train=records,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    address = rf.Address("event", "label")
    snapshotter = _StateMetricSnapshotter(address)

    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=2,
        callbacks=[snapshotter],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    trainer.fit(model=model, datamodule=datamodule)

    # Both epoch-start snapshots must show a freshly reset tracker.
    # If Lightning's auto-reset failed on the nested ModuleDict, epoch 1's
    # snapshot would carry over epoch 0's batch count.
    assert snapshotter.snapshots == [(0, 0), (1, 0)], (
        f"Expected [(0, 0), (1, 0)] (fresh trackers at each epoch start); "
        f"got {snapshotter.snapshots}. "
        "This indicates Lightning did not reset the stateful state-metric "
        "tracker between epochs; epoch-level accuracy would drift because "
        "history accumulates across epochs."
    )

from __future__ import annotations

import time
from collections import defaultdict
from functools import partialmethod
from typing import TYPE_CHECKING, Any

import torch
from lightning import Callback, Trainer

from relflow.structs.enums import Metric, Strata

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.datasets.base import EncodedInput


def observation_count(batch: EncodedInput) -> int:
    """Return the leading batch dimension from an encoded input."""

    if batch.batch_size:
        return batch.batch_size[0]

    for field in batch.values():
        batch_size = getattr(field, "batch_size", ())
        if batch_size:
            return batch_size[0]

    raise ValueError("encoded batch has no batch dimension")


class ThroughputLogger(Callback):
    def __init__(self):
        super().__init__()

        self.timestamp: dict[Strata, float] = defaultdict(time.perf_counter)
        self.observations: dict[Strata, int] = defaultdict(int)
        self.throughput: dict[Strata, float] = {}

    def start(self, trainer: Trainer, pl_module: Model, strata: Strata) -> None:
        self.timestamp[strata] = time.perf_counter()
        self.observations[strata] = 0

    def count(
        self,
        trainer: Trainer,
        pl_module: Model,
        outputs: Any,
        batch: EncodedInput,
        batch_idx: int,
        dataloader_idx: int = 0,
        *,
        strata: Strata,
    ) -> None:
        self.observations[strata] += observation_count(batch)

    def end(self, trainer: Trainer, pl_module: Model, strata: Strata) -> None:
        device = getattr(pl_module, "device", torch.device("cpu"))
        observations = torch.tensor(self.observations[strata], dtype=torch.int64, device=device)
        elapsed = torch.tensor(time.perf_counter() - self.timestamp[strata], dtype=torch.float32, device=device)

        strategy = getattr(trainer, "strategy", None)
        if strategy is not None:
            observations = strategy.reduce(observations, reduce_op="sum")
            elapsed = strategy.reduce(elapsed, reduce_op="max")

        elapsed_seconds = float(elapsed.item())
        throughput = float(observations.item()) / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        self.throughput[strata] = throughput

        # Lightning does not register a result collection for prediction hooks,
        # so LightningModule.log()/Model.track() raises at predict epoch end.
        # Send the scalar straight to an attached logger and retain it on this
        # callback for runtimes without a logger.
        if strata == Strata.predict:
            logger = getattr(trainer, "logger", None)
            if logger is not None and getattr(trainer, "is_global_zero", True):
                logger.log_metrics(
                    {f"{Metric.throughput.value}/{strata.value}": throughput},
                    step=getattr(trainer, "global_step", None),
                )
            return

        pl_module.log(
            f"{Metric.throughput.value}/{strata.value}",
            torch.tensor(throughput, device=device),
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

    on_train_epoch_start = partialmethod(start, strata=Strata.train)
    on_validation_epoch_start = partialmethod(start, strata=Strata.validate)
    on_test_epoch_start = partialmethod(start, strata=Strata.test)
    on_predict_epoch_start = partialmethod(start, strata=Strata.predict)

    on_train_batch_end = partialmethod(count, strata=Strata.train)
    on_validation_batch_end = partialmethod(count, strata=Strata.validate)
    on_test_batch_end = partialmethod(count, strata=Strata.test)
    on_predict_batch_end = partialmethod(count, strata=Strata.predict)

    on_train_epoch_end = partialmethod(end, strata=Strata.train)
    on_validation_epoch_end = partialmethod(end, strata=Strata.validate)
    on_test_epoch_end = partialmethod(end, strata=Strata.test)
    on_predict_epoch_end = partialmethod(end, strata=Strata.predict)

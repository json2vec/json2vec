from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning.pytorch import Callback

from relflow.distributed import all_reduce_sum, is_distributed, synchronize_epoch_metrics
from relflow.structs.tree import Address

if TYPE_CHECKING:
    from lightning.pytorch import Trainer

    from relflow.architecture.root import Model


class Counter(torch.nn.Module):
    def __init__(self, address: Address, size: int):
        super().__init__()

        self.size: int = size

        # init with ones to avoid division by zero
        # it doesn't matter much since we will normalize over time
        self.counts: torch.Tensor
        self._pending_counts: torch.Tensor
        self.register_buffer("counts", torch.ones(size, dtype=torch.int64))
        self.register_buffer("_pending_counts", torch.zeros(size, dtype=torch.int64), persistent=False)
        self.is_full: bool = False

    def __str__(self) -> str:
        counts = self.counts.detach().cpu().tolist()
        return "\n".join(
            (
                f"size: {self.size}",
                f"is_full: {self.is_full}",
                f"counts: {counts}",
            )
        )

    @torch.no_grad()
    def observe(self, values: torch.Tensor) -> torch.Tensor:
        if self.training and not self.is_full:
            self.learn(tally(values, self.size))

        return values

    @torch.no_grad()
    def learn(self, counts: torch.Tensor) -> None:
        """Apply one already-counted pristine observation."""

        if not isinstance(counts, torch.Tensor):
            raise TypeError(f"counter counts must be a tensor, got {type(counts).__name__}")
        if tuple(counts.shape) != tuple(self.counts.shape):
            raise ValueError(f"counter counts must have shape {tuple(self.counts.shape)}, got {tuple(counts.shape)}")
        if counts.is_floating_point() or counts.is_complex():
            raise TypeError(f"counter counts must use an integer dtype, got {counts.dtype}")
        if (counts < 0).any():
            raise ValueError("counter counts cannot be negative")
        if self.is_full:
            return

        update = counts.to(device=self.counts.device, dtype=self.counts.dtype)
        remaining = torch.iinfo(self.counts.dtype).max - self.counts
        if (update >= remaining).any():
            self.is_full = True
            self._pending_counts.zero_()
            return

        self.counts += update
        self._pending_counts += update

    @torch.no_grad()
    def sync(self) -> None:
        local_update = self._pending_counts.clone()
        global_update = all_reduce_sum(self._pending_counts.clone())
        self._pending_counts.zero_()

        if self.is_full:
            return

        folded_update = global_update - local_update
        if not bool(folded_update.any().item()):
            return

        remaining = torch.iinfo(self.counts.dtype).max - self.counts
        could_overflow = bool((folded_update >= remaining).any().item())

        if could_overflow:
            # if we are approaching the max value, we stop counting and assume the counts are full
            self.is_full = True
            return

        self.counts += folded_update

    @torch.no_grad()
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.observe(values)

    @property
    @torch.no_grad()
    def weight(self) -> torch.Tensor:
        counts = self.counts.to(dtype=torch.float32)
        weights = counts.rsqrt()
        return weights * (counts.sum() / (weights * counts).sum())


class CounterUpdateCallback(Callback):
    @torch.no_grad()
    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: Model,
    ) -> None:  # ty:ignore[invalid-method-override]
        resources: dict[tuple[Address, str], Counter] = {}

        for address, node in pl_module.nodes.items():
            embedder = getattr(node, "embedder", None)
            if embedder is None:
                continue

            counter = getattr(embedder, "counter", None)
            if isinstance(counter, Counter):
                resources[(address, "counter")] = counter

            counter_map = getattr(embedder, "counters", None)
            if counter_map is None:
                continue

            for name, item in counter_map.items():
                if isinstance(item, Counter):
                    resources[(address, str(name))] = item

        if resources and is_distributed():
            synchronize_epoch_metrics(trainer)

        for _, counter in sorted(resources.items(), key=lambda item: (str(item[0][0]), item[0][1])):
            counter.sync()


def tally(values: torch.Tensor, size: int) -> torch.Tensor:
    """Count valid integer classes into one fixed-width vector."""

    if not isinstance(values, torch.Tensor):
        raise TypeError(f"counter values must be a tensor, got {type(values).__name__}")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError(f"counter size must be a positive integer, got {size!r}")

    observed = values.reshape(-1).to(dtype=torch.int64)
    observed = observed.masked_select(observed.ge(0) & observed.lt(size))
    return torch.bincount(observed, minlength=size)


__all__ = ["Counter", "CounterUpdateCallback", "tally"]

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lightning.pytorch import Callback

from json2vec.distributed import all_reduce_sum
from json2vec.structs.enums import TensorKey, Tokens
from json2vec.structs.tree import Address

if TYPE_CHECKING:
    from lightning.pytorch import Trainer
    from tensordict import TensorDict

    from json2vec.architecture.root import Model


class Counter(torch.nn.Module):
    def __init__(self, address: Address, size: int):
        super().__init__()

        self.size: int = size

        # init with ones to avoid division by zero
        # it doesn't matter much since we will normalize over time
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
            update = torch.bincount(values.view(-1), minlength=self.counts.shape[0]).to(self.counts.dtype)

            remaining = torch.iinfo(self.counts.dtype).max - self.counts
            could_overflow = bool((update >= remaining).any().item())

            if could_overflow:
                # if we are approaching the max value, we stop counting and assume the counts are full
                self.is_full = True
                self._pending_counts.zero_()
                return values

            self.counts += update
            self._pending_counts += update

        return values

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
    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: Model,
        batch: TensorDict,
        batch_idx: int,
    ) -> None:
        for address in pl_module.hyperparameters.active_requests:
            field = batch[address]
            embedder = pl_module.nodes[address].embedder
            observed: set[int] = set()

            def values_for(key: TensorKey) -> torch.Tensor | None:
                targets = getattr(field, "targets", None)
                if targets is not None and key in targets.keys():
                    return targets[key]

                value = getattr(field, key.name, None)
                return value if isinstance(value, torch.Tensor) else None

            def observe(counter: torch.nn.Module, values: torch.Tensor) -> None:
                identity = id(counter)
                if identity in observed:
                    return

                observed.add(identity)
                counter(values)

            state = values_for(TensorKey.state)

            if state is not None and hasattr(embedder, "counter"):
                observe(counter=embedder.counter, values=state)

            counters = getattr(embedder, "counters", None)
            if counters is None:
                continue

            if state is not None and TensorKey.state.name in counters:
                observe(counter=counters[TensorKey.state.name], values=state)

            if state is None or TensorKey.content.name not in counters:
                continue

            content = values_for(TensorKey.content)
            if content is None or content.shape != state.shape:
                continue

            values = content.masked_select(state.eq(Tokens.valued.value))
            observe(counter=counters[TensorKey.content.name], values=values)

    @torch.no_grad()
    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: Model,
    ) -> None:
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

        for _, counter in sorted(resources.items(), key=lambda item: (str(item[0][0]), item[0][1])):
            counter.sync()


__all__ = ["Counter", "CounterUpdateCallback"]

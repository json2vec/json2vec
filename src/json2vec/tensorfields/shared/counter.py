from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch import Callback

from json2vec.distributed import all_reduce_sum
from json2vec.structs.enums import TensorKey, Tokens
from json2vec.structs.tree import Address

if TYPE_CHECKING:
    from lightning.pytorch import Trainer
    from tensordict import TensorDict

    from json2vec.architecture.root import Model
    from json2vec.tensorfields.base import TensorFieldBase


class Counter(torch.nn.Module):
    def __init__(self, address: Address, size: int):
        super().__init__()

        self.size: int = size

        # init with ones to avoid division by zero
        # it doesn't matter much since we will normalize over time
        self.register_buffer("counts", torch.ones(size, dtype=torch.int64))
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
    def forward(self, values: torch.Tensor):
        if self.training and not self.is_full:
            update = torch.bincount(values.view(-1), minlength=self.counts.shape[0]).to(self.counts.dtype)
            update = all_reduce_sum(update)

            remaining = torch.iinfo(self.counts.dtype).max - self.counts
            could_overflow = bool((update >= remaining).any().item())

            if could_overflow:
                # if we are approaching the max value, we stop counting and assume the counts are full
                self.is_full = True
                return values

            self.counts += update

        return values

    @property
    @torch.no_grad()
    def weight(self) -> torch.Tensor:
        counts = self.counts.to(dtype=torch.float32)
        weights = counts.rsqrt()
        return weights * (counts.sum() / (weights * counts).sum())


def counter_value(field: TensorFieldBase, key: TensorKey) -> torch.Tensor | None:
    targets = getattr(field, "targets", None)
    if targets is not None and key in targets.keys():
        return targets[key]

    value = getattr(field, key.name, None)
    if isinstance(value, torch.Tensor):
        return value

    return None


def _observe(counter: torch.nn.Module, values: torch.Tensor, observed: set[int]) -> None:
    identity = id(counter)
    if identity in observed:
        return

    observed.add(identity)
    counter(values)


def _update_component_counters(
    component: Any,
    field: TensorFieldBase,
    observed: set[int],
) -> None:
    state = counter_value(field=field, key=TensorKey.state)

    if state is not None and hasattr(component, "counter"):
        _observe(counter=component.counter, values=state, observed=observed)

    counters = getattr(component, "counters", None)
    if counters is None:
        return

    if state is not None and TensorKey.state.name in counters:
        _observe(counter=counters[TensorKey.state.name], values=state, observed=observed)

    if state is None or TensorKey.content.name not in counters:
        return

    content = counter_value(field=field, key=TensorKey.content)
    if content is None or content.shape != state.shape:
        return

    values = content.masked_select(state.eq(Tokens.valued.value))
    _observe(counter=counters[TensorKey.content.name], values=values, observed=observed)


@torch.no_grad()
def observe_counters(component: Any, field: TensorFieldBase) -> None:
    _update_component_counters(component=component, field=field, observed=set())


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
            observe_counters(component=embedder, field=field)


__all__ = ["Counter", "CounterUpdateCallback", "counter_value", "observe_counters"]

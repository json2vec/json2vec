from __future__ import annotations

import torch

from json2vec.distributed import all_reduce_sum
from json2vec.structs.tree import Address


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


__all__ = ["Counter"]

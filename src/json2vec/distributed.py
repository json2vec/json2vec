from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 0

    return dist.get_rank()


def world_size() -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return 1

    return dist.get_world_size()


def is_rank_zero() -> bool:
    return rank() == 0


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


def all_gather_object(value: Any) -> list[Any]:
    if not is_distributed():
        return [value]

    gathered: list[Any] = [None for _ in range(world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def broadcast_object(value: Any, src: int = 0) -> Any:
    if not is_distributed():
        return value

    payload = [value if rank() == src else None]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]

from __future__ import annotations

import functools
from collections.abc import Iterator
from typing import Any

import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# Match torch.nn.parallel.DistributedDataParallel's default ``bucket_cap_mb``.
_DDP_DEFAULT_BUCKET_BYTES: int = 25 * 1024 * 1024


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


@functools.cache
def _backend_supports_avg() -> bool:
    # ReduceOp.AVG exists since PyTorch 1.10 but is only wired up for NCCL.
    if not hasattr(dist.ReduceOp, "AVG"):
        return False
    try:
        backend = dist.get_backend()
    except RuntimeError:
        return False
    return backend == dist.Backend.NCCL


def _bucketize(
    grads: list[torch.Tensor],
    max_bytes: int,
    element_bytes: int,
) -> Iterator[list[torch.Tensor]]:
    # Greedy first-fit so a single oversized grad still gets its own bucket.
    bucket: list[torch.Tensor] = []
    bucket_bytes: int = 0
    for grad in grads:
        grad_bytes = grad.numel() * element_bytes
        if bucket and bucket_bytes + grad_bytes > max_bytes:
            yield bucket
            bucket = []
            bucket_bytes = 0
        bucket.append(grad)
        bucket_bytes += grad_bytes
    if bucket:
        yield bucket


def mean_all_reduce_grads(
    module: torch.nn.Module,
    *,
    bucket_bytes: int = _DDP_DEFAULT_BUCKET_BYTES,
) -> None:
    """Bucketed async mean all-reduce of parameter gradients.

    Approximates DDP's reducer (bucketed, pipelined, ``ReduceOp.AVG`` on NCCL)
    without its C++ post-hooks, which are incompatible with TorchJD's vmap'd
    autograd. Unlike DDP this cannot overlap with the backward pass.
    """
    if not is_distributed():
        return

    size = world_size()
    use_avg = _backend_supports_avg()
    op = dist.ReduceOp.AVG if use_avg else dist.ReduceOp.SUM

    grouped: dict[tuple[torch.dtype, torch.device], list[torch.Tensor]] = {}
    for parameter in module.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        grouped.setdefault((grad.dtype, grad.device), []).append(grad)

    pending: list[tuple[Any, torch.Tensor, list[torch.Tensor]]] = []
    for (dtype, _device), grads in grouped.items():
        element_bytes = torch.tensor([], dtype=dtype).element_size()
        for chunk in _bucketize(grads, bucket_bytes, element_bytes):
            flat = _flatten_dense_tensors(chunk)
            work = dist.all_reduce(flat, op=op, async_op=True)
            pending.append((work, flat, chunk))

    for work, flat, chunk in pending:
        if work is not None:
            work.wait()
        if not use_avg:
            flat.div_(size)
        for original, synced in zip(chunk, _unflatten_dense_tensors(flat, chunk), strict=True):
            original.copy_(synced)


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

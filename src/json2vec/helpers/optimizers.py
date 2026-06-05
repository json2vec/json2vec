"""Optimizer factories for json2vec models."""

from collections.abc import Callable
from typing import Any

import torch
from torch.nn import Parameter

OptimizerFactory = Callable[[torch.nn.Module], torch.optim.Optimizer]

__all__ = ["OptimizerFactory", "adamw"]


def _uses_weight_decay(
    name: str,
    parameter: Parameter,
    *,
    decay_bias: bool,
    decay_1d: bool,
    no_decay_name_fragments: tuple[str, ...],
) -> bool:
    if not decay_bias and name.endswith(".bias"):
        return False
    if not decay_1d and parameter.ndim <= 1:
        return False

    lower_name = name.lower()
    return not any(fragment in lower_name for fragment in no_decay_name_fragments)


def adamw(
    learning_rate: float,
    *,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    decay_bias: bool = False,
    decay_1d: bool = False,
    no_decay_name_fragments: tuple[str, ...] = ("norm",),
) -> OptimizerFactory:
    """Build an AdamW optimizer with common no-decay parameter grouping."""

    fragments = tuple(fragment.lower() for fragment in no_decay_name_fragments if fragment)

    def build(module: torch.nn.Module) -> torch.optim.Optimizer:
        decay_parameters: list[Parameter] = []
        no_decay_parameters: list[Parameter] = []

        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue

            if _uses_weight_decay(
                name,
                parameter,
                decay_bias=decay_bias,
                decay_1d=decay_1d,
                no_decay_name_fragments=fragments,
            ):
                decay_parameters.append(parameter)
            else:
                no_decay_parameters.append(parameter)

        parameter_groups: list[dict[str, Any]] = []
        if decay_parameters:
            parameter_groups.append({"params": decay_parameters, "weight_decay": weight_decay})
        if no_decay_parameters:
            parameter_groups.append({"params": no_decay_parameters, "weight_decay": 0.0})

        return torch.optim.AdamW(
            params=parameter_groups,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

    return build

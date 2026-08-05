from collections import OrderedDict
from collections.abc import Iterable

import torch

from relflow.helpers.optimizers import adamw


def _module() -> torch.nn.Module:
    return torch.nn.Sequential(
        OrderedDict(
            [
                ("linear", torch.nn.Linear(3, 2)),
                ("norm", torch.nn.LayerNorm(2)),
            ]
        )
    )


def _parameter_ids(parameters: Iterable[torch.nn.Parameter]) -> set[int]:
    return {id(parameter) for parameter in parameters}


def test_adamw_groups_bias_1d_and_named_parameters_without_decay() -> None:
    module = _module()
    parameters = dict(module.named_parameters())

    optimizer = adamw(
        learning_rate=0.123,
        weight_decay=0.5,
        betas=(0.8, 0.9),
        eps=1e-7,
    )(module)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 0.123
    assert optimizer.defaults["betas"] == (0.8, 0.9)
    assert optimizer.defaults["eps"] == 1e-7
    assert optimizer.defaults["weight_decay"] == 0.5
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.5, 0.0]

    decay_group, no_decay_group = optimizer.param_groups
    assert _parameter_ids(decay_group["params"]) == {id(parameters["linear.weight"])}
    assert _parameter_ids(no_decay_group["params"]) == {
        id(parameters["linear.bias"]),
        id(parameters["norm.weight"]),
        id(parameters["norm.bias"]),
    }


def test_adamw_can_decay_bias_and_1d_parameters() -> None:
    module = _module()
    parameters = dict(module.named_parameters())

    optimizer = adamw(
        learning_rate=1e-3,
        weight_decay=0.2,
        decay_bias=True,
        decay_1d=True,
        no_decay_name_fragments=(),
    )(module)

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 0.2
    assert _parameter_ids(optimizer.param_groups[0]["params"]) == {id(parameter) for parameter in parameters.values()}


def test_adamw_skips_frozen_parameters() -> None:
    module = _module()
    parameters = dict(module.named_parameters())
    parameters["linear.weight"].requires_grad_(False)

    optimizer = adamw(learning_rate=1e-3)(module)

    optimized_parameters = {
        parameter_id for group in optimizer.param_groups for parameter_id in _parameter_ids(group["params"])
    }
    assert id(parameters["linear.weight"]) not in optimized_parameters

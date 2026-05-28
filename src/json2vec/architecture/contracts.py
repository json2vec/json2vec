"""Generic runtime contracts for model forward inputs."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TENSORFIELDS, TensorFieldBase

if TYPE_CHECKING:
    from json2vec.architecture.root import Model


class ForwardContractError(ValueError):
    """Raised when a forward batch violates a model input contract."""


INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


ContractSignature = tuple[Any, ...]
ContractScope = tuple[str, int, int, ContractSignature]


@dataclass
class ContractScheduler:
    """Deterministic backoff scheduler for expensive forward contract checks."""

    periodic_interval: int = 1024
    _counts: dict[ContractScope, int] = field(default_factory=dict)

    def reset(self) -> None:
        self._counts.clear()

    def should_check(
        self,
        module: "Model",
        inputs: Any,
        *,
        strata: Strata,
        dataloader_idx: int,
    ) -> bool:
        generation = int(getattr(module, "_contract_generation", 0))
        scope = (
            str(strata),
            dataloader_idx,
            generation,
            batch_signature(module, inputs),
        )
        count = self._counts.get(scope, 0)
        self._counts[scope] = count + 1
        return is_backoff_index(count, periodic_interval=self.periodic_interval)


def sanitize(
    module: "Model",
    inputs: TensorDict[Address, TensorFieldBase],
    *,
    strata: Strata | str,
    dataloader_idx: int = 0,
) -> None:
    """Validate the generic forward-input contract before model execution."""
    normalized = Strata.normalize(strata)
    scheduler = getattr(module, "_contract_scheduler", None)
    if isinstance(scheduler, ContractScheduler) and not scheduler.should_check(
        module,
        inputs,
        strata=normalized,
        dataloader_idx=dataloader_idx,
    ):
        return

    if not isinstance(inputs, TensorDict):
        raise TypeError(f"forward inputs must be a TensorDict, got {type(inputs).__name__}")

    require_forward_addresses(module, inputs, strata=normalized)

    for address in module.hyperparameters.active_requests:
        tensorfield = inputs[address]
        require_registered_tensorfield(module, address, tensorfield)
        require_core_tensors(module, address, tensorfield)
        require_tensor_devices(module, address, tensorfield)
        require_target_contract(module, address, tensorfield, strata=normalized)
        require_mask_contract(module, address, tensorfield, strata=normalized)


def is_backoff_index(index: int, *, periodic_interval: int) -> bool:
    if index == 0:
        return True

    if (index & (index - 1)) == 0:
        return True

    return periodic_interval > 0 and index % periodic_interval == 0


def batch_signature(module: "Model", inputs: Any) -> ContractSignature:
    if not isinstance(inputs, TensorDict):
        return ("inputs", qualified_name(type(inputs)))

    input_keys = tuple(sorted(str(key) for key in inputs.keys()))
    fields: list[tuple[Any, ...]] = []
    for address in sorted(module.hyperparameters.active_requests, key=str):
        if address not in inputs.keys():
            fields.append((str(address), "missing"))
            continue

        tensorfield = inputs[address]
        fields.append(
            (
                str(address),
                qualified_name(type(tensorfield)),
                tensor_signature(getattr(tensorfield, TensorKey.state, None)),
                tensor_signature(getattr(tensorfield, TensorKey.trainable, None)),
                tensor_tree_signature(getattr(tensorfield, TensorKey.content, None)),
                tensor_tree_signature(getattr(tensorfield, TensorKey.targets, None)),
            )
        )

    return (input_keys, tuple(fields))


def tensor_signature(value: Any) -> tuple[Any, ...]:
    if not torch.is_tensor(value):
        return ("object", qualified_name(type(value)))

    return (
        "tensor",
        tuple(value.shape),
        str(value.dtype),
        str(value.device),
    )


def tensor_tree_signature(value: Any) -> tuple[Any, ...]:
    if torch.is_tensor(value):
        return tensor_signature(value)

    if isinstance(value, TensorDict):
        return (
            "tensordict",
            tuple((str(key), tensor_tree_signature(value[key])) for key in sorted(value.keys(), key=str)),
        )

    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (str(key), tensor_tree_signature(item)) for key, item in sorted(value.items(), key=lambda x: str(x[0]))
            ),
        )

    return ("object", qualified_name(type(value)))


def require_forward_addresses(
    module: "Model",
    inputs: TensorDict[Address, TensorFieldBase],
    *,
    strata: Strata,
) -> None:
    keys = set(inputs.keys())
    metadata_keys = {key for key in keys if key == TensorKey.metadata}
    addresses = {Address(str(key)) for key in keys if key != TensorKey.metadata}
    expected = set(module.hyperparameters.active_requests)

    if metadata_keys and strata != Strata.predict:
        raise ForwardContractError(f"forward input contains {TensorKey.metadata} outside predict strata")

    missing = expected - addresses
    if missing:
        raise ForwardContractError(f"forward input is missing active request address(es): {format_addresses(missing)}")

    extra = addresses - expected
    if not extra:
        return

    arrays = extra & set(module.hyperparameters.arrays)
    if arrays:
        raise ForwardContractError(
            f"forward input contains array address(es); only active leaf request addresses are allowed: "
            f"{format_addresses(arrays)}"
        )

    inactive = {address for address in extra if address in module.hyperparameters.requests}
    if inactive:
        raise ForwardContractError(
            "forward input contains inactive request address(es): "
            f"{format_addresses(inactive)}. Inactive fields remain in the schema but must not be present in runtime input."
        )

    raise ForwardContractError(f"forward input contains unknown address(es): {format_addresses(extra)}")


def require_registered_tensorfield(module: "Model", address: Address, value: Any) -> None:
    if not isinstance(value, TensorFieldBase):
        raise TypeError(f"forward input '{address}' must be a TensorFieldBase, got {type(value).__name__}")

    request = module.hyperparameters.requests[address]
    expected = TENSORFIELDS[request.type].TensorField
    if not isinstance(value, expected):
        raise TypeError(
            f"forward input '{address}' must use tensorfield class {qualified_name(expected)}, "
            f"got {qualified_name(type(value))}"
        )


def require_core_tensors(module: "Model", address: Address, tensorfield: TensorFieldBase) -> None:
    state = require_tensor_attribute(address, tensorfield, TensorKey.state)
    trainable = require_tensor_attribute(address, tensorfield, TensorKey.trainable)
    content = require_tensor_tree(
        address,
        TensorKey.content,
        getattr(tensorfield, TensorKey.content, None),
    )
    targets = require_targets(address, tensorfield)

    field_shape = module.hyperparameters.shapes[address]
    if state.ndim != len(field_shape) + 1:
        raise ForwardContractError(
            f"forward input '{address}' state must have rank {len(field_shape) + 1}, got {state.ndim}"
        )

    expected_shape = (state.shape[0], *field_shape)
    if tuple(state.shape) != expected_shape:
        raise ForwardContractError(
            f"forward input '{address}' state must have shape {expected_shape}, got {tuple(state.shape)}"
        )

    if state.dtype not in INTEGER_DTYPES:
        raise TypeError(f"forward input '{address}' state must use an integer dtype, got {state.dtype}")

    if tuple(trainable.shape) != tuple(state.shape):
        raise ForwardContractError(
            f"forward input '{address}' trainable must have shape {tuple(state.shape)}, got {tuple(trainable.shape)}"
        )

    if trainable.dtype != torch.bool:
        raise TypeError(f"forward input '{address}' trainable must use bool dtype, got {trainable.dtype}")

    require_token_values(address, TensorKey.state, state)
    require_content_prefix_shapes(address, content, state)

    if TensorKey.state in targets.keys():
        target_state_name = f"{TensorKey.targets}[{TensorKey.state}]"
        target_state = require_tensor_tree(address, target_state_name, targets[TensorKey.state])
        require_matching_tree_shapes(
            address,
            actual_name=target_state_name,
            actual=target_state,
            expected_name=TensorKey.state,
            expected={(): state},
        )
        require_integer_tensors(address, target_state_name, target_state)
        require_token_values(address, target_state_name, targets[TensorKey.state])

    if TensorKey.content in targets.keys():
        target_content_name = f"{TensorKey.targets}[{TensorKey.content}]"
        target_content = require_tensor_tree(address, target_content_name, targets[TensorKey.content])
        require_matching_tree_shapes(
            address,
            actual_name=target_content_name,
            actual=target_content,
            expected_name=TensorKey.content,
            expected=content,
        )


def require_tensor_devices(module: "Model", address: Address, tensorfield: TensorFieldBase) -> None:
    tensors = list(iter_tensor_leaves(tensorfield))
    devices = {tensor.device for _, tensor in tensors}
    if len(devices) > 1:
        formatted = ", ".join(sorted(str(device) for device in devices))
        raise ForwardContractError(f"forward input '{address}' tensors must share one device, got {formatted}")

    module_device = getattr(module, "device", None)
    if isinstance(module_device, torch.device) and devices and next(iter(devices)) != module_device:
        raise ForwardContractError(
            f"forward input '{address}' tensors must be on module device {module_device}, got {next(iter(devices))}"
        )


def require_mask_contract(module: "Model", address: Address, tensorfield: TensorFieldBase, *, strata: Strata) -> None:
    state = tensorfield.state
    trainable = tensorfield.trainable
    is_masked = state.eq(Tokens.masked.value)
    is_target = address in module.hyperparameters.target

    if trainable.any() and not state.masked_select(trainable).eq(Tokens.masked.value).all():
        raise ForwardContractError(f"forward input '{address}' trainable positions must have masked state")

    if strata != Strata.predict and not is_target and (is_masked & ~trainable).any():
        raise ForwardContractError(f"forward input '{address}' has masked state where trainable is false")

    if not trainable.any():
        return

    targets = tensorfield.targets
    for key in (TensorKey.state, TensorKey.content):
        if key not in targets.keys():
            raise ForwardContractError(f"forward input '{address}' has trainable positions but lacks targets[{key}]")

    target_state = targets[TensorKey.state]
    if target_state.masked_select(trainable).eq(Tokens.masked.value).any():
        raise ForwardContractError(f"forward input '{address}' targets[{TensorKey.state}] must not be masked")


def require_target_contract(
    module: "Model",
    address: Address,
    tensorfield: TensorFieldBase,
    *,
    strata: Strata | None,
) -> None:
    if address not in module.hyperparameters.target:
        return

    if not tensorfield.state.eq(Tokens.masked.value).all():
        raise ForwardContractError(f"target field '{address}' must not contain visible input state")

    if strata in (Strata.train, Strata.validate, Strata.test) and not tensorfield.trainable.any():
        raise ForwardContractError(f"target field '{address}' must have trainable positions in {strata} strata")


def require_tensor_attribute(address: Address, tensorfield: TensorFieldBase, name: str) -> torch.Tensor:
    value = getattr(tensorfield, name, None)
    if not torch.is_tensor(value):
        raise TypeError(f"forward input '{address}' {name} must be a torch.Tensor, got {type(value).__name__}")

    return value


def require_targets(address: Address, tensorfield: TensorFieldBase) -> TensorDict:
    value = getattr(tensorfield, TensorKey.targets, None)
    if not isinstance(value, TensorDict):
        raise TypeError(
            f"forward input '{address}' {TensorKey.targets} must be a TensorDict, got {type(value).__name__}"
        )

    require_tensor_tree(address, TensorKey.targets, value, allow_empty=True)
    return value


def require_tensor_tree(
    address: Address,
    name: str,
    value: Any,
    *,
    allow_empty: bool = False,
) -> dict[tuple[str, ...], torch.Tensor]:
    tensors = dict(iter_tensor_leaves(value))
    if not tensors and not allow_empty:
        raise TypeError(f"forward input '{address}' {name} must contain at least one tensor")

    return tensors


def require_matching_tree_shapes(
    address: Address,
    *,
    actual_name: str,
    actual: dict[tuple[str, ...], torch.Tensor],
    expected_name: str,
    expected: dict[tuple[str, ...], torch.Tensor],
) -> None:
    if set(actual) != set(expected):
        raise ForwardContractError(
            f"forward input '{address}' {actual_name} keys must match {expected_name} keys: "
            f"expected {format_paths(expected)}, got {format_paths(actual)}"
        )

    for path, actual_tensor in actual.items():
        expected_tensor = expected[path]
        if tuple(actual_tensor.shape) != tuple(expected_tensor.shape):
            suffix = format_path(path)
            raise ForwardContractError(
                f"forward input '{address}' {actual_name}{suffix} must have shape "
                f"{tuple(expected_tensor.shape)}, got {tuple(actual_tensor.shape)}"
            )


def require_content_prefix_shapes(
    address: Address,
    content: dict[tuple[str, ...], torch.Tensor],
    state: torch.Tensor,
) -> None:
    state_shape = tuple(state.shape)
    state_rank = len(state_shape)
    for path, tensor in content.items():
        if len(tensor.shape) < state_rank or tuple(tensor.shape[:state_rank]) != state_shape:
            suffix = format_path(path)
            raise ForwardContractError(
                f"forward input '{address}' {TensorKey.content}{suffix} must start with {TensorKey.state} "
                f"shape {state_shape}, "
                f"got {tuple(tensor.shape)}"
            )


def require_integer_tensors(
    address: Address,
    name: str,
    tensors: dict[tuple[str, ...], torch.Tensor],
) -> None:
    for path, tensor in tensors.items():
        if tensor.dtype not in INTEGER_DTYPES:
            suffix = format_path(path)
            raise TypeError(f"forward input '{address}' {name}{suffix} must use an integer dtype, got {tensor.dtype}")


def require_token_values(address: Address, name: str, values: torch.Tensor) -> None:
    valid = torch.tensor([token.value for token in Tokens], device=values.device, dtype=values.dtype)
    invalid = ~torch.isin(values, valid)
    if invalid.any():
        value = values.masked_select(invalid).reshape(-1)[0].item()
        raise ForwardContractError(f"forward input '{address}' {name} contains invalid token id {value}")


def iter_tensor_leaves(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], torch.Tensor]]:
    if torch.is_tensor(value):
        yield path, value
        return

    if isinstance(value, TensorFieldBase):
        for name in (TensorKey.state, TensorKey.trainable, TensorKey.content, TensorKey.targets):
            yield from iter_tensor_leaves(getattr(value, name, None), (*path, name))
        return

    if isinstance(value, TensorDict):
        for key in value.keys():
            yield from iter_tensor_leaves(value[key], (*path, str(key)))
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from iter_tensor_leaves(item, (*path, str(key)))
        return

    raise TypeError(f"expected tensor tree at {format_path(path) or '<root>'}, got {type(value).__name__}")


def format_addresses(addresses: set[Address]) -> str:
    return ", ".join(sorted(str(address) for address in addresses))


def format_paths(values: Mapping[tuple[str, ...], Any]) -> str:
    return ", ".join(format_path(path) or "<tensor>" for path in sorted(values))


def format_path(path: tuple[str, ...]) -> str:
    return "".join(f"[{part}]" for part in path)


def qualified_name(cls: type[Any]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"

"""Arrow batch encoding and tensor masking stages."""

from __future__ import annotations

import inspect
import random
from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import Any, cast

from beartype import beartype
from tensordict import TensorDict

from relflow.data.arrow import Batch
from relflow.data.datasets.base import EncodedInput, InterprocessEncodingContext
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata
from relflow.structs.experiment import Schema
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS, TensorFieldBase


def encode(
    batch: Batch,
    schema: Schema,
    strata: Strata,
    interprocess_encoding_context: InterprocessEncodingContext,
    defer_target_masking: bool = False,
) -> EncodedInput:
    out: dict[Address, TensorFieldBase] = {}
    target_addresses = set(schema.target)

    hash_salt = random.getrandbits(64) if strata in {Strata.train, Strata.validate} else 0

    ragged_fields = coalesce(batch, schema=schema, strata=strata)

    for address, request in schema.active_requests.items():
        plugin = TENSORFIELDS[request.type]
        TensorField = cast(type[TensorFieldBase], getattr(plugin, "TensorField"))

        if (strata == Strata.predict) & (address in target_addresses):
            out[address] = TensorField.empty(
                batch_size=len(batch),
                address=address,
                schema=schema,
            )
            continue

        field = ragged_fields.pop(address)
        field = replace(field, values=plugin.prepare(field.values, address=address))
        kwargs: dict[str, Any] = dict(
            field=field,
            address=address,
            schema=schema,
            strata=strata,
        )
        parameters = inspect.signature(TensorField.new).parameters
        if "interprocess_encoding_context" in parameters:
            kwargs["interprocess_encoding_context"] = interprocess_encoding_context.get(address)

        if "salt" in parameters:
            kwargs["salt"] = hash_salt

        out[address] = TensorField.new(**kwargs)
        out[address].check_nullable(address=address, schema=schema)

        if not defer_target_masking and strata != Strata.predict and address in target_addresses:
            out[address].mask(p_prune=1.0)

    inputs = cast(EncodedInput, TensorDict(source=cast(Any, out)))

    return inputs


def policy(
    field: TensorFieldBase,
    *,
    p_mask: float,
    p_prune: float,
    branch_masks: tuple[Any, ...],
    address: Address,
    schema: Schema,
) -> None:
    parameters = inspect.signature(field.mask).parameters
    supports_policy_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    supports_policy_kwargs |= any(name in parameters for name in ("p_prune", "branch_masks", "schema"))

    if supports_policy_kwargs:
        field.mask(
            p_mask=p_mask,
            p_prune=p_prune,
            branch_masks=branch_masks,
            address=address,
            schema=schema,
        )
        return

    if branch_masks:
        raise TypeError(f"tensorfield at '{address}' must accept mask(..., branch_masks=...) to use Branch masks")

    if p_mask > 0.0:
        field.mask(p_mask=p_mask)

    if p_prune > 0.0:
        field.target(p_prune=p_prune)


@beartype
def mask(
    pipe: Iterable[EncodedInput],
    schema: Schema,
    strata: Strata = Strata.train,
) -> Iterator[EncodedInput]:
    for item in pipe:
        if strata == Strata.predict:
            yield item
            continue

        for address, request in schema.active_requests.items():
            p_mask = float(request.p_mask or 0.0)
            p_prune = float(request.p_prune or 0.0)
            branch_masks = schema.branch_masks_for(address)
            if p_mask <= 0.0 and p_prune <= 0.0 and not branch_masks:
                continue

            policy(
                item[address],
                p_mask=p_mask,
                p_prune=p_prune,
                branch_masks=branch_masks,
                address=address,
                schema=schema,
            )

        yield item


@beartype
def target(
    pipe: Iterable[EncodedInput],
    schema: Schema,
) -> Iterator[EncodedInput]:
    for item in pipe:
        for address, request in schema.active_requests.items():
            p_prune = float(request.p_prune or 0.0)
            if p_prune <= 0.0:
                continue

            item[address].target(p_prune=p_prune)

        yield item


def mock(schema: Schema, batch_size: int) -> EncodedInput:
    out: dict[Address, TensorFieldBase] = {}

    for address, request in schema.active_requests.items():
        TensorField = cast(type[TensorFieldBase], getattr(TENSORFIELDS[request.type], "TensorField"))
        out[address] = TensorField.empty(
            batch_size=batch_size,
            address=address,
            schema=schema,
        )

    return cast(EncodedInput, TensorDict(source=cast(Any, out), batch_size=batch_size))

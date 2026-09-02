"""Composable iterable stages for fetching, preprocessing, and encoding data."""

from __future__ import annotations

import inspect
import random
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar, cast

from beartype import beartype
from tensordict import TensorDict

from relflow.data.datasets.base import (
    EncodedBatch,
    EncodedInput,
    InterprocessEncodingContext,
    PreprocessorConfig,
    ProcessedObservation,
    RawObservation,
)
from relflow.data.processors import Preprocessor
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, TensorKey
from relflow.structs.experiment import Schema
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS, TensorFieldBase

T = TypeVar("T")


@beartype
def process(
    pipe: Iterable[RawObservation],
    preprocessor: PreprocessorConfig.Value,
    strata: Strata,
    schema: Schema,
    interprocess_encoding_context: InterprocessEncodingContext,
) -> Iterator[ProcessedObservation]:
    resolved = Preprocessor.normalize(PreprocessorConfig.normalize(preprocessor))

    if resolved is None:
        for item in pipe:
            yield [item]
        return

    for item in pipe:
        yield from resolved.outputs(
            item,
            strata=strata,
            schema=schema,
            encoding_context=interprocess_encoding_context,
        )


@beartype
def batch(pipe: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    items: list[T] = []

    for item in pipe:
        items.append(item)
        if len(items) == batch_size:
            yield items
            items = []

    if items:
        yield items


@beartype
def sample(pipe: Iterable[T], sample_rate: float, strata: Strata) -> Iterator[T]:
    if strata == Strata.predict or sample_rate >= 1.0:
        yield from pipe
        return

    for item in pipe:
        if random.random() < sample_rate:
            yield item


@beartype
def shuffle(pipe: Iterable[T], size: int, strata: Strata) -> Iterator[T]:
    if strata == Strata.predict:
        yield from pipe
        return

    iterable = iter(pipe)
    buffer: list[T] = []
    exhausted = False

    for _ in range(size):
        try:
            buffer.append(next(iterable))
        except StopIteration:
            exhausted = True
            break

    while buffer:
        idx = random.randrange(len(buffer))
        item = buffer[idx]

        if exhausted:
            buffer.pop(idx)
        else:
            try:
                buffer[idx] = next(iterable)
            except StopIteration:
                exhausted = True
                buffer.pop(idx)

        yield item


def encode(
    batch: EncodedBatch,
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
        TensorField = cast(type[TensorFieldBase], getattr(TENSORFIELDS[request.type], "TensorField"))

        if (strata == Strata.predict) & (address in target_addresses):
            out[address] = TensorField.empty(
                batch_size=len(batch),
                address=address,
                schema=schema,
            )
            continue

        kwargs: dict[str, Any] = dict(
            field=ragged_fields.pop(address),
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

    if strata == Strata.predict:
        inputs[TensorKey.metadata] = batch

    return inputs


@beartype
def transform(
    pipe: Iterable[EncodedBatch],
    schema: Schema,
    strata: Strata,
    interprocess_encoding_context: InterprocessEncodingContext,
) -> Iterator[EncodedInput]:
    for item in pipe:
        yield encode(
            batch=item,
            schema=schema,
            strata=strata,
            interprocess_encoding_context=interprocess_encoding_context,
            defer_target_masking=True,
        )


def _apply_mask_policy(
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

            _apply_mask_policy(
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

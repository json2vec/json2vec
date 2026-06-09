"""Composable iterable stages for fetching, preprocessing, and encoding data."""

from __future__ import annotations

import inspect
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from functools import cache
from typing import Annotated, Any, TypeVar, cast

import jmespath
import pydantic
from beartype import beartype
from tensordict import TensorDict

from json2vec.data.datasets.base import (
    EncodedBatch,
    EncodedInput,
    InterprocessEncodingContext,
    PreprocessorConfig,
    ProcessedObservation,
    RawObservation,
)
from json2vec.data.processing import MASK_LITERAL, contains_mask_literal
from json2vec.preprocessors.base import PREPROCESSORS, Preprocessor, PreprocessorMode
from json2vec.structs.enums import Strata, TensorKey
from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TENSORFIELDS, TensorFieldBase

T = TypeVar("T")


@beartype
def process(
    pipe: Iterable[RawObservation],
    preprocessor: PreprocessorConfig.Value,
    preprocessor_kwargs: dict[str, Any] | None,
    strata: Strata,
    interprocess_encoding_context: InterprocessEncodingContext,
) -> Iterator[ProcessedObservation]:
    preprocessor = PreprocessorConfig.normalize(preprocessor)
    kwargs = {} if preprocessor_kwargs is None else preprocessor_kwargs

    if preprocessor is None:
        for item in pipe:
            yield [item]
        return

    if isinstance(preprocessor, str):
        resolved = PREPROCESSORS[preprocessor]
    elif isinstance(preprocessor, Preprocessor):
        resolved = preprocessor
    else:
        resolved = Preprocessor(
            name=getattr(preprocessor, "__name__", type(preprocessor).__name__),
            func=preprocessor,
            mode=PreprocessorMode.transformation,
        )

    for item in pipe:
        yield from resolved.outputs(
            item,
            **kwargs,
            strata=strata,
            interprocess_encoding_context=interprocess_encoding_context,
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


@beartype
@cache
def query(expression: str) -> jmespath.parser.ParsedResult:
    """Compile a request-level JMESPath query for an encoded batch.

    Request queries are written relative to one processed observation, not the
    whole batch. This helper prepends the outer batch selector, so a request
    query like `[*].amount` is searched as `[*][*].amount` at encode time.
    Do not include both leading selectors in request definitions.
    """
    return jmespath.compile(expression=f"[*]{expression}")


@cache
def compile_query_extractor(expression: str) -> Callable[[Any], Any] | None:
    """Compile a direct extractor for simple JMESPath projection queries."""
    if not expression.startswith("[*]"):
        return None

    operations: list[tuple[str, str | None]] = [("projection", None), ("projection", None)]
    index = 3
    while index < len(expression):
        if expression.startswith("[*]", index):
            operations.append(("projection", None))
            index += 3
            continue

        if not expression.startswith(".", index):
            return None

        match = re.match(r"\.([A-Za-z_][A-Za-z0-9_]*)", expression[index:])
        if match is None:
            return None

        operations.append(("field", match.group(1)))
        index += len(match.group(0))

    def search(value: Any) -> Any:
        def apply(item: Any, operation_index: int) -> Any:
            if operation_index == len(operations):
                return item

            operation, key = operations[operation_index]
            if operation == "field":
                if not isinstance(item, dict) or key not in item:
                    return None
                return apply(item[key], operation_index + 1)

            if not isinstance(item, list):
                return None

            out = []
            for child in item:
                result = apply(child, operation_index + 1)
                if result is not None:
                    out.append(result)
            return out

        return apply(value, 0)

    return search


class JMESPathResolutionMonitor(pydantic.BaseModel):
    every: Annotated[int, pydantic.Field(gt=0)] = 1000

    _counts: Counter[Address] = pydantic.PrivateAttr(default_factory=Counter)

    def observe(self, *, address: Address, expression: str, result: Any) -> None:
        self._counts[address] += 1
        count = self._counts[address]

        if count % self.every != 0:
            return

        stack = [result]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
            elif isinstance(item, dict):
                stack.extend(item.values())
            elif item is None:
                continue
            elif isinstance(item, str) and item == "":
                continue
            else:
                return

        raise ValueError(f"JMESPath query returned empty result for address '{address}': {expression}")


def encode(
    batch: EncodedBatch,
    hyperparameters: Hyperparameters,
    strata: Strata,
    interprocess_encoding_context: InterprocessEncodingContext,
    jmespath_resolution_monitor: JMESPathResolutionMonitor | None = None,
    defer_target_masking: bool = False,
) -> EncodedInput:
    out: dict[Address, TensorFieldBase] = {}
    target_addresses = set(hyperparameters.target)

    if strata != Strata.predict and contains_mask_literal(batch):
        raise ValueError(f"{MASK_LITERAL!r} is only valid during predict strata")

    for address, request in hyperparameters.active_requests.items():
        TensorField = cast(type[TensorFieldBase], getattr(TENSORFIELDS[request.type], "TensorField"))

        if (strata == Strata.predict) & (address in target_addresses):
            out[address] = TensorField.empty(
                batch_size=len(batch),
                address=address,
                hyperparameters=hyperparameters,
            )
            continue

        expression = request.query
        if expression is None:
            raise ValueError(f"request '{address}' must define query")

        # `request.query` is relative to a processed observation. Direct
        # extractors add the outer batch selector; `query(...)` does the same
        # before JMESPath searches `batch`.
        extractor = compile_query_extractor(expression)
        result = extractor(batch) if extractor is not None else query(expression).search(batch)
        if jmespath_resolution_monitor is not None:
            jmespath_resolution_monitor.observe(address=address, expression=expression, result=result)

        kwargs: dict[str, Any] = dict(
            values=result,
            address=address,
            hyperparameters=hyperparameters,
            strata=strata,
        )
        if "interprocess_encoding_context" in inspect.signature(TensorField.new).parameters:
            kwargs["interprocess_encoding_context"] = interprocess_encoding_context.get(address)

        out[address] = TensorField.new(**kwargs)

        if not defer_target_masking and strata != Strata.predict and address in target_addresses:
            out[address].mask(p_prune=1.0)

    inputs = cast(EncodedInput, TensorDict(source=cast(Any, out)))

    if strata == Strata.predict:
        inputs[TensorKey.metadata] = batch

    return inputs


@beartype
def transform(
    pipe: Iterable[EncodedBatch],
    hyperparameters: Hyperparameters,
    strata: Strata,
    interprocess_encoding_context: InterprocessEncodingContext,
    jmespath_resolution_monitor: JMESPathResolutionMonitor | None = None,
) -> Iterator[EncodedInput]:
    for item in pipe:
        yield encode(
            batch=item,
            hyperparameters=hyperparameters,
            strata=strata,
            interprocess_encoding_context=interprocess_encoding_context,
            jmespath_resolution_monitor=jmespath_resolution_monitor,
            defer_target_masking=True,
        )


def _apply_mask_policy(
    field: TensorFieldBase,
    *,
    p_mask: float,
    p_prune: float,
    array_masks: tuple[Any, ...],
    address: Address,
    hyperparameters: Hyperparameters,
) -> None:
    parameters = inspect.signature(field.mask).parameters
    supports_policy_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    supports_policy_kwargs |= any(name in parameters for name in ("p_prune", "array_masks", "hyperparameters"))

    if supports_policy_kwargs:
        field.mask(
            p_mask=p_mask,
            p_prune=p_prune,
            array_masks=array_masks,
            address=address,
            hyperparameters=hyperparameters,
        )
        return

    if array_masks:
        raise TypeError(f"tensorfield at '{address}' must accept mask(..., array_masks=...) to use Array masks")

    if p_mask > 0.0:
        field.mask(p_mask=p_mask)

    if p_prune > 0.0:
        field.target(p_prune=p_prune)


@beartype
def mask(
    pipe: Iterable[EncodedInput],
    hyperparameters: Hyperparameters,
    strata: Strata = Strata.train,
) -> Iterator[EncodedInput]:
    for item in pipe:
        if strata == Strata.predict:
            yield item
            continue

        for address, request in hyperparameters.active_requests.items():
            p_mask = float(request.p_mask or 0.0)
            p_prune = float(request.p_prune or 0.0)
            array_masks = hyperparameters.array_masks_for(address)
            if p_mask <= 0.0 and p_prune <= 0.0 and not array_masks:
                continue

            _apply_mask_policy(
                item[address],
                p_mask=p_mask,
                p_prune=p_prune,
                array_masks=array_masks,
                address=address,
                hyperparameters=hyperparameters,
            )

        yield item


@beartype
def target(
    pipe: Iterable[EncodedInput],
    hyperparameters: Hyperparameters,
) -> Iterator[EncodedInput]:
    for item in pipe:
        for address, request in hyperparameters.active_requests.items():
            p_prune = float(request.p_prune or 0.0)
            if p_prune <= 0.0:
                continue

            item[address].target(p_prune=p_prune)

        yield item


def mock(hyperparameters: Hyperparameters, batch_size: int) -> EncodedInput:
    out: dict[Address, TensorFieldBase] = {}

    for address, request in hyperparameters.active_requests.items():
        TensorField = cast(type[TensorFieldBase], getattr(TENSORFIELDS[request.type], "TensorField"))
        out[address] = TensorField.empty(
            batch_size=batch_size,
            address=address,
            hyperparameters=hyperparameters,
        )

    return cast(EncodedInput, TensorDict(source=cast(Any, out), batch_size=batch_size))

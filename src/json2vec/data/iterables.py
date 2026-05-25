"""Composable iterable stages for fetching, preprocessing, and encoding data."""

from __future__ import annotations

import inspect
import random
from collections import Counter
from collections.abc import Iterable, Iterator
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
from json2vec.preprocessors.base import PREPROCESSORS, Preprocessor, PreprocessorMode
from json2vec.structs.enums import Strata
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
) -> EncodedInput:
    out: dict[Address, TensorFieldBase] = {}
    target_addresses = set(hyperparameters.target)

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

        # `request.query` is relative to a processed observation. `query(...)`
        # adds the outer batch selector before JMESPath searches `batch`.
        result = query(expression).search(batch)
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

        if address in target_addresses:
            out[address].target(p_prune=1.0)

    inputs = cast(EncodedInput, TensorDict(source=cast(Any, out)))

    if strata == Strata.predict:
        inputs["metadata"] = batch

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
        )


@beartype
def mask(
    pipe: Iterable[EncodedInput],
    hyperparameters: Hyperparameters,
) -> Iterator[EncodedInput]:
    for item in pipe:
        for address, request in hyperparameters.active_requests.items():
            p_mask = float(request.p_mask or 0.0)
            if p_mask <= 0.0:
                continue

            item[address].mask(p_mask=p_mask)

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

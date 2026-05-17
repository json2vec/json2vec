from __future__ import annotations

import enum
import inspect
from collections.abc import Callable, Iterator
from functools import cache
from typing import Any

import pluggy
import pydantic

from json2vec.processors.spec import PluginSpec

pm: pluggy.PluginManager = pluggy.PluginManager(project_name="processors")

pm.add_hookspecs(module_or_class=PluginSpec)


class ProcessorMode(enum.StrEnum):
    generator = "generator"
    transformation = "transformation"


class Processor(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)
    name: str
    func: Callable[..., Any]
    mode: ProcessorMode

    def __call__(self, observation: dict, **kwargs) -> Any:
        return self.func(observation, **_filter_supported_kwargs(self.func, kwargs))

    def outputs(self, observation: dict, **kwargs) -> Iterator[list[dict[str, Any]]]:
        result = self(observation, **kwargs)

        if self.mode == ProcessorMode.transformation:
            yield [self._normalize_object(result, mode=self.mode)]
            return

        if self.mode == ProcessorMode.generator:
            if isinstance(result, list):
                iterable: list[Any] | Iterator[Any] = result
            elif isinstance(result, Iterator):
                iterable = result
            else:
                raise TypeError(
                    f"generator processor '{self.name}' must yield dict objects or return a list of dict objects, "
                    f"got {type(result).__name__}"
                )

            for output in iterable:
                yield [self._normalize_object(output, mode=self.mode)]
            return

        raise ValueError(f"unsupported processor mode: {self.mode}")

    def _normalize_object(self, output: Any, *, mode: ProcessorMode) -> dict[str, Any]:
        if not isinstance(output, dict):
            raise TypeError(
                f"{mode} processor '{self.name}' must produce dict objects, got {type(output).__name__}"
            )

        return output


@cache
def _accepted_kwargs(func: Callable[..., Any]) -> tuple[bool, frozenset[str]]:
    signature = inspect.signature(func)
    accepts_variadic_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    accepted = frozenset(signature.parameters.keys())
    return accepts_variadic_kwargs, accepted


def _filter_supported_kwargs(func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    accepts_variadic_kwargs, accepted = _accepted_kwargs(func)
    if accepts_variadic_kwargs:
        return kwargs

    return {key: value for key, value in kwargs.items() if key in accepted}


PROCESSORS: dict[str, Processor] = {}


def _register(func: Callable[..., Any], *, mode: ProcessorMode) -> Callable[..., Any]:
    name = func.__name__

    if name in PROCESSORS:
        raise ValueError(f"Processor '{name}' is already registered.")

    PROCESSORS[name] = Processor(name=name, func=func, mode=mode)

    return func


def shim(
    func: Callable[..., Any] | None = None,
    *,
    yields: bool | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    if "yield" in kwargs:
        if yields is not None:
            raise TypeError("use either 'yields' or 'yield', not both")
        yields = kwargs.pop("yield")

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected shim keyword argument(s): {unexpected}")

    if yields is None:
        yields = False

    if not isinstance(yields, bool):
        raise TypeError("yields must be a boolean")

    mode = ProcessorMode.generator if yields else ProcessorMode.transformation

    def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
        return _register(inner, mode=mode)

    if func is None:
        return decorator

    if not callable(func):
        raise TypeError("shim can only decorate callables")

    return decorator(func)

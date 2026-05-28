"""Preprocessor decorator and registry implementation."""

from __future__ import annotations

import enum
import inspect
from collections.abc import Callable, Iterator
from functools import cache
from typing import Any

import pluggy
import pydantic

from json2vec.preprocessors.spec import PluginSpec

pm: pluggy.PluginManager = pluggy.PluginManager(project_name="preprocessors")

pm.add_hookspecs(module_or_class=PluginSpec)


class PreprocessorMode(enum.StrEnum):
    """Execution mode for a registered preprocessor."""

    generator = "generator"
    transformation = "transformation"

    @classmethod
    def from_yields(cls, yields: bool) -> "PreprocessorMode":
        if not isinstance(yields, bool):
            raise TypeError("yields must be a boolean")

        return cls.generator if yields else cls.transformation


class Preprocessor(pydantic.BaseModel):
    """Registered observation preprocessor.

    A transformation preprocessor returns one dict. A generator preprocessor
    yields or returns multiple dict objects, each of which becomes a processed
    observation.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)
    name: str
    func: Callable[..., Any]
    mode: PreprocessorMode

    @staticmethod
    @cache
    def accepted_kwargs(func: Callable[..., Any]) -> tuple[bool, frozenset[str]]:
        signature = inspect.signature(func)
        accepts_variadic_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
        accepted = frozenset(signature.parameters.keys())
        return accepts_variadic_kwargs, accepted

    @classmethod
    def filter_supported_kwargs(cls, func: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
        accepts_variadic_kwargs, accepted = cls.accepted_kwargs(func)
        if accepts_variadic_kwargs:
            return kwargs

        return {key: value for key, value in kwargs.items() if key in accepted}

    @classmethod
    def register(cls, func: Callable[..., Any], *, mode: PreprocessorMode) -> Callable[..., Any]:
        name = getattr(func, "__name__", type(func).__name__)
        PREPROCESSORS[name] = cls(name=name, func=func, mode=mode)
        return func

    def __call__(self, observation: dict, **kwargs) -> Any:
        return self.func(observation, **self.filter_supported_kwargs(self.func, kwargs))

    def outputs(self, observation: dict, **kwargs) -> Iterator[list[dict[str, Any]]]:
        """Yield normalized processed observations for one raw observation."""
        result = self(observation, **kwargs)

        if self.mode == PreprocessorMode.transformation:
            yield [self.require_object(result, mode=self.mode)]
            return

        if self.mode == PreprocessorMode.generator:
            if isinstance(result, list):
                iterable: list[Any] | Iterator[Any] = result
            elif isinstance(result, Iterator):
                iterable = result
            else:
                raise TypeError(
                    f"generator preprocessor '{self.name}' must yield dict objects or return a list of dict objects, "
                    f"got {type(result).__name__}"
                )

            for output in iterable:
                yield [self.require_object(output, mode=self.mode)]
            return

        raise ValueError(f"unsupported preprocessor mode: {self.mode}")

    def require_object(self, output: Any, *, mode: PreprocessorMode) -> dict[str, Any]:
        if not isinstance(output, dict):
            raise TypeError(f"{mode} preprocessor '{self.name}' must produce dict objects, got {type(output).__name__}")

        return output


PREPROCESSORS: dict[str, Preprocessor] = {}


def preprocess(
    func: Callable[..., Any] | None = None,
    *,
    yields: bool | None = None,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Register a callable as a JSON2Vec preprocessor.

    Args:
        func: Callable to register when used as `@preprocess`.
        yields: Set to `True` for generator preprocessors.
        **kwargs: Reserved for validation of unsupported decorator arguments.

    Returns:
        The original callable, after registering it in `PREPROCESSORS`.

    Example:
        ```python
        import json2vec as j2v

        @j2v.preprocess
        def normalize(record: dict) -> dict:
            return {**record, "amount": float(record["amount"])}
        ```
    """
    if "yield" in kwargs:
        if yields is not None:
            raise TypeError("use either 'yields' or 'yield', not both")
        yields = kwargs.pop("yield")

    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected preprocess keyword argument(s): {unexpected}")

    if yields is None:
        yields = False

    mode = PreprocessorMode.from_yields(yields)

    def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
        return Preprocessor.register(inner, mode=mode)

    if func is None:
        return decorator

    if not callable(func):
        raise TypeError("preprocess can only decorate callables")

    return decorator(func)

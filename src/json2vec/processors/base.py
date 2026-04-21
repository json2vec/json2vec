from __future__ import annotations

import ast
import enum
import inspect
import textwrap
from functools import cache
from typing import Any, Callable

import pluggy
import pydantic

from json2vec.processors.spec import PluginSpec

pm: pluggy.PluginManager = pluggy.PluginManager(project_name="processors")

pm.add_hookspecs(module_or_class=PluginSpec)


class ProcessorMode(enum.StrEnum):
    yielding = "yield"
    returning = "return"


def has_yield_expression(node: ast.AST, root: bool = False) -> bool:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.Yield, ast.YieldFrom)):
            return True

        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if root and has_yield_expression(child):
                return True
            continue

        if has_yield_expression(child):
            return True

    return False


def is_yielding_processor(func: Callable[..., Any]) -> bool:
    try:
        source: str = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        return inspect.isgeneratorfunction(func)

    module: ast.Module = ast.parse(source)
    candidates: list[ast.FunctionDef | ast.AsyncFunctionDef] = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    target = next((node for node in candidates if node.name == func.__name__), None)
    if target is None:
        return inspect.isgeneratorfunction(func)

    return has_yield_expression(target, root=True)


class Processor(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, frozen=True)
    name: str
    func: Callable[..., Any]
    mode: ProcessorMode

    def __call__(self, observation: dict, **kwargs) -> Any:
        return self.func(observation, **_filter_supported_kwargs(self.func, kwargs))


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


def register(func: Callable[..., Any]) -> Callable[..., Any]:
    name = func.__name__

    if name in PROCESSORS:
        raise ValueError(f"Processor '{name}' is already registered.")

    mode: ProcessorMode = ProcessorMode.yielding if is_yielding_processor(func) else ProcessorMode.returning
    PROCESSORS[name] = Processor(name=name, func=func, mode=mode)

    return func

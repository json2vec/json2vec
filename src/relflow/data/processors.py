"""Typed preprocessor and postprocessor callable wrappers."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import UnionType
from typing import Any, Self, TypeAlias, Union, get_args, get_origin, overload

from relflow.rich import strip_control_sequences
from relflow.structs.tree import Address

RawObservation: TypeAlias = dict[str, Any]
RawBatch: TypeAlias = list[RawObservation] | list[list[RawObservation]]
Metadata: TypeAlias = list[Any]
Predictions: TypeAlias = dict[Address, dict[str, Any]]
PostprocessorResult: TypeAlias = Mapping[str | Address, Any] | None

_EMPTY = inspect.Signature.empty
_DISPLAY_FIELD_LIMIT = 8
_DISPLAY_NAME_LIMIT = 80
_DISPLAY_SIGNATURE_LIMIT = 240


class PreprocessorProvider(StrEnum):
    """Pipeline-owned preprocessor parameter names."""

    strata = "strata"
    schema = "schema"
    encoding_context = "encoding_context"


class PostprocessorProvider(StrEnum):
    """Pipeline-owned postprocessor parameter names."""

    metadata = "metadata"
    input = "input"
    batch = "batch"
    observations = "observations"
    request = "request"
    batch_indices = "batch_indices"
    batch_idx = "batch_idx"
    dataloader_idx = "dataloader_idx"


ProviderName: TypeAlias = PreprocessorProvider | PostprocessorProvider
PREPROCESSOR_PROVIDERS = frozenset(PreprocessorProvider)
POSTPROCESSOR_PROVIDERS = frozenset(PostprocessorProvider)


def _bounded_name(value: str, *, limit: int = _DISPLAY_NAME_LIMIT) -> str:
    value = " ".join(strip_control_sequences(value).split()) or "<unnamed>"
    if len(value) <= limit:
        return value
    return f"{value[: limit - 1]}…"


def _bounded_names(values: Iterable[str]) -> tuple[str, ...]:
    names = sorted(_bounded_name(value) for value in values)
    visible = names[:_DISPLAY_FIELD_LIMIT]
    omitted = len(names) - len(visible)
    if omitted:
        visible.append(f"… +{omitted}")
    return tuple(visible)


def _observation_field_types(data: Mapping[str, Any]) -> tuple[dict[str, str], int]:
    fields: dict[str, str] = {}
    rendered = 0
    for key, value in data.items():
        if rendered >= _DISPLAY_FIELD_LIMIT:
            break
        key_label = key if isinstance(key, str) else f"<{type(key).__name__}>"
        safe_key = _bounded_name(key_label)
        fields[safe_key] = _bounded_name(type(value).__name__)
        rendered += 1
    return fields, max(0, len(data) - rendered)


@dataclass(frozen=True)
class Observation:
    """Processed model-facing observation emitted by a preprocessor."""

    data: Mapping[str, Any] = field(repr=False)

    def __repr__(self) -> str:
        fields, omitted = _observation_field_types(self.data)
        suffix = f", omitted={omitted}" if omitted else ""
        return f"Observation(count={len(self.data)}, fields={fields!r}{suffix})"

    def __rich_repr__(self):
        fields, omitted = _observation_field_types(self.data)
        yield "count", len(self.data)
        yield "fields", fields
        if omitted:
            yield "omitted", omitted


def _processor_name(func: Callable[..., Any]) -> str:
    return _bounded_name(str(getattr(func, "__name__", type(func).__name__)))


def _safe_signature(signature: inspect.Signature) -> str:
    """Render parameter structure without exposing defaults or annotations."""

    parameters = [parameter.replace(default=_EMPTY, annotation=_EMPTY) for parameter in signature.parameters.values()]
    visible = parameters[:_DISPLAY_FIELD_LIMIT]
    omitted = len(parameters) - len(visible)
    rendered = str(signature.replace(parameters=visible, return_annotation=_EMPTY))
    suffix = f" … +{omitted} parameters" if omitted else ""
    return f"{_bounded_name(rendered, limit=_DISPLAY_SIGNATURE_LIMIT - len(suffix))}{suffix}"


def _allows_none(annotation: Any) -> bool:
    if annotation is _EMPTY:
        return False
    if annotation is None or annotation is type(None):
        return True

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return any(item is type(None) for item in get_args(annotation))

    return False


def _is_optional(parameter: inspect.Parameter) -> bool:
    return parameter.default is not _EMPTY or _allows_none(parameter.annotation)


@dataclass(frozen=True)
class Processor:
    """Callable wrapper with explicit user parameter binding."""

    func: Callable[..., Any] = field(repr=False)
    provider_names: frozenset[ProviderName]
    primary_name: str
    kind: str
    signature: inspect.Signature = field(init=False)
    runtime_params: frozenset[str] = field(init=False)
    user_params: frozenset[str] = field(init=False)
    bound_params: Mapping[str, Any] = field(default_factory=dict, repr=False)
    name: str = field(init=False)
    decorated: bool = False

    def _missing_user_params(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in sorted(self.user_params)
            if name not in self.bound_params and not _is_optional(self.signature.parameters[name])
        )

    def __repr__(self) -> str:
        missing = self._missing_user_params()
        return (
            f"{type(self).__name__}(name={self.name!r}, signature={_safe_signature(self.signature)!r}, "
            f"ready={not missing!r}, runtime={_bounded_names(self.runtime_params)!r}, "
            f"bound={_bounded_names(self.bound_params)!r}, missing={_bounded_names(missing)!r})"
        )

    def __rich_repr__(self):
        missing = self._missing_user_params()
        yield "name", self.name
        yield "signature", _safe_signature(self.signature)
        yield "ready", not missing
        if self.runtime_params:
            yield "runtime", _bounded_names(self.runtime_params)
        if self.bound_params:
            yield "bound", _bounded_names(self.bound_params)
        if missing:
            yield "missing", _bounded_names(missing)

    def __post_init__(self) -> None:
        signature = inspect.signature(self.func)
        runtime_params, user_params = self._classify_signature(signature)
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "runtime_params", frozenset(runtime_params))
        object.__setattr__(self, "user_params", frozenset(user_params))
        object.__setattr__(self, "name", _processor_name(self.func))
        self._validate_bound_params()

    def _classify_signature(self, signature: inspect.Signature) -> tuple[set[str], set[str]]:
        parameters = list(signature.parameters.values())
        if not parameters:
            raise TypeError(f"{self.kind} '{_processor_name(self.func)}' must accept {self.primary_name!r}")

        first = parameters[0]
        if first.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"{self.kind} '{_processor_name(self.func)}' has invalid first parameter")

        if self.primary_name == "predictions" and first.name != "predictions":
            raise TypeError(f"postprocessor '{_processor_name(self.func)}' first parameter must be 'predictions'")

        provider_names = {provider.value for provider in self.provider_names}
        runtime_params: set[str] = set()
        user_params: set[str] = set()
        for parameter in parameters[1:]:
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TypeError(f"{self.kind} '{_processor_name(self.func)}' does not support *args")
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                raise TypeError(f"{self.kind} '{_processor_name(self.func)}' does not support **kwargs")
            if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
                raise TypeError(
                    f"{self.kind} '{_processor_name(self.func)}' parameter '{parameter.name}' must be keyword-only"
                )
            if self.kind == "postprocessor" and parameter.name == "context":
                raise TypeError("postprocessor context dictionaries were removed; request named providers instead")
            if parameter.name in provider_names:
                runtime_params.add(parameter.name)
            else:
                user_params.add(parameter.name)

        return runtime_params, user_params

    def _validate_bound_params(self) -> None:
        for name in self.bound_params:
            if name in self.runtime_params:
                raise ValueError(f"{self.kind} '{self.name}' parameter '{name}' is provided by the pipeline")
            if name not in self.user_params:
                raise ValueError(f"{self.kind} '{self.name}' has no user-bound parameter '{name}'")

    def partial(self, **values: Any) -> Self:
        for name in values:
            if name in self.runtime_params:
                raise ValueError(f"{self.kind} '{self.name}' parameter '{name}' is provided by the pipeline")
            if name not in self.user_params:
                raise ValueError(f"{self.kind} '{self.name}' has no user-bound parameter '{name}'")
            if name in self.bound_params:
                raise ValueError(f"{self.kind} '{self.name}' parameter '{name}' is already bound")

        return replace(self, bound_params={**self.bound_params, **values})

    def validate_ready(self) -> None:
        missing = self._missing_user_params()

        if missing:
            formatted = ", ".join(repr(name) for name in missing)
            raise ValueError(f"{self.kind} '{self.name}' requires unbound parameter(s): {formatted}")

    def _call(self, primary: Any, runtime_values: Mapping[str, Any]) -> Any:
        self.validate_ready()
        supplied = {name: runtime_values[name] for name in self.runtime_params if name in runtime_values}
        return self.func(primary, **dict(self.bound_params), **supplied)

    def __call__(self, primary: Any, **runtime_values: Any) -> Any:
        return self._call(primary, runtime_values)


@dataclass(frozen=True, repr=False)
class Preprocessor(Processor):
    """Callable preprocessor object returned by `@preprocess`."""

    func: Callable[..., Any]
    provider_names: frozenset[ProviderName] = field(default=PREPROCESSOR_PROVIDERS, init=False)
    primary_name: str = field(default="observation", init=False)
    kind: str = field(default="preprocessor", init=False)

    @classmethod
    def normalize(cls, value: "Preprocessor | None") -> "Preprocessor | None":
        if value is None:
            return None
        if isinstance(value, cls):
            value.validate_ready()
            return value

        raise TypeError(f"preprocessor must be a Preprocessor object or None, got {type(value).__name__}")

    def outputs(
        self,
        observation: RawObservation,
        *,
        strata: Any,
        schema: Any,
        encoding_context: Any,
    ) -> Iterable[list[RawObservation]]:
        result = self._call(
            observation,
            {
                PreprocessorProvider.strata.value: strata,
                PreprocessorProvider.schema.value: schema,
                PreprocessorProvider.encoding_context.value: encoding_context,
            },
        )
        yield from self._normalize_outputs(result)

    def _normalize_outputs(self, result: Any) -> Iterable[list[RawObservation]]:
        if result is None:
            return

        if isinstance(result, Observation):
            yield [dict(result.data)]
            return

        if isinstance(result, (str, bytes, Mapping)):
            raise TypeError(
                f"preprocessor '{self.name}' must return Observation, None, or an iterable of Observation | None; "
                f"got {type(result).__name__}"
            )
        if not isinstance(result, Iterable):
            raise TypeError(
                f"preprocessor '{self.name}' must return Observation, None, or an iterable of Observation | None; "
                f"got {type(result).__name__}"
            )

        for output in result:
            if output is None:
                continue
            if not isinstance(output, Observation):
                raise TypeError(
                    f"preprocessor '{self.name}' yielded {type(output).__name__}; expected Observation or None"
                )
            yield [dict(output.data)]


@dataclass(frozen=True, repr=False)
class Postprocessor(Processor):
    """Callable postprocessor object returned by `@postprocess`."""

    func: Callable[..., Any]
    provider_names: frozenset[ProviderName] = field(default=POSTPROCESSOR_PROVIDERS, init=False)
    primary_name: str = field(default="predictions", init=False)
    kind: str = field(default="postprocessor", init=False)

    @classmethod
    def normalize(cls, value: "Postprocessor | None") -> "Postprocessor | None":
        if value is None:
            return None
        if isinstance(value, cls):
            value.validate_ready()
            return value

        raise TypeError(f"postprocessor must be a Postprocessor object or None, got {type(value).__name__}")

    def run(self, predictions: Predictions, *, available: Mapping[str, Any]) -> PostprocessorResult:
        runtime_values: dict[str, Any] = {}
        for name in self.runtime_params:
            if name in available:
                runtime_values[name] = available[name]
                continue

            parameter = self.signature.parameters[name]
            if _is_optional(parameter):
                runtime_values[name] = None
                continue

            raise ValueError(
                f"postprocessor parameter '{name}' is not available in this runtime; "
                "make it optional or use this postprocessor only with a compatible runtime"
            )

        result = self._call(predictions, runtime_values)
        if result is None:
            return None
        if not isinstance(result, Mapping):
            raise TypeError(f"postprocessor '{self.name}' must return a mapping or None, got {type(result).__name__}")

        return result


@overload
def preprocess(func: Callable[..., Any], **kwargs: Any) -> Preprocessor: ...


@overload
def preprocess(
    func: None = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Preprocessor]: ...


def preprocess(
    func: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Preprocessor] | Preprocessor:
    """Validate and wrap a callable as a relflow preprocessor."""
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected preprocess keyword argument(s): {unexpected}")

    def decorator(inner: Callable[..., Any]) -> Preprocessor:
        if not callable(inner):
            raise TypeError("preprocess can only decorate callables")
        return Preprocessor(func=inner, decorated=True)

    if func is None:
        return decorator
    return decorator(func)


@overload
def postprocess(func: Callable[..., Any], **kwargs: Any) -> Postprocessor: ...


@overload
def postprocess(
    func: None = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Postprocessor]: ...


def postprocess(
    func: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Postprocessor] | Postprocessor:
    """Validate and wrap a callable as a relflow postprocessor."""
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected postprocess keyword argument(s): {unexpected}")

    def decorator(inner: Callable[..., Any]) -> Postprocessor:
        if not callable(inner):
            raise TypeError("postprocess can only decorate callables")
        return Postprocessor(func=inner, decorated=True)

    if func is None:
        return decorator
    return decorator(func)

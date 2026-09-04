"""Arrow-native preprocessor and postprocessor contracts."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal, Self, TypeAlias, overload

import pyarrow as pa

from relflow.data.arrow import Batch

EMPTY = inspect.Signature.empty
Scope = Literal["partition", "dataset"]


class PreprocessorProvider(StrEnum):
    """Pipeline-owned preprocessor parameter names."""

    strata = "strata"
    schema = "schema"
    encoding_context = "encoding_context"


PREPROCESSOR_PROVIDERS = frozenset(provider.value for provider in PreprocessorProvider)


def label(func: Callable[..., Any]) -> str:
    """Return a stable display name for a processor callable."""

    return getattr(func, "__name__", type(func).__name__)


def columns(value: tuple[str, ...] | None, *, name: str) -> tuple[str, ...] | None:
    """Validate one ordered top-level column declaration."""

    if value is None:
        return None
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of top-level column names or None")
    if any(not isinstance(item, str) or not item for item in value):
        raise TypeError(f"{name} entries must be non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} entries must be unique")
    return value


def equal(left: pa.Array | pa.ChunkedArray, right: pa.Array | pa.ChunkedArray) -> bool:
    """Compare Arrow arrays independent of chunk boundaries."""

    left_array = left.combine_chunks() if isinstance(left, pa.ChunkedArray) else left
    right_array = right.combine_chunks() if isinstance(right, pa.ChunkedArray) else right
    return left_array.equals(right_array)


@dataclass(frozen=True, slots=True, kw_only=True)
class Processor:
    """Callable wrapper with explicit pipeline and user parameter binding."""

    func: Callable[..., Any]
    kind: str
    providers: frozenset[str]
    bound: Mapping[str, Any] = field(default_factory=dict)
    requires: tuple[str, ...] | None = None
    produces: tuple[str, ...] = ()
    signature: inspect.Signature = field(init=False)
    runtime: frozenset[str] = field(init=False)
    user: frozenset[str] = field(init=False)
    name: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "requires", columns(self.requires, name="requires"))
        produces = columns(self.produces, name="produces")
        if produces is None:
            raise TypeError("produces must be a tuple of top-level column names")
        object.__setattr__(self, "produces", produces)

        signature = inspect.signature(self.func)
        runtime, user = self.classify(signature)
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "runtime", frozenset(runtime))
        object.__setattr__(self, "user", frozenset(user))
        object.__setattr__(self, "name", label(self.func))
        self.bindings()

    @classmethod
    def normalize(cls, value: Self | list[Self] | tuple[Self, ...] | None) -> tuple[Self, ...]:
        """Normalize optional processor configuration to an immutable pipeline."""

        if value is None:
            return ()
        if isinstance(value, cls):
            processors = (value,)
        elif isinstance(value, (list, tuple)):
            processors = tuple(value)
        else:
            raise TypeError(
                f"{cls.__name__.lower()} must be a {cls.__name__}, list, tuple, or None; got {type(value).__name__}"
            )

        for index, processor in enumerate(processors):
            if not isinstance(processor, cls):
                raise TypeError(
                    f"{cls.__name__.lower()} at index {index} must be a {cls.__name__}; got {type(processor).__name__}"
                )
            processor.ready()
        return processors

    def classify(self, signature: inspect.Signature) -> tuple[set[str], set[str]]:
        """Classify callable parameters as pipeline-owned or user-bound."""

        parameters = list(signature.parameters.values())
        if not parameters:
            raise TypeError(f"{self.kind} '{label(self.func)}' must accept 'batch'")

        first = parameters[0]
        if first.name != "batch" or first.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(f"{self.kind} '{label(self.func)}' first parameter must be 'batch'")

        runtime: set[str] = set()
        user: set[str] = set()
        for parameter in parameters[1:]:
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                raise TypeError(f"{self.kind} '{label(self.func)}' does not support *args")
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                raise TypeError(f"{self.kind} '{label(self.func)}' does not support **kwargs")
            if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
                raise TypeError(f"{self.kind} '{label(self.func)}' parameter '{parameter.name}' must be keyword-only")
            if parameter.name in self.providers:
                runtime.add(parameter.name)
            else:
                user.add(parameter.name)

        return runtime, user

    def bindings(self) -> None:
        """Validate explicitly bound user arguments."""

        for name in self.bound:
            if name in self.runtime:
                raise ValueError(f"{self.kind} '{self.name}' parameter '{name}' is provided by the pipeline")
            if name not in self.user:
                raise ValueError(f"{self.kind} '{self.name}' has no user-bound parameter '{name}'")

    def ready(self) -> None:
        """Validate that every required user argument is bound."""

        self.bindings()
        missing = [
            name
            for name in sorted(self.user)
            if name not in self.bound and self.signature.parameters[name].default is EMPTY
        ]
        if missing:
            formatted = ", ".join(repr(name) for name in missing)
            raise ValueError(f"{self.kind} '{self.name}' requires unbound parameter(s): {formatted}")

    def partial(self, **values: Any) -> Self:
        """Return a processor with additional immutable user configuration."""

        for name in values:
            if name in self.runtime:
                raise ValueError(f"{self.kind} '{self.name}' parameter '{name}' is provided by the pipeline")
            if name not in self.user:
                raise ValueError(f"{self.kind} '{self.name}' has no user-bound parameter '{name}'")
            if name in self.bound:
                raise ValueError(f"{self.kind} '{self.name}' parameter '{name}' is already bound")

        return replace(self, bound={**self.bound, **values})

    def call(self, batch: Batch, runtime: Mapping[str, Any]) -> Any:
        """Invoke the wrapped callable with validated arguments."""

        if not isinstance(batch, Batch):
            raise TypeError(f"{self.kind} '{self.name}' requires an rf.Batch, got {type(batch).__name__}")
        if self.requires is not None:
            missing = [name for name in self.requires if name not in batch.data.column_names]
            if missing:
                names = ", ".join(repr(name) for name in missing)
                raise KeyError(f"{self.kind} '{self.name}' requires absent column(s): {names}")
        self.ready()
        missing = sorted(self.runtime - runtime.keys())
        if missing:
            formatted = ", ".join(repr(name) for name in missing)
            raise ValueError(f"{self.kind} '{self.name}' is missing pipeline parameter(s): {formatted}")
        supplied = {name: runtime[name] for name in self.runtime}
        return self.func(batch, **dict(self.bound), **supplied)

    def verify(self, batch: Batch) -> None:
        """Validate the processor's declared output columns."""

        missing = [name for name in self.produces if name not in batch.data.column_names]
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise KeyError(f"{self.kind} '{self.name}' did not produce declared column(s): {names}")

    def __call__(self, batch: Batch, **runtime: Any) -> Any:
        return self.call(batch, runtime)


@dataclass(frozen=True, slots=True, kw_only=True)
class Preprocessor(Processor):
    """Arrow batch transform returned by :func:`preprocess`."""

    func: Callable[..., Any]
    scope: Scope = "partition"
    kind: str = field(default="preprocessor", init=False)
    providers: frozenset[str] = field(default=PREPROCESSOR_PROVIDERS, init=False)

    def __post_init__(self) -> None:
        if self.scope not in ("partition", "dataset"):
            raise ValueError("preprocessor scope must be 'partition' or 'dataset'")
        Processor.__post_init__(self)

    def run(
        self,
        batch: Batch,
        *,
        strata: Any,
        schema: Any,
        encoding_context: Any,
    ) -> Iterable[Batch]:
        """Yield the zero, one, or many Arrow batches produced by one call."""

        result = self.call(
            batch,
            {
                PreprocessorProvider.strata.value: strata,
                PreprocessorProvider.schema.value: schema,
                PreprocessorProvider.encoding_context.value: encoding_context,
            },
        )
        if result is None:
            return
        if isinstance(result, Batch):
            self.verify(result)
            yield result
            return
        if isinstance(result, (str, bytes, Mapping, pa.Table)) or not isinstance(result, Iterable):
            raise TypeError(
                f"preprocessor '{self.name}' must return Batch, Iterable[Batch], or None; got {type(result).__name__}"
            )

        for item in result:
            if not isinstance(item, Batch):
                raise TypeError(f"preprocessor '{self.name}' yielded {type(item).__name__}; expected Batch")
            self.verify(item)
            yield item


@dataclass(frozen=True, slots=True, kw_only=True)
class Postprocessor(Processor):
    """Same-row Arrow output transform returned by :func:`postprocess`."""

    func: Callable[..., Any]
    kind: str = field(default="postprocessor", init=False)
    providers: frozenset[str] = field(default_factory=frozenset, init=False)

    def run(self, batch: Batch) -> Batch:
        """Apply a same-row output transform and validate its Arrow contract."""

        result = self.call(batch, {})
        if not isinstance(result, Batch):
            raise TypeError(f"postprocessor '{self.name}' must return Batch, got {type(result).__name__}")
        if len(result) != len(batch):
            raise ValueError(f"postprocessor '{self.name}' returned {len(result)} rows; expected {len(batch)}")
        if not equal(result.identity, batch.identity):
            raise ValueError(f"postprocessor '{self.name}' must preserve Batch identity")
        self.verify(result)
        if result.data.num_columns == 0:
            raise ValueError(f"postprocessor '{self.name}' must return at least one Arrow column")
        return result


PreprocessorInput: TypeAlias = Preprocessor | list[Preprocessor] | tuple[Preprocessor, ...] | None
PostprocessorInput: TypeAlias = Postprocessor | list[Postprocessor] | tuple[Postprocessor, ...] | None


def apply(batch: Batch, processors: PostprocessorInput = ()) -> Batch:
    """Apply an ordered postprocessor pipeline to one Arrow batch."""

    result = batch
    for processor in Postprocessor.normalize(processors):
        result = processor.run(result)
    return result


@overload
def preprocess(
    func: Callable[..., Any],
    /,
    *,
    scope: Scope = "partition",
    requires: tuple[str, ...] | None = None,
    produces: tuple[str, ...] = (),
) -> Preprocessor: ...


@overload
def preprocess(
    func: None = None,
    /,
    *,
    scope: Scope = "partition",
    requires: tuple[str, ...] | None = None,
    produces: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Preprocessor]: ...


def preprocess(
    func: Callable[..., Any] | None = None,
    /,
    *,
    scope: Scope = "partition",
    requires: tuple[str, ...] | None = None,
    produces: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Preprocessor] | Preprocessor:
    """Wrap a callable as an Arrow preprocessor."""

    def decorate(inner: Callable[..., Any]) -> Preprocessor:
        if not callable(inner):
            raise TypeError("preprocess can only decorate callables")
        return Preprocessor(func=inner, scope=scope, requires=requires, produces=produces)

    if func is None:
        return decorate
    return decorate(func)


@overload
def postprocess(
    func: Callable[..., Any],
    /,
    *,
    requires: tuple[str, ...] | None = None,
    produces: tuple[str, ...] = (),
) -> Postprocessor: ...


@overload
def postprocess(
    func: None = None,
    /,
    *,
    requires: tuple[str, ...] | None = None,
    produces: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Postprocessor]: ...


def postprocess(
    func: Callable[..., Any] | None = None,
    /,
    *,
    requires: tuple[str, ...] | None = None,
    produces: tuple[str, ...] = (),
) -> Callable[[Callable[..., Any]], Postprocessor] | Postprocessor:
    """Wrap a callable as a same-row Arrow postprocessor."""

    def decorate(inner: Callable[..., Any]) -> Postprocessor:
        if not callable(inner):
            raise TypeError("postprocess can only decorate callables")
        return Postprocessor(func=inner, requires=requires, produces=produces)

    if func is None:
        return decorate
    return decorate(func)


__all__ = [
    "Postprocessor",
    "Preprocessor",
    "PreprocessorProvider",
    "postprocess",
    "preprocess",
]

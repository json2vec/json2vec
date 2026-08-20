"""Base model and active Pluggy registry for stateless metrics."""

from __future__ import annotations

import enum
import inspect
import re
import string
from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, TypeVar, cast

import pluggy
import pydantic
import torch
from pydantic_core import core_schema

from relflow.metrics.spec import PluginSpec, hookimpl
from relflow.structs.enums import Strata
from relflow.structs.tree import Address

if TYPE_CHECKING:
    from relflow.architecture.root import Model


class Trait(enum.Enum):
    """Common prepared-input contracts shared by compatible tensorfields."""

    classification = "classification"
    regression = "regression"
    cyclic = "cyclic"

    @property
    def metrics(self) -> list[Metric]:
        """Return fresh default configurations from the live metric registry."""
        return [
            cast(Metric, plugin.default).model_copy(deep=True)
            for _, plugin in sorted(METRICS.items())
            if self in plugin.traits
        ]


class Metric(pydantic.BaseModel, ABC):
    """Serializable configuration and stateless behavior for one metric."""

    model_config = pydantic.ConfigDict(
        defer_build=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    _formatter: ClassVar[string.Formatter] = string.Formatter()

    type: str
    name: str

    @pydantic.field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, name: str) -> str:
        available = set(cls.model_fields)
        try:
            parsed = list(cls._formatter.parse(name))
        except ValueError as error:
            raise ValueError(f"invalid metric name template: {error}") from error

        for _, field_name, format_spec, conversion in parsed:
            if field_name is None:
                continue
            if not field_name:
                raise ValueError("metric name templates may not use anonymous placeholders")
            if not field_name.isidentifier():
                raise ValueError("metric name template placeholders must be top-level field names")
            if field_name == "name":
                raise ValueError("metric name templates may not reference 'name'")
            if field_name not in available:
                raise ValueError(f"metric name template references unknown field {field_name!r}")
            if conversion is not None:
                raise ValueError("metric name templates may not use conversions")
            if "{" in format_spec or "}" in format_spec:
                raise ValueError("metric name templates may not use nested replacement fields")
        return name

    @pydantic.model_validator(mode="after")
    def validate_rendered_name(self) -> Metric:
        str(self)
        return self

    def __str__(self) -> str:
        fields = {
            field_name: getattr(self, field_name) for field_name in type(self).model_fields if field_name != "name"
        }
        try:
            rendered = self._formatter.vformat(self.name, (), fields)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError(f"could not render metric name template: {error}") from error

        if not rendered:
            raise ValueError("rendered metric name must not be empty")
        if rendered != rendered.lower():
            raise ValueError("rendered metric name must be lowercase")
        if "/" in rendered:
            raise ValueError("rendered metric name must not contain '/'")
        if any(character.isspace() for character in rendered):
            raise ValueError("rendered metric name must not contain whitespace")
        return rendered

    @abstractmethod
    def __call__(
        self,
        module: Model,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trainable: torch.Tensor,
        *,
        address: Address,
        strata: Strata,
        scope: tuple[str, ...],
    ) -> None:
        """Compute and track one stateless scalar metric."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: pydantic.GetCoreSchemaHandler,
    ) -> core_schema.CoreSchema:
        # Concrete subclasses need their normal generated schema. Dispatching
        # there would recurse through ``Concrete.model_validate`` forever.
        if "__get_pydantic_core_schema__" not in cls.__dict__ or source_type is not cls:
            return handler(source_type)

        metric_mapping_schema = core_schema.typed_dict_schema(
            {
                "type": core_schema.typed_dict_field(core_schema.str_schema(), required=True),
                "name": core_schema.typed_dict_field(core_schema.str_schema(), required=False),
            },
            extra_behavior="allow",
        )
        return core_schema.no_info_before_validator_function(
            registry.parse,
            core_schema.any_schema(),
            json_schema_input_schema=metric_mapping_schema,
        )


@dataclass(frozen=True)
class MetricPlugin:
    """Registry metadata contributed by a metric provider."""

    key: str
    Metric: type[Metric]
    traits: frozenset[Trait]
    data_types: frozenset[str]
    default: Metric | None

    @hookimpl
    def metric(self) -> MetricPlugin:
        return self

    def accepts(self, *, data_type: str, traits: Collection[Trait]) -> bool:
        return data_type in self.data_types or bool(self.traits.intersection(traits))


MetricT = TypeVar("MetricT", bound=Metric)
MetricSelector: TypeAlias = Trait | str


@dataclass(frozen=True)
class _MetricRegistrar:
    registry: MetricRegistry
    traits: frozenset[Trait]
    data_types: frozenset[str]

    def __call__(self, metric_class: type[MetricT]) -> type[MetricT]:
        return self.registry._register(
            metric_class,
            traits=self.traits,
            data_types=self.data_types,
        )


class MetricRegistry:
    """Own metric registration, Pluggy discovery, parsing, and validation."""

    _metric_key_pattern: ClassVar[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")
    _data_type_pattern: ClassVar[re.Pattern[str]] = re.compile(r"^[a-z0-9_]+$")

    def __init__(self) -> None:
        self.plugin_manager = pluggy.PluginManager(project_name="metrics")
        self.plugin_manager.add_hookspecs(PluginSpec)
        self.metrics: dict[str, MetricPlugin] = {}

    def parse(self, value: object) -> Metric:
        """Resolve one metric instance or serialized mapping."""
        if isinstance(value, Metric):
            plugin = self.metrics.get(value.type)
            if plugin is None or type(value) is not plugin.Metric:
                raise ValueError("metric instance does not match a registered metric class")
            return value

        if not isinstance(value, Mapping):
            raise ValueError("metric configuration must be a mapping")

        key = value.get("type")
        if not isinstance(key, str) or key not in self.metrics:
            raise ValueError(f"unknown metric type: {key!r}")
        metric = self.metrics[key].Metric.model_validate(dict(value))
        if metric.type != key:
            raise ValueError(f"metric type must remain {key!r}")
        return metric

    def validate_request(
        self,
        metrics: Sequence[Metric],
        *,
        data_type: str,
        traits: Collection[Trait],
    ) -> None:
        """Validate identity, eligibility, and rendered-name uniqueness."""
        rendered_names: set[str] = set()
        for metric in metrics:
            registered = self.parse(metric)
            plugin = self.metrics[registered.type]
            if not plugin.accepts(data_type=data_type, traits=traits):
                raise ValueError(
                    f"metric {registered.type!r} is not registered for tensorfield {data_type!r} or any of its traits"
                )

            rendered = str(registered)
            if rendered in rendered_names:
                raise ValueError(f"duplicate rendered metric name: {rendered!r}")
            rendered_names.add(rendered)

    def register(
        self,
        *types: MetricSelector,
    ) -> _MetricRegistrar:
        """Register a concrete metric for trait and/or datatype selectors."""
        if not types:
            raise TypeError("register requires at least one trait or datatype selector")

        traits: set[Trait] = set()
        data_types: set[str] = set()
        for selector in types:
            if isinstance(selector, Trait):
                traits.add(selector)
            elif not isinstance(selector, str):
                raise TypeError("metric selectors must be Trait members or datatype strings")
            elif self._data_type_pattern.fullmatch(selector) is None:
                raise ValueError("datatype selectors must contain only lowercase letters, numbers, and underscores")
            else:
                data_types.add(selector)

        return _MetricRegistrar(
            registry=self,
            traits=frozenset(traits),
            data_types=frozenset(data_types),
        )

    def _register(
        self,
        metric_class: type[MetricT],
        *,
        traits: frozenset[Trait],
        data_types: frozenset[str],
    ) -> type[MetricT]:
        if not isinstance(metric_class, type) or not issubclass(metric_class, Metric):
            raise TypeError("registered metrics must be Metric subclasses")
        if inspect.isabstract(metric_class):
            raise TypeError("registered metrics must be concrete")

        field = metric_class.model_fields.get("type")
        key = None if field is None else field.get_default(call_default_factory=True)
        if not isinstance(key, str) or self._metric_key_pattern.fullmatch(key) is None:
            raise ValueError(
                "metric type must start with a lowercase letter and contain only "
                "lowercase letters, numbers, and underscores"
            )
        if key in self.metrics:
            raise ValueError(f"metric {key!r} is already registered")

        try:
            default: Metric | None = metric_class.model_validate({})
        except pydantic.ValidationError as error:
            missing = [item for item in error.errors() if item["type"] == "missing"]
            if len(missing) != len(error.errors()):
                raise ValueError(str(error)) from error
            if traits:
                fields = ", ".join(".".join(str(part) for part in item["loc"]) for item in missing)
                detail = f"; missing fields: {fields}" if fields else ""
                raise TypeError(
                    f"trait-registered metric {key!r} must have a valid default configuration{detail}"
                ) from error
            default = None

        provider = MetricPlugin(
            key=key,
            Metric=metric_class,
            traits=traits,
            data_types=data_types,
            default=default,
        )
        self.plugin_manager.register(provider, name=key)
        try:
            rebuilt = self._build()
        except Exception:
            self.plugin_manager.unregister(plugin=provider)
            raise

        self.metrics.clear()
        self.metrics.update(rebuilt)
        return metric_class

    def unregister(self, key: str) -> MetricPlugin | None:
        """Remove one provider while preserving the live registry mapping."""
        provider = self.metrics.get(key)
        if provider is None:
            return None
        self.plugin_manager.unregister(plugin=provider)
        rebuilt = self._build()
        self.metrics.clear()
        self.metrics.update(rebuilt)
        return provider

    def _build(self) -> dict[str, MetricPlugin]:
        rebuilt: dict[str, MetricPlugin] = {}
        for provider in self.plugin_manager.hook.metric():
            if not isinstance(provider, MetricPlugin):
                raise TypeError("metric hook providers must return MetricPlugin instances")
            if provider.key in rebuilt:
                raise ValueError(f"metric {provider.key!r} is registered more than once")
            rebuilt[provider.key] = provider
        return dict(sorted(rebuilt.items()))


registry = MetricRegistry()

# These references are intentionally stable: imported registry mappings remain
# live, while the decorator keeps the familiar ``@register(...)`` spelling.
METRICS = registry.metrics
register = registry.register


__all__ = [
    "METRICS",
    "Metric",
    "MetricPlugin",
    "MetricRegistry",
    "Trait",
    "register",
    "registry",
]

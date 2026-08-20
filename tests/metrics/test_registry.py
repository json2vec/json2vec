import inspect
from types import ModuleType
from typing import Literal

import pydantic
import pytest

import relflow.metrics.base as metric_base
import relflow.metrics.extensions.classification as metric_classification
import relflow.metrics.extensions.dateparts as metric_dateparts
import relflow.metrics.extensions.regression as metric_regression
import relflow.metrics.spec as metric_spec
from relflow.metrics import MAE, RMSE, Accuracy, AngularMAE, Precision, Recall, Specificity
from relflow.metrics.base import (
    METRICS,
    Metric,
    Trait,
    register,
    registry,
)


@pytest.mark.parametrize(
    "module",
    [
        metric_base,
        metric_classification,
        metric_dateparts,
        metric_regression,
        metric_spec,
    ],
)
def test_metrics_modules_have_no_loose_functions(module: ModuleType):
    assert [
        name
        for name, function in inspect.getmembers(module, inspect.isfunction)
        if function.__module__ == module.__name__
    ] == []


def test_each_concrete_call_tracks_directly_without_a_base_helper():
    assert not hasattr(Metric, "_track")
    for metric_class in (Accuracy, Precision, Recall, Specificity, MAE, RMSE, AngularMAE):
        assert "module.track" in inspect.getsource(metric_class.__call__)


def _remove_metric(key: str) -> None:
    registry.unregister(key)


def test_public_registry_references_are_bound_to_the_registry_object():
    assert METRICS is registry.metrics
    assert register.__self__ is registry


def test_register_returns_the_class_and_records_variadic_selectors():
    class Sample(Metric):
        type: Literal["core_variadic_sample"] = "core_variadic_sample"
        name: str = "core_variadic_sample"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    try:
        result = register(Trait.classification, "category", "boolean")(Sample)

        assert result is Sample
        plugin = METRICS["core_variadic_sample"]
        assert plugin.Metric is Sample
        assert plugin.traits == frozenset({Trait.classification})
        assert plugin.data_types == frozenset({"category", "boolean"})
        assert registry.plugin_manager.get_plugin("core_variadic_sample") is plugin
    finally:
        _remove_metric("core_variadic_sample")


def test_register_rejects_empty_tuple_and_keyword_selector_forms():
    with pytest.raises(TypeError, match="at least one"):
        register()
    with pytest.raises(TypeError, match="Trait members or datatype strings"):
        register((Trait.classification,))
    with pytest.raises(TypeError, match="unexpected keyword"):
        register(type=Trait.classification)


def test_raw_datatype_name_remains_distinct_from_a_trait():
    @register("classification")
    class Exact(Metric):
        type: Literal["core_exact_classification"] = "core_exact_classification"
        name: str = "core_exact_classification"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    try:
        plugin = METRICS[Exact().type]
        assert plugin.traits == frozenset()
        assert plugin.data_types == frozenset({"classification"})
        assert all(type(metric) is not Exact for metric in Trait.classification.metrics)
    finally:
        _remove_metric("core_exact_classification")


def test_duplicate_registration_is_transactional():
    @register("category")
    class First(Metric):
        type: Literal["core_duplicate"] = "core_duplicate"
        name: str = "core_duplicate"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    original_registry = dict(METRICS)
    original_provider = METRICS["core_duplicate"]

    class Second(Metric):
        type: Literal["core_duplicate"] = "core_duplicate"
        name: str = "core_duplicate_second"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    try:
        with pytest.raises(ValueError, match="already registered"):
            register("category")(Second)

        assert METRICS == original_registry
        assert METRICS["core_duplicate"] is original_provider
        assert registry.plugin_manager.get_plugin("core_duplicate") is original_provider
    finally:
        _remove_metric("core_duplicate")


def test_exact_registration_may_require_configuration_but_trait_registration_may_not():
    class RequiredExact(Metric):
        type: Literal["core_required_exact"] = "core_required_exact"
        value: int
        name: str = "core_required_exact@{value}"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    class RequiredTrait(Metric):
        type: Literal["core_required_trait"] = "core_required_trait"
        value: int
        name: str = "core_required_trait@{value}"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    try:
        register("category")(RequiredExact)
        assert METRICS["core_required_exact"].default is None

        with pytest.raises(TypeError, match="default configuration"):
            register(Trait.classification)(RequiredTrait)
        assert "core_required_trait" not in METRICS
        assert registry.plugin_manager.get_plugin("core_required_trait") is None
    finally:
        _remove_metric("core_required_exact")


def test_exact_registration_validates_a_name_template_before_required_configuration_exists():
    class InvalidTemplate(Metric):
        type: Literal["core_invalid_required_name"] = "core_invalid_required_name"
        value: int
        name: str = "core_invalid_required_name@{missing}"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    with pytest.raises(ValueError, match="unknown field 'missing'"):
        register("category")(InvalidTemplate)

    assert "core_invalid_required_name" not in METRICS
    assert registry.plugin_manager.get_plugin("core_invalid_required_name") is None


def test_metric_annotation_dispatches_registered_mappings_and_preserves_subclass_fields():
    @register("category")
    class Configured(Metric):
        type: Literal["core_dynamic_parse"] = "core_dynamic_parse"
        threshold: float = 0.5
        name: str = "core_dynamic_parse@{threshold:.2f}"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    class Container(pydantic.BaseModel):
        metrics: list[Metric]

    try:
        container = Container.model_validate({"metrics": [{"type": "core_dynamic_parse", "threshold": 0.75}]})
        assert type(container.metrics[0]) is Configured
        assert container.metrics[0].threshold == 0.75

        dumped = container.model_dump()
        assert dumped == {
            "metrics": [
                {
                    "type": "core_dynamic_parse",
                    "name": "core_dynamic_parse@{threshold:.2f}",
                    "threshold": 0.75,
                }
            ]
        }

        restored = Container.model_validate_json(container.model_dump_json())
        assert type(restored.metrics[0]) is Configured
        assert restored == container
    finally:
        _remove_metric("core_dynamic_parse")


def test_metric_annotation_has_a_useful_base_json_schema():
    class Container(pydantic.BaseModel):
        metrics: list[Metric]

    item_schema = Container.model_json_schema()["properties"]["metrics"]["items"]

    assert item_schema["properties"]["type"]["type"] == "string"
    assert item_schema["properties"]["name"]["type"] == "string"
    assert item_schema["required"] == ["type"]


def test_metric_annotation_wraps_parser_type_errors_as_validation_errors():
    class Container(pydantic.BaseModel):
        metrics: list[Metric]

    with pytest.raises(pydantic.ValidationError, match="metric configuration must be a mapping"):
        Container(metrics=[123])


def test_nested_metric_serialization_preserves_enclosing_options():
    @register("category")
    class Configured(Metric):
        type: Literal["core_serialization_options"] = "core_serialization_options"
        value: int = pydantic.Field(default=1, serialization_alias="v")
        optional: int | None = None
        name: str = "core_serialization_options@{value}"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    class Container(pydantic.BaseModel):
        metrics: list[Metric]

    try:
        container = Container(metrics=[Configured(value=2)])

        assert container.model_dump(by_alias=True, exclude_none=True) == {
            "metrics": [
                {
                    "type": "core_serialization_options",
                    "name": "core_serialization_options@{value}",
                    "v": 2,
                }
            ]
        }
        assert container.model_dump(exclude_defaults=True) == {"metrics": [{"value": 2}]}
    finally:
        _remove_metric("core_serialization_options")


def test_registry_parse_rejects_unknown_or_mismatched_instances():
    class Unregistered(Metric):
        type: Literal["core_unregistered"] = "core_unregistered"
        name: str = "core_unregistered"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    with pytest.raises(ValueError, match="unknown metric type"):
        registry.parse({"type": "does_not_exist"})
    with pytest.raises(ValueError, match="registered metric class"):
        registry.parse(Unregistered())


def test_request_validation_uses_or_eligibility_and_rejects_name_collisions():
    @register(Trait.classification, "special")
    class Eligible(Metric):
        type: Literal["core_eligible"] = "core_eligible"
        suffix: int = 1
        name: str = "core_eligible@{suffix}"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    try:
        registry.validate_request(
            [Eligible()],
            data_type="category",
            traits=frozenset({Trait.classification}),
        )
        registry.validate_request([Eligible()], data_type="special", traits=frozenset())

        with pytest.raises(ValueError, match="not registered"):
            registry.validate_request([Eligible()], data_type="number", traits={Trait.regression})
        with pytest.raises(ValueError, match="duplicate rendered metric name"):
            registry.validate_request(
                [Eligible(), Eligible()],
                data_type="category",
                traits={Trait.classification},
            )
    finally:
        _remove_metric("core_eligible")


@pytest.mark.parametrize(
    "selector",
    ["Upper", "has-hyphen", "has space", ""],
)
def test_register_rejects_invalid_datatype_selectors(selector):
    with pytest.raises(ValueError, match="datatype selectors"):
        register(selector)


def test_register_rejects_abstract_metrics_before_mutating_pluggy():
    class AbstractMetric(Metric):
        type: Literal["core_abstract"] = "core_abstract"
        name: str = "core_abstract"

    with pytest.raises(TypeError, match="concrete"):
        register("category")(AbstractMetric)

    assert "core_abstract" not in METRICS
    assert registry.plugin_manager.get_plugin("core_abstract") is None

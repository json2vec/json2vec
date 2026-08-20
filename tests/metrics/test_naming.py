from typing import Literal

import pydantic
import pytest

from relflow.metrics.base import Metric


class ConfiguredMetric(Metric):
    type: Literal["configured_for_naming"] = "configured_for_naming"
    threshold: float | None = 0.5
    top_k: int = 3
    name: str = "configured@{threshold:.2f}"

    def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
        return None


def test_name_is_an_ordinary_serialized_string_and_str_renders_it():
    metric = ConfiguredMetric()

    assert metric.name == "configured@{threshold:.2f}"
    assert str(metric) == "configured@0.50"
    assert metric.model_dump()["name"] == "configured@{threshold:.2f}"
    assert '"name":"configured@{threshold:.2f}"' in metric.model_dump_json()


def test_name_may_reference_any_other_top_level_model_field():
    metric = ConfiguredMetric(name="{type}@top{top_k}")

    assert str(metric) == "configured_for_naming@top3"


@pytest.mark.parametrize(
    "name",
    [
        "configured@{}",
        "configured@{0}",
        "configured@{missing}",
        "configured@{name}",
        "configured@{threshold.real}",
        "configured@{threshold[0]}",
        "configured@{threshold:{top_k}}",
        "configured@{threshold!r}",
    ],
)
def test_name_rejects_unsupported_placeholders(name):
    with pytest.raises(pydantic.ValidationError):
        ConfiguredMetric(name=name)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "Configured",
        "configured/value",
        "configured value",
        "configured\nvalue",
    ],
)
def test_name_rejects_invalid_rendered_suffixes(name):
    with pytest.raises(pydantic.ValidationError):
        ConfiguredMetric(name=name)


def test_name_rejects_a_format_spec_incompatible_with_the_value():
    with pytest.raises(pydantic.ValidationError, match="could not render"):
        ConfiguredMetric(name="configured@{threshold:d}")


def test_metric_models_are_frozen():
    metric = ConfiguredMetric()

    with pytest.raises(pydantic.ValidationError, match="frozen"):
        metric.name = "replacement"

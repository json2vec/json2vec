from typing import Literal

from relflow.metrics.base import METRICS, Metric, Trait, register, registry


def _remove_metric(key: str) -> None:
    registry.unregister(key)


def test_trait_metrics_is_a_live_sorted_deep_copied_default_catalog():
    @register(Trait.cyclic)
    class ZMetric(Metric):
        type: Literal["core_trait_z"] = "core_trait_z"
        options: list[int] = [1]
        name: str = "core_trait_z"

        def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
            return None

    try:
        first = [metric for metric in Trait.cyclic.metrics if metric.type in {"core_trait_a", "core_trait_z"}]
        assert [metric.type for metric in first] == ["core_trait_z"]

        @register(Trait.cyclic)
        class AMetric(Metric):
            type: Literal["core_trait_a"] = "core_trait_a"
            options: list[int] = [2]
            name: str = "core_trait_a"

            def __call__(self, module, predictions, targets, trainable, *, address, strata, scope) -> None:
                return None

        second = [metric for metric in Trait.cyclic.metrics if metric.type in {"core_trait_a", "core_trait_z"}]
        third = [metric for metric in Trait.cyclic.metrics if metric.type in {"core_trait_a", "core_trait_z"}]

        assert [metric.type for metric in second] == ["core_trait_a", "core_trait_z"]
        assert second is not third
        assert second[0] is not third[0]
        assert second[0].options is not third[0].options

        second[0].options.append(9)
        assert third[0].options == [2]
        assert METRICS["core_trait_a"].default.options == [2]
    finally:
        _remove_metric("core_trait_a")
        _remove_metric("core_trait_z")


def test_trait_is_a_plain_enum_not_a_string_enum():
    assert Trait.classification.value == "classification"
    assert Trait.classification != "classification"

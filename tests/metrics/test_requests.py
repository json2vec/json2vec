from __future__ import annotations

import pydantic
import pytest

import relflow as rf
from relflow.metrics import MAE, Accuracy, Metric, Trait


def test_tensorfield_requests_have_independent_metric_lists() -> None:
    first = rf.Category()
    second = rf.Category()

    assert first.metrics == []
    assert second.metrics == []
    assert first.metrics is not second.metrics


def test_category_accepts_configured_metric_instances_in_order() -> None:
    request = rf.Category(
        metrics=[
            Accuracy(0.5),
            Accuracy(0.75),
        ]
    )

    assert all(isinstance(metric, Metric) for metric in request.metrics)
    assert [str(metric) for metric in request.metrics] == [
        "accuracy@0.50",
        "accuracy@0.75",
    ]


def test_request_parses_serialized_metrics_and_preserves_subclass_fields() -> None:
    request = rf.Category.model_validate(
        {
            "size": 128,
            "metrics": [
                {
                    "type": "accuracy",
                    "name": "accuracy@{threshold:.2f}",
                    "threshold": 0.75,
                }
            ],
        }
    )

    metric = request.metrics[0]
    assert isinstance(metric, Accuracy)
    assert metric.threshold == 0.75
    assert metric.name == "accuracy@{threshold:.2f}"

    dumped = request.model_dump(mode="python", round_trip=True)
    assert dumped["metrics"][0]["threshold"] == 0.75
    assert dumped["metrics"][0]["name"] == "accuracy@{threshold:.2f}"

    restored = rf.Category.model_validate_json(request.model_dump_json(round_trip=True))
    restored_metric = restored.metrics[0]
    assert isinstance(restored_metric, Accuracy)
    assert restored_metric == metric


def test_schema_round_trip_preserves_concrete_metrics() -> None:
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        label=rf.Category(
            target=True,
            size=8,
            metrics=[Accuracy(0.5)],
        ),
    )

    restored = rf.Schema.model_validate(model.schema.model_dump(mode="python", round_trip=True))
    metric = next(iter(restored.requests.values())).metrics[0]

    assert isinstance(metric, Accuracy)
    assert metric.threshold == 0.5
    assert metric.name == "accuracy@{threshold:.2f}"


def test_request_rejects_singular_metric_option() -> None:
    with pytest.raises(pydantic.ValidationError, match=r"use metrics=\[\.\.\.\]"):
        rf.Category(metric=Accuracy())


def test_request_rejects_ineligible_metric() -> None:
    with pytest.raises(pydantic.ValidationError, match="is not registered"):
        rf.Category(metrics=[MAE()])


def test_request_rejects_duplicate_rendered_metric_names() -> None:
    with pytest.raises(pydantic.ValidationError, match="duplicate rendered metric name"):
        rf.Category(metrics=[Accuracy(0.5), Accuracy(threshold=0.5)])


def test_request_post_bind_validation_still_requires_query() -> None:
    request = rf.Category(metrics=[Accuracy()])

    with pytest.raises(ValueError, match="must define query"):
        request.post_bind_validate()


def test_tensorfield_plugins_declare_content_traits() -> None:
    classification = {"boolean", "category", "cluster", "hash", "set"}
    regression = {"number", "text", "vector"}

    for name in classification:
        assert rf.TENSORFIELDS[name].traits == frozenset({Trait.classification})
    for name in regression:
        assert rf.TENSORFIELDS[name].traits == frozenset({Trait.regression})
    assert rf.TENSORFIELDS["dateparts"].traits == frozenset({Trait.cyclic})


def test_trait_defaults_can_be_selected_as_a_request_snapshot() -> None:
    selected = Trait.classification.metrics
    request = rf.Category(metrics=selected)

    assert request.metrics is not selected
    assert [metric.type for metric in request.metrics] == [metric.type for metric in selected]

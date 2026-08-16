from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchmetrics import Metric as TorchMetric

from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.metric import Metric
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions.boolean import BooleanCounter, Decoder, Embedder, TensorField, loss, write

ADDRESS = "root/groups/items/enabled"


def _schema(*, threshold: float | list[float] = 0.5) -> Schema:
    return Schema.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "branch",
                "fields": [
                    {
                        "name": "groups",
                        "type": "branch",
                        "length": 2,
                        "fields": [
                            {
                                "name": "items",
                                "type": "branch",
                                "length": 3,
                                "fields": [
                                    {
                                        "name": "enabled",
                                        "type": "boolean",
                                        "query": "[*].groups[*].items[*].enabled",
                                        "threshold": threshold,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        }
    )


def test_boolean_tensorfield_encodes_nested_values_without_vocabulary():
    schema = _schema()
    field = TensorField.new(
        values=[[[[False, True, None], [np.bool_(True)]]]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )

    assert field.content.tolist() == [[[[-1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]]
    assert field.state.tolist() == [
        [
            [
                [Tokens.valued.value, Tokens.valued.value, Tokens.null.value],
                [Tokens.valued.value, Tokens.padded.value, Tokens.padded.value],
            ]
        ]
    ]


def test_boolean_tensorfield_rejects_integer_lookalikes():
    with pytest.raises(TypeError, match="bool or None"):
        TensorField.new(
            values=[[[[1]]]],
            address=ADDRESS,
            schema=_schema(),
            strata=Strata.train,
        )


def test_boolean_embedder_is_vocabulary_free_and_registers_fixed_value_buffer():
    schema = _schema()
    embedder = Embedder(schema=schema, address=ADDRESS)

    assert not hasattr(embedder, "vocab")
    assert torch.equal(
        embedder.content,
        torch.tensor([[-1.0] * schema.d_model, [0.0] * schema.d_model, [1.0] * schema.d_model]),
    )
    assert "content" in dict(embedder.named_buffers())
    assert "content" not in dict(embedder.named_parameters())
    assert "content" in embedder.state_dict()


def test_boolean_embedder_maps_false_zero_and_true_to_initialized_rows():
    schema = _schema()
    embedder = Embedder(schema=schema, address=ADDRESS)
    field = TensorField.new(
        values=[[[[False, True, None]]]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )

    content_only = embedder(field).payload - embedder.state(field.state)

    assert torch.allclose(content_only[0, 0, 0, 0], torch.full((schema.d_model,), -1.0))
    assert torch.allclose(content_only[0, 0, 0, 1], torch.full((schema.d_model,), 1.0))
    assert torch.allclose(content_only[0, 0, 0, 2], torch.zeros(schema.d_model))


class _TrackingModule:
    def __init__(self, schema: Schema, embedder: Embedder, decoder: Decoder):
        self.schema = schema
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder, decoder=decoder)}
        self.tracked: dict[tuple, torch.Tensor | TorchMetric] = {}

    def track(self, names: tuple[str, ...], value: torch.Tensor | TorchMetric):
        self.tracked[names] = value
        return value


def test_boolean_loss_tracks_binary_torchmetrics():
    schema = _schema(threshold=[0.25, 0.75])
    field = TensorField.new(
        values=[[[[False, True]]], [[[True, False]]]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )
    field.target(1.0)
    module = _TrackingModule(schema, Embedder(schema, ADDRESS), Decoder(schema, ADDRESS))
    state_logits = torch.zeros(*field.state.shape, len(Tokens))
    content_logits = torch.zeros(*field.content.shape, 1)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: state_logits, TensorKey.content: content_logits},
            batch_size=field.batch_size,
        ),
    )

    result = loss(module, prediction, field, Strata.train)

    assert torch.isfinite(result)
    assert torch.equal(
        module.nodes[ADDRESS].embedder.counters[TensorKey.state.name].counts,
        torch.ones(len(Tokens), dtype=torch.int64),
    )
    assert torch.equal(
        module.nodes[ADDRESS].embedder.counters[TensorKey.content.name].counts,
        torch.ones(2, dtype=torch.int64),
    )
    expected_names = {
        Metric.auc.value,
        *(
            f"{metric.value}@{threshold}"
            for threshold in (0.25, 0.75)
            for metric in (
                Metric.accuracy,
                Metric.precision,
                Metric.recall,
                Metric.specificity,
            )
        ),
    }
    for metric_name in expected_names:
        tracked = module.tracked[(ADDRESS, Strata.train, metric_name, TensorKey.content)]
        assert isinstance(tracked, TorchMetric)


def test_boolean_embedder_has_state_and_content_counters():
    embedder = Embedder(schema=_schema(), address=ADDRESS)

    assert embedder.counters[TensorKey.state.name].size == len(Tokens)
    assert isinstance(embedder.counters[TensorKey.content.name], BooleanCounter)
    assert embedder.counters[TensorKey.content.name].size == 2


def test_boolean_counter_maps_fixed_content_to_false_and_true_classes():
    counter = BooleanCounter(address=ADDRESS, size=2)

    counter(torch.tensor([-1.0, -1.0, 1.0]))

    assert counter.counts.tolist() == [3, 2]


@pytest.mark.parametrize(
    ("threshold", "expected"),
    [
        (0.8, (0.8,)),
        ([0.8, 0.2, 0.8, 0.5, 0.2], (0.8, 0.2, 0.5)),
    ],
)
def test_boolean_request_normalizes_scalar_and_ordered_unique_thresholds(
    threshold: float | list[float],
    expected: tuple[float, ...],
):
    request = _schema(threshold=threshold).requests[ADDRESS]

    assert request.thresholds == expected


def test_boolean_thresholds_configure_unique_metrics_and_one_auc():
    decoder = Decoder(_schema(threshold=[0.8, 0.2, 0.8]), ADDRESS)

    metrics = decoder.metrics[f"{Strata.train.value}_metrics"]
    assert set(metrics) == {Metric.auc.value, "threshold_0", "threshold_1"}

    expected_metric_names = {
        Metric.accuracy.value,
        Metric.precision.value,
        Metric.recall.value,
        Metric.specificity.value,
    }
    for index, threshold in enumerate((0.8, 0.2)):
        threshold_metrics = metrics[f"threshold_{index}"]
        assert set(threshold_metrics) == expected_metric_names
        assert all(metric.threshold == threshold for metric in threshold_metrics.values())

    named_metrics = list(decoder.content_metrics(Strata.train))
    names = [name for name, _ in named_metrics]
    assert names.count(Metric.auc.value) == 1
    assert len(names) == len(set(names)) == 9
    assert set(names) == {
        Metric.auc.value,
        *(f"{name}@{threshold}" for threshold in (0.8, 0.2) for name in expected_metric_names),
    }
    assert all(isinstance(metric, TorchMetric) for _, metric in named_metrics)


def test_boolean_metrics_ignore_null_and_padded_targets():
    schema = _schema()
    field = TensorField.new(
        values=[[[[False, True, None]]]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )
    field.target(1.0)
    decoder = Decoder(schema, ADDRESS)
    module = _TrackingModule(schema, Embedder(schema, ADDRESS), decoder)
    content_logits = torch.full((*field.content.shape, 1), 20.0)
    content_logits[..., 0, 0, 0, 0] = -20.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: content_logits,
            },
            batch_size=field.batch_size,
        ),
    )

    loss(module, prediction, field, Strata.train)

    metrics: dict[str, TorchMetric] = dict(decoder.content_metrics(Strata.train))
    assert set(metrics) == {
        Metric.auc.value,
        f"{Metric.accuracy.value}@0.5",
        f"{Metric.precision.value}@0.5",
        f"{Metric.recall.value}@0.5",
        f"{Metric.specificity.value}@0.5",
    }
    assert all(metric.compute().item() == 1.0 for metric in metrics.values())  # ty: ignore[missing-argument]


def test_boolean_non_valued_batch_only_trains_state_and_does_not_update_content_metrics():
    schema = _schema(threshold=[0.25, 0.75])
    field = TensorField.new(
        values=[[[[None]]]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )
    field.target(1.0)
    decoder = Decoder(schema, ADDRESS)
    module = _TrackingModule(schema, Embedder(schema, ADDRESS), decoder)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(*field.content.shape, 1),
            },
            batch_size=field.batch_size,
        ),
    )

    result = loss(module, prediction, field, Strata.train)

    assert torch.isfinite(result)
    assert (ADDRESS, Strata.train, Metric.loss, TensorKey.content) not in module.tracked
    metrics = tuple(decoder.content_metrics(Strata.train))
    assert len(metrics) == 9
    assert all(metric.update_count == 0 for _, metric in metrics)
    assert all((ADDRESS, Strata.train, metric_name, TensorKey.content) in module.tracked for metric_name, _ in metrics)


def test_boolean_write_preserves_nested_shape_and_emits_true_probability():
    schema = _schema()
    state_logits = torch.zeros(1, 1, 2, 3, len(Tokens))
    content_logits = torch.tensor([[[[[-2.0], [0.0], [2.0]], [[2.0], [-2.0], [0.0]]]]])
    output = write(
        SimpleNamespace(schema=schema),
        Prediction(
            address=ADDRESS,
            payload=TensorDict(
                {TensorKey.state: state_logits, TensorKey.content: content_logits},
                batch_size=[1],
            ),
        ),
    )

    content = output[TensorKey.content.name]
    assert set(content) == {TensorKey.probability.name}
    probabilities = content[TensorKey.probability.name]
    assert probabilities.shape == (1, 1, 2, 3)
    assert np.allclose(
        probabilities,
        [[[[0.11920292, 0.5, 0.880797], [0.880797, 0.11920292, 0.5]]]],
    )
    assert set(output[TensorKey.state.name]) == set(Tokens.__members__)


@pytest.mark.parametrize(
    "thresholds",
    [
        [0.8, 0.2],
        [0.2, 0.8],
    ],
)
def test_boolean_write_is_independent_of_evaluation_thresholds(thresholds: list[float]):
    schema = _schema(threshold=thresholds)
    output = write(
        SimpleNamespace(schema=schema),
        Prediction(
            address=ADDRESS,
            payload=TensorDict(
                {
                    TensorKey.state: torch.zeros(1, 1, 2, 3, len(Tokens)),
                    TensorKey.content: torch.full((1, 1, 2, 3, 1), 1.0),
                },
                batch_size=[1],
            ),
        ),
    )

    content = output[TensorKey.content.name]
    assert set(content) == {TensorKey.probability.name}
    probabilities = content[TensorKey.probability.name]
    assert probabilities.shape == (1, 1, 2, 3)
    assert np.allclose(probabilities, torch.tensor(1.0).sigmoid().item())

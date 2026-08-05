from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchmetrics import Metric as TorchMetric

from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions.boolean import BooleanCounter, Decoder, Embedder, TensorField, loss, write

ADDRESS = "root/groups/items/enabled"


def _schema(*, threshold: float = 0.5) -> Schema:
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
    schema = _schema()
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
    for metric in (Metric.accuracy, Metric.precision, Metric.recall, Metric.auc):
        tracked = module.tracked[(ADDRESS, Strata.train, metric, TensorKey.content)]
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


def test_boolean_threshold_configures_decision_metrics():
    decoder = Decoder(_schema(threshold=0.8), ADDRESS)

    metrics = decoder.metrics[f"{Strata.train.value}_metrics"]
    assert metrics[Metric.accuracy.value].threshold == 0.8
    assert metrics[Metric.precision.value].threshold == 0.8
    assert metrics[Metric.recall.value].threshold == 0.8


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

    metrics = decoder.metrics[f"{Strata.train.value}_metrics"]
    assert metrics[Metric.accuracy.value].compute().item() == 1.0
    assert metrics[Metric.precision.value].compute().item() == 1.0
    assert metrics[Metric.recall.value].compute().item() == 1.0
    assert metrics[Metric.auc.value].compute().item() == 1.0


def test_boolean_non_valued_batch_only_trains_state_and_does_not_update_content_metrics():
    schema = _schema()
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
    metrics = decoder.metrics[f"{Strata.train.value}_metrics"]
    assert all(metric.update_count == 0 for metric in metrics.values())
    assert all(
        (ADDRESS, Strata.train, metric_name, TensorKey.content) in module.tracked for metric_name in metrics.keys()
    )


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

    assert output[TensorKey.content.name][TensorKey.value.name].tolist() == [
        [[[False, True, True], [True, False, True]]]
    ]
    probabilities = output[TensorKey.content.name][TensorKey.probability.name]
    assert probabilities.shape == (1, 1, 2, 3)
    assert set(output[TensorKey.state.name]) == set(Tokens.__members__)


def test_boolean_write_uses_configured_threshold():
    schema = _schema(threshold=0.8)
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

    assert not output[TensorKey.content.name][TensorKey.value.name].any()

import pytest
import torch
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from relflow.structs.metric import Metric, Traits
from relflow.tensorfields.base import Plugin


def _fresh_metric(name: str, **kwargs) -> Metric:
    """Register a metric and drop it from the registry after the caller uses it."""
    metric = Metric(name, **kwargs)
    return metric


def _drop(metric: Metric) -> None:
    Metric._registry.pop(metric.name, None)


def test_metric_is_a_str_subclass_and_hashes_as_the_underlying_string():
    assert isinstance(Metric.loss, str)
    assert Metric.loss == "loss"
    assert hash(Metric.loss) == hash("loss")
    # Tuple keys built from Metric/StrEnum still work because Metric IS-A str.
    key = (Metric.loss, "state")
    assert key == ("loss", "state")


def test_metric_exposes_name_and_value_backward_compat():
    assert Metric.throughput.name == "throughput"
    assert Metric.throughput.value == "throughput"
    assert f"{Metric.throughput.value}/train" == "throughput/train"


def test_metric_registers_canonical_singletons_once():
    with pytest.raises(ValueError, match="already registered"):
        Metric("loss")


def test_metric_rejects_empty_name():
    with pytest.raises(TypeError):
        Metric("")


def test_metric_dispatches_on_prediction_rank():
    prediction_1d = torch.tensor([0, 1, 1, 0])
    target_1d = torch.tensor([0, 1, 0, 0])
    assert Metric.accuracy(prediction_1d, target_1d).item() == pytest.approx(0.75)

    # torchmetrics multiclass accuracy is macro-averaged by default.
    # Class 0: 1/1 correct = 1.0. Class 1: 1/2 correct = 0.5. Macro mean = 0.75.
    logits_2d = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 0.0]])
    target = torch.tensor([0, 1, 1])
    assert Metric.accuracy(logits_2d, target).item() == pytest.approx(0.75)


def test_metric_any_rank_impl_is_used_when_no_rank_specific_impl():
    prediction = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 4.0])
    assert Metric.mae(prediction, target).item() == pytest.approx(1 / 3)

    # Same MAE works on 2D too because it was registered without an ndim.
    prediction_2d = prediction.reshape(1, 3)
    target_2d = target.reshape(1, 3)
    assert Metric.mae(prediction_2d, target_2d).item() == pytest.approx(1 / 3)


def test_metric_reports_missing_impl_by_rank():
    metric = _fresh_metric("_test_rank_missing", traits=(Traits.discrete,))
    try:
        @metric.register(ndim=2)
        def _(prediction, target):
            return torch.tensor(0.0)

        with pytest.raises(NotImplementedError, match="ndim=1"):
            metric(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0]))
    finally:
        _drop(metric)


def test_metric_reports_no_impl_at_all():
    with pytest.raises(NotImplementedError, match="no registered functional implementation"):
        Metric.loss(torch.tensor(0.0), torch.tensor(0.0))


def test_metric_register_rejects_duplicate_ranks():
    metric = _fresh_metric("_test_dup_rank")
    try:
        @metric.register(ndim=1)
        def _(a, b):
            return torch.tensor(0.0)

        with pytest.raises(ValueError, match="ndim=1"):
            @metric.register(ndim=1)
            def _(a, b):
                return torch.tensor(0.0)
    finally:
        _drop(metric)


def test_traits_carry_over_from_construction():
    assert Metric.accuracy.traits == frozenset({Traits.discrete})
    assert Metric.mae.traits == frozenset({Traits.continuous})
    assert Metric.cosine_similarity.traits == frozenset({Traits.embedding})
    assert Metric.loss.traits == frozenset()


def test_metric_qualifies_by_plugin_traits():
    continuous_plugin = Plugin(name="_test_continuous_plugin", traits=(Traits.continuous,))
    try:
        assert Metric.mae.qualifies(continuous_plugin)
        assert not Metric.accuracy.qualifies(continuous_plugin)
        # Metric with no required traits always qualifies.
        assert Metric.loss.qualifies(continuous_plugin)
    finally:
        from relflow.tensorfields.base import TENSORFIELDS
        TENSORFIELDS.pop("_test_continuous_plugin", None)


def test_metric_qualifies_accepts_bare_trait_and_iterable():
    assert Metric.accuracy.qualifies(Traits.discrete)
    assert Metric.accuracy.qualifies("discrete")
    assert Metric.accuracy.qualifies([Traits.discrete, Traits.embedding])
    assert not Metric.accuracy.qualifies([])
    assert not Metric.accuracy.qualifies(Traits.continuous)


def test_metric_get_and_with_traits_lookup():
    assert Metric.get("mae") is Metric.mae
    with pytest.raises(KeyError):
        Metric.get("does_not_exist")

    discrete_metrics = Metric.with_traits(Traits.discrete)
    assert Metric.accuracy in discrete_metrics
    assert Metric.auc in discrete_metrics
    assert Metric.mae not in discrete_metrics
    # `loss` requires no traits, so it qualifies under any candidate trait set.
    assert Metric.loss in discrete_metrics


def test_plugin_accepts_traits_kwarg_and_exposes_qualifies_for():
    plugin = Plugin(name="_test_qualifies_for", traits=(Traits.continuous,))
    try:
        assert plugin.traits == frozenset({Traits.continuous})
        assert plugin.qualifies_for(Metric.mae)
        assert not plugin.qualifies_for(Metric.accuracy)
    finally:
        from relflow.tensorfields.base import TENSORFIELDS
        TENSORFIELDS.pop("_test_qualifies_for", None)


def test_plugin_defaults_to_empty_traits():
    plugin = Plugin(name="_test_default_traits")
    try:
        assert plugin.traits == frozenset()
        assert not plugin.qualifies_for(Metric.mae)
    finally:
        from relflow.tensorfields.base import TENSORFIELDS
        TENSORFIELDS.pop("_test_default_traits", None)


def test_stateful_factory_returns_a_torchmetric_instance():
    tracker = Metric.mae.stateful()
    assert isinstance(tracker, TorchMetric)
    assert isinstance(tracker, MeanAbsoluteError)


def test_stateful_factory_dispatches_on_rank_for_classification():
    binary_tracker = Metric.accuracy.stateful(ndim=1)
    assert isinstance(binary_tracker, BinaryAccuracy)

    multi_tracker = Metric.accuracy.stateful(ndim=2, num_classes=4)
    assert isinstance(multi_tracker, MulticlassAccuracy)
    assert multi_tracker.num_classes == 4


def test_stateful_factory_forwards_kwargs():
    tracker = Metric.rmse.stateful()
    # RMSE is MeanSquaredError with squared=False; verify the flag propagates.
    assert isinstance(tracker, MeanSquaredError)
    assert tracker.squared is False


def test_stateful_factory_missing_reports_error():
    metric = _fresh_metric("_test_no_factory")
    try:
        with pytest.raises(NotImplementedError, match="no registered stateful factory"):
            metric.stateful()
    finally:
        _drop(metric)


def test_stateful_factory_rejects_non_torchmetric_returns():
    metric = _fresh_metric("_test_bad_factory")
    try:
        metric.factory(lambda **_: object())
        with pytest.raises(TypeError, match="expected a torchmetrics.Metric instance"):
            metric.stateful()
    finally:
        _drop(metric)


def test_stateful_tracker_aggregates_across_updates():
    tracker = Metric.mae.stateful()
    tracker.update(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
    tracker.update(torch.tensor([3.0]), torch.tensor([2.0]))
    # (|0| + |1| + |1|) / 3 = 2/3
    assert tracker.compute().item() == pytest.approx(2 / 3)


def test_accumulates_history_flags_memory_intensive_metrics():
    assert Metric.auroc.accumulates_history is True
    assert Metric.auc.accumulates_history is True
    # Constant-state metrics keep the default False.
    assert Metric.mae.accumulates_history is False
    assert Metric.mse.accumulates_history is False
    assert Metric.accuracy.accumulates_history is False
    assert Metric.precision.accumulates_history is False


def test_factory_registration_rejects_duplicate_ranks():
    metric = _fresh_metric("_test_dup_factory")
    try:
        metric.factory(MeanAbsoluteError)
        with pytest.raises(ValueError, match="already has a stateful factory"):
            metric.factory(MeanAbsoluteError)
    finally:
        _drop(metric)


def test_matched_functional_and_stateful_produce_the_same_scalar():
    prediction = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([1.0, 2.0, 4.0, 6.0])

    functional_value = Metric.mae(prediction, target)
    tracker = Metric.mae.stateful()
    tracker.update(prediction, target)

    assert tracker.compute().item() == pytest.approx(functional_value.item())

from typing import Any, cast

import deal
import pytest
import torch
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from relflow.structs.enums import Strata
from relflow.structs.metric import HitsTotalAccuracy, Metric, Traits
from relflow.tensorfields.base import TENSORFIELDS, Plugin


def _fresh_metric(name: str, **kwargs) -> Metric:
    """Register a metric and drop it from the registry after the caller uses it."""
    metric = Metric(name, **kwargs)
    return metric


def _drop(metric: Metric) -> None:
    Metric._registry.pop(metric.name, None)


def _drop_plugin(name: str) -> None:
    TENSORFIELDS.pop(name, None)


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

    # Multiclass accuracy defaults to `average="micro"` so it matches the plain
    # `argmax.eq(target).mean()` semantics we log across the extensions.
    logits_2d = torch.tensor([[2.0, 0.0], [0.0, 3.0], [1.0, 0.0]])
    target = torch.tensor([0, 1, 1])
    assert Metric.accuracy(logits_2d, target).item() == pytest.approx(2 / 3)


def test_multiclass_accuracy_micro_matches_manual_hits_over_total():
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    target = torch.tensor([0, 1, 1, 2])
    manual = logits.argmax(dim=-1).eq(target).float().mean()

    tracker = Metric.accuracy.stateful(ndim=2, num_classes=3)
    tracker.update(logits, target)
    assert tracker.compute().item() == pytest.approx(manual.item())


def test_multiclass_accuracy_stateful_footprint_is_constant():
    small = Metric.accuracy.stateful(ndim=2, num_classes=8)
    huge = Metric.accuracy.stateful(ndim=2, num_classes=10_000)
    small_total = sum(s.numel() for s in small.state_dict().values() if hasattr(s, "numel"))
    huge_total = sum(s.numel() for s in huge.state_dict().values() if hasattr(s, "numel"))
    assert small_total == huge_total, (
        "micro-average accuracy state must not scale with num_classes; "
        f"got small={small_total} huge={huge_total}"
    )


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


# ---------------------------------------------------------------------------
# Plugin tracking / loss metric sets + uniform registry helper
# ---------------------------------------------------------------------------


def test_plugin_accepts_tracking_and_losses_sets():
    # tracking/losses now live on the Request (per-instance), not the Plugin.
    # The Plugin's role is to declare traits that gate metric eligibility.
    plugin = Plugin(name="_test_tracking_losses", traits=(Traits.continuous,))
    try:
        eligible = plugin.trait_eligible_metrics()
        assert Metric.mae in eligible
        assert Metric.mse in eligible
        assert Metric.accuracy not in eligible
    finally:
        _drop_plugin("_test_tracking_losses")


def test_plugin_rejects_tracking_metric_without_stateful_factory():
    orphan = _fresh_metric("_test_no_factory_metric", traits=(Traits.continuous,))
    plugin = Plugin(name="_test_orphan_tracking", traits=(Traits.continuous,))
    try:
        with pytest.raises((KeyError, RuntimeError, ValueError, TypeError)):
            plugin.build_metric_registry([Strata.train], tracking={orphan})
    finally:
        _drop_plugin("_test_orphan_tracking")
        _drop(orphan)


def test_plugin_build_metric_registry_produces_strata_indexed_moduledict():
    plugin = Plugin(name="_test_registry", traits=(Traits.continuous,))
    try:
        registry = plugin.build_metric_registry(
            [Strata.train, Strata.validate],
            tracking={Metric.mae, Metric.mse},
        )
        assert isinstance(registry, torch.nn.ModuleDict)
        assert set(registry.keys()) == {"train_metrics", "validate_metrics"}
        for stratum_key in ("train_metrics", "validate_metrics"):
            per_metric = registry[stratum_key]
            assert set(per_metric.keys()) == {"mae", "mse"}
            assert isinstance(per_metric["mae"], MeanAbsoluteError)
            assert isinstance(per_metric["mse"], MeanSquaredError)
    finally:
        _drop_plugin("_test_registry")


def test_plugin_build_metric_registry_forwards_ndim_and_overrides():
    plugin = Plugin(name="_test_registry_binary", traits=(Traits.discrete,))
    try:
        registry = plugin.build_metric_registry(
            [Strata.train],
            tracking={Metric.accuracy, Metric.auc},
            ndim=1,
            overrides={Metric.accuracy: {"threshold": 0.7}},
        )
        accuracy = registry["train_metrics"]["accuracy"]
        auc = registry["train_metrics"]["auc"]
        assert isinstance(accuracy, BinaryAccuracy)
        assert accuracy.threshold == 0.7
        assert isinstance(auc, TorchMetric)
    finally:
        _drop_plugin("_test_registry_binary")


def test_plugin_iter_tracked_pairs_metrics_with_trackers_in_deterministic_order():
    plugin = Plugin(name="_test_iter_tracked", traits=(Traits.continuous,))
    try:
        registry = plugin.build_metric_registry(
            [Strata.train],
            tracking={Metric.mae, Metric.mse, Metric.rmse},
        )
        pairs = list(plugin.iter_tracked(registry, Strata.train))
        assert [m.name for m, _ in pairs] == ["mae", "mse", "rmse"]
        for metric, tracker in pairs:
            assert isinstance(metric, Metric)
            assert isinstance(tracker, TorchMetric)
    finally:
        _drop_plugin("_test_iter_tracked")


# ---------------------------------------------------------------------------
# Deal contract enforcement
# ---------------------------------------------------------------------------


def test_metric_register_rejects_negative_ndim_via_deal_pre_contract():
    metric = _fresh_metric("_test_neg_ndim_register")
    try:
        with pytest.raises(deal.PreContractError):
            metric.register(lambda p, t: p, ndim=-1)
    finally:
        _drop(metric)


def test_metric_factory_rejects_non_callable_via_deal_pre_contract():
    metric = _fresh_metric("_test_bad_factory_type")
    try:
        with pytest.raises(deal.PreContractError):
            metric.factory(42)  # type: ignore[arg-type]
    finally:
        _drop(metric)


def test_metric_stateful_post_contract_returns_torchmetric():
    tracker = Metric.mae.stateful()
    assert isinstance(tracker, TorchMetric)


# ---------------------------------------------------------------------------
# Tracker-driven loss composition
# ---------------------------------------------------------------------------


class _StubModule:
    def __init__(self) -> None:
        self.tracked: dict[tuple, Any] = {}

    def track(self, names: tuple, value: Any) -> Any:
        self.tracked[names] = value
        return value


def test_plugin_rejects_loss_metric_without_stateful_factory():
    functional_only = Metric(
        "_test_functional_only_loss",
        traits=(Traits.continuous,),
    )
    functional_only.register(lambda pred, target: (pred - target).abs().mean())
    plugin = Plugin(name="_test_rejects_functional_loss", traits=(Traits.continuous,))
    try:
        with pytest.raises((KeyError, RuntimeError, ValueError, TypeError)):
            plugin.build_metric_registry(
                [Strata.train],
                tracking={Metric.mae, functional_only},
            )
    finally:
        _drop_plugin("_test_rejects_functional_loss")
        _drop(functional_only)


def test_tracker_driven_loss_contribution_carries_gradient():
    plugin = Plugin(name="_test_tracker_loss_grad", traits=(Traits.continuous,))
    losses = frozenset({Metric.mae})
    try:
        registry = plugin.build_metric_registry(
            (Strata.train,),
            tracking={Metric.mae},
            ndim=1,
        )
        module = _StubModule()

        pred = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        target = torch.tensor([0.0, 0.0, 3.0, 3.0])

        loss = pred.new_zeros(())
        for metric, tracker in plugin.iter_tracked(registry, Strata.train):
            batch_value = tracker(pred, target)
            if metric in losses:
                loss = loss + batch_value
            module.track(("root/x", Strata.train, metric, "content"), value=tracker)

        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0
        logged_keys = {names for names in module.tracked}
        assert ("root/x", Strata.train, Metric.mae, "content") in logged_keys
    finally:
        _drop_plugin("_test_tracker_loss_grad")


def test_tracker_forward_runs_exactly_once_per_step_and_matches_batch_value():
    class _CountingMAE(TorchMetric):
        # Minimal MAE tracker that counts how many times `forward` is invoked.
        forward_calls: int

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.add_state("sum_abs", default=torch.zeros(()), dist_reduce_fx="sum")
            self.add_state("n", default=torch.zeros(()), dist_reduce_fx="sum")
            self.forward_calls = 0

        def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return super().forward(preds, target)

        def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
            diff = (preds - target).abs()
            self.sum_abs += diff.sum()
            self.n += diff.new_tensor(diff.numel())

        def compute(self) -> torch.Tensor:
            return self.sum_abs / self.n.clamp_min(1.0)

    counting_metric = Metric(
        "_test_counting_mae",
        traits=(Traits.continuous,),
    )
    counting_metric.register(lambda pred, target: (pred - target).abs().mean())
    counting_metric.factory(_CountingMAE)
    plugin = Plugin(name="_test_forward_once", traits=(Traits.continuous,))
    losses = frozenset({counting_metric})
    try:
        registry = plugin.build_metric_registry(
            (Strata.train,),
            tracking={counting_metric},
            ndim=1,
        )
        module = _StubModule()

        pred = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        target = torch.tensor([0.0, 0.0, 3.0, 3.0])

        loss = pred.new_zeros(())
        seen_values: list[torch.Tensor] = []
        for metric, tracker in plugin.iter_tracked(registry, Strata.train):
            batch_value = tracker(pred, target)
            seen_values.append(batch_value)
            if metric in losses:
                loss = loss + batch_value
            module.track(("root/x", Strata.train, metric, "content"), value=tracker)

        strata_registry = registry[f"{Strata.train.value}_metrics"]
        tracker_instance = cast(_CountingMAE, strata_registry[counting_metric.name])
        assert tracker_instance.forward_calls == 1, (
            f"tracker.forward must run exactly once per step; got {tracker_instance.forward_calls}"
        )
        expected = (pred - target).abs().mean().detach()
        assert torch.allclose(loss.detach(), expected)
        assert torch.allclose(seen_values[0].detach(), expected)
    finally:
        _drop_plugin("_test_forward_once")
        _drop(counting_metric)


def test_tracker_driven_empty_branch_still_logs_and_contributes_zero_loss():
    plugin = Plugin(name="_test_empty_branch", traits=(Traits.continuous,))
    losses = frozenset({Metric.mae})
    try:
        registry = plugin.build_metric_registry(
            (Strata.train,),
            tracking={Metric.mae},
            ndim=1,
        )
        module = _StubModule()

        pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        loss = pred.sum() * 0.0
        mask_any = False

        for metric, tracker in plugin.iter_tracked(registry, Strata.train):
            if mask_any:
                batch_value = tracker(pred, pred)
                if metric in losses:
                    loss = loss + batch_value
            module.track(("root/x", Strata.train, metric, "content"), value=tracker)

        assert torch.equal(loss.detach(), torch.zeros(()))
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum().item() == 0
        assert ("root/x", Strata.train, Metric.mae, "content") in module.tracked
    finally:
        _drop_plugin("_test_empty_branch")


# ---------------------------------------------------------------------------
# HitsTotalAccuracy — opt-in O(1)-state multiclass accuracy.
# ---------------------------------------------------------------------------


def test_hits_total_accuracy_matches_argmax_eq_mean_on_logits():
    torch.manual_seed(0)
    logits = torch.randn(64, 10)
    targets = torch.randint(0, 10, (64,))

    tracker = Metric.hits_total_accuracy.stateful()
    tracker.update(logits, targets)

    expected = logits.argmax(dim=-1).eq(targets).float().mean()
    assert torch.allclose(tracker.compute(), expected)


def test_hits_total_accuracy_accepts_prealready_argmaxed_predictions():
    predicted = torch.tensor([0, 1, 2, 3, 2])
    targets = torch.tensor([0, 1, 0, 3, 2])

    tracker = Metric.hits_total_accuracy.stateful()
    tracker.update(predicted, targets)
    assert torch.allclose(tracker.compute(), torch.tensor(4.0 / 5.0))


def test_hits_total_accuracy_aggregates_across_updates_and_resets():
    tracker = Metric.hits_total_accuracy.stateful()
    tracker.update(torch.tensor([0, 1, 2, 3]), torch.tensor([0, 1, 2, 3]))  # 4/4
    tracker.update(torch.tensor([0, 0, 0, 0]), torch.tensor([1, 1, 1, 1]))  # 0/4
    assert torch.allclose(tracker.compute(), torch.tensor(0.5))

    tracker.reset()
    tracker.update(torch.tensor([9, 9]), torch.tensor([9, 9]))
    assert torch.allclose(tracker.compute(), torch.tensor(1.0))


def test_hits_total_accuracy_state_is_o1_regardless_of_num_classes():
    small = Metric.hits_total_accuracy.stateful(num_classes=8)
    large = Metric.hits_total_accuracy.stateful(num_classes=10_000_000)

    def _numel(tracker: TorchMetric) -> int:
        # `_defaults` is torchmetrics' registry of add_state entries and is
        # authoritative regardless of whether states have been mutated yet.
        return sum(t.numel() for t in tracker._defaults.values())

    assert _numel(small) == _numel(large) == 2, (
        "HitsTotalAccuracy must retain exactly two scalar state tensors "
        "(hits + total) regardless of num_classes."
    )


def test_hits_total_accuracy_functional_impl_matches_stateful():
    torch.manual_seed(0)
    logits = torch.randn(32, 5)
    targets = torch.randint(0, 5, (32,))

    tracker = Metric.hits_total_accuracy.stateful()
    tracker.update(logits, targets)

    assert torch.allclose(tracker.compute(), Metric.hits_total_accuracy(logits, targets))


def test_hits_total_accuracy_class_is_a_torchmetric_subclass():
    tracker = HitsTotalAccuracy()
    assert isinstance(tracker, TorchMetric)


# ---------------------------------------------------------------------------
# New classification and regression metrics
# ---------------------------------------------------------------------------


_BINARY_CLS_METRICS = ("specificity", "sensitivity", "f1", "mcc", "cohen_kappa")
_MULTICLASS_CLS_METRICS = (
    "specificity",
    "sensitivity",
    "f1",
    "mcc",
    "cohen_kappa",
    "specificity_at_sensitivity",
    "sensitivity_at_specificity",
)
_REGRESSION_METRICS = ("mape", "r2", "pearson")


@pytest.mark.parametrize("name", _BINARY_CLS_METRICS)
def test_binary_classification_metric_produces_scalar_tracker(name):
    tracker = Metric.get(name).stateful(ndim=1)
    assert isinstance(tracker, TorchMetric)

    torch.manual_seed(0)
    pred = torch.rand(32)
    target = torch.randint(0, 2, (32,))
    tracker.update(pred, target)
    result = tracker.compute()
    assert torch.is_tensor(result) and result.dim() == 0


@pytest.mark.parametrize("name", _BINARY_CLS_METRICS)
def test_binary_classification_metric_functional_matches_stateful(name):
    torch.manual_seed(1)
    pred = torch.rand(32)
    target = torch.randint(0, 2, (32,))

    tracker = Metric.get(name).stateful(ndim=1)
    tracker.update(pred, target)
    functional = Metric.get(name)(pred, target)

    assert torch.allclose(tracker.compute(), functional, atol=1e-6)


@pytest.mark.parametrize("name", _MULTICLASS_CLS_METRICS)
def test_multiclass_classification_metric_produces_scalar_tracker(name):
    tracker = Metric.get(name).stateful(ndim=2, num_classes=4)
    assert isinstance(tracker, TorchMetric)

    torch.manual_seed(0)
    pred = torch.randn(32, 4).softmax(dim=-1)
    target = torch.randint(0, 4, (32,))
    tracker.update(pred, target)
    result = tracker.compute()
    assert torch.is_tensor(result) and result.dim() == 0


@pytest.mark.parametrize("name", _REGRESSION_METRICS)
def test_regression_metric_produces_scalar_tracker(name):
    tracker = Metric.get(name).stateful()
    assert isinstance(tracker, TorchMetric)

    torch.manual_seed(0)
    pred = torch.randn(32)
    target = torch.randn(32)
    tracker.update(pred, target)
    result = tracker.compute()
    assert torch.is_tensor(result) and result.dim() == 0


def test_sensitivity_and_recall_produce_identical_values():
    torch.manual_seed(2)
    pred = torch.rand(64)
    target = torch.randint(0, 2, (64,))
    assert torch.allclose(
        Metric.sensitivity(pred, target),
        Metric.recall(pred, target),
    )


def test_alias_inherits_traits_and_dispositions_by_default():
    source = _fresh_metric(
        "_alias_source",
        traits=(Traits.discrete,),
        higher_is_better=True,
        accumulates_history=True,
    )
    try:
        alias = Metric.alias("_alias_target", of=source)
        try:
            assert alias.traits == source.traits
            assert alias.higher_is_better == source.higher_is_better
            assert alias.accumulates_history == source.accumulates_history
            assert alias.name == "_alias_target"
        finally:
            _drop(alias)
    finally:
        _drop(source)


def test_alias_can_override_traits_higher_is_better_and_history():
    source = _fresh_metric(
        "_alias_override_source",
        traits=(Traits.discrete,),
        higher_is_better=True,
        accumulates_history=False,
    )
    try:
        alias = Metric.alias(
            "_alias_override_target",
            of=source,
            traits=(Traits.continuous,),
            higher_is_better=False,
            accumulates_history=True,
        )
        try:
            assert alias.traits == frozenset({Traits.continuous})
            assert alias.higher_is_better is False
            assert alias.accumulates_history is True
        finally:
            _drop(alias)
    finally:
        _drop(source)


def test_users_can_wire_a_custom_penalty_into_tracking_and_losses():
    """End-to-end: a user defines their own gradient-producing penalty, registers
    functional + stateful forms, and wires it into a Plugin's tracking set
    alongside built-in metrics."""
    penalty = _fresh_metric(
        "_user_penalty",
        traits=(Traits.discrete,),
        higher_is_better=False,
    )

    class _PenaltyState(TorchMetric):
        is_differentiable = True
        higher_is_better = False

        def __init__(self, **_: Any) -> None:
            super().__init__()
            self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

        def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
            per_row = -(prediction.log_softmax(dim=-1).gather(-1, target.unsqueeze(-1))).squeeze(-1)
            self.total = self.total + per_row.sum()
            self.count = self.count + torch.tensor(per_row.numel())

        def compute(self) -> torch.Tensor:
            return self.total / self.count.clamp(min=1)

    @penalty.register(ndim=2)
    def _penalty_functional(
        prediction: torch.Tensor,
        target: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        return -(prediction.log_softmax(dim=-1).gather(-1, target.unsqueeze(-1))).squeeze(-1).mean()

    penalty.factory(_PenaltyState, ndim=2)

    plugin = Plugin(name="_user_penalty_plugin", traits=(Traits.discrete,))
    try:
        registry = plugin.build_metric_registry(
            [Strata.train],
            tracking={Metric.accuracy, penalty},
            ndim=2,
        )
        tracker = registry["train_metrics"]["_user_penalty"]

        pred = torch.randn(8, 4, requires_grad=True)
        target = torch.randint(0, 4, (8,))
        tracker.update(pred.detach(), target)
        assert torch.is_tensor(tracker.compute())

        loss = penalty(pred, target)
        loss.backward()
        assert pred.grad is not None
    finally:
        _drop_plugin("_user_penalty_plugin")
        _drop(penalty)



from __future__ import annotations

import enum
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, TypeAlias, cast

import deal
import torch
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCohenKappa,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    BinarySensitivityAtSpecificity,
    BinarySpecificity,
    BinarySpecificityAtSensitivity,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCohenKappa,
    MulticlassF1Score,
    MulticlassMatthewsCorrCoef,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSensitivityAtSpecificity,
    MulticlassSpecificity,
    MulticlassSpecificityAtSensitivity,
    MultilabelAccuracy,
)
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_cohen_kappa,
    binary_f1_score,
    binary_matthews_corrcoef,
    binary_precision,
    binary_recall,
    binary_specificity,
    multiclass_accuracy,
    multiclass_auroc,
    multiclass_cohen_kappa,
    multiclass_f1_score,
    multiclass_matthews_corrcoef,
    multiclass_precision,
    multiclass_recall,
    multiclass_specificity,
    multilabel_accuracy,
)
from torchmetrics.functional.regression import (
    cosine_similarity,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    normalized_root_mean_squared_error,
    pearson_corrcoef,
    r2_score,
)
from torchmetrics.regression import (
    CosineSimilarity,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
    NormalizedRootMeanSquaredError,
    PearsonCorrCoef,
    R2Score,
)
from torchmetrics.text import Perplexity

if TYPE_CHECKING:
    from relflow.tensorfields.base import Plugin


class Traits(enum.StrEnum):
    discrete = "discrete"
    continuous = "continuous"
    embedding = "embedding"


TraitsInput: TypeAlias = Iterable[Traits | str] | Traits | str | None


@deal.pure
@deal.post(lambda result: isinstance(result, frozenset))
def _coerce_traits(value: TraitsInput) -> frozenset[Traits]:
    if value is None:
        return frozenset()
    if isinstance(value, (Traits, str)):
        return frozenset({Traits(value)})
    return frozenset(Traits(item) for item in value)


MetricImpl = Callable[..., torch.Tensor]
StatefulFactory = Callable[..., TorchMetric]
_ANY_RANK: int = -1


class Metric(str):
    _registry: ClassVar[dict[str, "Metric"]] = {}

    name: str
    value: str
    traits: frozenset[Traits]

    higher_is_better: bool
    accumulates_history: bool
    _impls: dict[int, MetricImpl]
    _factories: dict[int, StatefulFactory]

    loss: ClassVar["Metric"]
    sigma: ClassVar["Metric"]
    throughput: ClassVar["Metric"]
    accuracy: ClassVar["Metric"]
    precision: ClassVar["Metric"]
    recall: ClassVar["Metric"]
    sensitivity: ClassVar["Metric"]
    specificity: ClassVar["Metric"]
    specificity_at_sensitivity: ClassVar["Metric"]
    sensitivity_at_specificity: ClassVar["Metric"]
    f1: ClassVar["Metric"]
    mcc: ClassVar["Metric"]
    cohen_kappa: ClassVar["Metric"]
    auc: ClassVar["Metric"]
    auroc: ClassVar["Metric"]
    perplexity: ClassVar["Metric"]
    multilabel_accuracy: ClassVar["Metric"]
    hits_total_accuracy: ClassVar["Metric"]
    mae: ClassVar["Metric"]
    mse: ClassVar["Metric"]
    rmse: ClassVar["Metric"]
    nrmse: ClassVar["Metric"]
    mape: ClassVar["Metric"]
    r2: ClassVar["Metric"]
    pearson: ClassVar["Metric"]
    cosine_similarity: ClassVar["Metric"]

    ce: ClassVar["Metric"]
    bce: ClassVar["Metric"]
    cosine_dissimilarity: ClassVar["Metric"]

    @deal.raises(TypeError, ValueError)
    def __new__(
        cls,
        name: str,
        *,
        traits: TraitsInput = None,
        higher_is_better: bool = False,
        accumulates_history: bool = False,
    ) -> "Metric":
        if not isinstance(name, str) or not name:
            raise TypeError("Metric name must be a non-empty string")

        if name in Metric._registry:
            raise ValueError(f"Metric '{name}' already registered")

        instance = super().__new__(cls, name)
        instance.name = name
        instance.value = name  # legacy StrEnum callsites read `.value`
        instance.traits = _coerce_traits(traits)
        instance.higher_is_better = higher_is_better
        instance.accumulates_history = accumulates_history
        instance._impls = {}
        instance._factories = {}
        Metric._registry[name] = instance
        return instance


    @deal.raises(TypeError, ValueError)
    @deal.pre(lambda self, fn=None, ndim=None: fn is None or callable(fn))
    @deal.pre(lambda self, fn=None, ndim=None: ndim is None or (isinstance(ndim, int) and ndim >= 0))
    def register(
        self,
        fn: MetricImpl | None = None,
        *,
        ndim: int | None = None,
    ) -> MetricImpl | Callable[[MetricImpl], MetricImpl]:
        if fn is None:
            def wrapper(func: MetricImpl) -> MetricImpl:
                self._install_impl(func, ndim=ndim)
                return func

            return wrapper

        self._install_impl(fn, ndim=ndim)
        return fn

    def _install_impl(self, fn: MetricImpl, *, ndim: int | None) -> None:
        if not callable(fn):
            raise TypeError(f"Metric '{self.name}' implementations must be callable")

        key = _normalize_rank(ndim)
        if key in self._impls:
            existing = "any rank" if key == _ANY_RANK else f"ndim={key}"
            raise ValueError(f"Metric '{self.name}' already has an implementation for {existing}")

        self._impls[key] = fn

    @deal.raises(NotImplementedError, TypeError)
    @deal.post(lambda result: torch.is_tensor(result))
    def __call__(self, prediction: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not self._impls:
            raise NotImplementedError(f"Metric '{self.name}' has no registered functional implementation")

        if not torch.is_tensor(prediction):
            raise TypeError(f"Metric '{self.name}' requires a tensor as the first argument, got {type(prediction).__name__}")

        impl = self._impls.get(prediction.ndim) or self._impls.get(_ANY_RANK)
        if impl is None:
            raise NotImplementedError(_missing_rank_message(self, prediction.ndim, self._impls))

        return impl(prediction, *args, **kwargs)

    @deal.raises(TypeError, ValueError)
    @deal.pre(lambda self, fn=None, ndim=None: fn is None or callable(fn))
    @deal.pre(lambda self, fn=None, ndim=None: ndim is None or (isinstance(ndim, int) and ndim >= 0))
    def factory(
        self,
        fn: StatefulFactory | None = None,
        *,
        ndim: int | None = None,
    ) -> StatefulFactory | Callable[[StatefulFactory], StatefulFactory]:
        if fn is None:
            def wrapper(func: StatefulFactory) -> StatefulFactory:
                self._install_factory(func, ndim=ndim)
                return func

            return wrapper

        self._install_factory(fn, ndim=ndim)
        return fn

    def _install_factory(self, fn: StatefulFactory, *, ndim: int | None) -> None:
        if not callable(fn):
            raise TypeError(f"Metric '{self.name}' factories must be callable")

        key = _normalize_rank(ndim)
        if key in self._factories:
            existing = "any rank" if key == _ANY_RANK else f"ndim={key}"
            raise ValueError(f"Metric '{self.name}' already has a stateful factory for {existing}")

        self._factories[key] = fn

    @deal.raises(NotImplementedError, TypeError, ValueError)
    @deal.post(lambda result: isinstance(result, TorchMetric))
    def stateful(self, *, ndim: int | None = None, **kwargs: Any) -> TorchMetric:
        if not self._factories:
            raise NotImplementedError(f"Metric '{self.name}' has no registered stateful factory")

        key = _normalize_rank(ndim)
        factory = self._factories.get(key) or self._factories.get(_ANY_RANK)
        if factory is None:
            raise NotImplementedError(_missing_rank_message(self, key, self._factories))

        instance = factory(**kwargs)
        if not isinstance(instance, TorchMetric):
            raise TypeError(f"Metric '{self.name}' factory returned {type(instance).__name__}, expected a torchmetrics.Metric instance")

        return instance

    @deal.post(lambda result: isinstance(result, bool))
    def qualifies(self, source: "Plugin | Iterable[Traits | str] | Traits | str") -> bool:
        traits_attr = getattr(source, "traits", None)
        if traits_attr is not None and not isinstance(source, (Traits, str)):
            candidate = _coerce_traits(cast(TraitsInput, traits_attr))
        else:
            candidate = _coerce_traits(cast(TraitsInput, source))

        return self.traits.issubset(candidate)

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return (Metric.get, (self.name,))

    @classmethod
    def get(cls, name: str) -> "Metric":
        try:
            return cls._registry[name]
        except KeyError:
            raise KeyError(f"no metric named {name!r}") from None

    @classmethod
    @deal.raises(TypeError, ValueError)
    def alias(
        cls,
        name: str,
        *,
        of: "Metric",
        traits: TraitsInput = None,
        higher_is_better: bool | None = None,
        accumulates_history: bool | None = None,
    ) -> "Metric":
        instance = cls(
            name,
            traits=of.traits if traits is None else traits,
            higher_is_better=of.higher_is_better if higher_is_better is None else higher_is_better,
            accumulates_history=of.accumulates_history if accumulates_history is None else accumulates_history,
        )
        instance._impls = dict(of._impls)
        instance._factories = dict(of._factories)
        return instance

    @classmethod
    def with_traits(cls, *traits: Traits | str) -> tuple["Metric", ...]:
        candidate = _coerce_traits(traits)
        return tuple(metric for metric in cls._registry.values() if metric.traits.issubset(candidate))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any):
        from pydantic_core import core_schema

        del source_type, handler
        return core_schema.no_info_after_validator_function(
            cls.get,
            core_schema.str_schema(),
        )

    def __repr__(self) -> str:
        traits = ", ".join(sorted(t.value for t in self.traits)) or "-"
        return f"Metric({self.name!r}, traits=[{traits}])"


def _normalize_rank(ndim: int | None) -> int:
    if ndim is None:
        return _ANY_RANK

    key = int(ndim)
    if key < 0:
        raise ValueError(f"ndim must be a non-negative integer or None, got {ndim}")

    return key


def _missing_rank_message(metric: Metric, ndim: int, table: dict[int, Any]) -> str:
    registered = sorted(k for k in table if k != _ANY_RANK)
    return (
        f"Metric '{metric.name}' has no implementation for ndim={ndim}. Registered ranks: {registered or 'none'}"
    )


def _register_binary_multiclass_pair(
    metric: Metric,
    *,
    binary_functional: MetricImpl,
    binary_stateful: StatefulFactory,
    multiclass_functional: MetricImpl,
    multiclass_stateful: StatefulFactory,
    average: str | None = None,
    num_key: str = "num_classes",
) -> None:
    metric.register(binary_functional, ndim=1)
    metric.factory(binary_stateful, ndim=1)

    def _multiclass_impl(
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        if average is not None:
            kwargs.setdefault("average", average)
        if kwargs.get(num_key) is None:
            kwargs[num_key] = int(prediction.shape[-1])
        return multiclass_functional(prediction, target, **kwargs)

    metric.register(_multiclass_impl, ndim=2)
    multiclass_stateful_factory: StatefulFactory = (
        partial(multiclass_stateful, average=average) if average is not None else multiclass_stateful
    )
    metric.factory(multiclass_stateful_factory, ndim=2)


class HitsTotalAccuracy(TorchMetric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    hits: torch.Tensor
    total: torch.Tensor

    def __init__(self, num_classes: int | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        del num_classes
        self.add_state("hits", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros((), dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if preds.ndim > target.ndim:
            preds = preds.argmax(dim=-1)
        self.hits += preds.eq(target).sum().to(torch.long)
        self.total += target.new_tensor(target.numel(), dtype=torch.long)

    def compute(self) -> torch.Tensor:
        return self.hits.float() / self.total.clamp_min(1).float()


class BinarySpecificityAtSensitivityScalar(BinarySpecificityAtSensitivity):
    def __init__(self, *, min_sensitivity: float = 0.9, **kwargs: Any) -> None:
        super().__init__(min_sensitivity=min_sensitivity, **kwargs)

    def compute(self) -> torch.Tensor:
        specificity, _ = super().compute()
        return specificity


class MulticlassSpecificityAtSensitivityScalar(MulticlassSpecificityAtSensitivity):
    def __init__(self, *, min_sensitivity: float = 0.9, **kwargs: Any) -> None:
        super().__init__(min_sensitivity=min_sensitivity, **kwargs)

    def compute(self) -> torch.Tensor:
        specificity, _ = super().compute()
        return specificity.mean()


class BinarySensitivityAtSpecificityScalar(BinarySensitivityAtSpecificity):
    def __init__(self, *, min_specificity: float = 0.9, **kwargs: Any) -> None:
        super().__init__(min_specificity=min_specificity, **kwargs)

    def compute(self) -> torch.Tensor:
        sensitivity, _ = super().compute()
        return sensitivity


class MulticlassSensitivityAtSpecificityScalar(MulticlassSensitivityAtSpecificity):
    def __init__(self, *, min_specificity: float = 0.9, **kwargs: Any) -> None:
        super().__init__(min_specificity=min_specificity, **kwargs)

    def compute(self) -> torch.Tensor:
        sensitivity, _ = super().compute()
        return sensitivity.mean()


Metric.loss = Metric("loss")
Metric.sigma = Metric("sigma")
Metric.throughput = Metric("throughput")

Metric.accuracy = Metric("accuracy", traits=(Traits.discrete,), higher_is_better=True)
Metric.precision = Metric("precision", traits=(Traits.discrete,), higher_is_better=True)
Metric.recall = Metric("recall", traits=(Traits.discrete,), higher_is_better=True)
Metric.specificity = Metric("specificity", traits=(Traits.discrete,), higher_is_better=True)
Metric.specificity_at_sensitivity = Metric(
    "specificity_at_sensitivity", traits=(Traits.discrete,), higher_is_better=True
)
Metric.sensitivity_at_specificity = Metric(
    "sensitivity_at_specificity", traits=(Traits.discrete,), higher_is_better=True
)
Metric.f1 = Metric("f1", traits=(Traits.discrete,), higher_is_better=True)
Metric.mcc = Metric("mcc", traits=(Traits.discrete,), higher_is_better=True)
Metric.cohen_kappa = Metric("cohen_kappa", traits=(Traits.discrete,), higher_is_better=True)

Metric.auroc = Metric(
    "auroc",
    traits=(Traits.discrete,),
    higher_is_better=True,
    accumulates_history=True,
)
Metric.perplexity = Metric("perplexity", traits=(Traits.discrete,))

Metric.multilabel_accuracy = Metric(
    "multilabel_accuracy",
    traits=(Traits.discrete,),
    higher_is_better=True,
)

Metric.hits_total_accuracy = Metric(
    "hits_total_accuracy",
    traits=(Traits.discrete,),
    higher_is_better=True,
)

Metric.mae = Metric("mae", traits=(Traits.continuous,))
Metric.mse = Metric("mse", traits=(Traits.continuous,))
Metric.rmse = Metric("rmse", traits=(Traits.continuous,))
Metric.nrmse = Metric("nrmse", traits=(Traits.continuous,))
Metric.mape = Metric("mape", traits=(Traits.continuous,))
Metric.r2 = Metric("r2", traits=(Traits.continuous,), higher_is_better=True)
Metric.pearson = Metric("pearson", traits=(Traits.continuous,), higher_is_better=True)

Metric.cosine_similarity = Metric(
    "cosine_similarity",
    traits=(Traits.embedding,),
    higher_is_better=True,
)


Metric.mae.register(mean_absolute_error)
Metric.mae.factory(MeanAbsoluteError)

Metric.mse.register(mean_squared_error)
Metric.mse.factory(MeanSquaredError)

Metric.rmse.register(partial(mean_squared_error, squared=False))
Metric.rmse.factory(partial(MeanSquaredError, squared=False))

Metric.nrmse.register(normalized_root_mean_squared_error)
Metric.nrmse.factory(NormalizedRootMeanSquaredError)


@Metric.cosine_similarity.register
def _cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return cosine_similarity(prediction, target, reduction="mean")


Metric.cosine_similarity.factory(CosineSimilarity)

_register_binary_multiclass_pair(
    Metric.accuracy,
    binary_functional=binary_accuracy,
    binary_stateful=BinaryAccuracy,
    multiclass_functional=multiclass_accuracy,
    multiclass_stateful=MulticlassAccuracy,
    average="micro",
)
_register_binary_multiclass_pair(
    Metric.precision,
    binary_functional=binary_precision,
    binary_stateful=BinaryPrecision,
    multiclass_functional=multiclass_precision,
    multiclass_stateful=MulticlassPrecision,
    average="micro",
)
_register_binary_multiclass_pair(
    Metric.recall,
    binary_functional=binary_recall,
    binary_stateful=BinaryRecall,
    multiclass_functional=multiclass_recall,
    multiclass_stateful=MulticlassRecall,
    average="micro",
)
_register_binary_multiclass_pair(
    Metric.auroc,
    binary_functional=binary_auroc,
    binary_stateful=BinaryAUROC,
    multiclass_functional=multiclass_auroc,
    multiclass_stateful=MulticlassAUROC,
)

Metric.perplexity.factory(Perplexity)


@Metric.multilabel_accuracy.register
def _multilabel_accuracy_functional(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_labels: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    kwargs.setdefault("average", "micro")
    return multilabel_accuracy(
        prediction,
        target,
        num_labels=num_labels if num_labels is not None else int(prediction.shape[-1]),
        **kwargs,
    )


Metric.multilabel_accuracy.factory(partial(MultilabelAccuracy, average="micro"))


@Metric.hits_total_accuracy.register
def _hits_total_accuracy_functional(
    prediction: torch.Tensor,
    target: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    del kwargs
    if prediction.ndim > target.ndim:
        prediction = prediction.argmax(dim=-1)
    return prediction.eq(target).float().mean()


Metric.hits_total_accuracy.factory(HitsTotalAccuracy)


Metric.ce = Metric("ce")
Metric.bce = Metric("bce")
Metric.cosine_dissimilarity = Metric("cosine_dissimilarity")


@Metric.ce.register
def _ce_functional(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    if mask is not None:
        per_sample = torch.nn.functional.cross_entropy(
            input=logits,
            target=target,
            weight=weight,
            reduction="none",
        )
        return per_sample.masked_select(mask).mean()

    return torch.nn.functional.cross_entropy(
        input=logits,
        target=target,
        weight=weight,
        reduction=reduction,
    )


@Metric.bce.register
def _bce_functional(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    per_element = torch.nn.functional.binary_cross_entropy_with_logits(
        input=logits,
        target=target.float(),
        reduction="none",
    )
    if weight is not None:
        per_element = per_element * weight[target.long()]

    if mask is not None:
        return per_element.masked_select(mask).mean()

    return per_element.mean()


@Metric.cosine_dissimilarity.register
def _cosine_dissimilarity_functional(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    per_sample = 1.0 - (prediction * target).sum(dim=-1)
    if mask is not None:
        return per_sample.masked_select(mask).mean()

    return per_sample.mean()


_register_binary_multiclass_pair(
    Metric.specificity,
    binary_functional=binary_specificity,
    binary_stateful=BinarySpecificity,
    multiclass_functional=multiclass_specificity,
    multiclass_stateful=MulticlassSpecificity,
    average="micro",
)
_register_binary_multiclass_pair(
    Metric.f1,
    binary_functional=binary_f1_score,
    binary_stateful=BinaryF1Score,
    multiclass_functional=multiclass_f1_score,
    multiclass_stateful=MulticlassF1Score,
    average="micro",
)
_register_binary_multiclass_pair(
    Metric.mcc,
    binary_functional=binary_matthews_corrcoef,
    binary_stateful=BinaryMatthewsCorrCoef,
    multiclass_functional=multiclass_matthews_corrcoef,
    multiclass_stateful=MulticlassMatthewsCorrCoef,
)
_register_binary_multiclass_pair(
    Metric.cohen_kappa,
    binary_functional=binary_cohen_kappa,
    binary_stateful=BinaryCohenKappa,
    multiclass_functional=multiclass_cohen_kappa,
    multiclass_stateful=MulticlassCohenKappa,
)

Metric.specificity_at_sensitivity.factory(BinarySpecificityAtSensitivityScalar, ndim=1)
Metric.specificity_at_sensitivity.factory(MulticlassSpecificityAtSensitivityScalar, ndim=2)

Metric.sensitivity_at_specificity.factory(BinarySensitivityAtSpecificityScalar, ndim=1)
Metric.sensitivity_at_specificity.factory(MulticlassSensitivityAtSpecificityScalar, ndim=2)

Metric.mape.register(mean_absolute_percentage_error)
Metric.mape.factory(MeanAbsolutePercentageError)

Metric.r2.register(r2_score)
Metric.r2.factory(R2Score)

Metric.pearson.register(pearson_corrcoef)
Metric.pearson.factory(PearsonCorrCoef)

Metric.sensitivity = Metric.alias("sensitivity", of=Metric.recall)
Metric.auc = Metric.alias("auc", of=Metric.auroc)

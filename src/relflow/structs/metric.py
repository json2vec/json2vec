"""Metric registry, traits, and rank-dispatched implementations.

Each `Metric` is a singleton registry object exposing two complementary APIs
backed by torchmetrics:

* **Functional** — `Metric.mae(prediction, target)` calls the pure function
  from `torchmetrics.functional.*` and returns a scalar tensor. No state is
  retained; the value reflects only the batch just computed. This is the right
  choice for per-batch loss terms and lightweight per-step logging where cross-
  batch aggregation is handled elsewhere (or explicitly not needed).

* **Stateful** — `Metric.mae.stateful(**kwargs)` returns a fresh, DDP-safe
  `torchmetrics.Metric` instance from `torchmetrics.classification` /
  `torchmetrics.regression`. Owners must attach it to an `nn.Module` (typically
  a decoder's `ModuleDict`) so that Lightning's `.log()` handles
  `sync_dist`/`sync_on_compute` when reducing across ranks. The instance is
  updated per batch and computed at epoch end.

Statefulness carries a memory cost that scales with how many
(address × strata × metric) triples a model tracks. Most torchmetrics we bind
here use *constant* state (running sum + count). A few — notably AUROC and PR-
curve metrics — accumulate raw predictions and targets across the full epoch
and therefore grow linearly in observations. Those are flagged with
`accumulates_history=True` so callers can opt into the extra cost knowingly.

Rank dispatch selects the correct implementation based on the primary tensor's
`ndim`: rank 1 typically maps to the binary variant (e.g. `BinaryAccuracy`) and
rank 2 to the multiclass variant (`MulticlassAccuracy`). Rank-agnostic metrics
(regression) register a single implementation with `ndim=None`.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterable, TypeAlias, cast

import torch
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_precision,
    binary_recall,
    multiclass_accuracy,
    multiclass_auroc,
    multiclass_precision,
    multiclass_recall,
)
from torchmetrics.functional.regression import (
    cosine_similarity,
    mean_absolute_error,
    mean_squared_error,
    normalized_root_mean_squared_error,
)
from torchmetrics.regression import (
    CosineSimilarity,
    MeanAbsoluteError,
    MeanSquaredError,
    NormalizedRootMeanSquaredError,
)
from torchmetrics.text import Perplexity

if TYPE_CHECKING:
    from relflow.tensorfields.base import Plugin


class Traits(enum.StrEnum):
    """Capability tags a tensorfield plugin advertises to metric consumers."""

    discrete = "discrete"
    continuous = "continuous"
    embedding = "embedding"


TraitsInput: TypeAlias = Iterable[Traits | str] | Traits | str | None


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
    """Named metric with trait requirements and torchmetrics-backed dispatch.

    `Metric` subclasses `str` so instances remain compatible with callsites
    that expect `tuple[str, ...]` tracking keys (as the legacy `StrEnum`
    provided). `metric == "loss"` and `hash(metric) == hash("loss")` both hold,
    letting Metric instances substitute directly for strings.
    """

    _registry: ClassVar[dict[str, "Metric"]] = {}

    # Instance attributes (populated in __new__).
    name: str
    value: str
    traits: frozenset[Traits]
    higher_is_better: bool
    # True when the stateful form retains O(N) memory in observations
    # (AUROC, PR-curve metrics). Constant-state metrics leave this False.
    accumulates_history: bool
    _impls: dict[int, MetricImpl]
    _factories: dict[int, StatefulFactory]

    # Canonical metric instances are assigned as class attributes below.
    loss: ClassVar["Metric"]
    sigma: ClassVar["Metric"]
    throughput: ClassVar["Metric"]
    accuracy: ClassVar["Metric"]
    precision: ClassVar["Metric"]
    recall: ClassVar["Metric"]
    auc: ClassVar["Metric"]
    auroc: ClassVar["Metric"]
    perplexity: ClassVar["Metric"]
    mae: ClassVar["Metric"]
    mse: ClassVar["Metric"]
    rmse: ClassVar["Metric"]
    nrmse: ClassVar["Metric"]
    cosine_similarity: ClassVar["Metric"]

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

    # -- functional registration & dispatch -------------------------------

    def register(
        self,
        fn: MetricImpl | None = None,
        *,
        ndim: int | None = None,
    ) -> MetricImpl | Callable[[MetricImpl], MetricImpl]:
        """Register a stateless implementation, optionally scoped to a tensor rank.

        Usage:

            @Metric.mae.register
            def _(prediction, target): ...

            @Metric.accuracy.register(ndim=2)
            def _(prediction, target, *, num_classes: int): ...
        """
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

    def __call__(self, prediction: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not self._impls:
            raise NotImplementedError(f"Metric '{self.name}' has no registered functional implementation")

        if not torch.is_tensor(prediction):
            raise TypeError(
                f"Metric '{self.name}' requires a tensor as the first argument, "
                f"got {type(prediction).__name__}"
            )

        impl = self._impls.get(prediction.ndim) or self._impls.get(_ANY_RANK)
        if impl is None:
            raise NotImplementedError(_missing_rank_message(self, prediction.ndim, self._impls))

        return impl(prediction, *args, **kwargs)

    # -- stateful factory registration & instantiation --------------------

    def factory(
        self,
        fn: StatefulFactory | None = None,
        *,
        ndim: int | None = None,
    ) -> StatefulFactory | Callable[[StatefulFactory], StatefulFactory]:
        """Register a factory that produces a fresh torchmetrics `Metric` instance.

        Accepts either a `torchmetrics.Metric` subclass or a callable that
        returns one.

        Usage:

            Metric.mae.factory(MeanAbsoluteError)
            Metric.accuracy.factory(MulticlassAccuracy, ndim=2)

            @Metric.perplexity.factory
            def _(**kwargs): return Perplexity(**kwargs)
        """
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

    def stateful(self, *, ndim: int | None = None, **kwargs: Any) -> TorchMetric:
        """Instantiate a stateful `torchmetrics.Metric` for the given rank.

        The returned instance must be attached to an `nn.Module` for DDP-safe
        aggregation; Lightning's `self.log()` handles cross-rank sync when the
        module owns it. See `accumulates_history` before choosing this path for
        metrics that grow linearly in observations.
        """
        if not self._factories:
            raise NotImplementedError(
                f"Metric '{self.name}' has no registered stateful factory"
            )

        key = _normalize_rank(ndim)
        factory = self._factories.get(key) or self._factories.get(_ANY_RANK)
        if factory is None:
            raise NotImplementedError(_missing_rank_message(self, key, self._factories))

        instance = factory(**kwargs)
        if not isinstance(instance, TorchMetric):
            raise TypeError(
                f"Metric '{self.name}' factory returned {type(instance).__name__}, "
                "expected a torchmetrics.Metric instance"
            )

        return instance

    # -- trait qualification ---------------------------------------------

    def qualifies(self, source: "Plugin | Iterable[Traits | str] | Traits | str") -> bool:
        traits_attr = getattr(source, "traits", None)
        if traits_attr is not None and not isinstance(source, (Traits, str)):
            candidate = _coerce_traits(cast(TraitsInput, traits_attr))
        else:
            candidate = _coerce_traits(cast(TraitsInput, source))

        return self.traits.issubset(candidate)

    # -- registry lookup --------------------------------------------------

    @classmethod
    def get(cls, name: str) -> "Metric":
        try:
            return cls._registry[name]
        except KeyError:
            raise KeyError(f"no metric named {name!r}") from None

    @classmethod
    def with_traits(cls, *traits: Traits | str) -> tuple["Metric", ...]:
        """Return every registered metric whose traits are covered by `traits`."""
        candidate = _coerce_traits(traits)
        return tuple(metric for metric in cls._registry.values() if metric.traits.issubset(candidate))

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
        f"Metric '{metric.name}' has no implementation for ndim={ndim}. "
        f"Registered ranks: {registered or 'none'}"
    )



# Bookkeeping / label-only metrics (used purely as tracking keys).
Metric.loss = Metric("loss")
Metric.sigma = Metric("sigma")
Metric.throughput = Metric("throughput")

# Discrete-label scoring metrics.
Metric.accuracy = Metric("accuracy", traits=(Traits.discrete,), higher_is_better=True)
Metric.precision = Metric("precision", traits=(Traits.discrete,), higher_is_better=True)
Metric.recall = Metric("recall", traits=(Traits.discrete,), higher_is_better=True)
# AUC/AUROC store per-sample predictions across the epoch to compute the curve,
# so their stateful form is O(N) in observations.
Metric.auc = Metric(
    "auc",
    traits=(Traits.discrete,),
    higher_is_better=True,
    accumulates_history=True,
)
Metric.auroc = Metric(
    "auroc",
    traits=(Traits.discrete,),
    higher_is_better=True,
    accumulates_history=True,
)
Metric.perplexity = Metric("perplexity", traits=(Traits.discrete,))

# Continuous-value scoring metrics.
Metric.mae = Metric("mae", traits=(Traits.continuous,))
Metric.mse = Metric("mse", traits=(Traits.continuous,))
Metric.rmse = Metric("rmse", traits=(Traits.continuous,))
Metric.nrmse = Metric("nrmse", traits=(Traits.continuous,))

# Embedding-comparison metrics.
Metric.cosine_similarity = Metric(
    "cosine_similarity",
    traits=(Traits.embedding,),
    higher_is_better=True,
)


# ---------------------------------------------------------------------------
# torchmetrics-backed implementations
# ---------------------------------------------------------------------------

# Regression / embedding (rank-agnostic).

Metric.mae.register(mean_absolute_error)
Metric.mae.factory(MeanAbsoluteError)

Metric.mse.register(mean_squared_error)
Metric.mse.factory(MeanSquaredError)


@Metric.rmse.register
def _rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return mean_squared_error(prediction, target, squared=False)


@Metric.rmse.factory
def _rmse_factory(**kwargs: Any) -> MeanSquaredError:
    return MeanSquaredError(squared=False, **kwargs)


Metric.nrmse.register(normalized_root_mean_squared_error)
Metric.nrmse.factory(NormalizedRootMeanSquaredError)


@Metric.cosine_similarity.register
def _cosine(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return cosine_similarity(prediction, target, reduction="mean")


Metric.cosine_similarity.factory(CosineSimilarity)


# Classification (rank-dispatched).
# ndim=1: binary. ndim=2: multiclass; caller supplies `num_classes` for
# the stateful factory (or falls back to `prediction.shape[-1]` for functional).

Metric.accuracy.register(binary_accuracy, ndim=1)
Metric.accuracy.factory(BinaryAccuracy, ndim=1)


@Metric.accuracy.register(ndim=2)
def _accuracy_multiclass(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    return multiclass_accuracy(
        prediction,
        target,
        num_classes=num_classes if num_classes is not None else int(prediction.shape[-1]),
        **kwargs,
    )


Metric.accuracy.factory(MulticlassAccuracy, ndim=2)

Metric.precision.register(binary_precision, ndim=1)
Metric.precision.factory(BinaryPrecision, ndim=1)


@Metric.precision.register(ndim=2)
def _precision_multiclass(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    return multiclass_precision(
        prediction,
        target,
        num_classes=num_classes if num_classes is not None else int(prediction.shape[-1]),
        **kwargs,
    )


Metric.precision.factory(MulticlassPrecision, ndim=2)

Metric.recall.register(binary_recall, ndim=1)
Metric.recall.factory(BinaryRecall, ndim=1)


@Metric.recall.register(ndim=2)
def _recall_multiclass(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    return multiclass_recall(
        prediction,
        target,
        num_classes=num_classes if num_classes is not None else int(prediction.shape[-1]),
        **kwargs,
    )


Metric.recall.factory(MulticlassRecall, ndim=2)

Metric.auroc.register(binary_auroc, ndim=1)
Metric.auroc.factory(BinaryAUROC, ndim=1)


@Metric.auroc.register(ndim=2)
def _auroc_multiclass(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    return multiclass_auroc(
        prediction,
        target,
        num_classes=num_classes if num_classes is not None else int(prediction.shape[-1]),
        **kwargs,
    )


Metric.auroc.factory(MulticlassAUROC, ndim=2)

# `auc` is a legacy alias for AUROC in this codebase; wire it to the same
# torchmetrics implementations so downstream extensions keep working.
Metric.auc.register(binary_auroc, ndim=1)
Metric.auc.factory(BinaryAUROC, ndim=1)


@Metric.auc.register(ndim=2)
def _auc_multiclass(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    num_classes: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    return multiclass_auroc(
        prediction,
        target,
        num_classes=num_classes if num_classes is not None else int(prediction.shape[-1]),
        **kwargs,
    )


Metric.auc.factory(MulticlassAUROC, ndim=2)

Metric.perplexity.factory(Perplexity)

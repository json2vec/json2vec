# Stateless Metric Plugin Specification

Status: Slice 1 implemented; loss-path invocation remains proposed for Slice 2

Scope: registry, request configuration, and stateless metric scaffolding first;
runtime migration second

## Summary

RelFlow now has a `relflow.metrics` package whose registered unit is a
frozen Pydantic `Metric` configuration model. A metric owns:

1. its serializable configuration;
2. a formattable `name` string; and
3. a stateless `__call__` implementation that computes one scalar and calls
   `Model.track` itself.

There is no TorchMetric factory, stateful module, stage `ModuleDict`, or
`MetricContribution` in this version.

Metric definitions use the requested registration syntax:

```python
@register(Trait.classification)
class Accuracy(Metric):
    ...


@register("geopoint", "trajectory")
class PathDistance(Metric):
    ...
```

The decorator accepts built-in `Trait` enum members, exact tensorfield plugin
names, or both. Registration makes a metric available; it does not implicitly
enable that metric for every compatible datatype. Trait groups are queryable as
default configurations:

```python
Trait.classification.metrics  # list[Metric]
```

Metrics are selected and configured on a tensorfield request:

```python
import relflow as rf

category = rf.Category(
    size=128,
    metrics=[rf.metrics.Accuracy(0.5)],
)
```

The positional value above is `Accuracy` shorthand for `threshold=0.5`.
Positional construction is only ergonomic sugar: request and checkpoint
serialization use the ordinary named Pydantic fields.

## Goals

- Add a real Pluggy-backed `METRICS` registry under `relflow.metrics`.
- Register frozen Pydantic metric models with `@register(*types)`.
- Make every metric stateless and callable.
- Require `Metric.__call__` to perform its own `Model.track` call and return
  `None`.
- Make `name` a normal string-valued Pydantic field that may reference
  other configuration fields, for example
  `"accuracy@{threshold:.2f}"`.
- Serialize `name` as that ordinary template string.
- Make `str(metric)` render that template into the concrete logging name.
- Categorize metrics with a small `Trait` enum while allowing exact datatype
  names as an extension escape hatch.
- Expose each trait's live, deterministic default catalog through
  `Trait.<member>.metrics`.
- Let every tensorfield request select configured metrics with
  `metrics=[Metric(...), ...]` and preserve those configurations through schema
  and checkpoint serialization.
- Scaffold the stateless predictive metrics already computed directly in
  tensorfield losses.

## Non-goals

- TorchMetric modules or any other metric-owned state.
- Train/validate/test module containers, reset behavior, or accumulator state
  in checkpoints. Metric configuration remains part of the serialized schema.
- Exact dataset-level metrics that require retained history or nonlinear
  finalization.
- AUC in this first version. Its useful epoch semantics require state.
- Using metric results as differentiable training objectives.
- Automatically enabling every registered compatible metric.
- Entry-point discovery for third-party packages.
- Backwards compatibility with the existing `Metric` string enum.

## Proposed package layout

```text
src/relflow/metrics/
├── __init__.py
├── base.py
├── spec.py
└── extensions/
    ├── __init__.py
    ├── classification.py
    ├── dateparts.py
    └── regression.py

tests/metrics/
├── test_registry.py
├── test_naming.py
├── test_requests.py
├── test_traits.py
└── extensions/
    ├── test_classification.py
    ├── test_dateparts.py
    └── test_regression.py
```

| File | Responsibility |
| --- | --- |
| `spec.py` | Pluggy markers and the metric-provider hookspec. |
| `base.py` | `Trait`, `Metric`, `MetricPlugin`, the `MetricRegistry` object, naming, and dynamic parsing. |
| `extensions/*` | Eagerly imported built-in stateless metric models. |

## Public API

The first slice exports these names from `relflow.metrics`:

```python
from relflow.metrics import (
    METRICS,
    Accuracy,
    MAE,
    Metric,
    MetricRegistry,
    RMSE,
    Trait,
    register,
    registry,
)

classification_defaults: list[Metric] = Trait.classification.metrics
```

The root package exports the module itself, so the supported user-facing form
is:

```python
import relflow as rf

field = rf.Category(
    metrics=[rf.metrics.Accuracy(0.5)],
)

all_classification_metrics = rf.metrics.Trait.classification.metrics
```

`relflow.__init__` explicitly binds and exports `metrics`; this syntax does not
depend on an incidental Python submodule import side effect.

The existing `relflow.structs.enums.Metric` conflicts with the new Pydantic
base. Because backwards compatibility is not a goal, implementation renames
remaining internal logging labels to `LogKey` and makes the Pydantic model the
public `relflow.Metric`.

## The `name` field

`name` is an ordinary Pydantic `str` field containing a restricted Python
format string:

```python
class Accuracy(Metric):
    threshold: float = 0.5
    name: str = "accuracy@{threshold:.2f}"
```

`Accuracy(0.5).name` is therefore `"accuracy@{threshold:.2f}"`. The threshold
is configuration for the tensorfield adapter; `Accuracy.__call__` receives
the adapter's prepared decisions and effective trainable mask.

There is no custom runtime string type. Pydantic serializes and deserializes
the template exactly as a normal string:

```python
metric = Accuracy(0.5)

metric.name
# "accuracy@{threshold:.2f}"

str(metric)
# "accuracy@0.50"

metric.model_dump()["name"]
# "accuracy@{threshold:.2f}"

metric.model_dump_json()
# ... "name":"accuracy@{threshold:.2f}" ...
```

The serialized value is the template source, not the rendered logging key.
This preserves a parameterized name across a configuration round trip.

### Allowed formatting

Templates use the standard format mini-language for values but a deliberately
restricted field grammar. V1 permits only a top-level Pydantic field name:

```text
accuracy@{threshold:.2f}
{type}
```

The following are rejected:

- anonymous or positional fields such as `{}` and `{0}`;
- unknown fields;
- `{name}` recursion;
- attribute access such as `{threshold.real}`;
- index access such as `{options[threshold]}`;
- nested replacement fields inside a format specifier; and
- conversions such as `!r`, `!s`, or `!a` in V1.

`Metric` validates `name` with a Pydantic field validator using
`string.Formatter.parse`. Each placeholder must be a simple identifier present
in the concrete metric's `model_fields`, excluding `name`.

There are two validation phases:

1. Pydantic validates the template against the concrete model fields.
2. Instance validation renders the template against the fully validated
   metric and catches incompatible specifications such as applying `:d` to a
   decimal value.

`Metric.__str__` performs rendering with actual field values, not
`model_dump(mode="json")`, so ordinary Python formatting behavior is retained:

```python
def __str__(self) -> str:
    fields = {
        field_name: getattr(self, field_name)
        for field_name in type(self).model_fields
        if field_name != "name"
    }
    rendered = self._formatter.vformat(self.name, (), fields)
    if not rendered or rendered != rendered.lower():
        raise ValueError("metric names must be nonempty and lowercase")
    if "/" in rendered or any(character.isspace() for character in rendered):
        raise ValueError("metric names may not contain '/' or whitespace")
    return rendered
```

The rendered value must be nonempty, lowercase, and free of `/` and
whitespace. Periods are allowed because decimal names such as
`accuracy@0.50` require them. Request and schema validation reject collisions
using the fully rendered log suffix rather than the unrendered template.

`name` may be overridden by a user or third-party extension. An override must
obey the same placeholder and rendered-name validation and round-trips
unchanged.

## Pydantic `Metric` contract

```python
class Metric(pydantic.BaseModel, ABC):
    model_config = pydantic.ConfigDict(
        defer_build=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )

    type: str
    name: str

    @pydantic.field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, name: str) -> str:
        # Reject placeholders outside the concrete model's fields.
        ...
        return name

    def __str__(self) -> str:
        fields = {
            field_name: getattr(self, field_name)
            for field_name in type(self).model_fields
            if field_name != "name"
        }
        rendered = self._formatter.vformat(self.name, (), fields)
        return rendered

    @abstractmethod
    def __call__(
        self,
        module: Model,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        trainable: torch.Tensor,
        *,
        address: Address,
        strata: Strata,
        scope: tuple[str, ...],
    ) -> None:
        """Compute and track one stateless scalar metric."""
```

Every concrete implementation must compute one scalar from its trait-prepared
inputs after applying `trainable`, call `module.track` with the complete key,
and return `None`. It does not rediscover or validate tensorfield semantics.

The standard call is:

```python
module.track(
    (address, strata, str(self), *scope),
    value=value,
)
return None
```

The caller passes prepared and decoded predictions and targets without
preselecting masked entries. The tensorfield also passes the effective
trainable/valued/known mask as `trainable`, in a shape compatible with the
metric's per-item values or count inputs. Metrics apply
`masked_select(trainable)` to those per-item values/count inputs before their
reduction. Tensorfields own construction of the effective mask,
sigmoid/threshold application, argmax/top-k decoding, and reshaping. Metrics do
not infer the datatype from tensor rank or values.

Metric calls run under `torch.no_grad()`, either on the concrete method or in
the generic invocation loop. Metric computations cannot retain the training
graph or contribute to the differentiable loss.

Metric configuration is keyword-based by default. A concrete built-in may
provide a documented positional convenience when it has one obvious primary
parameter. `Accuracy` supports exactly one such argument:

```python
_MISSING = object()


class Accuracy(Metric):
    def __init__(self, threshold=_MISSING, /, **data):
        if threshold is not _MISSING:
            if "threshold" in data:
                raise TypeError("threshold was provided both positionally and by keyword")
            data["threshold"] = threshold
        super().__init__(**data)
```

Consequently `Accuracy(0.5)` and `Accuracy(threshold=0.5)` produce equal
Pydantic models. More than one positional argument is rejected by Python, and
third-party metrics remain keyword-only unless they deliberately implement and
document equivalent sugar.

## Example definition

```python
from typing import Annotated, Literal


@register(Trait.classification)
class Accuracy(Metric):
    type: Literal["accuracy"] = "accuracy"
    name: str = "accuracy@{threshold:.2f}"
    threshold: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] = 0.5

    def __call__(
        self,
        module,
        predictions,
        targets,
        trainable: torch.Tensor,
        *,
        address,
        strata,
        scope,
    ) -> None:
        value = predictions.eq(targets).masked_select(trainable).float().mean()
        module.track(
            (address, strata, str(self), *scope),
            value=value,
        )
        return None
```

This model serializes the selected template; an instance with threshold `0.5`
stores `"accuracy@{threshold:.2f}"` and renders `"accuracy@0.50"` immediately
before tracking. The datatype adapter consumes the threshold while preparing
Boolean decisions; categorical adapters supply decoded class ids.

## Pluggy registry

The metric package uses a distinct Pluggy project named `"metrics"`:

```python
hookspec = pluggy.HookspecMarker("metrics")
hookimpl = pluggy.HookimplMarker("metrics")


class PluginSpec:
    @hookspec
    def metric(self) -> "MetricPlugin":
        """Contribute one registered metric definition."""
```

Unlike the current tensorfield manager, the metric manager is active:

1. `@register` creates a `MetricPlugin` provider.
2. `MetricRegistry` registers the provider with its owned Pluggy manager.
3. The same object rebuilds its live mapping from the metric hook results.

The registry stores definitions, never runtime values:

```python
@dataclass(frozen=True)
class MetricPlugin:
    key: str
    Metric: type[Metric]
    traits: frozenset[Trait]
    data_types: frozenset[str]
    default: Metric | None

    @hookimpl
    def metric(self) -> "MetricPlugin":
        return self

    def accepts(self, *, data_type: str, traits: frozenset[Trait]) -> bool:
        return data_type in self.data_types or bool(self.traits & traits)


class MetricRegistry:
    def __init__(self):
        self.plugin_manager = pluggy.PluginManager("metrics")
        self.metrics: dict[str, MetricPlugin] = {}

    def register(self, *types): ...
    def parse(self, value): ...
    def validate_request(self, metrics, *, data_type, traits): ...


registry = MetricRegistry()
METRICS = registry.metrics
register = registry.register
```

`METRICS` is analogous to `TENSORFIELDS`: its key is the serialized metric
discriminator, while the value contains the registered class and applicability
metadata.

Registration is transactional. Validation and duplicate checks occur before
Pluggy mutation. If rebuilding the temporary index fails, the new provider is
unregistered and the existing `METRICS` dictionary remains unchanged.
Successful rebuilds sort by discriminator and update the existing dictionary
with `clear()`/`update()` rather than rebinding it. This keeps imported registry
references live and gives `Trait.<member>.metrics` deterministic results.

Built-ins are made available by eager imports from
`relflow.metrics.extensions`, matching the current tensorfield import model.

## Trait and datatype selectors

`Trait` lives in `metrics/base.py` beside `METRICS`. The initial trait set is a
closed plain enum, and each member reads that module-global registry directly
to provide a live, deterministically ordered view of its default metrics:

```python
class Trait(enum.Enum):
    classification = "classification"
    regression = "regression"
    cyclic = "cyclic"

    @property
    def metrics(self) -> list["Metric"]:
        return [
            cast(Metric, plugin.default).model_copy(deep=True)
            for _, plugin in sorted(METRICS.items())
            if self in plugin.traits
        ]
```

The public lookup is therefore:

```python
Trait.classification.metrics
# [Accuracy()]


@register(Trait.classification)
class BalancedAccuracy(Metric):
    ...
```

`.metrics` queries the live registry on every access and returns a new list of
deep-copied frozen defaults sorted by discriminator. Callers may therefore
modify the list without mutating registry state or sharing nested defaults. A
metric registered only to an exact datatype name does not appear in a trait
group; a metric registered to both a trait and a datatype appears once in that
trait's list.

To make `Trait.<member>.metrics` reliably return `list[Metric]`, every class
registered to at least one trait must be default-constructible. Registration
validates `MetricSubclass.model_validate({})` once and stores that canonical
default. Exact-datatype-only registrations may retain required fields and use
`default=None`, because they never appear in a trait group. Configured variants
are still created normally—for example `Accuracy(0.75)` or
`Accuracy(threshold=0.75)`.

That canonical default must implement the common input forms for every
datatype advertising the trait. A metric whose default only works for Boolean,
for example, registers to the exact `"boolean"` datatype rather than
`Trait.classification`. Built-in tests exercise every trait default against
every built-in datatype carrying that trait. This makes
`rf.Category(metrics=Trait.classification.metrics)` a usable catalog snapshot,
not a list that predictably fails at runtime.

The registry's decorator method accepts one or more selectors as positional
arguments. The public `register` name is its bound-method alias:

```python
MetricSelector = Trait | str

registry = MetricRegistry()
register = registry.register
```

Passing a tuple as one selector is rejected; callers unpack an existing
sequence explicitly with `@register(*selectors)`.

Enum members and datatype-name strings are normalized into separate sets. A
plain `Enum` is intentional: `Trait.classification` and the exact datatype
name `"classification"` must remain distinct.

Multiple selectors use inclusive OR semantics. A metric registered with:

```python
@register(Trait.regression, "geopoint")
```

may be invoked for a regression scope or for the exact `geopoint` tensorfield.
Matching both selectors still yields one invocation.

Datatype strings must satisfy `re.fullmatch(r"[a-z0-9_]+", name)`. They are not
resolved against `TENSORFIELDS` during decoration, so metric and datatype
extensions can be imported in either order. When a request is constructed, its
validated `type` is the canonical `TENSORFIELDS` key and is compared exactly;
schema binding repeats that check.

An exact datatype selector changes eligibility only. It does not allow a
different Python call signature: every metric still receives
`(module, predictions, targets, trainable, *, address, strata, scope)`.

## Decorator validation

`@register(*types)` will:

1. normalize and validate the trait/datatype selectors;
2. require a concrete `Metric` subclass;
3. require a default string discriminator matching
   `^[a-z][a-z0-9_]*$`;
4. let normal Pydantic construction validate metric fields and `name`;
5. if any trait selector is present, default-construct and store the canonical
   metric instance used by `Trait.<member>.metrics`;
6. reject a duplicate discriminator rather than replacing it;
7. register the provider with Pluggy and rebuild `METRICS`; and
8. return the decorated class unchanged.

Metric discriminator keys use `^[a-z][a-z0-9_]*$`.

## Dynamic Pydantic parsing

Metric configurations resolve through the live registry instead of a static
discriminated union. Registry dispatch is part of the `Metric` base model's
Pydantic schema, so requests expose the user-facing type directly:

```python
class RequestBase(Leaf):
    metrics: list[Metric] = pydantic.Field(default_factory=list)
```

The base schema delegates to the registry object's parser before validating a
`Metric` value:

```python
class MetricRegistry:
    def parse(self, value: object) -> Metric:
        if isinstance(value, Metric):
            plugin = self.metrics.get(value.type)
            if plugin is None or type(value) is not plugin.Metric:
                raise ValueError("metric instance does not match its registration")
            return value

        if not isinstance(value, Mapping):
            raise ValueError("metric configuration must be a mapping")

        key = value.get("type")
        plugin = self.metrics.get(key)
        if plugin is None:
            raise ValueError(f"unknown metric type: {key!r}")
        return plugin.Metric.model_validate(value)
```

The custom Pydantic schema installs this dispatch only for the abstract base.
Concrete subclasses must use their normal generated schema, preventing
`registry.parse -> Accuracy.model_validate -> registry.parse` recursion:

```python
@classmethod
def __get_pydantic_core_schema__(cls, source_type, handler):
    if source_type is not Metric:
        return handler(source_type)
    return core_schema.no_info_before_validator_function(
        registry.parse,
        core_schema.any_schema(),
        json_schema_input_schema=metric_mapping_schema,
    )
```

The inner `any_schema` supplies Pydantic's native polymorphic serialization,
so concrete configuration fields and enclosing serialization options are
preserved without a custom serializer function.

There is no separate `MetricConfig` public type: `Metric` is already the
configuration. `name` remains a normal string in both direct and nested
serialized output.

## Tensorfield request configuration

The current `RequestBase` alias becomes a concrete subclass of `Leaf`. It owns
the metric field once for every built-in and third-party tensorfield request:

```python
class RequestBase(Leaf):
    metrics: list[Metric] = pydantic.Field(default_factory=list)

    @pydantic.model_validator(mode="after")
    def validate_metrics(self):
        registry.validate_request(
            self.metrics,
            data_type=self.type,
            traits=TENSORFIELDS[self.type].traits,
        )
        return self

    def post_bind_validate(self):
        super().post_bind_validate()  # preserve Leaf's required-query check
        registry.validate_request(
            self.metrics,
            data_type=self.type,
            traits=TENSORFIELDS[self.type].traits,
        )
```

Tensorfield component registration changes its `Request` check from “subclass
of `Node`” to “subclass of `RequestBase`”. This guarantees that every
third-party tensorfield request has the same metric field and validation
contract. The metrics package does not import tensorfield implementations;
RelFlow eagerly initializes metrics before tensorfields at the package root,
so the dependency direction does not form an initialization cycle.

The public constructor accepts configured instances:

```python
request = rf.Category(
    size=128,
    metrics=[
        rf.metrics.Accuracy(0.5),
        rf.metrics.Accuracy(0.75),
    ],
)
```

It also accepts the equivalent serialized mappings when a schema is loaded:

```python
request = rf.Category.model_validate(
    {
        "type": "category",
        "size": 128,
        "metrics": [
            {
                "type": "accuracy",
                "name": "accuracy@{threshold:.2f}",
                "threshold": 0.5,
            }
        ],
    }
)

assert isinstance(request.metrics[0], rf.metrics.Accuracy)
```

The list has these semantics:

- omission produces an independent empty list;
- `metrics=[]` selects no registry-backed content metrics; existing hard-coded
  datatype logging remains unchanged until Slice 2;
- declaration order is preserved through validation and serialization;
- multiple instances of the same metric type are allowed when their rendered
  names differ;
- a repeated rendered name is rejected on the request; and
- the singular legacy-looking key `metric=` is rejected explicitly rather
  than being retained as an unused extra field.

Registration means *available*, not *selected*. Importing a third-party metric
does not mutate existing request defaults. A user may deliberately select the
current live defaults for a group:

```python
request = rf.Category(
    metrics=rf.metrics.Trait.classification.metrics,
)
```

That expression takes a snapshot at request construction. Later registrations
do not mutate `request.metrics`. Supported schema mutation APIs revalidate the
complete list; mutating the list after model binding is unsupported. Built-in
tensorfields may declare explicit default factories during migration, but
those factories must name the metric configurations directly; they must not
read a live trait catalog and thereby let an import change model behavior.

### Datatype traits and eligibility

Tensorfield plugins gain immutable content-trait metadata:

```python
category = Plugin(name="category", traits=(Trait.classification,))
number = Plugin(name="number", traits=(Trait.regression,))
dateparts = Plugin(name="dateparts", traits=(Trait.cyclic,))
```

The initial content-trait declarations are:

| Tensorfield | Content trait | Existing content metrics represented by the scaffold |
| --- | --- | --- |
| Boolean | `Trait.classification` | Accuracy, precision, recall, specificity; AUC deferred. |
| Category | `Trait.classification` | Accuracy and configured top-k accuracy. |
| Set | `Trait.classification` | Multilabel accuracy. |
| Hash | `Trait.classification` | Multiclass accuracy over hash lanes. |
| Cluster | `Trait.classification` | Accuracy after the datatype derives vocabulary scores. |
| Number | `Trait.regression` | MAE and RMSE on the original scale. |
| Vector, Text | `Trait.regression` | MAE and RMSE on prepared vectors. |
| DateParts | `Trait.cyclic` | Angular MAE per configured date part. |

The trait says which common invocation family is eligible; the tensorfield
still owns construction of the effective trainable/valued/known mask,
reshaping, vocabulary projection, and other preparation. The metric owns
applying that mask to its per-item reduction inputs.

For each request item, `registry.validate_request` resolves the registered
`MetricPlugin` and accepts it when either:

1. the request's exact `type` occurs in `MetricPlugin.data_types`; or
2. the tensorfield plugin's traits intersect `MetricPlugin.traits`.

The metric is invoked once even when both conditions match. Exact datatype
selectors are not resolved when the metric decorator executes, preserving
third-party import-order independence. The request validator checks the
schema-resolved datatype, and `post_bind_validate` repeats the check
defensively before model construction.

This first API checks registry identity and broad eligibility at request
construction. Shape-dependent restrictions, such as whether a particular
classification configuration accepts `[N]` scores or `[N, C]` scores, are
validated by the tensorfield adapter and metric implementation when their
concrete input contract is available.

### Scope

The flat V1 list is reserved for the tensorfield's public `content` metric
scope. Slice 1 stores and validates that intent but does not invoke the list.
It does not configure the internal five-way state classifier. This avoids
making one `Accuracy` instance ambiguously run against both state and content.
State accuracy and operational diagnostics remain direct internal logging.

In Slice 2, most tensorfields will invoke each configured metric once with
`scope=(TensorKey.content,)`. A structured content adapter may expand one
configuration deterministically: DateParts is expected to invoke it once per
configured part with `scope=(TensorKey.content, datepart)`. Slice 1 rejects
collisions among the rendered names in the flat request list; expanded,
scope-aware collision validation belongs to that integration work.

The existing schema paths require no alternate representation:
`Schema.request_from_leaf`, schema updates, Python/JSON dumps, and checkpoint
restoration all carry the request's metric list. Round-trip tests must verify
that the concrete metric class, all subclass fields, and the unrendered `name`
template survive each path.

Restoring a checkpoint that names a third-party metric requires importing that
metric's registration module first. Built-ins are eager; automatic third-party
entry-point discovery remains outside this slice.

## Stateless built-ins

The first scaffold registers only functional metrics:

| Configuration | Selector | Rendered name form(s) | Per-call computation |
| --- | --- | --- | --- |
| `Accuracy` | `Trait.classification` | `accuracy@0.50` | Equality mean over masked prepared decisions or class ids. |
| `Precision` | `"boolean"` | `precision@0.50` | Micro precision over masked prepared Boolean count inputs. |
| `Recall` | `"boolean"` | `recall@0.50` | Micro recall over masked prepared Boolean count inputs. |
| `Specificity` | `"boolean"` | `specificity@0.50` | Micro specificity over masked prepared Boolean count inputs. |
| `MAE` | `Trait.regression` | `mae` | Mean masked absolute error for one prepared input pair. |
| `RMSE` | `Trait.regression` | `rmse` | Root mean masked squared error for one prepared input pair. |
| `AngularMAE` | `Trait.cyclic`, `"dateparts"` | `mae` | Mean masked angular error for normalized sin/cos pairs. |

Classification metrics do not inspect ranks or recover datatype semantics.
The tensorfield supplies prepared, same-shaped decisions and targets plus the
effective trainable/valued/known mask in a compatible shape. `Accuracy`
computes equality, applies `masked_select(trainable)`, and then takes the mean.
Precision, recall, and specificity apply the same selection to the prediction
and target inputs used to form their counts. A Boolean adapter uses
`metric.threshold` while preparing decisions; a categorical adapter owns
argmax; Category retains its existing direct top-k logging until a separate
top-k metric contract is requested. Precision, recall, and specificity are
registered only for Boolean because no multiclass averaging contract has been
specified.

Adapters skip invocation when the effective mask selects no items. Precision,
recall, and specificity use zero when their conditional denominator is zero.

The implementations may use ordinary tensor operations or
`torchmetrics.functional`; they must not construct `torchmetrics.Metric`
objects.

### Semantics of this first slice

These are per-invocation values. They delegate epoch aggregation to the
existing `Model.track`/Lightning behavior. Therefore:

- masked or differently sized invocations are not guaranteed to produce an
  observation-weighted global mean;
- scalar metrics retain the current rank-zero logging behavior; and
- `RMSE` is the mean of logged per-invocation RMSE values, matching the current
  style rather than a newly promised global RMSE.

Those limitations are explicit consequences of the stateless-only scope.
Correct global nonlinear aggregation and distributed metric state belong to a
future stateful design, not this scaffold.

Current `AUC` is excluded because averaging batch AUC values is not a useful
replacement for its existing history-bearing TorchMetric behavior. `loss`,
`throughput`, `sigma`, vocabulary size, and cluster diagnostics also remain
outside the predictive metric registry.

## Slice 2 invocation (deferred)

Stateless request metrics will not need a binding record, collection, or
PyTorch module container. The validated request will remain the source of
truth, and invocation will preserve its configured order:

```python
request = module.schema.requests[address]

with torch.no_grad():
    predictions = predictions.detach()
    targets = targets.detach()
    trainable = trainable.detach()
    for metric in request.metrics:
        metric(
            module,
            predictions,
            targets,
            trainable,
            address=address,
            strata=strata,
            scope=scope,
        )
```

The loss/adapter does not call `Model.track`; the metric does. Tests enforce
that every invocation makes exactly one tracking call and returns `None`.

Each tensorfield loss remains responsible for producing the canonical
predictions, targets, and effective trainable/valued/known mask. Boolean
applies sigmoid and the configured threshold before passing Boolean decisions;
Category passes argmax class ids and a mask incorporating known targets;
Number passes original-scale values and a compatible effective mask; DateParts
passes normalized sin/cos pairs and a per-part compatible effective mask. The
metric, rather than the tensorfield, applies `masked_select(trainable)` to its
per-item values/count inputs. The integration slice may extract that
preparation into a datatype adapter, but the adapter API is not required for
the registry scaffold.

Slice 2 schema binding will render the complete suffix for every request metric
and every scope the datatype will emit, then reject collisions before
training. It must not reorder the request list or copy metrics into a second
runtime registry.

Registered does not mean invoked. Importing a third-party metric changes
`METRICS` only; it must not alter datatype defaults or existing model logs.

## Validation and errors

| Condition | Required behavior |
| --- | --- |
| Decorated object is not a concrete `Metric` subclass | `TypeError` |
| Empty or malformed selector input | `TypeError` |
| Malformed exact datatype selector | `ValueError` |
| Invalid or duplicate metric discriminator | `ValueError` |
| Missing or malformed `name` field | Pydantic validation error |
| Trait-registered metric cannot be default-constructed | `TypeError` at registration naming the required fields |
| Unknown, numeric/anonymous, nested, or traversing template placeholder | `ValueError` |
| Format specification is incompatible with the instance value | `ValueError` during model validation |
| Rendered name is empty, uppercase, or contains slash/whitespace | `ValueError` during model validation |
| Missing `__call__` override | `TypeError` at registration because the metric remains abstract |
| Unknown serialized metric discriminator | `ValueError` |
| Existing metric instance does not match the registered class | `ValueError` |
| `Accuracy` threshold is passed both positionally and by keyword | `TypeError` |
| Tensorfield `Request` component does not subclass `RequestBase` | `TypeError` during tensorfield registration |
| Request `metrics` cannot be parsed as a sequence of registered metric configurations | Pydantic validation error |
| Singular request option `metric=` is supplied | `ValueError` during request validation |
| Neither trait nor exact datatype name matches the request | `ValueError` during request validation |
| Colliding fully rendered log suffix | `ValueError` during request or schema validation |
| Metric returns a value or fails to call `Model.track` exactly once | test failure; built-ins must obey the contract |

## Test plan

### Registry

- The decorator returns the original class.
- Single and multiple trait/name selectors normalize into separate immutable
  sets.
- `Trait.classification` remains distinct from raw `"classification"`.
- `Trait.classification.metrics` is live, sorted by discriminator, returns a
  fresh list of deep-copied `Metric` defaults, and includes each matching
  registration once.
- An exact-datatype-only registration is absent from every trait group, while a
  newly imported trait registration appears on the next property access.
- Trait registration rejects a metric with required configuration and no
  default instance; exact-datatype-only registration permits it.
- Every built-in trait default executes against the prepared input forms and
  compatible effective masks of every built-in tensorfield carrying that
  trait.
- Metric and datatype modules may import in either order.
- Pluggy owns the provider and `METRICS` is rebuilt from the hook results.
- Duplicate and partially failed registrations roll back transactionally.
- Registration does not enable a metric for existing datatypes.

### Pydantic and naming

- `name` is an ordinary `str` at runtime and in Python/JSON serialization.
- JSON and containing-model round trips restore the registered subclass and
  template value.
- The dynamic `list[Metric]` request item JSON Schema describes the stable
  `type` and `name` fields as strings and permits concrete configuration
  fields. It does not claim a static discriminated union over a live registry.
- Valid templates render expected values, including
  `str(Accuracy(0.5)) == "accuracy@0.50"`.
- Unknown fields, numeric/anonymous fields, recursion, traversal, nested format
  fields, conversions, and incompatible format specs fail.
- User-overridden templates validate and round-trip.
- Rendered log-key collisions fail before runtime.

### Tensorfield requests

- `rf.Category(metrics=[rf.metrics.Accuracy(0.5)])` constructs successfully and
  stores an `Accuracy` instance in declaration order.
- `Accuracy(0.5)` equals `Accuracy(threshold=0.5)`; excess positional arguments
  and positional-plus-keyword duplication fail.
- Omitting `metrics` gives each request an independent empty list, and
  `metrics=[]` remains empty.
- Direct instances and tagged dictionaries both parse into the registered
  concrete metric subclass.
- Python dump, JSON dump/load, `Schema.from_tree`, schema update, and checkpoint
  reconstruction preserve `type`, the unrendered `name`, and every subclass
  configuration field.
- A trait-compatible metric and an exact-datatype metric are accepted; an
  ineligible metric is rejected during request construction and the invariant
  is checked again during schema binding.
- Request order is preserved, while duplicate rendered content names are
  rejected.
- `metric=` is rejected rather than silently retained through `extra="allow"`.
- Tensorfield registration rejects a third-party `Request` that bypasses
  `RequestBase`.
- Passing `Trait.classification.metrics` takes an independent snapshot; a later
  registry change does not mutate the request.

### Stateless execution

- Every built-in is called with a spy model and `trainable` tensor and makes
  exactly one `Model.track` call.
- The call key is `(address, strata, str(metric), *scope)`.
- `__call__` returns `None`.
- Values match direct tensor reference calculations after
  `masked_select(trainable)`.
- Inputs requiring gradients do not leave a retained graph in the tracked
  value.
- Prepared decisions and compatible effective masks match the same direct
  tensor reductions already used by the tensorfields.
- No metric creates a TorchMetric, `torch.nn.Module`, or mutable accumulator.
- AUC is absent from the initial registry and documented as deferred.

### Public API and regression

- `relflow.metrics` exposes the base, decorator, registry, traits, and
  built-ins.
- `import relflow as rf` exposes the module as `rf.metrics`.
- Top-level `relflow.Metric` is the Pydantic base after the old enum migration.
- Existing tensorfield tests remain green after their logging labels move to
  `LogKey`.
- The complete suite and type checks pass.

## Implementation sequence

### Slice 1: stateless registry scaffold

1. Put restricted formatting, validation, and serialization behavior on
   `Metric`; each concrete `__call__` invokes `Model.track` directly.
2. Add the active Pluggy hookspec and a `MetricRegistry` object that owns
   provider registration, the live `METRICS` index, and dynamic parsing.
3. Define `Trait` beside the registry in `metrics/base.py`, with its direct live
   `.metrics` property and exact datatype selector normalization.
4. Replace the `RequestBase = Leaf` alias with a concrete `RequestBase(Leaf)`
   carrying `metrics: list[Metric]` and request validation.
5. Add immutable content traits to tensorfield `Plugin` metadata and declare
   them on the built-ins.
6. Add stateless built-in configuration models, including the documented
   `Accuracy(0.5)` constructor shorthand, and their callable behavior.
7. Add eager built-in imports and export the `rf.metrics` namespace.
8. Rename the old logging enum to avoid the `Metric` name collision.
9. Add isolated registry, naming, selector, request-round-trip, and callable
   tests.
10. Run `uv run pytest`, then
   `uv run ty check src/relflow --output-format concise`.

### Slice 2: tensorfield integration

1. Have each tensorfield prepare its canonical content predictions, targets,
   and compatible effective trainable/valued/known mask.
2. Invoke `request.metrics` from the loss path; each metric calls
   `Model.track` itself.
3. Migrate configurable content metrics one datatype at a time.
4. Declare any built-in request defaults explicitly rather than reading the
   live trait catalogs.
5. Keep internal state accuracy and operational diagnostics on their direct
   logging paths until scoped request metrics are designed.
6. Remove Boolean's decoder-owned TorchMetric tree; leave AUC unavailable until
   a separate stateful design exists.
7. Update evaluation and datatype documentation.

## Acceptance criteria

The scaffold is complete when:

1. A frozen Pydantic metric can be registered by trait or exact datatype name.
2. `name` is a validated, formattable `str` field and serializes unchanged as
   a normal string.
3. `str(metric)` renders the concrete logging name from that template.
4. Templates may reference only fields available on their concrete metric
   configuration.
5. Every registered metric implements the one stateless `__call__` signature,
   including `trainable: torch.Tensor`, calls `Model.track` itself, and returns
   `None`.
6. `METRICS` is derived from actual Pluggy providers.
7. Dynamic configuration parsing preserves third-party subclass fields.
8. Every tensorfield request accepts `metrics=[...]`, including the exact
   `rf.Category(metrics=[rf.metrics.Accuracy(0.5)])` construction, and the
   configured concrete models survive schema/checkpoint round trips.
9. Request validation accepts trait or exact-datatype eligibility, preserves
   declaration order, and rejects duplicate rendered content names.
10. Multiple configurations of one metric render distinct logging names and
   coexist without collision.
11. `Trait.classification.metrics` and the other trait properties return live,
   sorted `list[Metric]` snapshots of their registered defaults.
12. The metrics package contains no TorchMetric factories, runtime modules,
   `ModuleDict`s, or `MetricContribution` abstraction.

## Deferred stateful design

A later, separate proposal may add:

- TorchMetric factories;
- exact dataset-level AUC and global RMSE;
- per-stage module ownership;
- distributed state synchronization;
- empty-rank behavior;
- reset and checkpoint lifecycle; and
- metrics requiring more than one stateless prediction/target invocation.

None of those concerns are part of this scaffold.

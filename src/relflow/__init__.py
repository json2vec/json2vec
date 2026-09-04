"""Public `relflow` SDK surface.

The top-level package exports the constructors and helpers used by most
applications: `Model(...)` for model construction, tensorfield
request constructors such as `Category` and `Number`, data modules, schema
mutation predicates, and the `@preprocess` and `@postprocess` decorators.
"""

from typing import TYPE_CHECKING, Any

from relflow._version import __version__
from relflow.architecture.checkpoint import RollbackCheckpoint
from relflow.architecture.mutations import MutationLockCallback, RuntimePlacementCallback
from relflow.architecture.root import (
    Model,
    OptimizerConfig,
    SchedulerConfig,
)
from relflow.data.arrow import Batch
from relflow.data.datasets import ArrowDataModule, CustomDataModule, PolarsDataModule, SyntheticDataModule
from relflow.data.processors import (
    Postprocessor,
    Preprocessor,
    PreprocessorProvider,
    postprocess,
    preprocess,
)
from relflow.data.ragged import RaggedField
from relflow.helpers import Jitter
from relflow.inference.callback import Writer
from relflow.structs.enums import (
    AttentionMode,
    Component,
    Metric,
    Overflow,
    Strata,
    TensorKey,
    Tokens,
)
from relflow.structs.experiment import (
    NodeAttribute,
    NodePredicate,
    Schema,
    SchemaField,
    predicate,
    where,
)
from relflow.structs.structure import Branch, Mask
from relflow.structs.tree import Address, Leaf
from relflow.tensorfields import (
    TENSORFIELDS,
    Context,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
    TensorInput,
)
from relflow.tensorfields.extensions.boolean import Request as Boolean
from relflow.tensorfields.extensions.category import Request as Category
from relflow.tensorfields.extensions.cluster import Request as Cluster
from relflow.tensorfields.extensions.dateparts import Request as DateParts
from relflow.tensorfields.extensions.hashable import Request as Hash
from relflow.tensorfields.extensions.number import Request as Number
from relflow.tensorfields.extensions.set import Request as Set
from relflow.tensorfields.extensions.text import Request as Text
from relflow.tensorfields.extensions.vector import Request as Vector
from relflow.tensorfields.shared.vocabulary import VocabularySyncCallback

if TYPE_CHECKING:
    from relflow.inference.deployment import (
        Accelerator,
        Deployment,
        Input,
        JSONBackend,
        ModelSource,
        UpdateOperation,
    )

_SERVING_EXPORTS = {
    "Accelerator",
    "Deployment",
    "Input",
    "JSONBackend",
    "ModelSource",
    "UpdateOperation",
}


def __getattr__(name: str) -> Any:
    if name not in _SERVING_EXPORTS:
        raise AttributeError(f"module 'relflow' has no attribute {name!r}")

    try:
        from relflow.inference import deployment
    except ModuleNotFoundError as error:
        if error.name in {"fastapi", "orjson", "pydantic_settings", "uvicorn"}:
            raise ModuleNotFoundError(
                f"relflow.{name} requires the serving extra; install with `pip install relflow[serving]`."
            ) from error
        raise

    value = getattr(deployment, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_SERVING_EXPORTS])


__all__ = [
    "__version__",
    "Address",
    "Accelerator",
    "Branch",
    "Boolean",
    "AttentionMode",
    "ArrowDataModule",
    "Batch",
    "Cluster",
    "Category",
    "Component",
    "Context",
    "CustomDataModule",
    "DateParts",
    "DecoderBase",
    "Deployment",
    "EmbedderBase",
    "Hash",
    "Schema",
    "Input",
    "JSONBackend",
    "Jitter",
    "Leaf",
    "Metric",
    "Mask",
    "Model",
    "ModelSource",
    "MutationLockCallback",
    "NodeAttribute",
    "NodePredicate",
    "Number",
    "OptimizerConfig",
    "Overflow",
    "Extension",
    "PolarsDataModule",
    "Postprocessor",
    "Preprocessor",
    "PreprocessorProvider",
    "RaggedField",
    "RequestBase",
    "RollbackCheckpoint",
    "RuntimePlacementCallback",
    "Set",
    "SchedulerConfig",
    "SchemaField",
    "SyntheticDataModule",
    "Strata",
    "TENSORFIELDS",
    "TensorFieldBase",
    "TensorInput",
    "TensorKey",
    "Text",
    "Tokens",
    "UpdateOperation",
    "Vector",
    "VocabularySyncCallback",
    "Writer",
    "predicate",
    "postprocess",
    "preprocess",
    "where",
]

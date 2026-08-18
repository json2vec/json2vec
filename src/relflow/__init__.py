"""Public `relflow` SDK surface.

The top-level package exports the constructors and helpers used by most
applications: `Model(...)` for model construction, tensorfield
request constructors such as `Category` and `Number`, data modules, schema
mutation predicates, and the `@preprocess` decorator.
"""

from typing import TYPE_CHECKING, Any

from relflow import helpers as helpers
from relflow import rich as rich
from relflow.architecture.checkpoint import RollbackCheckpoint
from relflow.architecture.mutations import MutationLockCallback, RuntimePlacementCallback
from relflow.architecture.root import (
    Model,
    OptimizerConfig,
    SchedulerConfig,
)
from relflow.data.datasets import CustomDataModule, PolarsDataModule, StreamingDataModule, SyntheticDataModule
from relflow.data.nested import MASK_LITERAL, MaskLiteral
from relflow.data.processors import (
    Metadata,
    Observation,
    Postprocessor,
    PostprocessorProvider,
    PostprocessorResult,
    Predictions,
    Preprocessor,
    PreprocessorProvider,
    RawBatch,
    RawObservation,
    postprocess,
    preprocess,
)
from relflow.inference.callback import Writer
from relflow.structs.enums import (
    AttentionMode,
    Component,
    Metric,
    Overflow,
    ShardingStrategy,
    Strata,
    Suffix,
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
from relflow.tensorfields import TENSORFIELDS, DecoderBase, EmbedderBase, Plugin, RequestBase, TensorFieldBase
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
    "Address",
    "Accelerator",
    "Branch",
    "Boolean",
    "AttentionMode",
    "Cluster",
    "Category",
    "Component",
    "CustomDataModule",
    "DateParts",
    "DecoderBase",
    "Deployment",
    "EmbedderBase",
    "Hash",
    "helpers",
    "Schema",
    "Input",
    "JSONBackend",
    "Leaf",
    "Metric",
    "MASK_LITERAL",
    "Mask",
    "MaskLiteral",
    "Metadata",
    "Model",
    "ModelSource",
    "MutationLockCallback",
    "NodeAttribute",
    "NodePredicate",
    "Number",
    "Observation",
    "OptimizerConfig",
    "Overflow",
    "Plugin",
    "PolarsDataModule",
    "Postprocessor",
    "PostprocessorProvider",
    "PostprocessorResult",
    "Predictions",
    "Preprocessor",
    "PreprocessorProvider",
    "RawBatch",
    "RawObservation",
    "RequestBase",
    "RollbackCheckpoint",
    "RuntimePlacementCallback",
    "Set",
    "SchedulerConfig",
    "SchemaField",
    "ShardingStrategy",
    "StreamingDataModule",
    "SyntheticDataModule",
    "Strata",
    "Suffix",
    "TENSORFIELDS",
    "TensorFieldBase",
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

rich.install_tracebacks()

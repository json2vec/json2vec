"""Public `json2vec` SDK surface.

The top-level package exports the constructors and helpers used by most
applications: `Model(...)` for model construction, tensorfield
request constructors such as `Category` and `Number`, data modules, schema
mutation predicates, and the `@preprocess` decorator.
"""

from typing import TYPE_CHECKING, Any

from json2vec import helpers as helpers
from json2vec.architecture.checkpoint import RollbackCheckpoint
from json2vec.architecture.mutations import MutationLockCallback, RuntimePlacementCallback
from json2vec.architecture.root import (
    Model,
    OptimizerConfig,
    SchedulerConfig,
)
from json2vec.data.datasets import CustomDataModule, PolarsDataModule, StreamingDataModule
from json2vec.data.nested import MASK_LITERAL, MaskLiteral
from json2vec.data.processors import (
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
from json2vec.inference.callback import Writer
from json2vec.structs.enums import (
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
from json2vec.structs.experiment import (
    NodeAttribute,
    NodePredicate,
    Schema,
    SchemaField,
    predicate,
    where,
)
from json2vec.structs.structure import Branch, Mask
from json2vec.structs.tree import Address, Leaf
from json2vec.tensorfields import TENSORFIELDS, DecoderBase, EmbedderBase, Plugin, RequestBase, TensorFieldBase
from json2vec.tensorfields.extensions.boolean import Request as Boolean
from json2vec.tensorfields.extensions.category import Request as Category
from json2vec.tensorfields.extensions.dateparts import Request as DateParts
from json2vec.tensorfields.extensions.entity import Request as Entity
from json2vec.tensorfields.extensions.number import Request as Number
from json2vec.tensorfields.extensions.set import Request as Set
from json2vec.tensorfields.extensions.staticEntity import Request as StaticEntity
from json2vec.tensorfields.extensions.text import Request as Text
from json2vec.tensorfields.extensions.vector import Request as Vector
from json2vec.tensorfields.shared.vocabulary import VocabularySyncCallback

if TYPE_CHECKING:
    from json2vec.inference.deployment import (
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
        raise AttributeError(f"module 'json2vec' has no attribute {name!r}")

    try:
        from json2vec.inference import deployment
    except ModuleNotFoundError as error:
        if error.name in {"fastapi", "orjson", "pydantic_settings", "uvicorn"}:
            raise ModuleNotFoundError(
                f"json2vec.{name} requires the serving extra; install with `pip install json2vec[serving]`."
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
    "Category",
    "Component",
    "CustomDataModule",
    "DateParts",
    "DecoderBase",
    "Deployment",
    "EmbedderBase",
    "Entity",
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
    "StaticEntity",
    "StreamingDataModule",
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

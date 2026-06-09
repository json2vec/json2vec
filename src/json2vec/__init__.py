"""Public `json2vec` SDK surface.

The top-level package exports the constructors and helpers used by most
applications: `Model.from_schema(...)` for model construction, tensorfield
request constructors such as `Category` and `Number`, data modules, schema
mutation predicates, and the `@preprocess` decorator.
"""

from typing import TYPE_CHECKING, Any

from json2vec.architecture.checkpoint import RollbackCheckpoint
from json2vec.architecture.mutations import MutationLockCallback, RuntimePlacementCallback
from json2vec.architecture.root import (
    Model,
    OptimizerConfig,
    SchedulerConfig,
)
from json2vec.data.datasets import CustomDataModule, PolarsDataModule, StreamingDataModule
from json2vec.data.processing import MASK_LITERAL, MaskLiteral
from json2vec.inference.callback import Postprocessor, Writer
from json2vec.preprocessors import PREPROCESSORS, Preprocessor, PreprocessorMode, preprocess
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
    Hyperparameters,
    NodeAttribute,
    NodePredicate,
    SchemaField,
    predicate,
    where,
)
from json2vec.structs.structure import Array, Mask
from json2vec.structs.tree import Address, Leaf
from json2vec.tensorfields import TENSORFIELDS, DecoderBase, EmbedderBase, Plugin, RequestBase, TensorFieldBase
from json2vec.tensorfields.extensions.category import Request as Category
from json2vec.tensorfields.extensions.dateparts import Request as DateParts
from json2vec.tensorfields.extensions.entity import Request as Entity
from json2vec.tensorfields.extensions.number import Request as Number
from json2vec.tensorfields.extensions.set import Request as Set
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
    "Array",
    "AttentionMode",
    "Category",
    "Component",
    "CustomDataModule",
    "DateParts",
    "DecoderBase",
    "Deployment",
    "EmbedderBase",
    "Entity",
    "Hyperparameters",
    "Input",
    "JSONBackend",
    "Leaf",
    "Metric",
    "MASK_LITERAL",
    "Mask",
    "MaskLiteral",
    "Model",
    "ModelSource",
    "MutationLockCallback",
    "NodeAttribute",
    "NodePredicate",
    "Number",
    "OptimizerConfig",
    "Overflow",
    "PREPROCESSORS",
    "Plugin",
    "PolarsDataModule",
    "Postprocessor",
    "Preprocessor",
    "PreprocessorMode",
    "RequestBase",
    "RollbackCheckpoint",
    "RuntimePlacementCallback",
    "Set",
    "SchedulerConfig",
    "SchemaField",
    "ShardingStrategy",
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
    "preprocess",
    "where",
]

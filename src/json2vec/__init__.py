"""Public JSON2Vec SDK surface.

The top-level package exports the constructors and helpers used by most
applications: `Model.from_schema(...)` for model construction, tensorfield
request constructors such as `Category` and `Number`, data modules, schema
mutation predicates, and the `@preprocess` decorator.
"""

from json2vec.architecture.root import (
    Model,
    MutationLockCallback,
    OptimizerConfig,
    RuntimePlacementCallback,
    SchedulerConfig,
)
from json2vec.data.datasets import Dataset, PolarsDataModule, StreamingDataModule
from json2vec.preprocessors import PREPROCESSORS, Preprocessor, PreprocessorMode, preprocess
from json2vec.structs.enums import Component, Metric, ShardingStrategy, Strata, Suffix, TensorKey, Tokens
from json2vec.structs.experiment import (
    Hyperparameters,
    NodeAttribute,
    NodePredicate,
    SchemaField,
    predicate,
    where,
)
from json2vec.structs.structure import Array
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

__all__ = [
    "Address",
    "Array",
    "Category",
    "Component",
    "DateParts",
    "Dataset",
    "DecoderBase",
    "EmbedderBase",
    "Entity",
    "Hyperparameters",
    "Leaf",
    "Metric",
    "Model",
    "MutationLockCallback",
    "NodeAttribute",
    "NodePredicate",
    "Number",
    "OptimizerConfig",
    "PREPROCESSORS",
    "Plugin",
    "PolarsDataModule",
    "Preprocessor",
    "PreprocessorMode",
    "RequestBase",
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
    "Vector",
    "VocabularySyncCallback",
    "predicate",
    "preprocess",
    "where",
]

from __future__ import annotations

import functools
import math
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Type, TypeAlias

import pluggy
import torch
from tensordict import TensorDict

from json2vec.architecture.pool import LearnedQueryCrossAttention
from json2vec.structs.enums import Component, Strata, TensorKey
from json2vec.structs.packages import Parcel, Prediction
from json2vec.structs.tree import Address, Leaf, Node
from json2vec.tensorfields.spec import PluginSpec

if TYPE_CHECKING:
    from json2vec.architecture.plot import Pane
    from json2vec.architecture.root import JSON2Vec
    from json2vec.structs.experiment import Session
    from json2vec.structs.structure import Structure

pm: pluggy.PluginManager = pluggy.PluginManager(project_name="tensorfields")

pm.add_hookspecs(module_or_class=PluginSpec)

RequestBase: TypeAlias = Leaf


def default_plot(
    module: "JSON2Vec",
    address: Address,
    branch: "Pane",
    detail: bool,
) -> None:
    pass


class EmbedderBase(torch.nn.Module):
    def __init__(self, structure: Structure, address: Address):
        super().__init__()


class DecoderBase(torch.nn.Module):
    def __init__(self, structure: Structure, address: Address):
        super().__init__()

        self.address: Address = address
        self.sigma: torch.Tensor = torch.nn.Parameter(torch.zeros(1))

        self.pool = LearnedQueryCrossAttention(
            n_context=math.prod(structure.shapes[address]),
            d_model=structure.d_model,
            nhead=structure.requests[address].n_heads,
            dropout=structure.dropout,
            n_linear=structure.requests[address].n_linear,
        )

    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        raise NotImplementedError("decoder must implement decode(pooled)")

    def forward(self, parcels: list[Parcel]) -> Prediction:
        if len(parcels) == 0:
            raise ValueError("decoder requires at least one parcel")

        N, *_, C = parcels[0].payload.shape
        stacked = torch.cat([parcel.payload.reshape(N, -1, C) for parcel in parcels], dim=1)
        pooled = self.pool(stacked)

        payload = self.decode(pooled)
        return Prediction(
            payload=payload,
            address=self.address,
            batch_size=pooled.shape[0],
        )


class TensorFieldBase(ABC):
    content: torch.Tensor
    state: torch.Tensor
    trainable: torch.Tensor
    targets: TensorDict[TensorKey, torch.Tensor]

    @classmethod
    @abstractmethod
    def new(
        cls,
        values: list,
        address: Address,
        session: Session,
        strata: Strata,
        state: Any,
    ) -> "TensorFieldBase":
        raise NotImplementedError

    @abstractmethod
    def mask(self, p_mask: float):
        raise NotImplementedError

    @abstractmethod
    def prune(cls, p_prune: float):
        raise NotImplementedError


TENSORFIELDS: dict[str, "Plugin"] = {}


class Plugin:
    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError("Plugin name must be a string")

        # should start with a letter and contain only lowercase letters, numbers, and underscores
        if not re.match(r"^[a-z0-9_]+$", name):
            raise ValueError("Plugin name must consist of lowercase letters, numbers, and underscores only")

        self.name: str = name
        self.components: dict[Component, Callable | Type] = {}

        if name in TENSORFIELDS:
            raise ValueError(f"Plugin '{name}' already registered")

        TENSORFIELDS[name] = self

    def register(self, obj: Type | Callable) -> Type | Callable:
        if not hasattr(obj, "__name__"):
            raise NameError(f"Object {obj} does not have a name")

        name: str = str(obj.__name__)

        if name in self.components:
            raise ValueError(f"Component '{name}' already registered in plugin '{self.name}'")

        match name:
            case Component.Request:
                if not isinstance(obj, type):
                    raise TypeError("Request must be a class type")

                if not issubclass(obj, Node):
                    raise TypeError("Request must be a subclass of Node")

                # for attr in Leaf.__annotations__.keys():
                #     if not hasattr(obj, attr):
                #         raise AttributeError(f"Request class must have a '{attr}' attribute")

                # if getattr(obj, "type") != self.name:
                #     raise ValueError(
                #         f"Request class 'type' attribute must be '{self.name}', got '{getattr(obj, 'type')}'"
                #     )

            case Component.TensorField:
                if not isinstance(obj, type):
                    raise TypeError("TensorField must be a class type")

                if not issubclass(obj, TensorFieldBase):
                    raise TypeError("TensorField must be a subclass of TensorFieldBase")

            case Component.Embedder:
                if not isinstance(obj, type):
                    raise TypeError("Embedder must be a class type")

                if not issubclass(obj, EmbedderBase):
                    raise TypeError("Embedder must be a subclass of EmbedderBase")

                # confirm the init method is expecting structure and address
                init_params = list(obj.__init__.__annotations__.keys())
                if "structure" not in init_params or "address" not in init_params:
                    raise TypeError("Embedder __init__ method must accept 'structure' and 'address' parameters")

            case Component.Decoder:
                if not isinstance(obj, type):
                    raise TypeError("Decoder must be a class type")

                if not issubclass(obj, DecoderBase):
                    raise TypeError("Decoder must be a subclass of DecoderBase")

                init_params = list(obj.__init__.__annotations__.keys())
                if "structure" not in init_params or "address" not in init_params:
                    raise TypeError("Decoder __init__ method must accept 'structure' and 'address' parameters")

            case Component.loss:
                if not callable(obj):
                    raise TypeError("Loss must be a callable function")

                expected_params: list[str] = ["module", "prediction", "batch", "strata"]
                func_params: list[str] = list(obj.__annotations__.keys())

                if not set(expected_params).issubset(set(func_params)):
                    raise TypeError(
                        f"Loss function must accept the following parameters: {expected_params}, got {func_params}"
                    )

            case Component.write:
                if not callable(obj):
                    raise TypeError("Write must be a callable function")

                # check the signature of the function
                expected_params: list[str] = ["module", "prediction"]
                func_params: list[str] = list(obj.__annotations__.keys())

                if func_params != expected_params:
                    raise TypeError(
                        f"Write function must accept the following parameters: {expected_params}, got {func_params}"
                    )

            case Component.plot:
                if not callable(obj):
                    raise TypeError("Plot must be a callable function")

                expected_params: list[str] = ["module", "address", "branch", "detail"]
                func_params: list[str] = list(obj.__annotations__.keys())

                if func_params != expected_params:
                    raise TypeError(
                        f"Plot function must accept the following parameters: {expected_params}, got {func_params}"
                    )

        self.components[name] = obj

        return obj

    @functools.cache
    def __getattr__(self, key: Component) -> Callable | Type:
        try:
            component = Component(key)
        except ValueError:
            raise ValueError(f"Component '{key}' is not a valid Component enum value")

        if component in self.components:
            return self.components[component]

        if component == Component.plot:
            return default_plot

        raise AttributeError(f"Plugin '{self.name}' has no component '{component}'")

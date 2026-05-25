"""Tensorfield plugin base classes and registry."""

from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Type, TypeAlias

import pluggy
import torch
from lightning.pytorch import Callback
from tensordict import TensorDict

from json2vec.architecture.pool import LearnedQueryCrossAttention, MeanPool
from json2vec.structs.enums import Component, Strata, TensorKey
from json2vec.structs.packages import Parcel, Prediction
from json2vec.structs.tree import Address, Leaf, Node
from json2vec.tensorfields.spec import PluginSpec

if TYPE_CHECKING:
    from json2vec.architecture.plot import Pane
    from json2vec.architecture.root import Model
    from json2vec.structs.experiment import Hyperparameters

pm: pluggy.PluginManager = pluggy.PluginManager(project_name="tensorfields")

pm.add_hookspecs(module_or_class=PluginSpec)

RequestBase: TypeAlias = Leaf
CallbackFactory: TypeAlias = type[Callback] | Callable[[], Callback]


def default_plot(
    module: "Model",
    address: Address,
    branch: "Pane",
    detail: bool,
) -> None:
    return None


def default_write(module: "Model", prediction: Prediction) -> None:
    return None


class EmbedderBase(torch.nn.Module):
    """Base class for tensorfield embedders."""

    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__()


class DecoderBase(torch.nn.Module):
    """Base class for tensorfield decoders."""

    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__()

        self.address: Address = address
        self.sigma: torch.Tensor = torch.nn.Parameter(torch.zeros(1))

        request = hyperparameters.requests[address]
        n_context = 1
        for dimension in hyperparameters.shapes[address]:
            n_context *= dimension
        match request.pooling:
            case "query":
                self.pool = LearnedQueryCrossAttention(
                    n_context=n_context,
                    d_model=hyperparameters.d_model,
                    nhead=request.n_heads,
                    dropout=float(request.dropout or 0.0),
                    n_linear=request.n_linear,
                )
            case "mean":
                self.pool = MeanPool(n_context=n_context)
            case _:
                raise ValueError(f"unsupported decoder pooling: {request.pooling}")

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
    """Tensorized field values plus trainable target state."""

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
        hyperparameters: Hyperparameters,
        strata: Strata,
    ) -> "TensorFieldBase":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def empty(
        cls,
        batch_size: int,
        address: Address,
        hyperparameters: Hyperparameters,
    ) -> "TensorFieldBase":
        raise NotImplementedError

    @abstractmethod
    def mask(self, p_mask: float):
        raise NotImplementedError

    @abstractmethod
    def target(self, p_prune: float):
        raise NotImplementedError


TENSORFIELDS: dict[str, "Plugin"] = {}


class Plugin:
    """Registry object for a tensorfield implementation.

    Register request, tensorfield, embedder, decoder, loss, write, and plot
    components with `@plugin.register`. Creating a plugin with an existing
    name replaces the registry entry and emits a warning.
    """

    def __init__(self, name: str):
        if not isinstance(name, str):
            raise TypeError("Plugin name must be a string")

        # should start with a letter and contain only lowercase letters, numbers, and underscores
        if not re.match(r"^[a-z0-9_]+$", name):
            raise ValueError("Plugin name must consist of lowercase letters, numbers, and underscores only")

        self.name: str = name
        self.components: dict[Component, Callable | Type | None] = {}
        self.callback_factories: list[CallbackFactory] = []

        if name in TENSORFIELDS:
            warnings.warn(
                f"Plugin '{name}' already registered; overriding existing tensorfield plugin",
                UserWarning,
                stacklevel=2,
            )

        TENSORFIELDS[name] = self

    def register(self, obj: Type | Callable | None, component: Component | str | None = None) -> Type | Callable | None:
        """Register one tensorfield component with this plugin."""
        if obj is None:
            if component is None:
                raise TypeError("component must be provided when registering None")

            key = Component(component)
            if key not in {Component.write, Component.plot}:
                raise TypeError("only write and plot may be registered as None")

            if key in self.components:
                raise ValueError(f"Component '{key}' already registered in plugin '{self.name}'")

            self.components[key] = None
            return None

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

                # confirm the init method is expecting hyperparameters and address
                init_params = list(obj.__init__.__annotations__.keys())
                if "hyperparameters" not in init_params or "address" not in init_params:
                    raise TypeError("Embedder __init__ method must accept 'hyperparameters' and 'address' parameters")

            case Component.Decoder:
                if not isinstance(obj, type):
                    raise TypeError("Decoder must be a class type")

                if not issubclass(obj, DecoderBase):
                    raise TypeError("Decoder must be a subclass of DecoderBase")

                init_params = list(obj.__init__.__annotations__.keys())
                if "hyperparameters" not in init_params or "address" not in init_params:
                    raise TypeError("Decoder __init__ method must accept 'hyperparameters' and 'address' parameters")

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
                if obj is not None and not callable(obj):
                    raise TypeError("Write must be a callable function")

                # check the signature of the function
                expected_params: list[str] = ["module", "prediction"]
                func_params: list[str] = list(obj.__annotations__.keys())

                if func_params != expected_params:
                    raise TypeError(
                        f"Write function must accept the following parameters: {expected_params}, got {func_params}"
                    )

            case Component.plot:
                if obj is not None and not callable(obj):
                    raise TypeError("Plot must be a callable function")

                expected_params: list[str] = ["module", "address", "branch", "detail"]
                func_params: list[str] = list(obj.__annotations__.keys())

                if func_params != expected_params:
                    raise TypeError(
                        f"Plot function must accept the following parameters: {expected_params}, got {func_params}"
                    )

        self.components[name] = obj

        return obj

    def callback(self, factory: CallbackFactory) -> CallbackFactory:
        """Register a Lightning callback factory for this tensorfield."""
        callback = factory()
        if not isinstance(callback, Callback):
            raise TypeError(f"Plugin callback factory for '{self.name}' must produce a Lightning Callback")

        self.callback_factories.append(factory)
        return factory

    def callbacks(self) -> list[Callback]:
        """Instantiate all registered callback factories."""
        return [factory() for factory in self.callback_factories]

    def __getattr__(self, key: Component) -> Callable | Type:
        try:
            component = Component(key)
        except ValueError:
            raise ValueError(f"Component '{key}' is not a valid Component enum value")

        if component in self.components:
            value = self.components[component]
            if value is not None:
                return value

        if component == Component.plot:
            return default_plot

        if component == Component.write:
            return default_write

        raise AttributeError(f"Plugin '{self.name}' has no component '{component}'")

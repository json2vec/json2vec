"""Tensorfield plugin base classes and registry."""

from __future__ import annotations

import re
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, TypeAlias, TypeVar, cast, overload

import pluggy
import torch
from lightning.pytorch import Callback
from rich.text import Text
from tensordict import TensorDict

from json2vec.architecture.pool import LearnedQueryCrossAttention, MeanPool
from json2vec.structs.enums import Component, Strata, TensorKey, Tokens
from json2vec.structs.packages import Parcel, Prediction
from json2vec.structs.tree import Address, Leaf, Node, Renderable
from json2vec.tensorfields.spec import PluginSpec

if TYPE_CHECKING:
    from json2vec.architecture.root import Model
    from json2vec.structs.experiment import Hyperparameters
    from json2vec.structs.structure import Mask

pm: pluggy.PluginManager = pluggy.PluginManager(project_name="tensorfields")

pm.add_hookspecs(module_or_class=PluginSpec)

RequestBase: TypeAlias = Leaf
CallbackFactory: TypeAlias = type[Callback] | Callable[[], Callback]
ComponentValue: TypeAlias = Callable[..., Any] | type[Any]
RegisterT = TypeVar("RegisterT", bound=ComponentValue)
ArrayMaskApplication: TypeAlias = tuple[Address, "Mask"]


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

    def forward(self, parcels: list[Parcel], *, embed: bool = False) -> Prediction:
        if len(parcels) == 0:
            raise ValueError("decoder requires at least one parcel")

        N, *_, C = parcels[0].payload.shape
        stacked = torch.cat([parcel.payload.reshape(N, -1, C) for parcel in parcels], dim=1)
        pooled = self.pool(stacked)

        payload = self.decode(pooled)
        if embed:
            payload[TensorKey.embedding] = pooled

        return Prediction(
            payload=payload,
            address=self.address,
            batch_size=pooled.shape[0],
        )


class TensorFieldBase(Renderable):
    """Tensorized field values plus trainable target state."""

    STATE_PREVIEW_LIMIT: int = 80
    STATE_LABELS: dict[int, str] = {
        Tokens.valued.value: "V",
        Tokens.null.value: "N",
        Tokens.padded.value: "P",
        Tokens.masked.value: "M",
        Tokens.other.value: "O",
    }
    STATE_STYLES: dict[int, str] = {
        Tokens.valued.value: "bold green",
        Tokens.null.value: "bold yellow",
        Tokens.padded.value: "dim",
        Tokens.masked.value: "bold magenta",
        Tokens.other.value: "bold cyan",
    }

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
    def mask(self, p_mask: float = 0.0, **kwargs: Any):
        raise NotImplementedError

    @abstractmethod
    def target(self, p_prune: float = 1.0):
        raise NotImplementedError

    def hide(self, selected: torch.Tensor, *, cache_targets: bool = True, trainable: bool = True) -> None:
        raise NotImplementedError

    def __rich_console__(self, console, options):
        state = getattr(self, TensorKey.state, None)
        trainable = getattr(self, TensorKey.trainable, None)
        targets = getattr(self, TensorKey.targets, None)

        heading = Text()
        heading.append(type(self).__name__, style=self.RICH_NAME_STYLE)
        heading.append(" ")
        heading.append("[tensorfield]", style=self.RICH_TYPE_STYLE)

        if torch.is_tensor(state):
            heading.append(" ")
            heading.append("state=", style="dim")
            heading.append(str(tuple(state.shape)), style="cyan")
            heading.append(" ")
            heading.append("device=", style="dim")
            heading.append(str(state.device), style="cyan")

        if torch.is_tensor(trainable):
            heading.append(" ")
            heading.append("trainable=", style="dim")
            heading.append(str(int(trainable.sum().item())), style="cyan")

        yield heading

        if torch.is_tensor(state):
            yield self._state_counts_text(state)
            yield self._state_preview_text(state)

        if isinstance(targets, TensorDict) and targets.keys():
            text = Text(" targets=", style="dim")
            text.append(", ".join(str(key) for key in sorted(targets.keys(), key=str)), style="cyan")
            yield text

    def _state_counts_text(self, state: torch.Tensor) -> Text:
        values = state.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
        text = Text(" counts ", style="dim")
        for token in Tokens:
            count = int(values.eq(token.value).sum().item())
            text.append(self.STATE_LABELS[token.value], style=self.STATE_STYLES[token.value])
            text.append(f"={count} ", style="dim")

        return text

    def _state_preview_text(self, state: torch.Tensor) -> Text:
        preview = self._first_root_slice(state)
        text = Text(" state ", style="dim")

        if preview.ndim <= 1:
            rows = preview.reshape(1, -1)
            row_prefix = " "
        else:
            rows = preview.reshape(-1, preview.shape[-1])
            row_prefix = "\n       "

        limit = min(int(preview.numel()), self.STATE_PREVIEW_LIMIT)
        count = 0

        for row_index, row in enumerate(rows):
            if count >= limit:
                break
            if row_index:
                text.append(row_prefix, style="dim")

            for column_index, value in enumerate(row.tolist()):
                if count >= limit:
                    break
                if column_index:
                    text.append(" ")

                token = int(value)
                text.append(self.STATE_LABELS.get(token, str(token)), style=self.STATE_STYLES.get(token, "bold red"))
                count += 1

        if int(preview.numel()) > limit:
            text.append(" ...", style="dim")

        return text

    def _first_root_slice(self, tensor: torch.Tensor) -> torch.Tensor:
        values = tensor.detach().to(device="cpu")
        if values.ndim > 0:
            values = values[0]
        if values.ndim > 0:
            values = values[0]

        return values.to(dtype=torch.int64)


def _broadcast_to_state(selected: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    while selected.ndim < state.ndim:
        selected = selected.unsqueeze(-1)

    return selected.expand_as(state)


def _array_mask_candidates(
    state: torch.Tensor,
    *,
    address: Address,
    array_address: Address,
    mask: Mask,
    hyperparameters: Hyperparameters,
) -> torch.Tensor:
    occupied = state.ne(torch.as_tensor(Tokens.padded.value, device=state.device, dtype=state.dtype))
    request = hyperparameters.requests[address]
    array_nodes = [node for node in request.path if getattr(node, "type", None) == "array"]
    array_index = next(
        index for index, array in enumerate(array_nodes) if Address(str(array.address)) == Address(str(array_address))
    )
    array_dim = array_index + 1

    occupied_at_array = occupied
    while occupied_at_array.ndim > array_dim + 1:
        occupied_at_array = occupied_at_array.any(dim=-1)

    slot_count = occupied_at_array.shape[-1]
    positions = torch.arange(slot_count, device=state.device).reshape(
        *((1,) * (occupied_at_array.ndim - 1)),
        slot_count,
    )
    lengths = occupied_at_array.sum(dim=-1, keepdim=True)

    if mask.array:
        extent = torch.full_like(lengths, slot_count)
    else:
        extent = lengths

    offset = torch.as_tensor(mask.offset, device=state.device, dtype=positions.dtype)
    if mask.window is None:
        if mask.start:
            lower = torch.full_like(extent, mask.offset)
            upper = extent
        else:
            lower = torch.zeros_like(extent)
            upper = extent - offset
    elif mask.start:
        lower = torch.full_like(extent, mask.offset)
        upper = lower + mask.window
    else:
        upper = extent - offset
        lower = upper - mask.window

    lower = lower.clamp(min=0, max=slot_count)
    upper = upper.clamp(min=0, max=slot_count)
    candidates = (positions >= lower) & (positions < upper) & occupied_at_array

    if mask.rate is not None:
        selected_at_array = torch.rand(candidates.shape, device=state.device, dtype=torch.float).lt(mask.rate)
        selected_at_array &= candidates
    else:
        selected_at_array = torch.zeros_like(candidates, dtype=torch.bool)
        count = int(mask.count or 0)
        if count > 0 and candidates.any():
            flattened = candidates.reshape(-1, slot_count)
            selected_flat = selected_at_array.reshape(-1, slot_count)
            for row_index, row in enumerate(flattened):
                candidate_indexes = torch.nonzero(row, as_tuple=False).reshape(-1)
                if candidate_indexes.numel() == 0:
                    continue

                chosen_count = min(count, int(candidate_indexes.numel()))
                permutation = torch.randperm(candidate_indexes.numel(), device=state.device)[:chosen_count]
                selected_flat[row_index, candidate_indexes[permutation]] = True

    selected = _broadcast_to_state(selected_at_array, state)
    return selected & state.ne(torch.as_tensor(Tokens.padded.value, device=state.device, dtype=state.dtype))


def _prune_selection(state: torch.Tensor, p_prune: float) -> torch.Tensor:
    return torch.rand(state.size(0), *([1] * (len(state.shape) - 1)), device=state.device).lt(p_prune).expand_as(state)


def apply_mask_policies(
    tensorfield: TensorFieldBase,
    p_mask: float = 0.0,
    *,
    p_prune: float = 0.0,
    array_masks: tuple[ArrayMaskApplication, ...] = (),
    address: Address | None = None,
    hyperparameters: Hyperparameters | None = None,
) -> None:
    """Apply array masks, random masks, and prune masks from one snapshot."""
    state = tensorfield.state.clone()

    if array_masks and (address is None or hyperparameters is None):
        raise ValueError("array masks require address and hyperparameters")

    for array_address, mask in array_masks:
        selected = _array_mask_candidates(
            state,
            address=cast(Address, address),
            array_address=array_address,
            mask=mask,
            hyperparameters=hyperparameters,
        )
        if selected.any():
            tensorfield.hide(selected, cache_targets=True, trainable=True)

    if p_mask > 0.0:
        selected = torch.rand_like(input=state, dtype=torch.float).lt(other=p_mask)
        tensorfield.hide(selected, cache_targets=True, trainable=True)

    if p_prune > 0.0:
        selected = _prune_selection(state, p_prune=p_prune)
        tensorfield.hide(selected, cache_targets=True, trainable=True)


TENSORFIELDS: dict[str, "Plugin"] = {}


class Plugin:
    """Registry object for a tensorfield implementation.

    Register request, tensorfield, embedder, decoder, loss, and write
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
        self.components: dict[Component, ComponentValue | None] = {}
        self.callback_factories: list[CallbackFactory] = []

        if name in TENSORFIELDS:
            warnings.warn(
                f"Plugin '{name}' already registered; overriding existing tensorfield plugin",
                UserWarning,
                stacklevel=2,
            )

        TENSORFIELDS[name] = self

    @overload
    def register(self, obj: None, component: Component | str) -> None: ...

    @overload
    def register(self, obj: RegisterT, component: Component | str | None = None) -> RegisterT: ...

    def register(
        self,
        obj: RegisterT | None,
        component: Component | str | None = None,
    ) -> RegisterT | None:
        """Register one tensorfield component with this plugin."""
        if obj is None:
            if component is None:
                raise TypeError("component must be provided when registering None")

            key = Component(component)
            if key != Component.write:
                raise TypeError("only write may be registered as None")

            if key in self.components:
                raise ValueError(f"Component '{key}' already registered in plugin '{self.name}'")

            self.components[key] = None
            return None

        if not hasattr(obj, "__name__"):
            raise NameError(f"Object {obj} does not have a name")

        name: str = str(obj.__name__)
        try:
            key = Component(name)
        except ValueError:
            raise ValueError(f"Component '{name}' is not a valid Component enum value") from None

        if key in self.components:
            raise ValueError(f"Component '{key}' already registered in plugin '{self.name}'")

        match key:
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

        self.components[key] = obj

        return obj

    @overload
    def callback(self, factory: CallbackFactory, /) -> CallbackFactory: ...

    @overload
    def callback(self, factory: CallbackFactory, *factories: CallbackFactory) -> tuple[CallbackFactory, ...]: ...

    def callback(self, factory: CallbackFactory, *factories: CallbackFactory):
        """Register one or more Lightning callback factories for this tensorfield."""
        registered = (factory, *factories)
        for callback_factory in registered:
            callback = callback_factory()
            if not isinstance(callback, Callback):
                raise TypeError(f"Plugin callback factory for '{self.name}' must produce a Lightning Callback")

        self.callback_factories.extend(registered)
        return factory if len(registered) == 1 else registered

    @property
    def callbacks(self) -> list[Callback]:
        """Instantiate all registered callback factories."""
        return [factory() for factory in self.callback_factories]

    def _component(self, component: Component) -> ComponentValue:
        """Return a registered component, falling back to optional defaults."""
        if component in self.components:
            value = self.components[component]
            if value is not None:
                return value

        if component == Component.write:
            return default_write

        raise AttributeError(f"Plugin '{self.name}' has no component '{component}'")

    @property
    def Request(self) -> type[RequestBase]:
        return cast(type[RequestBase], self._component(Component.Request))

    @property
    def TensorField(self) -> type[TensorFieldBase]:
        return cast(type[TensorFieldBase], self._component(Component.TensorField))

    @property
    def Embedder(self) -> type[EmbedderBase]:
        return cast(type[EmbedderBase], self._component(Component.Embedder))

    @property
    def Decoder(self) -> type[DecoderBase]:
        return cast(type[DecoderBase], self._component(Component.Decoder))

    @property
    def loss(self) -> Callable[..., Any]:
        return cast(Callable[..., Any], self._component(Component.loss))

    @property
    def write(self) -> Callable[..., Any]:
        return cast(Callable[..., Any], self._component(Component.write))

    def __getattr__(self, key: str) -> ComponentValue:
        try:
            component = Component(key)
        except ValueError:
            raise ValueError(f"Component '{key}' is not a valid Component enum value") from None

        return self._component(component)

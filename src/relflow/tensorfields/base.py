"""Tensorfield plugin base classes and registry."""

from __future__ import annotations

import inspect
import re
import warnings
from abc import abstractmethod
from collections.abc import Iterator, Mapping
from datetime import date, datetime, time, timedelta
from types import MappingProxyType, UnionType
from typing import TYPE_CHECKING, Any, Callable, TypeAlias, TypeVar, cast, get_args, overload

import numpy as np
import pluggy
import pyarrow as pa
import pyarrow.compute as pc
import torch
from lightning.pytorch import Callback
from rich.text import Text
from tensordict import TensorDict

from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool
from relflow.data.arrow import variants
from relflow.structs.enums import Component, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address, Leaf, Renderable
from relflow.tensorfields.spec import PluginSpec

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.ragged import RaggedField
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Mask

pm: pluggy.PluginManager = pluggy.PluginManager(project_name="tensorfields")

pm.add_hookspecs(module_or_class=PluginSpec)

RequestBase: TypeAlias = Leaf
CallbackFactory: TypeAlias = type[Callback] | Callable[[], Callback]
ComponentValue: TypeAlias = Callable[..., Any] | type[Any]
RegisterT = TypeVar("RegisterT", bound=ComponentValue)
BranchMaskApplication: TypeAlias = tuple[Address, "Mask"]
ValueTypeFamily: TypeAlias = type[Any] | UnionType
ArrowMatcher: TypeAlias = Callable[[pa.DataType], bool]
MATCHERS: Mapping[type[Any], ArrowMatcher] = MappingProxyType(
    {
        object: lambda datatype: True,
        bool: pa.types.is_boolean,
        int: pa.types.is_integer,
        float: pa.types.is_floating,
        str: lambda datatype: pa.types.is_string(datatype) or pa.types.is_large_string(datatype),
        bytes: lambda datatype: (
            pa.types.is_binary(datatype)
            or pa.types.is_large_binary(datatype)
            or pa.types.is_fixed_size_binary(datatype)
        ),
        datetime: pa.types.is_timestamp,
        date: pa.types.is_date,
        time: pa.types.is_time,
        timedelta: pa.types.is_duration,
        dict: lambda datatype: pa.types.is_struct(datatype) or pa.types.is_map(datatype),
    }
)


def default_output(module: "Model", address: Address) -> None:
    return None


def default_write(
    module: "Model",
    prediction: Prediction,
    datatype: pa.StructType | None,
) -> None:
    return None


class EmbedderBase(torch.nn.Module):
    """Base class for tensorfield embedders."""

    def __init__(self, schema: Schema, address: Address):
        super().__init__()


class DecoderBase(torch.nn.Module):
    """Base class for tensorfield decoders."""

    def __init__(self, schema: Schema, address: Address):
        super().__init__()

        self.address: Address = address
        self.sigma: torch.Tensor = torch.nn.Parameter(torch.zeros(1))

        request = schema.requests[address]
        n_context = 1
        for dimension in schema.shapes[address]:
            n_context *= dimension
        match request.pooling:
            case "query":
                self.pool = LearnedQueryCrossAttention(
                    n_context=n_context,
                    d_model=schema.d_model,
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
        field: RaggedField,
        address: Address,
        schema: Schema,
        strata: Strata,
    ) -> "TensorFieldBase":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def empty(
        cls,
        batch_size: int,
        address: Address,
        schema: Schema,
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

    def check_nullable(self, *, address: Address, schema: Schema) -> None:
        request = schema.requests[address]
        if request.nullable:
            return

        nulls = self.state.eq(torch.as_tensor(Tokens.null.value, device=self.state.device, dtype=self.state.dtype))
        if not nulls.any():
            return

        count = int(nulls.sum().item())
        raise ValueError(f"request '{address}' has nullable=False but input contains {count} null value(s)")

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


def _branch_mask_candidates(
    state: torch.Tensor,
    *,
    address: Address,
    branch_address: Address,
    mask: Mask,
    schema: Schema,
) -> torch.Tensor:
    occupied = state.ne(torch.as_tensor(Tokens.padded.value, device=state.device, dtype=state.dtype))
    request = schema.requests[address]
    branch_nodes = [node for node in request.path if getattr(node, "type", None) == "branch"]
    branch_index = next(
        index
        for index, branch in enumerate(branch_nodes)
        if Address(str(branch.address)) == Address(str(branch_address))
    )
    branch_dim = branch_index + 1

    occupied_at_branch = occupied
    while occupied_at_branch.ndim > branch_dim + 1:
        occupied_at_branch = occupied_at_branch.any(dim=-1)

    slot_count = occupied_at_branch.shape[-1]
    positions = torch.arange(slot_count, device=state.device).reshape(
        *((1,) * (occupied_at_branch.ndim - 1)),
        slot_count,
    )
    lengths = occupied_at_branch.sum(dim=-1, keepdim=True)

    if mask.branch:
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
    candidates = (positions >= lower) & (positions < upper) & occupied_at_branch

    if mask.rate is not None:
        selected_at_branch = torch.rand(candidates.shape, device=state.device, dtype=torch.float).lt(mask.rate)
        selected_at_branch &= candidates
    else:
        selected_at_branch = torch.zeros_like(candidates, dtype=torch.bool)
        count = int(mask.count or 0)
        if count > 0 and candidates.any():
            flattened = candidates.reshape(-1, slot_count)
            selected_flat = selected_at_branch.reshape(-1, slot_count)
            for row_index, row in enumerate(flattened):
                candidate_indexes = torch.nonzero(row, as_tuple=False).reshape(-1)
                if candidate_indexes.numel() == 0:
                    continue

                chosen_count = min(count, int(candidate_indexes.numel()))
                permutation = torch.randperm(candidate_indexes.numel(), device=state.device)[:chosen_count]
                selected_flat[row_index, candidate_indexes[permutation]] = True

    selected = _broadcast_to_state(selected_at_branch, state)
    return selected & state.ne(torch.as_tensor(Tokens.padded.value, device=state.device, dtype=state.dtype))


def _prune_selection(state: torch.Tensor, p_prune: float) -> torch.Tensor:
    return torch.rand(state.size(0), *([1] * (len(state.shape) - 1)), device=state.device).lt(p_prune).expand_as(state)


def apply_mask_policies(
    tensorfield: TensorFieldBase,
    p_mask: float = 0.0,
    *,
    p_prune: float = 0.0,
    branch_masks: tuple[BranchMaskApplication, ...] = (),
    address: Address | None = None,
    schema: Schema | None = None,
) -> None:
    """Apply branch masks, random masks, and prune masks from one snapshot."""
    state = tensorfield.state.clone()

    if branch_masks and (address is None or schema is None):
        raise ValueError("branch masks require address and schema")

    for branch_address, mask in branch_masks:
        selected = _branch_mask_candidates(
            state,
            address=cast(Address, address),
            branch_address=branch_address,
            mask=mask,
            schema=schema,
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

    Register request, tensorfield, embedder, decoder, loss, output, and write
    components with `@plugin.register`. ``types`` names the Python equivalents
    of accepted canonical Arrow terminal families. Separate tuple entries are
    incompatible; types joined in one PEP 604 union may share one Arrow column.
    ``arrow`` supplies a physical-type matcher for custom Python atoms or
    overrides a standard matcher. Creating a plugin with an existing name
    replaces the registry entry and emits a warning.
    """

    def __init__(
        self,
        name: str,
        *,
        types: tuple[ValueTypeFamily, ...],
        arrow: Mapping[type[Any], ArrowMatcher] | None = None,
    ):
        if not isinstance(name, str):
            raise TypeError("Plugin name must be a string")

        # should start with a letter and contain only lowercase letters, numbers, and underscores
        if not re.match(r"^[a-z0-9_]+$", name):
            raise ValueError("Plugin name must consist of lowercase letters, numbers, and underscores only")

        if not isinstance(types, tuple):
            raise TypeError("Plugin types must be a tuple of types or PEP 604 unions")
        if not types:
            raise ValueError("Plugin types must contain at least one value type family")

        families: list[tuple[type[Any], ...]] = []
        type_to_family: dict[type[Any], int] = {}
        for family_index, family in enumerate(types):
            members = get_args(family) if isinstance(family, UnionType) else (family,)
            if not members or any(member is Any or not isinstance(member, type) for member in members):
                raise TypeError("Plugin types must contain only concrete types or PEP 604 unions of concrete types")
            if type(None) in members:
                raise TypeError("Plugin types must not include NoneType; null is represented by field state")
            if any(issubclass(member, (list, tuple, np.ndarray, np.generic, Iterator)) for member in members):
                raise TypeError(
                    "Plugin types declare canonical terminal Python atoms; "
                    "sequences, iterators, ndarrays, and NumPy scalar classes are prepared structurally"
                )

            for member in members:
                if member in type_to_family:
                    raise ValueError(
                        f"Plugin value type {member.__name__} appears in more than one compatibility family"
                    )
                type_to_family[member] = family_index
            families.append(cast(tuple[type[Any], ...], tuple(members)))

        if arrow is not None and not isinstance(arrow, Mapping):
            raise TypeError("Plugin arrow must be a mapping from declared value types to Arrow matchers")
        declared = {} if arrow is None else dict(arrow)
        unknown = [member for member in declared if member not in type_to_family]
        if unknown:
            names = ", ".join(getattr(member, "__name__", repr(member)) for member in unknown)
            raise ValueError(f"Plugin arrow matcher type(s) are absent from types: {names}")
        invalid = [member for member, matcher in declared.items() if not callable(matcher)]
        if invalid:
            names = ", ".join(member.__name__ for member in invalid)
            raise TypeError(f"Plugin Arrow matcher(s) must be callable: {names}")
        missing = [member for member in type_to_family if member not in MATCHERS and member not in declared]
        if missing:
            names = ", ".join(member.__name__ for member in missing)
            raise TypeError(f"Plugin custom value type(s) require arrow matchers: {names}")

        self.name: str = name
        self.value_types: tuple[ValueTypeFamily, ...] = types
        self.families: tuple[tuple[type[Any], ...], ...] = tuple(families)
        self.family_by_type: Mapping[type[Any], int] = MappingProxyType(type_to_family)
        self.matchers: Mapping[type[Any], ArrowMatcher] = MappingProxyType(
            {member: declared[member] if member in declared else MATCHERS[member] for member in type_to_family}
        )
        self.components: dict[Component, ComponentValue | None] = {}
        self.callback_factories: list[CallbackFactory] = []

        if name in TENSORFIELDS:
            warnings.warn(
                f"Plugin '{name}' already registered; overriding existing tensorfield plugin",
                UserWarning,
                stacklevel=2,
            )

        TENSORFIELDS[name] = self

    @property
    def types(self) -> tuple[ValueTypeFamily, ...]:
        """Registered raw atom compatibility families."""
        return self.value_types

    def terminals(self, datatype: pa.DataType) -> tuple[pa.DataType, ...]:
        """Return terminal Arrow atom types beneath structural containers."""

        if pa.types.is_dictionary(datatype):
            return self.terminals(datatype.value_type)
        if pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype):
            return self.terminals(datatype.value_type)
        if pa.types.is_union(datatype):
            return tuple(child for field in datatype for child in self.terminals(field.type))
        return (datatype,)

    def matches(self, member: type[Any], datatype: pa.DataType) -> bool:
        """Match one registered Python atom to one Arrow terminal type."""

        if pa.types.is_null(datatype):
            return True
        matcher = self.matchers[member]
        if matcher(datatype):
            return True
        if isinstance(datatype, pa.ExtensionType):
            return any(self.matches(member, terminal) for terminal in self.terminals(datatype.storage_type))
        return False

    def accepts(self, datatype: pa.DataType) -> bool:
        """Return whether one Arrow value type belongs to an allowed family."""

        families: set[int] = set()
        for terminal in self.terminals(datatype):
            if pa.types.is_null(terminal):
                continue
            matches = {
                family
                for family, members in enumerate(self.families)
                if any(self.matches(member, terminal) for member in members)
            }
            if not matches:
                return False
            families.update(matches)
        return len(families) <= 1

    def canonical(self, datatype: pa.DataType) -> pa.DataType:
        """Resolve Arrow encodings that wrap a plugin's declared atom types."""

        if pa.types.is_dictionary(datatype):
            return self.canonical(datatype.value_type)
        if pa.types.is_union(datatype):
            children = [self.canonical(field.type) for field in datatype]
            try:
                return (
                    pa.unify_schemas(
                        [pa.schema([pa.field("value", child)]) for child in children],
                        promote_options="permissive",
                    )
                    .field("value")
                    .type
                )
            except pa.ArrowException:
                return datatype
        if pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype):
            child = self.canonical(datatype.value_type)
            if child == datatype.value_type:
                return datatype
            field = pa.field(
                datatype.value_field.name,
                child,
                nullable=datatype.value_field.nullable,
                metadata=datatype.value_field.metadata,
            )
            if pa.types.is_large_list(datatype):
                return pa.large_list(field)
            if pa.types.is_fixed_size_list(datatype):
                return pa.list_(field, datatype.list_size)
            return pa.list_(field)
        if isinstance(datatype, pa.ExtensionType):
            direct = any(matcher(datatype) for matcher in self.matchers.values())
            return datatype if direct else self.canonical(datatype.storage_type)
        return datatype

    def normalize(
        self,
        values: pa.Array | pa.ChunkedArray,
        datatype: pa.DataType,
    ) -> pa.Array | pa.ChunkedArray:
        """Normalize Arrow wrappers to one codec-ready physical representation."""

        if values.type == datatype:
            return values
        if isinstance(values, pa.ChunkedArray):
            return pa.chunked_array(
                [self.normalize(chunk, datatype) for chunk in values.chunks],
                type=datatype,
            )
        if pa.types.is_dictionary(values.type):
            decoded = pc.take(values.dictionary, values.indices)
            return self.normalize(decoded, datatype)
        if isinstance(values, pa.ExtensionArray) and not isinstance(datatype, pa.ExtensionType):
            return self.normalize(values.storage, datatype)
        if pa.types.is_union(values.type) and not pa.types.is_union(datatype):
            codes, offsets = variants(values)
            result = pa.nulls(len(values), type=datatype)
            for index, code in enumerate(values.type.type_codes):
                selected = codes == code
                positions = np.flatnonzero(selected).astype(np.int64, copy=False)
                if not len(positions):
                    continue
                indices = offsets[selected] if offsets is not None else positions
                child = pc.take(values.field(index), pa.array(indices, type=pa.int64()))
                normalized = self.normalize(child, datatype)
                placed = pc.scatter(normalized, pa.array(positions, type=pa.int64()), max_index=len(values) - 1)
                result = pc.coalesce(result, placed)
            return result
        if (
            pa.types.is_list(values.type)
            or pa.types.is_large_list(values.type)
            or pa.types.is_fixed_size_list(values.type)
        ) and (pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype)):
            mask = pc.is_null(values)
            if pa.types.is_fixed_size_list(values.type):
                start = values.offset * values.type.list_size
                length = len(values) * values.type.list_size
                child = values.values.slice(start, length)
            else:
                start = values.offsets[0].as_py()
                stop = values.offsets[-1].as_py()
                child = values.values.slice(start, stop - start)
            child = self.normalize(child, datatype.value_type)
            if pa.types.is_fixed_size_list(datatype):
                return pa.FixedSizeListArray.from_arrays(child, type=datatype, mask=mask)
            offsets = pc.subtract(values.offsets, values.offsets[0])
            if pa.types.is_large_list(datatype):
                return pa.LargeListArray.from_arrays(offsets, child, type=datatype, mask=mask)
            return pa.ListArray.from_arrays(pc.cast(offsets, pa.int32()), child, type=datatype, mask=mask)
        return pc.cast(values, datatype, safe=True)

    def prepare(
        self,
        values: pa.Array | pa.ChunkedArray,
        *,
        address: Address,
    ) -> pa.Array | pa.ChunkedArray:
        """Validate one whole Arrow leaf column at the plugin boundary."""

        if not isinstance(values, (pa.Array, pa.ChunkedArray)):
            raise TypeError(
                f"plugin '{self.name}' at address '{address}' requires an Arrow array, got {type(values).__name__}"
            )
        if not self.accepts(values.type):
            expected = ", ".join(" | ".join(member.__name__ for member in members) for members in self.families)
            raise TypeError(
                f"plugin '{self.name}' at address '{address}' does not accept Arrow type {values.type}; "
                f"expected terminal atoms compatible with {expected}; normalize it in a preprocessor"
            )
        datatype = self.canonical(values.type)
        try:
            return self.normalize(values, datatype)
        except pa.ArrowException as error:
            raise TypeError(
                f"plugin '{self.name}' at address '{address}' cannot safely normalize Arrow type "
                f"{values.type} to {datatype}; normalize it in a preprocessor"
            ) from error

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
            if key not in {Component.output, Component.write}:
                raise TypeError("only output and write may be registered as None")

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

                if not issubclass(obj, Leaf):
                    raise TypeError("Request must be a subclass of RequestBase")

            case Component.TensorField:
                if not isinstance(obj, type):
                    raise TypeError("TensorField must be a class type")

                if not issubclass(obj, TensorFieldBase):
                    raise TypeError("TensorField must be a subclass of TensorFieldBase")

                new_params = inspect.signature(obj.new).parameters
                required = {"field", "address", "schema", "strata"}
                if not required.issubset(new_params):
                    raise TypeError("TensorField.new must accept 'field', 'address', 'schema', and 'strata' parameters")

            case Component.Embedder:
                if not isinstance(obj, type):
                    raise TypeError("Embedder must be a class type")

                if not issubclass(obj, EmbedderBase):
                    raise TypeError("Embedder must be a subclass of EmbedderBase")

                # confirm the init method is expecting schema and address
                init_params = list(obj.__init__.__annotations__.keys())
                if "schema" not in init_params or "address" not in init_params:
                    raise TypeError("Embedder __init__ method must accept 'schema' and 'address' parameters")

            case Component.Decoder:
                if not isinstance(obj, type):
                    raise TypeError("Decoder must be a class type")

                if not issubclass(obj, DecoderBase):
                    raise TypeError("Decoder must be a subclass of DecoderBase")

                init_params = list(obj.__init__.__annotations__.keys())
                if "schema" not in init_params or "address" not in init_params:
                    raise TypeError("Decoder __init__ method must accept 'schema' and 'address' parameters")

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

                expected_params: list[str] = ["module", "prediction", "datatype"]
                func_params = list(inspect.signature(obj).parameters)

                if func_params != expected_params:
                    raise TypeError(
                        f"Write function must accept the following parameters: {expected_params}, got {func_params}"
                    )

            case Component.output:
                if not callable(obj):
                    raise TypeError("Output must be a callable function")

                expected_params = ["module", "address"]
                func_params = list(inspect.signature(obj).parameters)

                if func_params != expected_params:
                    raise TypeError(
                        f"Output function must accept the following parameters: {expected_params}, got {func_params}"
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

        if component == Component.output:
            return default_output

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
    def output(self) -> Callable[..., Any]:
        return cast(Callable[..., Any], self._component(Component.output))

    @property
    def write(self) -> Callable[..., Any]:
        return cast(Callable[..., Any], self._component(Component.write))

    def __getattr__(self, key: str) -> ComponentValue:
        try:
            component = Component(key)
        except ValueError:
            raise ValueError(f"Component '{key}' is not a valid Component enum value") from None

        return self._component(component)

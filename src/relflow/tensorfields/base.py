"""Tensorfield extension base classes and registry."""

from __future__ import annotations

import inspect
import math
import re
import warnings
from abc import abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from types import MappingProxyType, UnionType
from typing import TYPE_CHECKING, Any, Callable, TypeAlias, TypeVar, cast, get_args, overload

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from lightning.pytorch import Callback
from rich.text import Text
from tensordict import TensorClass, TensorDict

from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool
from relflow.data.arrow import variants
from relflow.structs.enums import Component, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address, Leaf, Renderable

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.ragged import RaggedField
    from relflow.structs.experiment import Schema

RequestBase: TypeAlias = Leaf
CallbackFactory: TypeAlias = type[Callback] | Callable[[], Callback]
ComponentValue: TypeAlias = Callable[..., Any] | type[Any]
RegisterT = TypeVar("RegisterT", bound=ComponentValue)
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


def default_observe(
    field: "RaggedField",
    *,
    address: Address,
    schema: "Schema",
    state: object | None,
    learn: bool,
) -> None:
    return None


def default_learn(
    module: "Model",
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    return None


@dataclass(frozen=True, slots=True)
class Context:
    """Per-field resources supplied uniformly during tensorization."""

    state: object | None = None
    salt: int = 0


class TensorInput(TensorClass):
    """Compact model input exposed to a tensorfield embedder."""

    state: torch.Tensor
    content: torch.Tensor | TensorDict


class EmbedderBase(torch.nn.Module):
    """Base class for tensorfield embedders."""

    def __init__(self, schema: Schema, address: Address):
        super().__init__()

        request = schema.requests[address]
        self.address = address
        self.destination = request.parent.address
        self.d_model = schema.d_model
        self.register_buffer("anchor", torch.zeros(()), persistent=False)

    @property
    def context(self) -> object | None:
        """Return the field-owned state shared with encoding workers."""

        return None

    def embed(self, field: "TensorFieldBase") -> Parcel:
        """Embed present coordinates and restore their fixed schema geometry."""

        shape = tuple(field.state.shape)
        present = field.present.reshape(-1)
        indices = present.nonzero(as_tuple=False).reshape(-1)
        count = int(indices.numel())

        if count:
            compact = self(field.take(indices))
            if not isinstance(compact, Parcel):
                raise TypeError(f"embedder for '{self.address}' must return a Parcel, got {type(compact).__name__}")
            expected = (count, self.d_model)
            if tuple(compact.payload.shape) != expected:
                raise ValueError(
                    f"embedder for '{self.address}' must return payload shape {expected}, "
                    f"got {tuple(compact.payload.shape)}"
                )
            if compact.present.dtype != torch.bool or tuple(compact.present.shape) != (count,):
                raise ValueError(
                    f"embedder for '{self.address}' must return bool presence shape {(count,)}, "
                    f"got {tuple(compact.present.shape)} with dtype {compact.present.dtype}"
                )
            if compact.origin != self.address or compact.destination != self.destination:
                raise ValueError(
                    f"compact parcel from '{self.address}' must route from '{self.address}' "
                    f"to '{self.destination}', got '{compact.origin}' to '{compact.destination}'"
                )

            compact_payload = compact.payload.masked_fill(~compact.present.unsqueeze(-1), 0.0)
            payload = compact.payload.new_zeros((math.prod(shape), self.d_model))
            payload = payload.index_copy(0, indices, compact_payload)
            restored = torch.zeros(math.prod(shape), dtype=torch.bool, device=compact.present.device)
            restored = restored.index_copy(0, indices, compact.present)
        else:
            payload = self.anchor.new_zeros((math.prod(shape), self.d_model))
            for parameter in self.parameters():
                payload = payload + parameter.sum() * 0.0
            restored = field.present.reshape(-1)

        return Parcel(
            payload=payload.reshape(*shape, self.d_model),
            present=restored.reshape(shape),
            origin=self.address,
            destination=self.destination,
            batch_size=shape[0],
        )


class DecoderBase(torch.nn.Module):
    """Base class for tensorfield decoders."""

    def __init__(self, schema: Schema, address: Address):
        super().__init__()

        self.address: Address = address
        self.register_buffer("anchor", torch.zeros(()), persistent=False)

        request = schema.requests[address]
        n_context = 1
        for dimension in schema.shapes[address]:
            n_context *= dimension
        self.n_context = n_context
        self.d_model = schema.d_model
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

    def forward(
        self,
        parcels: list[Parcel],
        *,
        batch_size: int,
        device: torch.device,
        embed: bool = False,
    ) -> Prediction:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 0:
            raise ValueError(f"decoder batch_size must be a non-negative integer, got {batch_size!r}")

        if not parcels:
            pooled = torch.zeros(
                (batch_size, self.n_context, self.d_model),
                device=device,
                dtype=self.anchor.dtype,
            )
            for parameter in self.pool.parameters():
                pooled = pooled + parameter.sum() * 0.0
        else:
            for parcel in parcels:
                if parcel.payload.shape[0] != batch_size:
                    raise ValueError(
                        f"decoder parcel from '{parcel.origin}' has batch size {parcel.payload.shape[0]}, "
                        f"expected {batch_size}"
                    )
                if tuple(parcel.present.shape) != tuple(parcel.payload.shape[:-1]):
                    raise ValueError(
                        f"decoder parcel from '{parcel.origin}' presence must have shape "
                        f"{tuple(parcel.payload.shape[:-1])}, got {tuple(parcel.present.shape)}"
                    )

            stacked = torch.cat(
                [parcel.payload.reshape(batch_size, -1, self.d_model) for parcel in parcels],
                dim=1,
            )
            present = torch.cat([parcel.present.reshape(batch_size, -1) for parcel in parcels], dim=1)
            pooled = self.pool(stacked, present=present)

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

    content: torch.Tensor | TensorDict
    state: torch.Tensor
    present: torch.Tensor
    trainable: torch.Tensor
    inferred: torch.Tensor
    targets: TensorDict[TensorKey, torch.Tensor]

    @classmethod
    @abstractmethod
    def new(
        cls,
        input: RaggedField,
        target: RaggedField,
        present: torch.Tensor,
        trainable: torch.Tensor,
        inferred: torch.Tensor,
        address: Address,
        schema: Schema,
        strata: Strata,
        context: Context,
    ) -> "TensorFieldBase":
        raise NotImplementedError

    def take(self, indices: torch.Tensor) -> TensorInput:
        """Gather one row-major coordinate prefix for embedding."""

        if indices.ndim != 1 or indices.dtype != torch.int64:
            raise TypeError("tensorfield take indices must be a one-dimensional int64 tensor")
        if indices.device != self.state.device:
            raise ValueError(
                f"tensorfield take indices must use state device {self.state.device}, got {indices.device}"
            )

        size = math.prod(self.state.shape)
        if indices.numel() and (indices.min() < 0 or indices.max() >= size):
            raise IndexError(f"tensorfield take indices must be between 0 and {size - 1}")

        def gather(value: Any) -> Any:
            if torch.is_tensor(value):
                if value.ndim < self.state.ndim or tuple(value.shape[: self.state.ndim]) != tuple(self.state.shape):
                    raise ValueError(
                        f"tensorfield content must start with state shape {tuple(self.state.shape)}, "
                        f"got {tuple(value.shape)}"
                    )
                trailing = tuple(value.shape[self.state.ndim :])
                return value.reshape(size, *trailing).index_select(0, indices)
            if isinstance(value, TensorDict):
                return TensorDict(
                    {key: gather(value[key]) for key in value.keys()},
                    batch_size=[indices.numel()],
                    device=value.device,
                )
            raise TypeError(f"tensorfield content must contain only tensors or TensorDicts, got {type(value).__name__}")

        return TensorInput(
            state=self.state.reshape(size).index_select(0, indices),
            content=gather(self.content),
            batch_size=[indices.numel()],
        )

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


TENSORFIELDS: dict[str, "Extension"] = {}


class Extension:
    """Registry object for a tensorfield implementation.

    Register request, tensorfield, embedder, decoder, loss, output, and write
    components with `@extension.register`. ``types`` names the Python equivalents
    of accepted canonical Arrow terminal families. Separate tuple entries are
    incompatible; types joined in one PEP 604 union may share one Arrow column.
    ``arrow`` supplies a physical-type matcher for custom Python atoms or
    overrides a standard matcher. Creating an extension with an existing name
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
            raise TypeError("Extension name must be a string")

        # should start with a letter and contain only lowercase letters, numbers, and underscores
        if not re.match(r"^[a-z0-9_]+$", name):
            raise ValueError("Extension name must consist of lowercase letters, numbers, and underscores only")

        if not isinstance(types, tuple):
            raise TypeError("Extension types must be a tuple of types or PEP 604 unions")
        if not types:
            raise ValueError("Extension types must contain at least one value type family")

        families: list[tuple[type[Any], ...]] = []
        type_to_family: dict[type[Any], int] = {}
        for family_index, family in enumerate(types):
            members = get_args(family) if isinstance(family, UnionType) else (family,)
            if not members or any(member is Any or not isinstance(member, type) for member in members):
                raise TypeError("Extension types must contain only concrete types or PEP 604 unions of concrete types")
            if type(None) in members:
                raise TypeError("Extension types must not include NoneType; null is represented by field state")
            if any(issubclass(member, (list, tuple, np.ndarray, np.generic, Iterator)) for member in members):
                raise TypeError(
                    "Extension types declare canonical terminal Python atoms; "
                    "sequences, iterators, ndarrays, and NumPy scalar classes are prepared structurally"
                )

            for member in members:
                if member in type_to_family:
                    raise ValueError(
                        f"Extension value type {member.__name__} appears in more than one compatibility family"
                    )
                type_to_family[member] = family_index
            families.append(cast(tuple[type[Any], ...], tuple(members)))

        if arrow is not None and not isinstance(arrow, Mapping):
            raise TypeError("Extension arrow must be a mapping from declared value types to Arrow matchers")
        declared = {} if arrow is None else dict(arrow)
        unknown = [member for member in declared if member not in type_to_family]
        if unknown:
            names = ", ".join(getattr(member, "__name__", repr(member)) for member in unknown)
            raise ValueError(f"Extension arrow matcher type(s) are absent from types: {names}")
        invalid = [member for member, matcher in declared.items() if not callable(matcher)]
        if invalid:
            names = ", ".join(member.__name__ for member in invalid)
            raise TypeError(f"Extension Arrow matcher(s) must be callable: {names}")
        missing = [member for member in type_to_family if member not in MATCHERS and member not in declared]
        if missing:
            names = ", ".join(member.__name__ for member in missing)
            raise TypeError(f"Extension custom value type(s) require arrow matchers: {names}")

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
                f"Extension '{name}' already registered; overriding existing tensorfield extension",
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
        """Resolve Arrow encodings that wrap an extension's declared atom types."""

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
        """Validate one whole Arrow leaf column at the extension boundary."""

        if not isinstance(values, (pa.Array, pa.ChunkedArray)):
            raise TypeError(
                f"extension '{self.name}' at address '{address}' requires an Arrow array, got {type(values).__name__}"
            )
        if not self.accepts(values.type):
            expected = ", ".join(" | ".join(member.__name__ for member in members) for members in self.families)
            raise TypeError(
                f"extension '{self.name}' at address '{address}' does not accept Arrow type {values.type}; "
                f"expected terminal atoms compatible with {expected}; normalize it in a preprocessor"
            )
        datatype = self.canonical(values.type)
        try:
            return self.normalize(values, datatype)
        except pa.ArrowException as error:
            raise TypeError(
                f"extension '{self.name}' at address '{address}' cannot safely normalize Arrow type "
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
        """Register one tensorfield component with this extension."""
        if obj is None:
            if component is None:
                raise TypeError("component must be provided when registering None")

            key = Component(component)
            if key not in {Component.output, Component.write}:
                raise TypeError("only output and write may be registered as None")

            if key in self.components:
                raise ValueError(f"Component '{key}' already registered in extension '{self.name}'")

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
            raise ValueError(f"Component '{key}' already registered in extension '{self.name}'")

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
                required = {
                    "input",
                    "target",
                    "present",
                    "trainable",
                    "inferred",
                    "address",
                    "schema",
                    "strata",
                    "context",
                }
                if not required.issubset(new_params):
                    names = ", ".join(sorted(required))
                    raise TypeError(f"TensorField.new must accept these parameters: {names}")

            case Component.Embedder:
                if not isinstance(obj, type):
                    raise TypeError("Embedder must be a class type")

                if not issubclass(obj, EmbedderBase):
                    raise TypeError("Embedder must be a subclass of EmbedderBase")

                # confirm the init method is expecting schema and address
                init_params = list(obj.__init__.__annotations__.keys())
                if "schema" not in init_params or "address" not in init_params:
                    raise TypeError("Embedder __init__ method must accept 'schema' and 'address' parameters")

                forward_params = inspect.signature(obj.forward).parameters
                if "inputs" not in forward_params:
                    raise TypeError("Embedder.forward must accept a compact 'inputs' parameter")

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

            case Component.observe:
                expected_params = ["field", "address", "schema", "state", "learn"]
                func_params = list(inspect.signature(obj).parameters)
                if func_params != expected_params:
                    raise TypeError(
                        f"Observe function must accept the following parameters: {expected_params}, got {func_params}"
                    )

            case Component.learn:
                expected_params = ["module", "observation", "address", "strata"]
                func_params = list(inspect.signature(obj).parameters)
                if func_params != expected_params:
                    raise TypeError(
                        f"Learn function must accept the following parameters: {expected_params}, got {func_params}"
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
                raise TypeError(f"Extension callback factory for '{self.name}' must produce a Lightning Callback")

        self.callback_factories.extend(registered)
        return factory if len(registered) == 1 else registered

    @property
    def callbacks(self) -> list[Callback]:
        """Instantiate all registered callback factories."""
        return [factory() for factory in self.callback_factories]

    def component(self, component: Component) -> ComponentValue:
        """Return a registered component, falling back to optional defaults."""
        if component in self.components:
            value = self.components[component]
            if value is not None:
                return value

        if component == Component.output:
            return default_output

        if component == Component.write:
            return default_write

        if component == Component.observe:
            return default_observe

        if component == Component.learn:
            return default_learn

        raise AttributeError(f"Extension '{self.name}' has no component '{component}'")

    @property
    def Request(self) -> type[RequestBase]:
        return cast(type[RequestBase], self.component(Component.Request))

    @property
    def TensorField(self) -> type[TensorFieldBase]:
        return cast(type[TensorFieldBase], self.component(Component.TensorField))

    @property
    def Embedder(self) -> type[EmbedderBase]:
        return cast(type[EmbedderBase], self.component(Component.Embedder))

    @property
    def Decoder(self) -> type[DecoderBase]:
        return cast(type[DecoderBase], self.component(Component.Decoder))

    @property
    def observe(self) -> Callable[..., TensorDict | None]:
        return cast(Callable[..., TensorDict | None], self.component(Component.observe))

    @property
    def learn(self) -> Callable[..., None]:
        return cast(Callable[..., None], self.component(Component.learn))

    @property
    def loss(self) -> Callable[..., Any]:
        return cast(Callable[..., Any], self.component(Component.loss))

    @property
    def output(self) -> Callable[..., Any]:
        return cast(Callable[..., Any], self.component(Component.output))

    @property
    def write(self) -> Callable[..., Any]:
        return cast(Callable[..., Any], self.component(Component.write))

    def __getattr__(self, key: str) -> ComponentValue:
        try:
            component = Component(key)
        except ValueError:
            raise ValueError(f"Component '{key}' is not a valid Component enum value") from None

        return self.component(component)

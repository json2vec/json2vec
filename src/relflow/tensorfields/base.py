"""Tensorfield plugin base classes and registry."""

from __future__ import annotations

import re
import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, TypeAlias, TypeVar, cast, overload

import pluggy
import torch
from lightning.pytorch import Callback
from rich.console import Console, Group
from rich.text import Text
from tensordict import TensorDict

from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool
from relflow.rich import bounded_path
from relflow.structs.enums import Component, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address, Leaf, Node, Renderable
from relflow.tensorfields.spec import PluginSpec

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Mask

pm: pluggy.PluginManager = pluggy.PluginManager(project_name="tensorfields")

pm.add_hookspecs(module_or_class=PluginSpec)

RequestBase: TypeAlias = Leaf
CallbackFactory: TypeAlias = type[Callback] | Callable[[], Callback]
ComponentValue: TypeAlias = Callable[..., Any] | type[Any]
RegisterT = TypeVar("RegisterT", bound=ComponentValue)
BranchMaskApplication: TypeAlias = tuple[Address, "Mask"]


def default_write(module: "Model", prediction: Prediction) -> None:
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
    STATE_PREVIEW_ROW_LIMIT: int = 10
    STATE_PREVIEW_COLUMN_LIMIT: int = 32
    STATE_AXIS_LIMIT: int = 8
    STATE_AXIS_NAME_LIMIT: int = 40
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

    def _rich_tensorfield_type(self) -> str:
        for name, plugin in TENSORFIELDS.items():
            if plugin.components.get(Component.TensorField) is type(self):
                return name
        return type(self).__module__.rsplit(".", maxsplit=1)[-1]

    def __rich_repr__(self):
        state = getattr(self, TensorKey.state, None)
        yield "type", self._rich_tensorfield_type()
        if torch.is_tensor(state):
            yield "shape", tuple(state.shape)
            yield "dtype", str(state.dtype).removeprefix("torch.")
            yield "device", str(state.device)

    @classmethod
    def _format_shape(cls, shape: torch.Size | tuple[int, ...]) -> str:
        dimensions = tuple(shape)
        if len(dimensions) <= cls.STATE_AXIS_LIMIT:
            return str(dimensions)

        head = ", ".join(str(dimension) for dimension in dimensions[:4])
        tail = ", ".join(str(dimension) for dimension in dimensions[-3:])
        omitted = len(dimensions) - 7
        return f"({head}, … +{omitted} dims, {tail})"

    @classmethod
    def _tensor_shape(cls, value: Any) -> str | None:
        if torch.is_tensor(value):
            return cls._format_shape(value.shape)
        if not isinstance(value, TensorDict):
            return None

        items: list[str] = []
        keys = sorted(value.keys(), key=str)
        for key in keys[:4]:
            tensor = value.get(key)
            shape = cls._format_shape(tensor.shape) if torch.is_tensor(tensor) else type(tensor).__name__
            items.append(f"{key}:{shape}")
        if len(keys) > 4:
            items.append(f"+{len(keys) - 4} more")
        return "{" + ", ".join(items) + "}"

    def __rich_console__(self, console: Console, options):
        state = getattr(self, TensorKey.state, None)
        content = getattr(self, TensorKey.content, None)
        trainable = getattr(self, TensorKey.trainable, None)
        targets = getattr(self, TensorKey.targets, None)

        heading = Text()
        heading.append(type(self).__name__, style=self._rich_style(console, self.RICH_NAME_STYLE))
        heading.append(" ")
        heading.append(
            f"[{self._rich_tensorfield_type()}]",
            style=self._rich_style(console, self.RICH_TYPE_STYLE),
        )

        if torch.is_tensor(state):
            heading.append(" ")
            heading.append("state=", style=self._rich_style(console, "relflow.dim"))
            heading.append(self._format_shape(state.shape), style=self._rich_style(console, "relflow.info"))
            heading.append(" ")
            heading.append("dtype=", style=self._rich_style(console, "relflow.dim"))
            heading.append(
                str(state.dtype).removeprefix("torch."),
                style=self._rich_style(console, "relflow.info"),
            )
            heading.append(" ")
            heading.append("device=", style=self._rich_style(console, "relflow.dim"))
            heading.append(str(state.device), style=self._rich_style(console, "relflow.info"))

        if torch.is_tensor(trainable):
            heading.append(" ")
            heading.append("trainable=", style=self._rich_style(console, "relflow.dim"))
            heading.append(self._format_shape(trainable.shape), style=self._rich_style(console, "relflow.info"))

        lines: list[Text] = [heading]

        content_shape = self._tensor_shape(content)
        if content_shape is not None:
            text = Text("  ")
            text.append("content=", style=self._rich_style(console, "relflow.dim"))
            text.append(content_shape, style=self._rich_style(console, "relflow.info"))
            lines.append(text)

        if torch.is_tensor(state):
            display_state, display_trainable, axes, singleton_axes = self._state_display(state, trainable)
            lines.append(self._state_axes_text(axes, singleton_axes=singleton_axes, console=console))

            if display_state.numel() == 0:
                lines.append(self._state_counts_text([], console=console, truncated=False))
                lines.append(self._state_preview_text([], axes=axes, console=console))
            elif display_state.ndim > 2:
                lines.append(self._state_slice_instruction(axes, console=console))
            else:
                preview = self._bounded_state_preview(
                    display_state,
                    display_trainable,
                    column_limit=self._state_preview_column_limit(axes, options.max_width),
                )
                if preview is None:
                    text = Text("  preview omitted for ", style=self._rich_style(console, "relflow.dim"))
                    text.append(str(state.device), style=self._rich_style(console, "relflow.info"))
                    text.append(
                        ", slice first and call .cpu() explicitly", style=self._rich_style(console, "relflow.dim")
                    )
                    lines.append(text)
                else:
                    rows, truncated = preview
                    lines.append(self._state_counts_text(rows, console=console, truncated=truncated))
                    lines.append(
                        self._state_preview_text(
                            rows,
                            axes=axes,
                            console=console,
                        )
                    )
                    lines.append(self._state_legend_text(console=console))

        if isinstance(targets, TensorDict) and targets.keys():
            text = Text("  targets=", style=self._rich_style(console, "relflow.dim"))
            text.append(
                ", ".join(str(key) for key in sorted(targets.keys(), key=str)),
                style=self._rich_style(console, "relflow.info"),
            )
            lines.append(text)

        yield Group(*lines)

    @classmethod
    def _state_axis_name(cls, value: Any, index: int) -> str:
        if value is None:
            return f"dim{index}"

        return bounded_path(str(value), limit=cls.STATE_AXIS_NAME_LIMIT)

    def _state_display(
        self,
        state: torch.Tensor,
        trainable: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[str, int]], list[tuple[str, int]]]:
        raw_names = list(getattr(self, "names", None) or ())
        axes: list[tuple[str, int]] = []
        for index, size in enumerate(state.shape):
            name = raw_names[index] if index < len(raw_names) else None
            axes.append((self._state_axis_name(name, index), size))

        singleton_indexes = [index for index, (_, size) in enumerate(axes) if size == 1]
        singleton_axes = [axes[index] for index in singleton_indexes]
        visible_axes = [axis for index, axis in enumerate(axes) if index not in singleton_indexes]

        display = state.detach()
        display_trainable: torch.Tensor | None = None
        if torch.is_tensor(trainable) and trainable.shape == state.shape:
            display_trainable = trainable.detach()

        for index in reversed(singleton_indexes):
            display = display.squeeze(index)
            if display_trainable is not None:
                display_trainable = display_trainable.squeeze(index)

        return display, display_trainable, visible_axes, singleton_axes

    @classmethod
    def _bounded_axes(
        cls,
        axes: list[tuple[str, int]],
    ) -> tuple[list[tuple[str, int]], list[tuple[str, int]], int]:
        if len(axes) <= cls.STATE_AXIS_LIMIT:
            return axes, [], 0

        head_count = cls.STATE_AXIS_LIMIT // 2
        tail_count = cls.STATE_AXIS_LIMIT - head_count
        return axes[:head_count], axes[-tail_count:], len(axes) - cls.STATE_AXIS_LIMIT

    def _state_axes_text(
        self,
        axes: list[tuple[str, int]],
        *,
        singleton_axes: list[tuple[str, int]],
        console: Console,
    ) -> Text:
        text = Text("  axes=", style=self._rich_style(console, "relflow.dim"))
        head_axes, tail_axes, omitted = self._bounded_axes(axes)
        if not head_axes:
            text.append("<scalar>", style=self._rich_style(console, "relflow.info"))
        else:
            for index, (name, size) in enumerate(head_axes):
                if index:
                    text.append(" × ", style=self._rich_style(console, "relflow.dim"))
                text.append(name, style=self._rich_style(console, "relflow.info"))
                text.append(f"={size}", style=self._rich_style(console, "relflow.dim"))
            if omitted:
                text.append(f" × … +{omitted} axes", style=self._rich_style(console, "relflow.dim"))
            for name, size in tail_axes:
                text.append(" × ", style=self._rich_style(console, "relflow.dim"))
                text.append(name, style=self._rich_style(console, "relflow.info"))
                text.append(f"={size}", style=self._rich_style(console, "relflow.dim"))

        if singleton_axes:
            head_singletons, tail_singletons, omitted_singletons = self._bounded_axes(singleton_axes)
            singleton_labels = [name for name, _ in head_singletons]
            if omitted_singletons:
                singleton_labels.append(f"… +{omitted_singletons}")
            singleton_labels.extend(name for name, _ in tail_singletons)
            text.append(" (singleton ", style=self._rich_style(console, "relflow.dim"))
            text.append(
                ", ".join(singleton_labels),
                style=self._rich_style(console, "relflow.info"),
            )
            text.append(" hidden)", style=self._rich_style(console, "relflow.dim"))

        return text

    def _state_slice_instruction(self, axes: list[tuple[str, int]], *, console: Console) -> Text:
        text = Text("  state preview omitted: ", style=self._rich_style(console, "relflow.dim"))
        text.append(str(len(axes)), style=self._rich_style(console, "relflow.info"))
        text.append(
            " non-singleton axes remain; slice to at most 2 to preview state",
            style=self._rich_style(console, "relflow.dim"),
        )
        return text

    def _state_counts_text(
        self,
        rows: list[list[tuple[int, bool]]],
        *,
        console: Console,
        truncated: bool,
    ) -> Text:
        cells = [cell for row in rows for cell in row]
        values = [value for value, _ in cells]
        text = Text("  preview counts ", style=self._rich_style(console, "relflow.dim"))
        for index, token in enumerate(Tokens):
            if index:
                text.append(" ", style=self._rich_style(console, "relflow.dim"))
            count = values.count(token.value)
            text.append(self.STATE_LABELS[token.value], style=self.STATE_STYLES[token.value])
            text.append(f"={count}", style=self._rich_style(console, "relflow.dim"))

        if truncated:
            text.append(" (sample)", style=self._rich_style(console, "relflow.dim"))

        trainable = sum(is_trainable for _, is_trainable in cells)
        if trainable:
            text.append(f" *={trainable} trainable", style=self._rich_style(console, "relflow.dim"))

        return text

    @staticmethod
    def _short_axis_name(name: str) -> str:
        return name.rstrip("/").rsplit("/", maxsplit=1)[-1]

    def _state_preview_column_limit(self, axes: list[tuple[str, int]], max_width: int) -> int:
        if len(axes) == 2:
            prefix_width = Text("          0 │ ").cell_len
        else:
            labels = " × ".join(self._short_axis_name(name) for name, _ in axes)
            prefix = f"  state [{labels}] " if labels else "  state "
            prefix_width = Text(prefix).cell_len

        available = max(1, max_width - prefix_width)
        # Every cell can be a two-character token such as ``M*`` and cells are
        # separated by one space.
        return max(1, min(self.STATE_PREVIEW_COLUMN_LIMIT, (available + 1) // 3))

    def _state_preview_text(
        self,
        rows: list[list[tuple[int, bool]]],
        *,
        axes: list[tuple[str, int]],
        console: Console,
    ) -> Text:
        text = Text("  state ", style=self._rich_style(console, "relflow.dim"))
        if axes:
            labels = " × ".join(self._short_axis_name(name) for name, _ in axes)
            text.append(f"[{labels}]", style=self._rich_style(console, "relflow.info"))
        if not rows:
            if axes:
                text.append(" ", style=self._rich_style(console, "relflow.dim"))
            text.append("<empty>", style=self._rich_style(console, "relflow.dim"))
            return text

        if len(axes) == 2:
            text.append("\n        ", style=self._rich_style(console, "relflow.dim"))
        elif axes:
            text.append(" ", style=self._rich_style(console, "relflow.dim"))

        for row_index, row in enumerate(rows):
            if row_index:
                text.append("\n        ", style=self._rich_style(console, "relflow.dim"))

            if len(axes) == 2:
                text.append(f"{row_index:>3} │ ", style=self._rich_style(console, "relflow.dim"))

            for column_index, (value, is_trainable) in enumerate(row):
                if column_index:
                    text.append(" ")

                token = int(value)
                label = self.STATE_LABELS.get(token, str(token))
                if is_trainable:
                    label = f"{label}*"
                text.append(label, style=self.STATE_STYLES.get(token, "bold red"))

        return text

    def _state_legend_text(self, *, console: Console) -> Text:
        text = Text("  legend ", style=self._rich_style(console, "relflow.dim"))
        descriptions = {
            Tokens.valued: "valued",
            Tokens.null: "null",
            Tokens.padded: "padded",
            Tokens.masked: "masked",
            Tokens.other: "other",
        }
        for index, token in enumerate(Tokens):
            if index:
                text.append("  ", style=self._rich_style(console, "relflow.dim"))
            text.append(self.STATE_LABELS[token.value], style=self.STATE_STYLES[token.value])
            text.append(f" {descriptions[token]}", style=self._rich_style(console, "relflow.dim"))
        text.append("  * trainable", style=self._rich_style(console, "relflow.dim"))
        return text

    def _bounded_state_preview(
        self,
        tensor: torch.Tensor,
        trainable: torch.Tensor | None,
        *,
        column_limit: int,
    ) -> tuple[list[list[tuple[int, bool]]], bool] | None:
        if tensor.device.type != "cpu":
            return None

        values = tensor.detach()
        trainable_values: torch.Tensor | None = None
        if trainable is not None and trainable.device.type == "cpu" and trainable.shape == tensor.shape:
            trainable_values = trainable.detach()

        if values.numel() == 0:
            return [], False

        if values.ndim == 0:
            is_trainable = bool(trainable_values.tolist()) if trainable_values is not None else False
            rows = [[(int(values.tolist()), is_trainable)]]
            return rows, False

        if values.ndim == 1:
            width = min(values.shape[0], self.STATE_PREVIEW_LIMIT, column_limit)
            bounded = values[:width]
            bounded_trainable = trainable_values[:width].tolist() if trainable_values is not None else [False] * width
            rows = [
                [
                    (int(value), bool(is_trainable))
                    for value, is_trainable in zip(bounded.tolist(), bounded_trainable, strict=True)
                ]
            ]
            return rows, values.numel() > width

        n_rows = min(values.shape[0], self.STATE_PREVIEW_ROW_LIMIT)
        n_columns = min(
            values.shape[1],
            column_limit,
            max(1, self.STATE_PREVIEW_LIMIT // max(1, n_rows)),
        )
        bounded = values[:n_rows, :n_columns]
        bounded_trainable = (
            trainable_values[:n_rows, :n_columns].tolist()
            if trainable_values is not None
            else [[False] * n_columns for _ in range(n_rows)]
        )
        rows = [
            [(int(value), bool(is_trainable)) for value, is_trainable in zip(row, trainable_row, strict=True)]
            for row, trainable_row in zip(bounded.tolist(), bounded_trainable, strict=True)
        ]
        return rows, values.numel() > bounded.numel()


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

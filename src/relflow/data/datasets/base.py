"""Shared data module type aliases and helpers."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable, Mapping
from functools import partial
from typing import Annotated, Any, TypeAlias, TypeVar

from beartype import beartype
from beartype.vale import Is
from rich.console import Console
from rich.text import Text
from rich.tree import Tree
from tensordict import TensorDict
from torch.utils.data import get_worker_info

from relflow.data.processors import Preprocessor
from relflow.distributed import rank as distributed_rank
from relflow.distributed import world_size as distributed_world_size
from relflow.rich import MimeBundleDisplay, bounded_path, render_html, render_text, strip_control_sequences
from relflow.structs.enums import Strata, TensorKey
from relflow.structs.tree import Address, Renderable
from relflow.tensorfields.base import TensorFieldBase

T = TypeVar("T")
StrataMap: TypeAlias = Mapping[Strata | str, T]
NonNegativeInt: TypeAlias = Annotated[int, Is[lambda value: not isinstance(value, bool) and value >= 0]]
PositiveInt: TypeAlias = Annotated[int, Is[lambda value: not isinstance(value, bool) and value >= 1]]
SampleRate: TypeAlias = Annotated[int | float, Is[lambda value: not isinstance(value, bool) and 0.0 < value <= 1.0]]
RawObservation: TypeAlias = dict[str, Any]
ProcessedObservation: TypeAlias = list[RawObservation]
EncodedBatch: TypeAlias = list[ProcessedObservation]
InterprocessEncodingContext: TypeAlias = dict[Address, Any]

DISPLAY_LABEL_LIMIT = 120

# Encoded batches are `list[list[dict]]`: outer batch, then records emitted for
# one processed observation. Request queries are written relative to the inner
# list; the encoder prepends the outer batch selector before JMESPath search.


def display_label(value: str, *, limit: int = DISPLAY_LABEL_LIMIT) -> str:
    """Return a bounded, single-line label for a user-controlled name."""

    label = " ".join(strip_control_sequences(value).split()) or "<unnamed>"
    if len(label) <= limit:
        return label
    return f"{label[: limit - 1]}…"


class EncodedInput(TensorDict, Renderable):
    """A batch of address-keyed tensorfields with a safe compact display."""

    FIELD_DISPLAY_LIMIT = 24
    AXIS_DISPLAY_LIMIT = 6
    ADDRESS_DISPLAY_LIMIT = 72

    def __repr__(self) -> str:
        return str(self)

    def __rich_console__(self, console: Console, options):
        fields = [
            (Address(str(address)), field) for address, field in self.items() if isinstance(field, TensorFieldBase)
        ]

        heading = Text()
        heading.append("EncodedInput", style=self._rich_style(console, self.RICH_NAME_STYLE))
        heading.append(" ")
        heading.append("[encoded]", style=self._rich_style(console, self.RICH_TYPE_STYLE))
        heading.append(" ")
        heading.append("batch_size=", style=self._rich_style(console, "relflow.dim"))
        heading.append(str(tuple(self.batch_size)), style=self._rich_style(console, "relflow.info"))
        heading.append(" ")
        heading.append("fields=", style=self._rich_style(console, "relflow.dim"))
        heading.append(str(len(fields)), style=self._rich_style(console, "relflow.info"))
        if TensorKey.metadata in self.keys():
            heading.append(" metadata=hidden", style=self._rich_style(console, "relflow.dim"))

        tree = Tree(heading, guide_style=self._rich_style(console, self.RICH_TREE_STYLE))
        ordered_fields = sorted(fields, key=lambda item: str(item[0]))
        for address, field in ordered_fields[: self.FIELD_DISPLAY_LIMIT]:
            state = getattr(field, TensorKey.state, None)
            line = Text()
            line.append(
                bounded_path(str(address), limit=self.ADDRESS_DISPLAY_LIMIT),
                style=self._rich_style(console, self.RICH_NAME_STYLE),
            )
            line.append(" ")
            line.append(
                f"[{field._rich_tensorfield_type()}]",
                style=self._rich_style(console, self.RICH_TYPE_STYLE),
            )
            if hasattr(state, "shape"):
                line.append(" state=", style=self._rich_style(console, "relflow.dim"))
                line.append(field._format_shape(state.shape), style=self._rich_style(console, "relflow.info"))

            names = [name for name in getattr(field, "names", ()) if name is not None]
            if names:
                line.append(" axes=", style=self._rich_style(console, "relflow.dim"))
                omitted_axes = max(0, len(names) - self.AXIS_DISPLAY_LIMIT)
                if omitted_axes:
                    names = [*names[:3], f"… +{omitted_axes}", *names[-3:]]
                axes = ", ".join(bounded_path(str(name), limit=48) for name in names)
                line.append(f"({axes})", style=self._rich_style(console, "relflow.info"))
            tree.add(line)

        omitted_fields = len(ordered_fields) - self.FIELD_DISPLAY_LIMIT
        if omitted_fields > 0:
            tree.add(
                Text(
                    f"… +{omitted_fields} fields",
                    style=self._rich_style(console, "relflow.dim"),
                )
            )

        yield tree


def compact_strata(configuration: Mapping[Strata, T], strata: tuple[Strata, ...]) -> Any:
    """Collapse uniform per-split configuration without inspecting its sources."""

    selected = [(stratum.value, configuration[stratum]) for stratum in strata]
    if not selected:
        return None
    first = selected[0][1]
    if all(value == first for _, value in selected[1:]):
        return first
    return dict(selected)


class DataModuleDisplay:
    """Safe Rich and notebook display shared by RelFlow data modules."""

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return render_text(self)

    def _display_(self) -> MimeBundleDisplay:
        return MimeBundleDisplay(self)

    def _repr_mimebundle_(self, include=None, exclude=None):
        requested = {"text/plain", "text/html"}
        if include is not None:
            requested.intersection_update(include)
        if exclude is not None:
            requested.difference_update(exclude)

        bundle: dict[str, str] = {}
        if "text/plain" in requested:
            bundle["text/plain"] = str(self)
        if "text/html" in requested:
            bundle["text/html"] = self._repr_html_()
        return bundle

    def _repr_html_(self) -> str:
        return render_html(self)

    def _mime_(self):
        return "text/html", self._repr_html_()

    def data_module_rich_repr(self, splits: Mapping[Strata, object]):
        strata = tuple(splits)
        details: object
        if all(value is None for value in splits.values()):
            details = tuple(stratum.value for stratum in strata)
        else:
            details = {stratum.value: value for stratum, value in splits.items()}

        yield "splits", details
        yield "batch_size", self.batch_size
        yield "preprocessor", self.preprocessor, None
        if not strata:
            return
        yield "num_workers", compact_strata(self.num_workers, strata), None
        yield "persistent_workers", compact_strata(self.persistent_workers, strata), True
        yield "pin_memory", compact_strata(self.pin_memory, strata), True
        yield "observation_buffer_size", compact_strata(self.observation_buffer_size, strata), 1
        yield "sample_rate", compact_strata(self.sample_rate, strata), 1.0


@beartype
class Pipeline:
    def __init__(self, **arguments: Any):
        self.arguments: dict[str, Any] = arguments
        self.steps: list[Callable[..., Any]] = []

    def __or__(self, function: Callable[..., Any]) -> "Pipeline":
        required = [name for name in inspect.signature(function).parameters.keys()]
        available = set(required) & set(self.arguments.keys())
        self.steps.append(partial(function, **{arg: self.arguments[arg] for arg in available}))
        return self

    def __repr__(self) -> str:
        return f"Pipeline(steps={len(self.steps)}, arguments={self.arguments!r})"

    def __iter__(self):
        stream = self.steps[0]()

        for step in self.steps[1:]:
            stream = step(stream)

        return iter(stream)


class PreprocessorConfig:
    Value: TypeAlias = Preprocessor | None

    @classmethod
    def normalize(cls, preprocessor: Value) -> Value:
        if preprocessor is None:
            return None

        if isinstance(preprocessor, Preprocessor):
            return preprocessor

        raise TypeError(f"preprocessor must be a Preprocessor object or None, got {type(preprocessor).__name__}")


@beartype
def sha256(string: str, bits: int = 64) -> int:
    if not (1 <= bits <= 256):
        raise ValueError("bits must be between 1 and 256")

    digest = hashlib.sha256(string.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") >> (256 - bits)


def _worker_identity(global_rank: int | None = None, world_size: int | None = None) -> tuple[int, int]:
    if global_rank is None:
        global_rank = distributed_rank()
    if world_size is None:
        world_size = distributed_world_size()

    worker_info = get_worker_info()
    if worker_info is None:
        return global_rank, max(1, world_size)

    worker_count = max(1, worker_info.num_workers)
    return (global_rank * worker_count) + worker_info.id, max(1, world_size) * worker_count


def _worker_buffer_size(size: int) -> int:
    """Divide an approximate per-rank buffer budget across local DataLoader workers."""
    worker_info = get_worker_info()
    if worker_info is None:
        return size

    worker_count = max(1, worker_info.num_workers)
    return max(1, (size + worker_count - 1) // worker_count)


def _is_assigned_to_worker(shard_key: str, worker_id: int, num_workers: int) -> bool:
    if num_workers <= 1:
        return True

    owner: int = sha256(shard_key) % num_workers
    return owner == worker_id


def share_interprocess_encoding_context(context: InterprocessEncodingContext) -> None:
    """Opt encoding context resources into multiprocessing-safe storage."""
    for field_context in context.values():
        share = getattr(field_context, "share", None)
        if callable(share):
            share()


def identity(data: Any) -> Any:
    return data

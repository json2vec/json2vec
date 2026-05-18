from __future__ import annotations

import io
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

import numpy as np
import pydantic
import rich.box
import torch
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.terminal_theme import DEFAULT_TERMINAL_THEME
from rich.text import Text

from json2vec.structs.tree import Address, Leaf, Node
from json2vec.tensorfields.base import TENSORFIELDS

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec
    from json2vec.structs.experiment import Hyperparameters
    from json2vec.tensorfields.shared.counter import Counter

PLOT_WIDTH = 220
PLOT_TITLE_STYLE = "bold"
PLOT_SECTION_STYLE = "bold"


class Pane(pydantic.BaseModel):
    title: str
    values: dict[str, Any] = pydantic.Field(default_factory=dict)
    sections: dict[str, Any] = pydantic.Field(default_factory=dict)
    children: list["Pane"] = pydantic.Field(default_factory=list)

    def add_section(self, title: str, values: Any) -> None:
        self.sections[title] = values

    def add_child(self, child: "Pane") -> None:
        self.children.append(child)


Pane.model_rebuild()


def plot(
    module: "JSON2Vec",
    address: Address | str | None = None,
    detail: bool = False,
    out: str | Path | None = None,
) -> str:
    hyperparameters = module.hyperparameters

    def build(node: Node) -> Pane:
        values: dict[str, Any] = {}

        if node.address and node.address != node.name:
            values["address"] = node.address

        values |= node.model_dump(mode="python", exclude={"fields", "type", "name"}, exclude_none=True)
        pane = Pane(title=f"{node.name} ({node.type})", values=values)

        for child in node.children:
            pane.add_child(build(child))

        if detail and isinstance(node, Leaf):
            extension = TENSORFIELDS[node.type]
            extension.plot(module=module, address=node.address, branch=pane, detail=detail)
            add_counter_details(pane=pane, module=module, address=node.address)

        return pane

    if address is None:
        pane = Pane(
            title="JSON2Vec",
            values=hyperparameters.model_dump(mode="python", exclude={"fields", "type", "name"}, exclude_none=True),
        )
        pane.add_child(build(hyperparameters.fields))
    else:
        pane = build(resolve_node(hyperparameters=hyperparameters, address=address))

    renderable = render_pane(pane, expand=True, depth=0)
    Console(width=PLOT_WIDTH).print(renderable)
    recorder = Console(file=io.StringIO(), record=True, width=PLOT_WIDTH, color_system="truecolor")
    recorder.print(renderable)
    rendered = recorder.export_html(theme=DEFAULT_TERMINAL_THEME, clear=False)

    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")

    return rendered


def resolve_node(hyperparameters: "Hyperparameters", address: Address | str) -> Node:
    key = Address(str(address))
    nodes: dict[Address, Node] = hyperparameters.arrays | hyperparameters.requests

    if key not in nodes:
        raise ValueError(f"address '{address}' was not found in the hyperparameters")

    return nodes[key]


def render_pane(pane: Pane, *, expand: bool, depth: int) -> Panel:
    color_index = depth % 8
    foreground_index = 0 if color_index == 7 else 15
    blocks: list[RenderableType] = []

    if pane.values:
        blocks.append(render_values(pane.values))

    for title, values in pane.sections.items():
        if blocks:
            blocks.append(Text())

        blocks.append(Text(title, style=PLOT_SECTION_STYLE))
        if isinstance(values, dict):
            section = render_values(values)
        else:
            formatted = format_value(values)
            if "\n" in formatted:
                section = Group(*(Text(line) for line in formatted.splitlines()))
            else:
                section = Text(formatted)

        blocks.append(Padding(section, (0, 0, 0, 2)))

    if pane.children:
        if blocks:
            blocks.append(Text())

        blocks.append(Text("children", style=PLOT_SECTION_STYLE))
        blocks.append(
            Padding(
                render_children(pane.children, depth=depth),
                (1, 0, 0, 0),
            )
        )

    content: RenderableType = Group(*blocks) if blocks else Text(" ")

    return Panel(
        content,
        title=Text(pane.title, style=PLOT_TITLE_STYLE),
        box=rich.box.ROUNDED,
        padding=(0, 1),
        expand=expand,
        title_align="left",
        border_style=f"color({foreground_index})",
        style=f"color({foreground_index}) on color({color_index})",
    )


def render_children(children: list[Pane], depth: int = 0) -> RenderableType:
    if len(children) == 1:
        return render_pane(children[0], expand=True, depth=depth + 1)

    columns = 2
    grid = Table.grid(expand=True, padding=(1, 3))

    for _ in range(columns):
        grid.add_column(ratio=1)

    row: list[RenderableType] = []

    for child in children:
        row.append(render_pane(child, expand=True, depth=depth + 1))

        if len(row) == columns:
            grid.add_row(*row)
            row = []

    if row:
        row.extend(Text("") for _ in range(columns - len(row)))
        grid.add_row(*row)

    return grid


def render_values(values: dict[str, Any]) -> RenderableType:
    lines: list[Text] = []

    for key, value in values.items():
        formatted = format_value(value)

        if "\n" not in formatted:
            lines.append(Text(f"{key}: {formatted}"))
            continue

        lines.append(Text(f"{key}:"))
        for line in formatted.splitlines():
            lines.append(Text(f"  {line}"))

    return Group(*lines) if lines else Text(" ")


def format_value(value: Any) -> str:
    normalized = normalize_value(value)
    if isinstance(normalized, str):
        return normalized
    return pformat(normalized, compact=True, sort_dicts=False, width=52)


def normalize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]

    if hasattr(value, "value"):
        return normalize_value(value.value)

    return value


def add_counter_details(
    pane: Pane,
    module: "JSON2Vec",
    address: Address,
) -> None:
    decoder = module.nodes[address].decoder
    counters: dict[str, "Counter"] = {}

    if hasattr(decoder, "counter"):
        counters["counter"] = decoder.counter

    if hasattr(decoder, "counters"):
        counters |= dict(decoder.counters.items())

    if not counters:
        return

    for name, counter in counters.items():
        title = "counter" if name == "counter" else f"counters.{name}"
        pane.add_section(title, str(counter))

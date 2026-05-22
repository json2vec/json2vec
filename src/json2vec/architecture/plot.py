from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

import numpy as np
import rich.box
import torch
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from json2vec.structs.tree import Address, Leaf, Node
from json2vec.tensorfields.base import TENSORFIELDS

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec
    from json2vec.structs.experiment import Hyperparameters
    from json2vec.tensorfields.shared.counter import Counter

PLOT_WIDTH = 120


@dataclass(slots=True)
class Pane:
    title: str
    marked: bool = False
    values: dict[str, Any] = field(default_factory=dict)
    sections: dict[str, Any] = field(default_factory=dict)
    children: list["Pane"] = field(default_factory=list)

    def add_section(self, title: str, values: Any) -> None:
        self.sections[title] = values


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
        if node.address in module.nodes:
            values |= parameter_counts(module.nodes[node.address])

        pane = Pane(
            title=f"{node.name} ({node.type})",
            marked=node.address in hyperparameters.target or node.address in hyperparameters.embed,
            values=values,
            children=[build(child) for child in node.children],
        )

        if detail and isinstance(node, Leaf):
            extension = TENSORFIELDS[node.type]
            extension.plot(module=module, address=node.address, branch=pane, detail=detail)
            add_counter_details(pane=pane, module=module, address=node.address)

        return pane

    if address is None:
        values = hyperparameters.model_dump(mode="python", exclude={"fields", "type", "name"}, exclude_none=True)
        values |= parameter_counts(module)
        pane = Pane(
            title="JSON2Vec",
            values=values,
            children=[build(hyperparameters.fields)],
        )
    else:
        pane = build(resolve_node(hyperparameters=hyperparameters, address=address))

    renderable = render_pane(pane)
    recorder = Console(file=io.StringIO(), record=True, width=PLOT_WIDTH)
    recorder.print(renderable)
    rendered = recorder.export_html(clear=False)

    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")

    return rendered


def parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    parameters = list(module.parameters())
    return {"parameters": sum(parameter.numel() for parameter in parameters)}


def resolve_node(hyperparameters: "Hyperparameters", address: Address | str) -> Node:
    key = Address(str(address))
    nodes: dict[Address, Node] = hyperparameters.arrays | hyperparameters.requests

    if key not in nodes:
        raise ValueError(f"address '{address}' was not found in the hyperparameters")

    return nodes[key]


def render_pane(pane: Pane) -> Panel:
    blocks: list[RenderableType] = []

    if pane.values:
        blocks.append(render_values(pane.values))

    for title, values in pane.sections.items():
        if blocks:
            blocks.append(Text())

        blocks.append(Text(title))
        if isinstance(values, dict):
            section = render_values(values)
        else:
            formatted = format_value(values)
            if "\n" in formatted:
                section = Group(*(Text(line) for line in formatted.splitlines()))
            else:
                section = Text(formatted)

        blocks.append(Padding(section, (0, 0, 0, 2)))

    for child in pane.children:
        if blocks:
            blocks.append(Text())

        blocks.append(
            Padding(
                render_pane(child),
                (0, 0, 0, 2),
            )
        )

    content: RenderableType = Group(*blocks) if blocks else Text(" ")

    return Panel(
        content,
        title=pane.title,
        box=rich.box.HEAVY if pane.marked else rich.box.ROUNDED,
        padding=(0, 1),
        expand=True,
        title_align="left",
    )


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

    inline_sequence = format_inline_sequence(normalized)
    if inline_sequence is not None:
        return inline_sequence

    return pformat(normalized, compact=True, sort_dicts=False, width=52)


def format_inline_sequence(value: Any) -> str | None:
    if not isinstance(value, list):
        return None

    if not all(item is None or isinstance(item, (str, int, float, bool)) for item in value):
        return None

    return "[" + ", ".join(str(item) for item in value) + "]"


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
    embedder = module.nodes[address].embedder
    counters: dict[str, "Counter"] = {}

    if hasattr(embedder, "counter"):
        counters["counter"] = embedder.counter

    if hasattr(embedder, "counters"):
        counters |= dict(embedder.counters.items())

    if not counters:
        return

    for name, counter in counters.items():
        title = "counter" if name == "counter" else f"counters.{name}"
        pane.add_section(title, str(counter))

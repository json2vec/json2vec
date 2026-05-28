from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import rich.box
import torch
from rich.console import Console, Group, RenderableType
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from json2vec.structs.tree import Address, Leaf, Node
from json2vec.tensorfields.base import TENSORFIELDS

if TYPE_CHECKING:
    from json2vec.architecture.root import Model
    from json2vec.structs.experiment import Hyperparameters
    from json2vec.tensorfields.shared.counter import Counter

PlotMode = Literal["schema", "state", "flow", "debug"]
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
    module: "Model",
    address: Address | str | None = None,
    detail: bool = False,
    out: str | Path | None = None,
    mode: PlotMode = "schema",
) -> None:
    """Print a Rich model visualization and optionally write it as text."""
    renderable = build_plot(module=module, address=address, detail=detail, mode=mode)
    Console(width=PLOT_WIDTH).print(renderable)

    if out is None:
        return

    recorder = Console(file=io.StringIO(), record=True, width=PLOT_WIDTH, force_jupyter=False)
    recorder.print(renderable)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(recorder.export_text(clear=False), encoding="utf-8")


def build_plot(
    module: "Model",
    address: Address | str | None,
    detail: bool,
    mode: PlotMode,
) -> RenderableType:
    match mode:
        case "schema":
            return render_schema_plot(module=module, address=address, detail=detail, state_focus=False)
        case "state":
            return render_schema_plot(module=module, address=address, detail=True, state_focus=True)
        case "flow":
            return render_flow_plot(module=module, address=address)
        case "debug":
            return render_debug_plot(module=module, address=address, detail=detail)
        case _:
            raise ValueError("plot mode must be one of: schema, state, flow, debug")


def render_schema_plot(
    module: "Model",
    address: Address | str | None,
    detail: bool,
    state_focus: bool,
) -> RenderableType:
    hyperparameters = module.hyperparameters
    root = hyperparameters.fields if address is None else resolve_node(hyperparameters=hyperparameters, address=address)
    title = "State" if state_focus else "Schema"

    tree = Tree(render_node_label(module=module, node=root, state_focus=state_focus), guide_style="dim")
    append_schema_children(tree=tree, module=module, node=root, detail=detail or state_focus, state_focus=state_focus)

    return Group(
        Text(title, style="bold dim"),
        tree,
    )


def render_flow_plot(module: "Model", address: Address | str | None) -> RenderableType:
    hyperparameters = module.hyperparameters
    root = hyperparameters.fields if address is None else resolve_node(hyperparameters=hyperparameters, address=address)
    fields = [node for node in root.descendants if isinstance(node, Leaf) and node.active]
    target_count = sum(1 for node in fields if node.address in hyperparameters.target)
    embed_count = sum(1 for node in [root, *root.descendants] if node.address in hyperparameters.embed)

    table = Table(box=rich.box.ROUNDED, expand=True)
    table.add_column("Step", style="bold")
    table.add_column("What Happens")
    table.add_column("Count", justify="right")
    table.add_row("JSON", "Raw nested records enter with the shape described by the schema.", "")
    table.add_row("Tensorfields", "Typed requests read values with JMESPath queries.", str(len(fields)))
    table.add_row(
        "Encoders", "Array nodes pool child embeddings into parent contexts.", str(len(hyperparameters.arrays))
    )
    table.add_row("Targets", "Target fields produce supervised predictions.", str(target_count))
    table.add_row("Embeddings", "Selected nodes expose reusable embeddings.", str(embed_count))

    return Group(Text(f"Flow from {root.address or root.name}", style="bold dim"), table)


def append_schema_children(
    tree: Tree,
    module: "Model",
    node: Node,
    detail: bool,
    state_focus: bool,
) -> None:
    if detail:
        append_detail_sections(tree=tree, module=module, node=node)

    for child in getattr(node, "children", ()):
        child_tree = tree.add(render_node_label(module=module, node=child, state_focus=state_focus))
        append_schema_children(tree=child_tree, module=module, node=child, detail=detail, state_focus=state_focus)


def render_node_label(module: "Model", node: Node, state_focus: bool) -> RenderableType:
    heading = Text()
    heading.append(node.name, style="bold")
    heading.append(" ")
    heading.append(f"[{node.type}]", style=type_style(node.type))

    for role in node_roles(module=module, node=node):
        heading.append(" ")
        heading.append(role, style=role_style(role))

    if node.address in module.nodes and node is not module.hyperparameters.fields:
        heading.append(" ")
        heading.append(f"{format_compact_number(parameter_count(module.nodes[node.address]))} params", style="dim")

    address = str(node.address) or node.name
    meta = render_metadata_line(module=module, node=node, state_focus=state_focus)
    lines: list[RenderableType] = [heading, Text(address, style="dim")]
    if meta.plain:
        lines.append(meta)

    return Group(*lines)


def render_metadata_line(module: "Model", node: Node, state_focus: bool) -> Text:
    values = schema_node_values(module=module, node=node)
    keys = node_metadata_keys(node=node, values=values, state_focus=state_focus)
    text = Text(style="dim")
    first = True
    for key in keys:
        if key not in values or should_hide_metadata(key, values[key]):
            continue

        if not first:
            text.append("  ")
        text.append(f"{key}=")
        text.append(format_metadata_value(values[key]), style="cyan")
        first = False

    return text


def schema_node_values(module: "Model", node: Node) -> dict[str, Any]:
    values = node.model_dump(mode="python", exclude={"fields", "type", "name"}, exclude_none=True)

    if node is not module.hyperparameters.fields:
        return values

    hyperparameters = module.hyperparameters
    return {
        "d_model": hyperparameters.d_model,
        **values,
        "batch_size": module.batch_size,
        "parameters": parameter_count(module),
        "arrays": len(hyperparameters.arrays),
        "fields": len(hyperparameters.active_requests),
        "targets": len(hyperparameters.target),
        "embeds": len(hyperparameters.embed),
    }


def append_detail_sections(tree: Tree, module: "Model", node: Node) -> None:
    sections = collect_detail_sections(module=module, node=node)
    description = getattr(node, "description", None)
    if description:
        sections = {"description": description} | sections

    if not sections:
        return

    details = tree.add(Text("details", style="dim bold"))
    for title, values in sections.items():
        details.add(
            Group(
                Text(title, style="dim bold"),
                Text(format_detail_value(values), style="dim"),
            )
        )


def node_roles(module: "Model", node: Node) -> list[str]:
    roles: list[str] = []
    if isinstance(node, Leaf) and not node.active:
        roles.append("inactive")
    if node.address in module.hyperparameters.target:
        roles.append("target")
    if node.address in module.hyperparameters.embed:
        roles.append("embed")
    return roles


def type_style(node_type: str) -> str:
    return {
        "array": "blue",
        "category": "magenta",
        "number": "green",
        "set": "cyan",
        "entity": "yellow",
        "text": "bright_blue",
        "vector": "bright_magenta",
    }.get(node_type, "white")


def role_style(role: str) -> str:
    return {
        "inactive": "bold red",
        "target": "bold yellow",
        "embed": "bold green",
    }.get(role, "bold")


def node_metadata_keys(node: Node, values: dict[str, Any], state_focus: bool) -> list[str]:
    if "d_model" in values and not isinstance(node, Leaf):
        preferred = [
            "d_model",
            "attention",
            "max_length",
            "n_outputs",
            "n_layers",
            "n_heads",
            "batch_size",
            "parameters",
            "arrays",
            "fields",
            "targets",
            "embeds",
        ]
    elif state_focus:
        preferred = ["query", "max_vocab_size", "topk", "p_mask", "p_prune", "weight"]
    elif isinstance(node, Leaf):
        preferred = ["query", "pooling", "max_vocab_size", "topk", "objective", "weight"]
    else:
        preferred = ["attention", "max_length", "n_outputs", "n_layers", "n_heads"]

    remaining = [key for key in values if key not in preferred]
    return preferred + remaining


def should_hide_metadata(key: str, value: Any) -> bool:
    return (key == "active" and value is True) or (key == "embed" and value is False) or key == "description"


def format_metadata_value(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return format_compact_number(value)

    rendered = format_value(value)
    return truncate(rendered.replace("\n", " "), width=82)


def format_detail_value(value: Any) -> str:
    summarized = summarize_value(normalize_value(value))
    return "\n".join(format_detail_lines(summarized))


def format_detail_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent

    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{prefix}{key}:")
                lines.extend(format_detail_lines(item, indent=indent + 2))
                continue

            lines.append(f"{prefix}{key}: {format_detail_inline(item)}")
        return lines

    if isinstance(value, str) and "\n" in value:
        return [prefix + truncate(line, width=100) for line in value.splitlines()]

    return [prefix + format_detail_inline(value)]


def format_detail_inline(value: Any) -> str:
    if isinstance(value, str):
        return truncate(value, width=100)

    if isinstance(value, list):
        return truncate(
            format_inline_sequence(value) or pformat(value, compact=True, sort_dicts=False, width=88), width=100
        )

    return truncate(pformat(value, compact=True, sort_dicts=False, width=88), width=100)


def summarize_value(value: Any, max_items: int = 8) -> Any:
    if isinstance(value, str):
        return truncate_multiline(value, width=100)

    if isinstance(value, dict):
        return {key: summarize_value(item, max_items=max_items) for key, item in value.items()}

    if isinstance(value, list):
        if len(value) <= max_items:
            return [summarize_value(item, max_items=max_items) for item in value]
        return [summarize_value(item, max_items=max_items) for item in value[:max_items]] + [
            f"... {len(value) - max_items} more"
        ]

    return value


def collect_detail_sections(module: "Model", node: Node) -> dict[str, Any]:
    if not isinstance(node, Leaf):
        return {}
    if not node.active or node.address not in module.nodes:
        return {}

    pane = Pane(title=node.name)
    extension = TENSORFIELDS[node.type]
    extension.plot(module=module, address=node.address, branch=pane, detail=True)
    add_counter_details(pane=pane, module=module, address=node.address)
    return pane.sections


def render_debug_plot(
    module: "Model",
    address: Address | str | None = None,
    detail: bool = False,
) -> RenderableType:
    hyperparameters = module.hyperparameters

    def build(node: Node) -> Pane:
        values: dict[str, Any] = {}

        if node.address and node.address != node.name:
            values["address"] = node.address

        values |= schema_node_values(module=module, node=node)
        if node.address in module.nodes and node is not hyperparameters.fields:
            values |= parameter_counts(module.nodes[node.address])

        pane = Pane(
            title=f"{node.name} ({node.type})",
            marked=node.address in hyperparameters.target or node.address in hyperparameters.embed,
            values=values,
            children=[build(child) for child in node.children],
        )

        if detail and isinstance(node, Leaf) and node.active and node.address in module.nodes:
            extension = TENSORFIELDS[node.type]
            extension.plot(module=module, address=node.address, branch=pane, detail=detail)
            add_counter_details(pane=pane, module=module, address=node.address)

        return pane

    if address is None:
        pane = build(hyperparameters.fields)
    else:
        pane = build(resolve_node(hyperparameters=hyperparameters, address=address))

    return render_pane(pane)


def parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    return {"parameters": parameter_count(module)}


def parameter_count(module: torch.nn.Module) -> int:
    parameters = list(module.parameters())
    return sum(parameter.numel() for parameter in parameters)


def format_compact_number(value: Any) -> str:
    if value is None:
        return ""

    if not isinstance(value, int):
        return str(value)

    return f"{value:,}"


def resolve_node(hyperparameters: "Hyperparameters", address: Address | str) -> Node:
    key = Address(str(address))
    leaves = {node.address: node for node in hyperparameters.descendants if isinstance(node, Leaf)}
    nodes: dict[Address, Node] = hyperparameters.arrays | leaves

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


def truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)] + "..."


def truncate_multiline(value: str, width: int) -> str:
    return "\n".join(truncate(line, width=width) for line in value.splitlines())


def add_counter_details(
    pane: Pane,
    module: "Model",
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

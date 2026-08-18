"""Structured schema nodes that group tensorfield requests."""

from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, TypeAlias, Union

import pydantic
from rich.console import Console, Group
from rich.text import Text
from rich.tree import Tree

from relflow.structs.enums import AttentionMode, Overflow
from relflow.structs.tree import Address, Leaf, Node, Rate
from relflow.tensorfields import extensions as _extensions  # noqa: F401
from relflow.tensorfields.base import TENSORFIELDS

if TYPE_CHECKING:
    RequestTypes: TypeAlias = Any
else:
    RequestTypes: TypeAlias = Annotated[
        Union[tuple([tensorfield.Request for tensorfield in TENSORFIELDS.values()])],
        pydantic.Field(discriminator="type"),
    ]


Dropout: TypeAlias = Rate
_RICH_MASK_LIMIT = 4


class Mask(pydantic.BaseModel):
    """Structured masking policy attached to a `Branch`."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, hide_input_in_errors=True)

    name: str | None = None
    rate: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] | None = None
    count: Annotated[int, pydantic.Field(ge=0)] | None = None
    window: Annotated[int, pydantic.Field(gt=0)] | None = None
    branch: bool = False
    start: bool = False
    offset: Annotated[int, pydantic.Field(ge=0)] = 0
    exclude: tuple[Address, ...] = pydantic.Field(default_factory=tuple)

    def __rich_repr__(self):
        if self.name is not None:
            yield "name", Node._rich_value(self.name)
        if self.rate is not None:
            yield "rate", self.rate
        if self.count is not None:
            yield "count", self.count
        if self.window is not None:
            yield "window", self.window
        yield "extent", "capacity" if self.branch else "occupied"
        yield "edge", "start" if self.start else "end"
        if self.offset:
            yield "offset", self.offset
        if self.exclude:
            yield "exclude", Node._rich_value(self.exclude)

    @pydantic.model_validator(mode="after")
    def check_rate_or_count(self):
        if (self.rate is None) == (self.count is None):
            raise ValueError("Mask requires exactly one of rate or count")

        return self

    @pydantic.field_validator("exclude", mode="before")
    @classmethod
    def normalize_exclude(cls, value: Any) -> tuple[Address, ...]:
        if value is None:
            return ()

        if isinstance(value, str):
            return (Address(value),)

        if isinstance(value, (list, tuple)):
            normalized: list[Address] = []
            for item in value:
                if not isinstance(item, str):
                    raise TypeError(f"Mask.exclude entries must be Address strings; got {type(item).__name__}")
                normalized.append(Address(item))
            return tuple(normalized)

        raise TypeError(f"Mask.exclude must be an Address, a tuple of Addresses, or None; got {type(value).__name__}")


class Branch(Node):
    """Repeated nested object group in a `relflow` schema.

    Positional children are treated as fields inside the branch.
    """

    name: str | None = None
    type: Annotated[Literal["branch"], pydantic.Field(default="branch")] = "branch"
    attention: AttentionMode = AttentionMode.mha
    length: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    overflow: Overflow = Overflow.head
    n_linear: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    n_layers: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    masks: list[Mask] = pydantic.Field(default_factory=list)
    fields: list[Self | RequestTypes | pydantic.InstanceOf[Leaf]] = pydantic.Field(default_factory=list)

    def __init__(self, *children: Self | RequestTypes | Leaf, **data):
        if "max_length" in data:
            raise ValueError("max_length was removed; use length")

        if data.get("type") not in (None, "branch"):
            super().__init__(**data)
            return

        config_names = set(type(self).model_fields) | {"mask"}
        keyword_children = {key: data.pop(key) for key in tuple(data) if key not in config_names}

        if children:
            if "fields" in data:
                raise TypeError("branch children were provided both positionally and by keyword")
            data["fields"] = list(children)

        if keyword_children:
            from relflow.structs.experiment import bind_tree_field

            data.setdefault("fields", [])
            data["fields"].extend(bind_tree_field(key, value) for key, value in keyword_children.items())

        super().__init__(**data)

    @pydantic.model_validator(mode="before")
    @classmethod
    def normalize_mask_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        values = dict(data)
        if "max_length" in values:
            raise ValueError("max_length was removed; use length")

        mask = values.pop("mask", None)
        if mask is None:
            return values

        if "masks" in values:
            raise ValueError("pass either mask or masks, not both")

        values["masks"] = [mask]
        return values

    def model_post_init(self, __context):
        for field in self.fields:
            field.parent: Self = self

    @pydantic.model_validator(mode="after")
    def check_unique_child_names(self):
        seen: set[str] = set()
        for field in self.fields:
            if field.name in seen:
                raise ValueError(f"duplicate field name: {field.name}")
            seen.add(field.name)

        return self

    def post_bind_validate(self):
        if len(self.masks) == 0:
            return None

        is_root = getattr(getattr(self, "parent", None), "type", None) == "schema"
        if is_root:
            raise ValueError("Mask on the generated root branch is not supported")

        active_leaves = [
            descendant
            for descendant in getattr(self, "descendants", ())
            if isinstance(descendant, Leaf) and getattr(descendant, "active", True)
        ]
        if not active_leaves:
            raise ValueError(f"branch '{self.address}' has masks but no active descendant leaves")

        prefix = f"{self.address}/"
        active_addresses = {str(leaf.address) for leaf in active_leaves}
        names = [mask.name for mask in self.masks if mask.name is not None]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"branch '{self.address}' has duplicate mask name(s): {duplicates}")

        for mask in self.masks:
            if mask.offset >= self.length:
                raise ValueError(f"branch '{self.address}' mask offset must be less than length={self.length}")

            excluded = {
                address if address.startswith(prefix) else f"{prefix}{address}" for address in map(str, mask.exclude)
            }
            if active_addresses <= excluded:
                label = f" '{mask.name}'" if mask.name is not None else ""
                raise ValueError(f"branch '{self.address}' mask{label} excludes every active descendant leaf")

        return None

    def __rich_repr__(self):
        yield "name", self.name
        yield "type", self._rich_type
        yield "length", self.length, 1
        yield "fields", len(self.fields)
        yield "embed", self.embed, False

    def _rich_heading(self, console: Console, *, name: str | None = None) -> Text:
        is_root = self._rich_type == "root"
        attributes = ("attention", "n_layers", "n_heads", "n_linear", "dropout")
        if not is_root:
            attributes = ("length", "overflow", *attributes)

        heading = self._rich_identity(console, name=name)
        if self.embed:
            heading.append(" ")
            heading.append("embed", style=self._rich_style(console, "relflow.info"))
        for attribute in attributes:
            value = getattr(self, attribute, None)
            if value is None:
                continue
            heading.append(" ")
            heading.append(f"{attribute}=", style=self._rich_style(console, "relflow.dim"))
            heading.append(self._rich_value(value), style=self._rich_style(console, "relflow.info"))
        if self.masks:
            heading.append(" ")
            heading.append("masks=", style=self._rich_style(console, "relflow.dim"))
            heading.append(str(len(self.masks)), style=self._rich_style(console, "relflow.info"))

        return heading

    def _rich_renderable(self, console: Console):
        lines: list[Text] = [self._rich_heading(console)]

        if self.description is not None:
            description = Text("  ")
            description.append(self._rich_value(self.description), style=self._rich_style(console, "relflow.dim"))
            lines.append(description)

        for mask in self.masks[:_RICH_MASK_LIMIT]:
            text = Text("  mask ", style=self._rich_style(console, "relflow.dim"))
            text.append(
                self._rich_value(mask.name) if mask.name is not None else "<unnamed>",
                style=self._rich_style(console, self.RICH_NAME_STYLE),
            )
            for name, value in (
                ("rate", mask.rate),
                ("count", mask.count),
                ("window", mask.window),
                ("extent", "capacity" if mask.branch else "occupied"),
                ("edge", "start" if mask.start else "end"),
                ("offset", mask.offset if mask.offset else None),
                ("exclude", mask.exclude if mask.exclude else None),
            ):
                if value is None:
                    continue
                text.append(" ")
                text.append(f"{name}=", style=self._rich_style(console, "relflow.dim"))
                text.append(self._rich_value(value), style=self._rich_style(console, "relflow.info"))
            lines.append(text)

        omitted = len(self.masks) - min(len(self.masks), _RICH_MASK_LIMIT)
        if omitted:
            text = Text(f"  … +{omitted} masks", style=self._rich_style(console, "relflow.dim"))
            lines.append(text)

        return Group(*lines)

    def _rich_add_to_tree(self, tree: Tree, console: Console) -> Tree:
        branch = tree.add(self._rich_renderable(console))
        self._rich_add_children(branch, console)
        return branch

    def _rich_add_children(self, tree: Tree, console: Console) -> None:
        for child in self.fields:
            if isinstance(child, Branch):
                child._rich_add_to_tree(tree, console)
            else:
                tree.add(child._rich_renderable(console))

    def _rich_tree(self, console: Console) -> Tree:
        tree = Tree(
            self._rich_renderable(console),
            guide_style=self._rich_style(console, self.RICH_TREE_STYLE),
        )
        self._rich_add_children(tree, console)
        return tree

    def _rich_selection_label(self, console: Console) -> Text:
        address = self._rich_address()
        return self._rich_heading(console, name=address or self._rich_name)

    def __rich_console__(self, console, options):
        yield self._rich_tree(console)

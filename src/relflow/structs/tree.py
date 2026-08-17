"""Schema tree node primitives used by models and tensorfields."""

import functools
import re
from abc import ABC
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, TypeAlias

import jmespath
import pydantic
from anytree import NodeMixin
from jmespath.exceptions import JMESPathError
from rich.console import Console, Group
from rich.text import Text

from relflow.rich import MimeBundleDisplay, render_html, render_text
from relflow.structs.enums import Overflow

Rate: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]
PruneRate: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, le=1.0)]


class Renderable(ABC):
    """Base class for objects rendered consistently through Rich."""

    RICH_NAME_STYLE: ClassVar[str] = "relflow.name"
    RICH_TYPE_STYLE: ClassVar[str] = "relflow.type"
    RICH_TREE_STYLE: ClassVar[str] = "relflow.dim"

    _RICH_STYLE_FALLBACKS: ClassVar[dict[str, str]] = {
        "relflow.info": "cyan",
        "relflow.warning": "yellow",
        "relflow.error": "bold red",
        "relflow.name": "bold white on #1f2937",
        "relflow.type": "bold yellow on #3f3f46",
        "relflow.dim": "dim",
    }

    @classmethod
    def _rich_style(cls, console: Console, name: str):
        """Resolve a semantic style on both RelFlow and ordinary Rich consoles."""
        return console.get_style(name, default=cls._RICH_STYLE_FALLBACKS.get(name, "none"))

    def __rich_console__(self, console, options):
        yield Text(repr(self))

    def __str__(self) -> str:
        return render_text(self)

    def _display_(self) -> MimeBundleDisplay:
        """Give Marimo a MIME bundle before type-specific formatters run."""
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

    def _repr_html_(self):
        return render_html(self)

    def _mime_(self):
        return "text/html", self._repr_html_()


class Selection(list, Renderable):
    """List-like selection result with readable Rich and pprint output."""

    __rich_repr__: ClassVar[None] = None

    def __repr__(self) -> str:
        return str(self)

    def __rich_console__(self, console, options):
        if not self:
            yield Text("[]")
            return

        dim = self._rich_style(console, self.RICH_TREE_STYLE)
        yield Text("[", style=dim)
        for index, node in enumerate(self):
            text = Text("  ")
            text.append_text(node._rich_selection_label(console))
            if index < len(self) - 1:
                text.append(",", style=dim)
            yield text
        yield Text("]", style=dim)


class Address(str):
    """Slash-delimited stable path to a schema node."""

    def __new__(cls, *parts: str) -> "Address":
        if len(parts) == 0:
            value = ""
        elif len(parts) == 1:
            value = parts[0]
        else:
            value = "/".join(parts)

        if not isinstance(value, str):
            raise TypeError("Address parts must be strings")

        return str.__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any):
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


class Node(NodeMixin, Renderable, pydantic.BaseModel):
    """Base schema tree node shared by branches and tensorfield requests."""

    model_config = pydantic.ConfigDict(extra="forbid")

    name: str | None = None
    type: str
    description: str | None = None
    embed: bool = False
    n_heads: Annotated[int, pydantic.Field(gt=0, default=4)] = 4
    dropout: Rate | None = None

    @classmethod
    def sanitize_name(cls, value: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z_-]+", "_", value).strip("_")
        return sanitized or "field"

    @functools.cached_property
    def address(self) -> Address:
        return Address(*(node.name for node in self.path[1:]))

    @functools.cached_property
    def heritage(self) -> list[Address]:
        return [node.address for node in self.path[1:]]

    @pydantic.model_validator(mode="after")
    def check_node_name(self):
        if self.name is None:
            return self

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

        if not all(c.isalnum() or c in "_-" for c in self.name):
            raise ValueError("name may contain only letters, digits, '_' or '-'")

        return self

    @pydantic.model_validator(mode="after")
    def check_n_heads_is_even(self):
        if not isinstance(self.n_heads, int):
            raise ValueError("n_heads must be an integer")

        if self.n_heads % 2 != 0:
            raise ValueError("n_heads must be even")

        return self

    @pydantic.field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: str | None):
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError("description must be a string when provided")

        normalized = value.strip()
        return normalized or None

    def post_bind_validate(self):
        return None

    @property
    def _rich_name(self) -> str:
        return self.name if self.name is not None else "<unnamed>"

    @property
    def _rich_type(self) -> str:
        is_root = getattr(getattr(self, "parent", None), "type", None) == "schema"
        return "root" if is_root else self.type

    @staticmethod
    def _rich_value(value: Any) -> str:
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        if isinstance(value, Enum):
            return str(value.value)
        return str(value)

    def _rich_identity(self, console: Console, *, name: str | None = None) -> Text:
        heading = Text()
        heading.append(name or self._rich_name, style=self._rich_style(console, self.RICH_NAME_STYLE))
        heading.append(" ")
        heading.append(f"[{self._rich_type}]", style=self._rich_style(console, self.RICH_TYPE_STYLE))
        return heading

    def _rich_renderable(self, console: Console):
        heading = self._rich_identity(console)
        if self.description is None:
            return heading

        description = Text("  ")
        description.append(self.description, style=self._rich_style(console, "relflow.dim"))
        return Group(heading, description)

    def _rich_address(self) -> str:
        """Return the display address without populating the cached property."""
        parts = tuple(node.name for node in self.path[1:])
        if not parts or any(not isinstance(part, str) or not part for part in parts):
            return ""
        return str(Address(*parts))

    def _rich_selection_label(self, console: Console) -> Text:
        address = self._rich_address()
        return self._rich_identity(console, name=address or self._rich_name)

    def __rich_repr__(self):
        """Compact nested representation; direct display uses ``__rich_console__``."""
        yield "name", self.name
        yield "type", self.type
        if self.description is not None:
            yield "description", self.description


class Leaf(Node):
    """Base tensorfield request node.

    Concrete tensorfield constructors such as `Number` and `Category` inherit
    from this class through their registered request models.
    """

    model_config = pydantic.ConfigDict(extra="allow")

    active: bool = True
    embed: bool = False
    name: str | None = None
    type: str
    query: str | None = None
    nullable: bool = True
    pooling: Literal["query", "mean"] = "query"
    weight: Annotated[float, pydantic.Field(gt=0.0, default=1.0)] = 1.0
    p_mask: Rate = 0.0
    p_prune: PruneRate = 0.0
    n_linear: Annotated[int, pydantic.Field(gt=0, default=1)] = 1

    def __init__(self, name: str | None = None, **data: Any):
        if name is not None:
            if "name" in data:
                raise TypeError("name was provided both positionally and by keyword")
            data["name"] = name
        super().__init__(**data)

    @property
    def target(self) -> bool:
        return self.p_prune == 1.0

    @target.setter
    def target(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("target must be a boolean")

        self.p_prune = 1.0 if value else 0.0
        self.model_fields_set.add("p_prune")

    @pydantic.model_validator(mode="before")
    @classmethod
    def resolve_role_shorthands(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        values = dict(data)
        target = values.pop("target", None)

        if target is None:
            return values

        if not isinstance(target, bool):
            raise ValueError("target must be a boolean")

        if target:
            if values.get("p_prune") not in (None, 1.0):
                raise ValueError("target=True is shorthand for p_prune=1.0")
            values["p_prune"] = 1.0
        else:
            if values.get("p_prune") not in (None, 0.0):
                raise ValueError("target=False is shorthand for p_prune=0.0")
            values["p_prune"] = 0.0

        return values

    @pydantic.model_validator(mode="before")
    @classmethod
    def merge_constructor_kwargs(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        values = dict(data)
        kwargs = values.pop("kwargs", None)

        if kwargs is None:
            return values
        if not isinstance(kwargs, Mapping):
            raise TypeError("kwargs must be a mapping")

        for key, value in kwargs.items():
            values.setdefault(key, value)

        return values

    @pydantic.field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        from relflow.tensorfields import extensions as _extensions  # noqa: F401
        from relflow.tensorfields.base import TENSORFIELDS

        if value not in TENSORFIELDS:
            raise ValueError(f"unknown tensor field type: {value}")

        return value

    @pydantic.model_validator(mode="after")
    def check_jmespath_query(self):
        if self.query is None:
            return self

        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")

        try:
            jmespath.compile(self.query)
        except JMESPathError as e:
            raise ValueError(f"invalid jmespath query: {e}") from e

        return self

    def post_bind_validate(self):
        if self.query is None:
            raise ValueError(f"request '{self.address}' must define query")

    def __rich_repr__(self):
        yield "name", self.name
        yield "type", self.type
        yield "active", self.active, True
        yield "embed", self.embed, False
        yield "target", self.target, False
        if self.query is not None:
            yield "query", self.query

    def _rich_heading(self, console: Console, *, name: str | None = None) -> Text:
        flags = ["active" if self.active else "inactive"]
        if self.embed:
            flags.append("embed")
        if self.target:
            flags.append("target")

        heading = self._rich_identity(console, name=name)
        for flag in flags:
            heading.append(" ")
            heading.append(
                flag,
                style={
                    "active": self._rich_style(console, "relflow.dim"),
                    "inactive": self._rich_style(console, "relflow.error"),
                    "embed": self._rich_style(console, "relflow.info"),
                    "target": self._rich_style(console, "relflow.warning"),
                }.get(flag, "bold"),
            )
        if self.query is not None:
            heading.append(" ")
            heading.append("query=", style=self._rich_style(console, "relflow.dim"))
            heading.append(self.query, style=self._rich_style(console, "relflow.info"))
        return heading

    def _rich_attributes(self, console: Console, names: tuple[str, ...]) -> Text:
        attributes = Text("  ")
        first = True
        for name in names:
            value = getattr(self, name, None)
            if value is None:
                continue
            if not first:
                attributes.append(" ")
            attributes.append(f"{name}=", style=self._rich_style(console, "relflow.dim"))
            attributes.append(self._rich_value(value), style=self._rich_style(console, "relflow.info"))
            first = False
        return attributes

    def _rich_renderable(self, console: Console):
        lines: list[Text] = [self._rich_heading(console)]

        if self.description is not None:
            description = Text("  ")
            description.append(self.description, style=self._rich_style(console, "relflow.dim"))
            lines.append(description)

        common_names = ("pooling", "weight", "p_mask", "p_prune", "n_heads", "n_linear", "dropout")
        common = self._rich_attributes(console, common_names)
        if not self.nullable:
            common.append(" ")
            common.append("nullable=", style=self._rich_style(console, "relflow.dim"))
            common.append("False", style=self._rich_style(console, "relflow.info"))
        if common.plain.strip():
            lines.append(common)

        excluded = {"name", "type", "description", "active", "embed", "query", "nullable", *common_names}
        specific = Text("  ")
        first = True
        for name, field in type(self).model_fields.items():
            if name in excluded:
                continue
            value = getattr(self, name, None)
            if value is None:
                continue
            if not first:
                specific.append(" ")
            label = str(field.serialization_alias or field.alias or name)
            specific.append(f"{label}=", style=self._rich_style(console, "relflow.dim"))
            specific.append(self._rich_value(value), style=self._rich_style(console, "relflow.info"))
            first = False
        if specific.plain.strip():
            lines.append(specific)

        return Group(*lines)

    def _rich_selection_label(self, console: Console) -> Text:
        address = self._rich_address()
        return self._rich_heading(console, name=address or self._rich_name)

    def __rich_console__(self, console, options):
        yield self._rich_renderable(console)

    @functools.cached_property
    def shape(self) -> tuple[int, ...]:
        out: list[int] = []

        for node in self.path:
            if node.type == "branch":
                out.append(node.length)

        return tuple(out)

    @functools.cached_property
    def overflows(self) -> tuple[Overflow, ...]:
        out: list[Overflow] = []

        for node in self.path:
            if node.type == "branch":
                out.append(node.overflow)

        return tuple(out)

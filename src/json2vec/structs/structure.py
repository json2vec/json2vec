"""Structured schema nodes that group tensorfield requests."""

from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, TypeAlias, Union

import pydantic
from rich.text import Text

from json2vec.structs.enums import AttentionMode, Overflow
from json2vec.structs.tree import Leaf, Node, Rate
from json2vec.tensorfields import extensions as _extensions  # noqa: F401
from json2vec.tensorfields.base import TENSORFIELDS

if TYPE_CHECKING:
    RequestTypes: TypeAlias = Any
else:
    RequestTypes: TypeAlias = Annotated[
        Union[tuple([tensorfield.Request for tensorfield in TENSORFIELDS.values()])],
        pydantic.Field(discriminator="type"),
    ]


Dropout: TypeAlias = Rate


class Array(Node):
    """Repeated nested object group in a `json2vec` schema.

    Positional children are treated as fields inside the array.
    """

    name: str
    type: Annotated[Literal["array"], pydantic.Field(default="array")] = "array"
    attention: AttentionMode = AttentionMode.mha
    max_length: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    overflow: Overflow = Overflow.head
    n_linear: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    n_layers: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    fields: list[Self | RequestTypes | pydantic.InstanceOf[Leaf]] = pydantic.Field(default_factory=list)

    def __init__(self, *children: Self | RequestTypes | Leaf, **data):
        if children:
            if "fields" in data:
                raise TypeError("array children were provided both positionally and by keyword")
            data["fields"] = list(children)

        super().__init__(**data)

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

    def __rich_console__(self, console, options):
        is_root = getattr(getattr(self, "parent", None), "type", None) == "hyperparameters"
        display_type = "root" if is_root else self.type
        attributes = ("attention", "n_layers", "n_heads", "n_linear", "dropout")
        if not is_root:
            attributes = ("max_length", "overflow", *attributes)

        heading = Text()
        heading.append(self.name, style=self.RICH_NAME_STYLE)
        heading.append(" ")
        heading.append(f"[{display_type}]", style=self.RICH_TYPE_STYLE)
        if self.embed:
            heading.append(" ")
            heading.append("embed", style="bold #065f46")
        for name in attributes:
            value = getattr(self, name, None)
            if value is None:
                continue
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            elif hasattr(value, "value"):
                value = value.value
            heading.append(" ")
            heading.append(f"{name}=", style="dim")
            heading.append(str(value), style="cyan")

        yield heading

        for index, child in enumerate(self.fields):
            connector = "`-- " if index == len(self.fields) - 1 else "|-- "
            continuation = "    " if index == len(self.fields) - 1 else "|   "
            lines = list(child.__rich_console__(console, options))
            if not lines:
                continue
            first = Text()
            first.append(connector, style=self.RICH_TREE_STYLE)
            if isinstance(lines[0], Text):
                first.append_text(lines[0])
            else:
                first.append(str(lines[0]))
            yield first
            for line in lines[1:]:
                nested = Text()
                nested.append(continuation, style=self.RICH_TREE_STYLE)
                if isinstance(line, Text):
                    nested.append_text(line)
                else:
                    nested.append(str(line))
                yield nested

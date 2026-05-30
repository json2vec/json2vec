"""Structured schema nodes that group tensorfield requests."""

from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, TypeAlias, Union

import pydantic

from json2vec.structs.enums import AttentionMode
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
    """Repeated nested object group in a JSON2Vec schema.

    Positional children are treated as fields inside the array.
    """

    name: str
    type: Annotated[Literal["array"], pydantic.Field(default="array")] = "array"
    attention: AttentionMode = AttentionMode.mha
    max_length: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
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

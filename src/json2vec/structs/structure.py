from typing import Annotated, Literal, Self, TypeAlias, Union

import pydantic

from json2vec.structs.tree import Node
from json2vec.tensorfields import extensions as _extensions  # noqa: F401
from json2vec.tensorfields.base import TENSORFIELDS

RequestTypes: TypeAlias = Annotated[
    Union[tuple([tensorfield.Request for tensorfield in TENSORFIELDS.values()])],
    pydantic.Field(discriminator="type"),
]


Dropout: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]


class Array(Node):
    name: str
    type: Annotated[Literal["array"], pydantic.Field(default="array")] = "array"
    attention: Literal["mha", "gqa", "mqa", "none"] = "mha"
    max_length: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    n_outputs: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    n_linear: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    n_layers: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    dropout: Dropout | None = None
    fields: list[Self | RequestTypes] = pydantic.Field(default_factory=list)

    def model_post_init(self, __context):
        for field in self.fields:
            field.parent: Self = self

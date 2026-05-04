import functools
from typing import Annotated, Literal, Self, TypeAlias, Union

import pydantic
from anytree import LevelOrderGroupIter, RenderTree

from json2vec.structs.tree import Address, Node
from json2vec.tensorfields.base import TENSORFIELDS

RequestTypes: TypeAlias = Annotated[
    Union[tuple([tensorfield.Request for tensorfield in TENSORFIELDS.values()])],
    pydantic.Field(discriminator="type"),
]


class Context(Node):
    name: str
    type: Annotated[Literal["context"], pydantic.Field(default="context")]
    context_size: Annotated[int, pydantic.Field(gt=0, default=1)]
    n_outputs: Annotated[int, pydantic.Field(gt=0)]
    n_linear: Annotated[int, pydantic.Field(gt=0, default=1)]
    n_layers: Annotated[int, pydantic.Field(gt=0, default=1)]
    fields: list[Self | RequestTypes] = pydantic.Field(default_factory=list)

    def model_post_init(self, __context):
        for field in self.fields:
            field.parent: Self = self


class Structure(Node):
    name: str
    type: Literal["structure"] = "structure"
    d_model: Annotated[int, pydantic.Field(gt=0, default=128)]
    batch_size: Annotated[int, pydantic.Field(gt=0)]
    dropout: Annotated[float, pydantic.Field(gt=0.0, lt=1.0)]
    fields: Context

    def model_post_init(self, __context):
        self.fields.parent: Self = self
        for request in self.requests.values():
            request.post_bind_validate()

    @functools.cached_property
    def contexts(self) -> dict[Address, Context]:
        return {node.address: node for node in self.descendants if isinstance(node, Context)}

    @functools.cached_property
    def requests(self) -> dict[Address, RequestTypes]:
        return {node.address: node for node in self.descendants if not isinstance(node, Context)}

    @functools.cached_property
    def shapes(self) -> dict[Address, tuple[int, ...]]:
        return {request.address: request.shape for request in self.requests.values()}

    @functools.cached_property
    def depthwise(self) -> list[list[Address]]:
        out: list[list[Address]] = []
        for depth in LevelOrderGroupIter(self.fields):
            contexts = [node.address for node in depth if isinstance(node, Context)]
            if contexts:
                out.append(contexts)

        return out

    def __str__(self) -> str:
        lines: list[str] = []
        for pre, _, node in RenderTree(self):
            lines.append(f"{pre}{node.name} ({node.type})")

        return "\n".join(lines)

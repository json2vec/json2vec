import functools
from typing import Annotated, Literal, Self

import pydantic
from anytree import LevelOrderGroupIter, RenderTree

from json2vec.structs.structure import Array, Dropout, RequestTypes
from json2vec.structs.tree import Address, Node


class Hyperparameters(Node):
    model_config = pydantic.ConfigDict(extra="forbid")

    name: Literal["hyperparameters"] = pydantic.Field(default="hyperparameters", exclude=True)
    type: Literal["hyperparameters"] = pydantic.Field(default="hyperparameters", exclude=True)
    description: Literal[None] = pydantic.Field(default=None, exclude=True)
    n_heads: Literal[4] = pydantic.Field(default=4, exclude=True)
    d_model: Annotated[int, pydantic.Field(gt=0, default=128)]
    dropout: Dropout | None = None
    fields: Array

    target: list[Address] = pydantic.Field(default_factory=list)
    reset: list[Address] = pydantic.Field(default_factory=list)
    embed: list[Address] = pydantic.Field(default_factory=list)

    p_target: Annotated[float, pydantic.Field(ge=0.0, lt=1.0, default=0.0)]
    p_mask: Annotated[float, pydantic.Field(ge=0.0, lt=1.0, default=0.0)]

    def model_post_init(self, __context):
        self.fields.parent: Self = self
        for request in self.requests.values():
            request.post_bind_validate()

    @functools.cached_property
    def arrays(self) -> dict[Address, Array]:
        return {node.address: node for node in self.descendants if isinstance(node, Array)}

    @functools.cached_property
    def requests(self) -> dict[Address, RequestTypes]:
        return {node.address: node for node in self.descendants if not isinstance(node, Array)}

    @functools.cached_property
    def shapes(self) -> dict[Address, tuple[int, ...]]:
        return {request.address: request.shape for request in self.requests.values()}

    @functools.cached_property
    def depthwise(self) -> list[list[Address]]:
        out: list[list[Address]] = []
        for depth in LevelOrderGroupIter(self.fields):
            arrays = [node.address for node in depth if isinstance(node, Array)]
            if arrays:
                out.append(arrays)

        return out

    def resolved_dropout(self, address: Address) -> float:
        if address in self.arrays:
            node: Node | None = self.arrays[address]
        elif address in self.requests:
            node = self.requests[address].parent
        else:
            raise ValueError(f"address '{address}' not found in hyperparameters")

        while node is not None:
            dropout = getattr(node, "dropout", None)
            if dropout is not None:
                return float(dropout)
            node = getattr(node, "parent", None)

        return 0.0

    @pydantic.model_validator(mode="after")
    def check_overriden_fields(self):
        for attribute in ["target", "reset"]:
            for field in getattr(self, attribute, []):
                if field not in self.requests:
                    raise ValueError(f"{attribute} field '{field}' not found in hyperparameter requests")

        for attribute in ["embed"]:
            for field in getattr(self, attribute, []):
                if field not in self.arrays and field not in self.requests:
                    raise ValueError(f"{attribute} target '{field}' not found in hyperparameter arrays or requests")

        return self

    def __str__(self) -> str:
        lines: list[str] = []
        for pre, _, node in RenderTree(self):
            lines.append(f"{pre}{node.name} ({node.type})")

        return "\n".join(lines)

import functools
from typing import Annotated, Any, ClassVar, Literal, Self

import pydantic
from anytree import LevelOrderGroupIter, RenderTree

from json2vec.structs.structure import Array, RequestTypes
from json2vec.structs.tree import Address, Node


class Hyperparameters(Node):
    model_config = pydantic.ConfigDict(extra="forbid")

    name: Literal["hyperparameters"] = pydantic.Field(default="hyperparameters", exclude=True)
    type: Literal["hyperparameters"] = pydantic.Field(default="hyperparameters", exclude=True)
    description: Literal[None] = pydantic.Field(default=None, exclude=True)
    d_model: Annotated[int, pydantic.Field(gt=0, default=128)]
    fields: Array

    target: list[Address]|Address = pydantic.Field(default_factory=list)
    embed: list[Address]|Address = pydantic.Field(default_factory=list)

    dropout: ClassVar[None] = None
    p_mask: ClassVar[None] = None
    p_target: ClassVar[None] = None

    @pydantic.field_validator("target", "embed", mode="before")
    @classmethod
    def normalize_address_list(cls, value: Any):
        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        return value

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

    def resolved_dropout(self, address: Address | str) -> float:
        return self._resolved_rate(address, "dropout")

    def _node_at(self, address: Address | str) -> Node:
        if address in self.arrays:
            return self.arrays[Address(str(address))]

        if address in self.requests:
            return self.requests[Address(str(address))]

        raise ValueError(f"address '{address}' not found in hyperparameters")

    def _resolved_rate(self, address: Address | str, name: Literal["dropout", "p_mask", "p_target"]) -> float:
        node: Node | None = self._node_at(address)

        while node is not None:
            rate = getattr(node, name, None)
            if rate is not None:
                return float(rate)
            node = getattr(node, "parent", None)

        return 0.0

    def resolved_p_mask(self, address: Address | str) -> float:
        return self._resolved_rate(address, "p_mask")

    def resolved_p_target(self, address: Address | str) -> float:
        return self._resolved_rate(address, "p_target")

    @pydantic.model_validator(mode="after")
    def check_overriden_fields(self):
        for field in self.target:
            if field not in self.requests:
                raise ValueError(f"target field '{field}' not found in hyperparameter requests")

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

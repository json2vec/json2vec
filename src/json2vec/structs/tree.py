import functools
from typing import Annotated, NewType

import jmespath
import pydantic
from anytree import NodeMixin
from jmespath.exceptions import JMESPathError

Address = NewType("Address", str)

class Node(NodeMixin, pydantic.BaseModel):
    name: str
    type: str
    description: str | None = None
    n_heads: Annotated[int, pydantic.Field(gt=0, default=4)]

    @functools.cached_property
    def address(self) -> Address:
        return "/".join(node.name for node in self.path[1:])

    @functools.cached_property
    def heritage(self) -> list[Address]:
        return [node.address for node in self.path[1:]]

    @pydantic.model_validator(mode="after")
    def check_node_name(self):

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


class Leaf(Node):
    name: str
    type: str
    query: str
    weight: Annotated[float, pydantic.Field(gt=0.0, default=1.0)]
    n_linear: Annotated[int, pydantic.Field(gt=0, default=1)]


    @pydantic.model_validator(mode="after")
    def check_jmespath_query(self):

        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be a non-empty string")

        try:
            jmespath.compile(self.query)
        except JMESPathError as e:
            raise ValueError(f"invalid jmespath query: {e}") from e

        return self

    @functools.cached_property
    def shape(self) -> tuple[int, ...]:
        out: list[int] = []

        for node in self.path:
            if node.type == "context":
                out.append(node.context_size)

        return tuple(out)

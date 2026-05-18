import functools
from typing import Annotated, Any, Literal, TypeAlias

import jmespath
import pydantic
from anytree import NodeMixin
from jmespath.exceptions import JMESPathError

Rate: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]


class Address(str):
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


class Node(NodeMixin, pydantic.BaseModel):
    name: str
    type: str
    description: str | None = None
    n_heads: Annotated[int, pydantic.Field(gt=0, default=4)] = 4
    dropout: Rate | None = None
    p_mask: Rate | None = None
    p_target: Rate | None = None

    @functools.cached_property
    def address(self) -> Address:
        return Address(*(node.name for node in self.path[1:]))

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
    pooling: Literal["query", "mean"] = "query"
    weight: Annotated[float, pydantic.Field(gt=0.0, default=1.0)] = 1.0
    n_linear: Annotated[int, pydantic.Field(gt=0, default=1)] = 1


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
            if node.type == "array":
                out.append(node.max_length)

        return tuple(out)

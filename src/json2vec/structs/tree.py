"""Schema tree node primitives used by models and tensorfields."""

import functools
from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypeAlias

import jmespath
import pydantic
from anytree import NodeMixin
from jmespath.exceptions import JMESPathError

Rate: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]
PruneRate: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, le=1.0)]


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


class Node(NodeMixin, pydantic.BaseModel):
    """Base schema tree node shared by arrays and tensorfield requests."""

    model_config = pydantic.ConfigDict(extra="allow")

    name: str
    type: str
    description: str | None = None
    embed: bool = False
    n_heads: Annotated[int, pydantic.Field(gt=0, default=4)] = 4
    dropout: Rate | None = None
    p_mask: Rate | None = None
    p_prune: PruneRate | None = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def resolve_role_shorthands(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        if "p_prune" not in cls.model_fields:
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

        return values

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
    """Base tensorfield request node.

    Concrete tensorfield constructors such as `Number` and `Category` inherit
    from this class through their registered request models.
    """

    embed: bool = False
    name: str
    type: str
    query: str | None = None
    pooling: Literal["query", "mean"] = "query"
    weight: Annotated[float, pydantic.Field(gt=0.0, default=1.0)] = 1.0
    n_linear: Annotated[int, pydantic.Field(gt=0, default=1)] = 1

    def __init__(self, name: str | None = None, **data: Any):
        if name is not None:
            if "name" in data:
                raise TypeError("name was provided both positionally and by keyword")
            data["name"] = name
        super().__init__(**data)

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
        from json2vec.tensorfields import extensions as _extensions  # noqa: F401
        from json2vec.tensorfields.base import TENSORFIELDS

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

    @functools.cached_property
    def shape(self) -> tuple[int, ...]:
        out: list[int] = []

        for node in self.path:
            if node.type == "array":
                out.append(node.max_length)

        return tuple(out)

"""Structured schema nodes that group tensorfield requests."""

from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, TypeAlias, Union

import pydantic
from rich.text import Text

from relflow.structs.enums import AttentionMode, Overflow
from relflow.structs.pooling import Attention, Mean, PoolingConfig
from relflow.structs.reference import Reference
from relflow.structs.tree import Leaf, Node, Rate
from relflow.tensorfields import extensions as _extensions  # noqa: F401
from relflow.tensorfields.base import TENSORFIELDS

if TYPE_CHECKING:
    RequestTypes: TypeAlias = Any
else:
    RequestTypes: TypeAlias = Annotated[
        Union[tuple([tensorfield.Request for tensorfield in TENSORFIELDS.values()])],
        pydantic.Field(discriminator="type"),
    ]


Dropout: TypeAlias = Rate


class Mask(pydantic.BaseModel):
    """Structured masking policy attached to a `Branch`."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: str | None = None
    rate: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] | None = None
    count: Annotated[int, pydantic.Field(ge=0)] | None = None
    window: Annotated[int, pydantic.Field(gt=0)] | None = None
    branch: bool = False
    start: bool = False
    offset: Annotated[int, pydantic.Field(ge=0)] = 0
    exclude: Any = None

    @pydantic.model_validator(mode="after")
    def check_rate_or_count(self):
        if (self.rate is None) == (self.count is None):
            raise ValueError("Mask requires exactly one of rate or count")

        return self

    @pydantic.field_validator("exclude", mode="before")
    @classmethod
    def normalize_exclude(cls, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, (list, tuple)):
            values = tuple(value)
        else:
            values = (value,)

        if any(isinstance(item, dict) and {"func", "key"}.issubset(item) for item in values):
            from relflow.structs.selectors import NodePredicate

            return tuple(
                NodePredicate.model_validate(item)
                if isinstance(item, dict) and {"func", "key"}.issubset(item)
                else item
                for item in values
            )

        return values

    @pydantic.field_serializer("exclude")
    def serialize_exclude(self, value: Any) -> Any:
        return value


class Branch(Node):
    """Repeated nested object group in a `relflow` schema.

    Positional children are treated as fields inside the branch.
    """

    name: str | None = None
    type: Annotated[Literal["branch"], pydantic.Field(default="branch")] = "branch"
    attention: AttentionMode = AttentionMode.mha
    length: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    overflow: Overflow = Overflow.head
    n_layers: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    pooling: PoolingConfig = pydantic.Field(default_factory=Attention)
    reference: Reference | tuple[Reference, ...] = ()
    masks: list[Mask] = pydantic.Field(default_factory=list)
    fields: list[Self | RequestTypes | pydantic.InstanceOf[Leaf]] = pydantic.Field(default_factory=list)

    def __init__(self, *children: Self | RequestTypes | Leaf, **data):
        if "max_length" in data:
            raise ValueError("max_length was removed; use length")
        if "n_linear" in data:
            raise ValueError("n_linear was removed; use pooling=Attention(n_layers=...)")
        if "references" in data:
            raise ValueError("references was removed; use the singular reference field")

        if data.get("type") not in (None, "branch"):
            super().__init__(**data)
            return

        config_names = set(type(self).model_fields) | {"mask"}
        keyword_children = {key: data.pop(key) for key in tuple(data) if key not in config_names}

        if children:
            if "fields" in data:
                raise TypeError("branch children were provided both positionally and by keyword")
            data["fields"] = list(children)

        if keyword_children:
            from relflow.structs.experiment import bind_tree_field

            data.setdefault("fields", [])
            data["fields"].extend(bind_tree_field(key, value) for key, value in keyword_children.items())

        super().__init__(**data)

    @pydantic.model_validator(mode="before")
    @classmethod
    def normalize_mask_shorthand(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        values = dict(data)
        if "max_length" in values:
            raise ValueError("max_length was removed; use length")
        if "n_linear" in values:
            raise ValueError("n_linear was removed; use pooling=Attention(n_layers=...)")
        if "references" in values:
            raise ValueError("references was removed; use the singular reference field")

        mask = values.pop("mask", None)
        if mask is None:
            return values

        if "masks" in values:
            raise ValueError("pass either mask or masks, not both")

        values["masks"] = [mask]
        return values

    @pydantic.field_validator("reference", mode="before")
    @classmethod
    def normalize_references(cls, value: Any) -> Any:
        if value is None:
            raise TypeError("reference must be a Reference or tuple of References")
        if isinstance(value, (list, tuple)):
            references = tuple(value)
            if not references:
                return ()
            if len(references) == 1:
                return references[0]
            return references
        return value

    @pydantic.field_serializer("reference")
    def serialize_references(self, value: Reference | tuple[Reference, ...], info: pydantic.SerializationInfo):
        references = (value,) if isinstance(value, Reference) else value
        dumped = [reference.model_dump(mode=info.mode, round_trip=info.round_trip) for reference in references]
        if not dumped:
            return []
        if len(dumped) == 1:
            return dumped[0]
        return dumped

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

    def post_bind_validate(self):
        if isinstance(self.pooling, Mean) and self.pooling.width not in (None, 1):
            raise ValueError(f"branch '{self.address}' Mean pooling width must be 1")

        if len(self.masks) == 0:
            return None

        is_root = getattr(getattr(self, "parent", None), "type", None) == "schema"
        if is_root:
            raise ValueError("Mask on the generated root branch is not supported")

        active_leaves = [
            descendant
            for descendant in getattr(self, "descendants", ())
            if isinstance(descendant, Leaf) and getattr(descendant, "active", True)
        ]
        if not active_leaves:
            raise ValueError(f"branch '{self.address}' has masks but no active descendant leaves")

        names = [mask.name for mask in self.masks if mask.name is not None]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"branch '{self.address}' has duplicate mask name(s): {duplicates}")

        for mask in self.masks:
            if mask.offset >= self.length:
                raise ValueError(f"branch '{self.address}' mask offset must be less than length={self.length}")

            excluded = self.excluded_leaves(mask)
            if len(excluded) == len(active_leaves):
                label = f" '{mask.name}'" if mask.name is not None else ""
                raise ValueError(f"branch '{self.address}' mask{label} excludes every active descendant leaf")

        return None

    def excluded_leaves(self, mask: Mask) -> tuple[Leaf, ...]:
        if mask.exclude is None:
            return ()

        from relflow.structs.selectors import NodePredicate

        predicates = tuple(NodePredicate.from_selector(item) for item in mask.exclude)
        return tuple(
            descendant
            for descendant in getattr(self, "descendants", ())
            if isinstance(descendant, Leaf)
            if getattr(descendant, "active", True)
            if any(predicate(descendant) for predicate in predicates)
        )

    def __rich_console__(self, console, options):
        is_root = getattr(getattr(self, "parent", None), "type", None) == "schema"
        display_type = "root" if is_root else self.type
        attributes = ("attention", "n_layers", "n_heads", "pooling", "dropout")
        if not is_root:
            attributes = ("length", "overflow", *attributes)

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
            elif name == "pooling":
                value = value.type
            elif hasattr(value, "value"):
                value = value.value
            heading.append(" ")
            heading.append(f"{name}=", style="dim")
            heading.append(str(value), style="cyan")
        if self.masks:
            heading.append(" ")
            heading.append("masks=", style="dim")
            heading.append(str(len(self.masks)), style="cyan")

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

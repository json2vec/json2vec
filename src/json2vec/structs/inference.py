"""Infer a `json2vec` schema directly from sample records.

`Model.from_schema(...)` expects hand-written field constructors. This module
derives those constructors from raw data instead, so a dataset can be turned
into a model without first describing its shape by hand.

Inference has two layers:

1. **Structure** — walk the record tree to discover nested objects and repeated
   child arrays. Nested ``dict`` values are flattened into leaves with dotted
   JMESPath queries; ``list`` of objects become :class:`~json2vec.Array` nodes.
2. **Typing** — profile each leaf column across the sample and pick a
   tensorfield (``Number``, ``Category``, ``Set``, ``Vector``, ``DateParts``,
   or optionally ``Text``) plus reasonable parameters from simple statistics
   (cardinality, dtype, null rate, list lengths).

The result is a list of field constructors ready to splat into
``Model.from_schema(*fields, ...)``. Inference is a best-effort starting point:
every threshold is tunable, and guesses can be corrected afterwards with
``model.update(...)`` / ``model.extend(...)``.
"""

from __future__ import annotations

import datetime as _datetime
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from json2vec.structs.enums import Overflow
from json2vec.structs.structure import Array
from json2vec.structs.tree import Leaf
from json2vec.tensorfields.extensions.category import Request as Category
from json2vec.tensorfields.extensions.dateparts import Request as DateParts
from json2vec.tensorfields.extensions.number import Request as Number
from json2vec.tensorfields.extensions.set import Request as Set
from json2vec.tensorfields.extensions.text import Request as Text
from json2vec.tensorfields.extensions.vector import Request as Vector

__all__ = ["InferenceConfig", "infer_schema"]

# Source-key substrings that mark an integer column as an identifier rather than
# a quantity, e.g. zip codes, phone numbers, or surrogate ids.
_DEFAULT_ID_HINTS: tuple[str, ...] = (
    "id",
    "zip",
    "postal",
    "code",
    "phone",
    "ssn",
    "uuid",
    "guid",
    "isbn",
    "sku",
)


@dataclass
class InferenceConfig:
    """Tunable thresholds for :func:`infer_schema`.

    Defaults are deliberately conservative; widen them for messier data. Every
    field maps to one decision in the typing layer so a single guess can be
    nudged without touching the rest.
    """

    sample_size: int = 10_000
    """Maximum number of top-level records profiled. Type inference is stable
    on a few thousand rows, so large datasets are sampled from the front."""

    category_max_cardinality: int = 50
    """Integer/ambiguous columns with at most this many distinct values are
    treated as categorical rather than numeric."""

    category_cardinality_ratio: float = 0.05
    """Alternative categorical test for larger samples: distinct / observed."""

    id_name_hints: tuple[str, ...] = _DEFAULT_ID_HINTS
    """Source-key substrings that force an integer column to ``Category``."""

    date_parse_min_ratio: float = 0.8
    """Fraction of non-null string values that must parse as ISO dates for a
    column to become ``DateParts``."""

    infer_text: bool = False
    """Allow high-cardinality free-text columns to become ``Text`` (requires
    the ``text`` extra). When ``False`` they fall back to ``Category``."""

    text_min_avg_tokens: float = 5.0
    """Mean whitespace-token count above which a string column may be ``Text``."""

    text_min_cardinality_ratio: float = 0.5
    """Minimum distinct / observed ratio for a string column to be ``Text``."""

    text_model_name: str = "distilbert-base-uncased"
    """HuggingFace model used when a column is inferred as ``Text``."""

    vocab_cap: int = 50_000
    """Upper bound on inferred ``max_vocab_size`` for ``Category`` / ``Set``."""

    array_length_quantile: float = 0.95
    """Quantile of observed list lengths used to size an ``Array.max_length``."""

    max_array_length: int = 1024
    """Hard cap on inferred ``Array.max_length``."""

    overflow: Overflow = Overflow.head
    """Default overflow policy for inferred arrays."""

    target: str | set[str] | None = None
    """Source key(s) to mark as supervised targets (``target=True``). Matched
    by the leaf's source key at any depth."""

    def __post_init__(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if not 0.0 < self.array_length_quantile <= 1.0:
            raise ValueError("array_length_quantile must be in (0, 1]")
        if self.max_array_length <= 0:
            raise ValueError("max_array_length must be positive")
        if self.vocab_cap < 2:
            raise ValueError("vocab_cap must be at least 2")
        self._targets: set[str] = (
            set()
            if self.target is None
            else {self.target}
            if isinstance(self.target, str)
            else set(self.target)
        )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _jmespath_member(value: str) -> str:
    """Render a source key as a JMESPath member access fragment."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        return value
    return json.dumps(value)


def _next_power_of_two(n: int) -> int:
    n = max(1, n)
    power = 1
    while power < n:
        power <<= 1
    return power


def _vocab_size(distinct: int, config: InferenceConfig) -> int:
    """Pick a ``max_vocab_size`` cap with headroom for unseen labels."""
    headroom = _next_power_of_two(distinct * 2 + 16)
    return max(2, min(config.vocab_cap, headroom))


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?")


def _parse_iso(value: str) -> _datetime.datetime | None:
    """Parse an ISO-8601 date or datetime string, or return ``None``."""
    text = value.strip()
    if not _ISO_DATE.match(text):
        return None
    candidate = text.replace(" ", "T", 1) if " " in text[:11] else text
    try:
        return _datetime.datetime.fromisoformat(candidate)
    except ValueError:
        try:
            return _datetime.datetime.combine(
                _datetime.date.fromisoformat(text[:10]), _datetime.time()
            )
        except ValueError:
            return None


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #


@dataclass
class _Column:
    """Accumulated observations for one source key at one nesting level."""

    key: str
    present: int = 0
    nulls: int = 0
    scalars: list[Any] = field(default_factory=list)
    objects: list[Mapping] = field(default_factory=list)
    list_lengths: list[int] = field(default_factory=list)
    list_scalar_items: list[Any] = field(default_factory=list)
    list_scalar_widths: set[int] = field(default_factory=set)
    list_object_items: list[Mapping] = field(default_factory=list)
    saw_dict: bool = False
    saw_list: bool = False
    saw_scalar: bool = False

    def observe(self, value: Any) -> None:
        self.present += 1
        if value is None:
            self.nulls += 1
            return
        if isinstance(value, Mapping):
            self.saw_dict = True
            self.objects.append(value)
            return
        if isinstance(value, (list, tuple)):
            self.saw_list = True
            items = list(value)
            self.list_lengths.append(len(items))
            object_items = [i for i in items if isinstance(i, Mapping)]
            scalar_items = [i for i in items if not isinstance(i, (Mapping, list, tuple)) and i is not None]
            if object_items:
                self.list_object_items.extend(object_items)
            if scalar_items:
                self.list_scalar_items.extend(scalar_items)
                self.list_scalar_widths.add(len([i for i in items if i is not None]))
            return
        self.saw_scalar = True
        self.scalars.append(value)


def _profile_level(records: Sequence[Mapping]) -> dict[str, _Column]:
    """Profile every key observed across a list of object records."""
    columns: dict[str, _Column] = {}
    for record in records:
        if not isinstance(record, Mapping):
            continue
        for key, value in record.items():
            column = columns.get(key)
            if column is None:
                column = columns[key] = _Column(key=str(key))
            column.observe(value)
    return columns


# --------------------------------------------------------------------------- #
# Typing decisions
# --------------------------------------------------------------------------- #


@dataclass
class _Decision:
    node: Array | Leaf | None
    reason: str
    query: str | None = None


def _scalar_type_breakdown(values: list[Any]) -> tuple[int, int, int, int]:
    """Return counts of (bool, int, float, str) among scalar values."""
    n_bool = sum(1 for v in values if isinstance(v, bool))
    n_int = sum(1 for v in values if isinstance(v, int) and not isinstance(v, bool))
    n_float = sum(1 for v in values if isinstance(v, float))
    n_str = sum(1 for v in values if isinstance(v, str))
    return n_bool, n_int, n_float, n_str


def _looks_categorical(distinct: int, observed: int, config: InferenceConfig) -> bool:
    if distinct <= config.category_max_cardinality:
        return True
    return observed > 0 and (distinct / observed) <= config.category_cardinality_ratio


def _decide_scalar(
    column: _Column,
    *,
    name: str,
    query: str,
    config: InferenceConfig,
) -> _Decision:
    values = column.scalars
    observed = len(values)
    distinct = len({v for v in values})
    n_bool, n_int, n_float, n_str = _scalar_type_breakdown(values)
    # `target=True` is exactly `p_prune=1.0`; set the declared field directly.
    p_prune = 1.0 if column.key in config._targets else 0.0

    # Booleans are categorical with a tiny vocabulary.
    if n_bool and n_bool == observed:
        return _Decision(
            Category(name=name, query=query, max_vocab_size=2, p_prune=p_prune),
            reason=f"boolean column ({observed} values)",
        )

    # Numeric columns: decide quantity vs. coded identifier.
    if n_str == 0 and (n_int + n_float) == observed and observed:
        id_hint = any(hint in name.casefold() for hint in config.id_name_hints)
        integral = n_float == 0
        if integral and (id_hint or _looks_categorical(distinct, observed, config)):
            why = "id-like name" if id_hint else f"low-cardinality int ({distinct} distinct)"
            return _Decision(
                Category(name=name, query=query, max_vocab_size=_vocab_size(distinct, config), p_prune=p_prune),
                reason=f"integer treated as category: {why}",
            )
        return _Decision(
            Number(name=name, query=query, p_prune=p_prune),
            reason=f"numeric column ({'int' if integral else 'float'}, {distinct} distinct)",
        )

    # String columns: dates, free text, or categories.
    if n_str == observed and observed:
        parsed = [_parse_iso(v) for v in values]
        n_dates = sum(1 for p in parsed if p is not None)
        if n_dates / observed >= config.date_parse_min_ratio:
            has_time = any(
                p is not None and (p.hour or p.minute or p.second) for p in parsed
            )
            parts = ["month_of_year", "day_of_month", "day_of_week"]
            if has_time:
                parts.append("hour_of_day")
            return _Decision(
                DateParts(name=name, query=query, dateparts=parts, p_prune=p_prune),
                reason=f"ISO dates ({n_dates}/{observed} parsed){' with time' if has_time else ''}",
            )

        avg_tokens = sum(len(v.split()) for v in values) / observed
        card_ratio = distinct / observed
        if (
            config.infer_text
            and avg_tokens >= config.text_min_avg_tokens
            and card_ratio >= config.text_min_cardinality_ratio
        ):
            return _Decision(
                Text(name=name, query=query, model_name=config.text_model_name, p_prune=p_prune),
                reason=f"free text (avg {avg_tokens:.1f} tokens, {card_ratio:.0%} unique)",
            )

        return _Decision(
            Category(name=name, query=query, max_vocab_size=_vocab_size(distinct, config), p_prune=p_prune),
            reason=f"string category ({distinct} distinct)",
        )

    return _Decision(None, reason=f"mixed/empty scalar types (bool={n_bool} int={n_int} float={n_float} str={n_str})")


def _decide_scalar_list(
    column: _Column,
    *,
    name: str,
    query: str,
    config: InferenceConfig,
) -> _Decision:
    items = column.list_scalar_items
    if not items:
        return _Decision(None, reason="empty scalar lists")
    n_bool, n_int, n_float, n_str = _scalar_type_breakdown(items)
    numeric = (n_int + n_float)
    p_prune = 1.0 if column.key in config._targets else 0.0

    # All-strings (or string-dominant) lists become a multi-label Set.
    if n_str == len(items):
        distinct = len(set(items))
        return _Decision(
            Set(name=name, query=query, max_vocab_size=_vocab_size(distinct, config), p_prune=p_prune),
            reason=f"list of labels → Set ({distinct} distinct)",
        )

    # Fixed-width numeric lists become a Vector.
    if numeric == len(items):
        if len(column.list_scalar_widths) == 1:
            width = next(iter(column.list_scalar_widths))
            if width > 0:
                return _Decision(
                    Vector(name=name, query=query, n_dim=width, p_prune=p_prune),
                    reason=f"fixed-width numeric list → Vector(n_dim={width})",
                )
        return _Decision(
            None,
            reason=f"variable-width numeric list (widths={sorted(column.list_scalar_widths)}); "
            "use a preprocessor to pad or a Set of binned labels",
        )

    return _Decision(None, reason="mixed-type scalar list; needs a preprocessor")


def _decide_column(
    column: _Column,
    *,
    query_prefix: str,
    name_prefix: str,
    config: InferenceConfig,
) -> _Decision:
    """Choose a node (or skip) for one profiled column."""
    name = _unique_name(name_prefix, column.key)
    member = _jmespath_member(column.key)

    kinds = sum((column.saw_dict, column.saw_list, column.saw_scalar))
    if kinds == 0:
        return _Decision(None, reason="only null values observed")
    if kinds > 1:
        return _Decision(None, reason="inconsistent kinds (dict/list/scalar mixed); needs a preprocessor")

    # Nested object: flatten into the current level with a dotted query prefix.
    if column.saw_dict:
        return _Decision(None, reason="object", query=f"{query_prefix}.{member}")

    # Repeated child objects: a new Array context.
    if column.saw_list and column.list_object_items:
        child_prefix = f"{query_prefix}.{member}[*]"
        child_columns = _profile_level(column.list_object_items)
        fields = _build_fields(child_columns, query_prefix=child_prefix, name_prefix="", config=config)
        if not fields:
            return _Decision(None, reason="array of objects with no typable fields")
        max_length = _infer_array_length(column.list_lengths, config)
        return _Decision(
            Array(*fields, name=name, max_length=max_length, overflow=config.overflow),
            reason=f"array of objects → Array(max_length={max_length})",
        )

    # Repeated scalars: Set or Vector.
    if column.saw_list:
        return _decide_scalar_list(column, name=name, query=f"{query_prefix}.{member}", config=config)

    # Plain scalar leaf.
    return _decide_scalar(column, name=name, query=f"{query_prefix}.{member}", config=config)


def _infer_array_length(lengths: list[int], config: InferenceConfig) -> int:
    if not lengths:
        return 1
    ordered = sorted(lengths)
    index = min(len(ordered) - 1, int(round(config.array_length_quantile * (len(ordered) - 1))))
    # Use the observed quantile length directly. Sequence length does not need to
    # be a power of two (that only matters for d_model / heads), so rounding up
    # would only pad fixed-length arrays — e.g. a constant length of 5 should be
    # max_length=5, not 8.
    chosen = max(1, ordered[index])
    return max(1, min(config.max_array_length, chosen))


# --------------------------------------------------------------------------- #
# Tree assembly
# --------------------------------------------------------------------------- #


def _unique_name(prefix: str, key: str) -> str:
    base = f"{prefix}_{key}" if prefix else key
    return Leaf.sanitize_name(base)


def _build_fields(
    columns: Mapping[str, _Column],
    *,
    query_prefix: str,
    name_prefix: str,
    config: InferenceConfig,
    explain: list[tuple[str, str, str]] | None = None,
) -> list[Array | Leaf]:
    """Build schema nodes for one container level, flattening nested objects."""
    fields: list[Array | Leaf] = []
    seen: set[str] = set()

    for column in columns.values():
        decision = _decide_column(column, query_prefix=query_prefix, name_prefix=name_prefix, config=config)

        # Flatten a nested object inline rather than nesting a node.
        if decision.node is None and decision.query is not None and column.saw_dict:
            nested_columns = _profile_level(column.objects)
            nested = _build_fields(
                nested_columns,
                query_prefix=decision.query,
                name_prefix=_unique_name(name_prefix, column.key),
                config=config,
                explain=explain,
            )
            fields.extend(nested)
            continue

        if explain is not None:
            label = type(decision.node).__name__ if decision.node is not None else "skipped"
            explain.append((column.key, label, decision.reason))

        if decision.node is None:
            warnings.warn(
                f"json2vec.infer_schema: skipping '{column.key}' — {decision.reason}",
                UserWarning,
                stacklevel=2,
            )
            continue

        if decision.node.name in seen:
            continue
        seen.add(decision.node.name)
        fields.append(decision.node)

    return fields


def _coerce_records(records: Any, sample_size: int) -> list[Mapping]:
    """Normalize supported inputs to a list of mapping records."""
    if hasattr(records, "to_dicts") and callable(records.to_dicts):  # polars-like frames
        head = getattr(records, "head", None)
        frame = head(sample_size) if callable(head) else records
        records = frame.to_dicts()
    if isinstance(records, Mapping):
        raise TypeError("infer_schema expects a sequence of records, not a single mapping")
    if not isinstance(records, Sequence):
        records = list(records)

    sample: list[Mapping] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each record must be a mapping (dict-like)")
        sample.append(record)
        if len(sample) >= sample_size:
            break
    return sample


def infer_schema(
    records: Any,
    *,
    config: InferenceConfig | None = None,
    explain: bool = False,
    **overrides: Any,
) -> list[Array | Leaf]:
    """Infer `json2vec` field constructors from sample records.

    Args:
        records: A sequence of dict-like records, an iterable of them, or a
            frame exposing ``.to_dicts()`` (e.g. a Polars ``DataFrame``).
        config: Full :class:`InferenceConfig`. When omitted a default config is
            built and any keyword ``overrides`` are applied to it.
        explain: When ``True``, print a table of each column's inferred type and
            the reason behind it.
        **overrides: Convenience overrides for individual
            :class:`InferenceConfig` fields when ``config`` is not given.

    Returns:
        A list of :class:`~json2vec.Array` and leaf request constructors ready
        to pass to ``Model.from_schema(*fields, ...)``.

    Raises:
        ValueError: If no records are provided or no typable fields are found.
    """
    if config is None:
        config = InferenceConfig(**overrides)
    elif overrides:
        raise TypeError("pass either config or keyword overrides, not both")

    sample = _coerce_records(records, config.sample_size)
    if not sample:
        raise ValueError("infer_schema requires at least one record")

    columns = _profile_level(sample)
    explanation: list[tuple[str, str, str]] | None = [] if explain else None
    fields = _build_fields(
        columns,
        query_prefix="[*]",
        name_prefix="",
        config=config,
        explain=explanation,
    )

    if explanation is not None:
        _print_explanation(explanation, n_records=len(sample))

    if not fields:
        raise ValueError("infer_schema could not derive any fields from the sample")

    return fields


def _print_explanation(rows: list[tuple[str, str, str]], *, n_records: int) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title=f"inferred schema ({n_records} records sampled)")
        table.add_column("source key", style="bold")
        table.add_column("inferred", style="cyan")
        table.add_column("reason", style="dim")
        for key, label, reason in rows:
            table.add_row(key, label, reason)
        Console().print(table)
    except ImportError:  # pragma: no cover - rich is a hard dependency
        for key, label, reason in rows:
            print(f"{key}: {label} — {reason}")

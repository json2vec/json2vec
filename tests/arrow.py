"""Small Arrow constructors shared by contract-focused tests."""

from collections.abc import Sequence
from typing import Any

import pyarrow as pa

from relflow.data.arrow import Batch
from relflow.data.datasets.arrow import convert


def table(rows: Sequence[dict[str, Any]], schema: pa.Schema | None = None) -> pa.Table:
    """Build the explicit Arrow boundary used by public model helpers."""

    return pa.Table.from_pylist(list(rows), schema=schema)


def batch(rows: Sequence[dict[str, Any]], schema: pa.Schema | None = None) -> Batch:
    """Build the internal carrier expected by low-level coalescing."""

    return convert(table(rows, schema=schema), namespace="test", offset=0)


__all__ = ["batch", "table"]

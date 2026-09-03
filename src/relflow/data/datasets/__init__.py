"""Public data modules over one Arrow-backed loader."""

from __future__ import annotations

from relflow.data.datasets.arrow import ArrowDataModule
from relflow.data.datasets.custom import CustomDataModule
from relflow.data.datasets.polars import PolarsDataModule
from relflow.data.datasets.synthetic import SyntheticDataModule

__all__ = [
    "ArrowDataModule",
    "CustomDataModule",
    "PolarsDataModule",
    "SyntheticDataModule",
]

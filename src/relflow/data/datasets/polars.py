"""Polars ingress for the canonical Arrow data module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import relflow
from relflow.data.datasets.arrow import ArrowDataModule, Retain
from relflow.data.processors import PreprocessorInput
from relflow.structs.enums import Strata

try:
    import polars as pl
except ImportError:
    pass


class PolarsDataModule(ArrowDataModule):
    """Convert in-memory Polars frames once, then use the Arrow pipeline.

    ``LazyFrame`` inputs are intentionally rejected. Collect them explicitly so
    users can see where query execution and materialization happen.
    """

    def __init__(
        self,
        model: relflow.Model,
        *,
        train: pl.DataFrame | None = None,
        validate: pl.DataFrame | None = None,
        test: pl.DataFrame | None = None,
        predict: pl.DataFrame | None = None,
        preprocessor: PreprocessorInput | Mapping[Strata | str, PreprocessorInput] = (),
        seed: int = 0,
        shuffle: bool | None | Mapping[Strata | str, bool | None] = None,
        sample: float | Mapping[Strata | str, float] = 1.0,
        replacement: bool | Mapping[Strata | str, bool] = False,
        epoch_size: int | None | Mapping[Strata | str, int | None] = None,
        shuffle_rows: int | None | Mapping[Strata | str, int | None] = None,
        drop_last: bool | Mapping[Strata | str, bool] = False,
        num_workers: int | Mapping[Strata | str, int] = 0,
        persistent_workers: bool | Mapping[Strata | str, bool] = False,
        pin_memory: bool | Mapping[Strata | str, bool] = False,
        retain: Retain | Mapping[Strata | str, Retain] = (),
    ):
        try:
            import polars as pl
        except ImportError as error:
            raise ImportError("PolarsDataModule requires `polars`; install `relflow[hash]`.") from error

        frames = {
            "train": train,
            "validate": validate,
            "test": test,
            "predict": predict,
        }
        converted: dict[str, Any] = {}
        for name, frame in frames.items():
            if frame is None:
                converted[name] = None
                continue
            if isinstance(frame, pl.LazyFrame):
                raise TypeError(f"{name} must be a collected polars.DataFrame, not LazyFrame")
            if not isinstance(frame, pl.DataFrame):
                raise TypeError(f"{name} must be a polars.DataFrame, got {type(frame).__name__}")
            converted[name] = frame.to_arrow()

        super().__init__(
            model=model,
            train=converted["train"],
            validate=converted["validate"],
            test=converted["test"],
            predict=converted["predict"],
            preprocessor=preprocessor,
            seed=seed,
            shuffle=shuffle,
            sample=sample,
            replacement=replacement,
            epoch_size=epoch_size,
            shuffle_rows=shuffle_rows,
            drop_last=drop_last,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            retain=retain,
        )


__all__ = ["PolarsDataModule"]

"""Polars-backed iterable datasets and Lightning data modules."""

from __future__ import annotations

import os
import random
import weakref
from collections.abc import Callable, Iterator, Mapping
from functools import partial, partialmethod
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import lightning.pytorch as lit
import polars as pl
import torch
from beartype import beartype
from torch.utils.data import DataLoader, IterableDataset

from json2vec.data.datasets.base import (
    InterprocessEncodingContext,
    NonNegativeInt,
    PositiveInt,
    PreprocessorConfig,
    RawObservation,
    SampleRate,
    StrataMap,
    _is_assigned_to_worker,
    _worker_identity,
    identity,
)
from json2vec.data.iterables import (
    JMESPathResolutionMonitor,
    batch,
    mask,
    process,
    sample,
    shuffle,
    target,
    transform,
)
from json2vec.data.processing import Pipeline
from json2vec.distributed import rank as distributed_rank
from json2vec.distributed import world_size as distributed_world_size
from json2vec.preprocessors.base import Preprocessor
from json2vec.structs.enums import ShardingStrategy, Strata
from json2vec.structs.experiment import Hyperparameters

if TYPE_CHECKING:
    from json2vec.architecture.root import Model
else:
    Model = "json2vec.architecture.root.Model"

DataFrameMap: TypeAlias = Mapping[Strata | str, pl.DataFrame]


def _dataframes_by_strata(dataframe: pl.DataFrame | DataFrameMap) -> dict[Strata, pl.DataFrame]:
    if not isinstance(dataframe, Mapping):
        return {strata: dataframe for strata in Strata}

    normalized: dict[Strata, pl.DataFrame] = {}
    for key, frame in cast(DataFrameMap, dataframe).items():
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"dataframe for strata '{key}' must be a polars DataFrame")
        normalized[Strata.normalize(key)] = frame

    if not normalized:
        raise ValueError("dataframe mapping must include at least one strata")

    return normalized


@beartype
def observe_polars(
    dataframe: pl.DataFrame,
    strata: Strata,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    replacement: bool = False,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> Iterator[RawObservation]:
    if replacement:
        rows = dataframe.to_dicts()
        if not rows:
            raise ValueError("no dataframe rows available for replacement sampling")

        while True:
            yield dict(random.choice(rows))

    worker_id, num_workers = _worker_identity(global_rank=global_rank, world_size=world_size)

    if sharding == ShardingStrategy.file:
        if not _is_assigned_to_worker(
            shard_key="dataframe:0",
            worker_id=worker_id,
            num_workers=num_workers,
        ):
            return

    if sharding == ShardingStrategy.record:
        for row_index, row in enumerate(dataframe.iter_rows(named=True)):
            if _is_assigned_to_worker(
                shard_key=f"dataframe:record:{row_index}",
                worker_id=worker_id,
                num_workers=num_workers,
            ):
                yield row
        return

    for chunk_index, offset in enumerate(range(0, dataframe.height, chunk_batch_size)):
        if sharding == ShardingStrategy.chunk:
            if not _is_assigned_to_worker(
                shard_key=f"dataframe:chunk:{chunk_index}",
                worker_id=worker_id,
                num_workers=num_workers,
            ):
                continue

        yield from dataframe.slice(offset, chunk_batch_size).to_dicts()


class PolarsBatchDataset(IterableDataset):
    def __init__(
        self,
        hyperparameters: Hyperparameters,
        dataframe: pl.DataFrame,
        preprocessor: PreprocessorConfig.Value,
        preprocessor_kwargs: dict[str, Any],
        interprocess_encoding_context: InterprocessEncodingContext,
        batch_size: int,
        strata: Strata,
        sharding: ShardingStrategy,
        chunk_batch_size: int,
        observation_buffer_size: int,
        sample_rate: float,
        replacement: bool = False,
        global_rank: int | None = None,
        world_size: int | None = None,
    ):
        super().__init__()

        self.hyperparameters = hyperparameters
        self.dataframe = dataframe
        self.preprocessor = preprocessor
        self.preprocessor_kwargs = preprocessor_kwargs
        self.interprocess_encoding_context = interprocess_encoding_context
        self.global_rank = distributed_rank() if global_rank is None else global_rank
        self.world_size = distributed_world_size() if world_size is None else world_size
        self.batch_size = batch_size
        self.strata = strata
        self.sharding = sharding
        self.chunk_batch_size = chunk_batch_size
        self.observation_buffer_size = observation_buffer_size
        self.sample_rate = sample_rate
        self.replacement = replacement

    def __iter__(self):
        for field_context in self.interprocess_encoding_context.values():
            if hasattr(field_context, "configure_distributed"):
                field_context.configure_distributed(global_rank=self.global_rank, world_size=self.world_size)

        yield from (
            Pipeline(
                hyperparameters=self.hyperparameters,
                dataframe=self.dataframe,
                preprocessor=self.preprocessor,
                preprocessor_kwargs=self.preprocessor_kwargs,
                strata=self.strata,
                interprocess_encoding_context=self.interprocess_encoding_context,
                jmespath_resolution_monitor=JMESPathResolutionMonitor(),
                sharding=self.sharding,
                chunk_batch_size=self.chunk_batch_size,
                sample_rate=self.sample_rate,
                replacement=self.replacement,
                batch_size=self.batch_size,
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            | observe_polars
            | process
            | sample
            | partial(shuffle, size=self.observation_buffer_size)
            | batch
            | transform
            | mask
            | target
        )


def polars_dataloader(
    hyperparameters: Hyperparameters,
    dataframe: pl.DataFrame,
    preprocessor: PreprocessorConfig.Value,
    preprocessor_kwargs: dict[str, Any],
    interprocess_encoding_context: InterprocessEncodingContext,
    batch_size: int,
    strata: Strata,
    num_workers: int | None,
    persistent_workers: bool,
    pin_memory: bool,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    observation_buffer_size: int,
    sample_rate: float,
    replacement: bool = False,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> DataLoader:
    workers = num_workers if num_workers is not None else (os.cpu_count() or 0)
    active_persistent_workers = persistent_workers and workers > 0
    active_pin_memory = pin_memory and strata != Strata.predict and torch.cuda.is_available()
    global_rank = distributed_rank() if global_rank is None else global_rank
    world_size = distributed_world_size() if world_size is None else world_size

    return DataLoader(
        dataset=PolarsBatchDataset(
            hyperparameters=hyperparameters,
            dataframe=dataframe,
            preprocessor=preprocessor,
            preprocessor_kwargs=preprocessor_kwargs,
            interprocess_encoding_context=interprocess_encoding_context,
            batch_size=batch_size,
            strata=strata,
            sharding=sharding,
            chunk_batch_size=chunk_batch_size,
            observation_buffer_size=observation_buffer_size,
            sample_rate=sample_rate,
            replacement=replacement,
            global_rank=global_rank,
            world_size=world_size,
        ),
        drop_last=False,
        batch_size=None,
        collate_fn=identity,
        num_workers=workers,
        persistent_workers=active_persistent_workers,
        pin_memory=active_pin_memory,
    )


class PolarsDataModule(lit.LightningDataModule):
    """Lightning data module for in-memory Polars DataFrames."""

    @beartype
    def __init__(
        self,
        model: Model,
        train: pl.DataFrame | None = None,
        validate: pl.DataFrame | None = None,
        test: pl.DataFrame | None = None,
        predict: pl.DataFrame | None = None,
        preprocessor: str | Callable[..., Any] | Preprocessor | None = None,
        dataframe: pl.DataFrame | DataFrameMap | None = None,
        num_workers: NonNegativeInt | None | StrataMap[NonNegativeInt | None] = None,
        persistent_workers: bool | StrataMap[bool] = True,
        pin_memory: bool | StrataMap[bool] = True,
        sharding: ShardingStrategy | str | StrataMap[ShardingStrategy | str] = ShardingStrategy.chunk,
        chunk_batch_size: PositiveInt | StrataMap[PositiveInt] = 4096,
        observation_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        sample_rate: SampleRate | StrataMap[SampleRate] = 1.0,
        replacement: bool | StrataMap[bool] = False,
        **kwargs: Any,
    ):
        super().__init__()

        if dataframe is not None and any(frame is not None for frame in (train, validate, test, predict)):
            raise ValueError("pass either dataframe or named splits, not both")

        if dataframe is None:
            dataframes = {
                strata: frame
                for strata, frame in {
                    Strata.train: train,
                    Strata.validate: validate,
                    Strata.test: test,
                    Strata.predict: predict,
                }.items()
                if frame is not None
            }
            if not dataframes:
                raise ValueError("at least one dataframe split is required")
        else:
            dataframes = _dataframes_by_strata(dataframe)

        self.dataframes = dataframes
        self.preprocessor = PreprocessorConfig.normalize(preprocessor)
        self.preprocessor_kwargs = dict(kwargs)
        try:
            self._model_ref = weakref.ref(model)
        except TypeError:
            self._model_ref = None
        self._hyperparameters = model.hyperparameters
        self._interprocess_encoding_context = model.interprocess_encoding_context
        self._batch_size = model.batch_size
        self.num_workers = Strata.expand(num_workers, default=None)
        self.persistent_workers = Strata.expand(persistent_workers, default=True)
        self.pin_memory = Strata.expand(pin_memory, default=True)
        self.sharding = ShardingStrategy.expand(sharding, default=ShardingStrategy.chunk)
        self.chunk_batch_size = Strata.expand(chunk_batch_size, default=4096)
        self.observation_buffer_size = Strata.expand(observation_buffer_size, default=1)
        self.sample_rate = {strata: float(rate) for strata, rate in Strata.expand(sample_rate, default=1.0).items()}
        self.replacement = Strata.expand(replacement, default=False)

    def _model(self) -> Model | None:
        if self._model_ref is None:
            return None

        return self._model_ref()

    @property
    def hyperparameters(self) -> Hyperparameters:
        model = self._model()
        if model is not None:
            return model.hyperparameters

        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters: Hyperparameters) -> None:
        self._model_ref = None
        self._hyperparameters = hyperparameters

    @property
    def batch_size(self) -> int:
        model = self._model()
        if model is not None:
            return model.batch_size

        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int) -> None:
        self._model_ref = None
        self._batch_size = batch_size

    @property
    def interprocess_encoding_context(self) -> InterprocessEncodingContext:
        model = self._model()
        if model is not None:
            return model.interprocess_encoding_context

        return self._interprocess_encoding_context

    @interprocess_encoding_context.setter
    def interprocess_encoding_context(self, context: InterprocessEncodingContext) -> None:
        self._model_ref = None
        self._interprocess_encoding_context = context

    def dataloader(self, strata: Strata, required: bool = True) -> DataLoader | None:
        strata = Strata.normalize(strata)
        trainer = getattr(self, "trainer", None)
        global_rank = getattr(trainer, "global_rank", None)
        world_size = getattr(trainer, "world_size", None)
        if strata not in self.dataframes:
            if not required:
                return None
            raise ValueError(f"no dataframe configured for strata: {strata}")

        return polars_dataloader(
            hyperparameters=self.hyperparameters,
            dataframe=self.dataframes[strata],
            preprocessor=self.preprocessor,
            preprocessor_kwargs=self.preprocessor_kwargs,
            interprocess_encoding_context=self.interprocess_encoding_context,
            batch_size=self.batch_size,
            strata=strata,
            num_workers=self.num_workers[strata],
            persistent_workers=self.persistent_workers[strata],
            pin_memory=self.pin_memory[strata],
            sharding=self.sharding[strata],
            chunk_batch_size=self.chunk_batch_size[strata],
            observation_buffer_size=self.observation_buffer_size[strata],
            sample_rate=self.sample_rate[strata],
            replacement=self.replacement[strata],
            global_rank=global_rank,
            world_size=world_size,
        )

    train_dataloader = partialmethod(dataloader, strata=Strata.train, required=False)
    val_dataloader = partialmethod(dataloader, strata=Strata.validate, required=False)
    test_dataloader = partialmethod(dataloader, strata=Strata.test, required=False)
    predict_dataloader = partialmethod(dataloader, strata=Strata.predict, required=False)

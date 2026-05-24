"""File-backed streaming datasets and Lightning data modules."""

from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING, Any

import lightning.pytorch as lit
import torch
from beartype import beartype
from torch.utils.data import DataLoader, IterableDataset

from json2vec.data.datasets.base import (
    Dataset,
    InterprocessEncodingContext,
    NonNegativeInt,
    PositiveInt,
    SampleRate,
    StrataMap,
    _by_strata,
    identity,
)
from json2vec.data.iterables import (
    JMESPathResolutionMonitor,
    batch,
    mask,
    observe,
    process,
    sample,
    shuffle,
    target,
    transform,
)
from json2vec.data.processing import Pipeline
from json2vec.distributed import rank as distributed_rank
from json2vec.distributed import world_size as distributed_world_size
from json2vec.structs.enums import ShardingStrategy, Strata
from json2vec.structs.experiment import Hyperparameters

if TYPE_CHECKING:
    from json2vec.architecture.root import Model


def _normalize_sharding(value: ShardingStrategy | str) -> ShardingStrategy:
    if isinstance(value, ShardingStrategy):
        return value

    return ShardingStrategy(value.strip().lower())


class BatchDataset(IterableDataset):
    def __init__(
        self,
        hyperparameters: Hyperparameters,
        dataset: Dataset,
        interprocess_encoding_context: InterprocessEncodingContext,
        batch_size: int,
        strata: Strata,
        sharding: ShardingStrategy,
        chunk_batch_size: int,
        file_buffer_size: int,
        observation_buffer_size: int,
        sample_rate: float,
        global_rank: int | None = None,
        world_size: int | None = None,
    ):
        super().__init__()

        self.hyperparameters = hyperparameters
        self.dataset = dataset
        self.interprocess_encoding_context = interprocess_encoding_context
        self.global_rank = distributed_rank() if global_rank is None else global_rank
        self.world_size = distributed_world_size() if world_size is None else world_size
        self.batch_size = batch_size
        self.strata = strata
        self.sharding = sharding
        self.chunk_batch_size = chunk_batch_size
        self.file_buffer_size = file_buffer_size
        self.observation_buffer_size = observation_buffer_size
        self.sample_rate = sample_rate

    def __iter__(self):
        for field_context in self.interprocess_encoding_context.values():
            if hasattr(field_context, "configure_distributed"):
                field_context.configure_distributed(global_rank=self.global_rank, world_size=self.world_size)

        yield from (
            Pipeline(
                hyperparameters=self.hyperparameters,
                dataset=self.dataset,
                strata=self.strata,
                interprocess_encoding_context=self.interprocess_encoding_context,
                jmespath_resolution_monitor=JMESPathResolutionMonitor(),
                sharding=self.sharding,
                chunk_batch_size=self.chunk_batch_size,
                file_buffer_size=self.file_buffer_size,
                sample_rate=self.sample_rate,
                batch_size=self.batch_size,
                global_rank=self.global_rank,
                world_size=self.world_size,
            )
            | observe
            | process
            | sample
            | partial(shuffle, size=self.observation_buffer_size)
            | batch
            | transform
            | mask
            | target
        )


def dataloader(
    hyperparameters: Hyperparameters,
    dataset: Dataset,
    interprocess_encoding_context: InterprocessEncodingContext,
    batch_size: int,
    strata: Strata,
    num_workers: int | None,
    persistent_workers: bool,
    pin_memory: bool,
    sharding: ShardingStrategy,
    chunk_batch_size: int,
    file_buffer_size: int,
    observation_buffer_size: int,
    sample_rate: float,
    global_rank: int | None = None,
    world_size: int | None = None,
) -> DataLoader:
    workers = num_workers if num_workers is not None else (os.cpu_count() or 0)
    active_persistent_workers = persistent_workers and workers > 0
    active_pin_memory = pin_memory and strata != Strata.predict and torch.cuda.is_available()
    global_rank = distributed_rank() if global_rank is None else global_rank
    world_size = distributed_world_size() if world_size is None else world_size

    return DataLoader(
        dataset=BatchDataset(
            hyperparameters=hyperparameters,
            dataset=dataset,
            interprocess_encoding_context=interprocess_encoding_context,
            batch_size=batch_size,
            strata=strata,
            sharding=sharding,
            chunk_batch_size=chunk_batch_size,
            file_buffer_size=file_buffer_size,
            observation_buffer_size=observation_buffer_size,
            sample_rate=sample_rate,
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


class StreamingDataModule(lit.LightningDataModule):
    """Lightning data module for streaming records from files.

    The dataset reads records from `Dataset.root`, applies the optional
    preprocessor, batches observations, and encodes them with model
    hyperparameters.
    """

    @beartype
    def __init__(
        self,
        hyperparameters: Hyperparameters,
        dataset: Dataset,
        interprocess_encoding_context: InterprocessEncodingContext,
        batch_size: PositiveInt,
        num_workers: NonNegativeInt | None | StrataMap[NonNegativeInt | None] = None,
        persistent_workers: bool | StrataMap[bool] = True,
        pin_memory: bool | StrataMap[bool] = True,
        sharding: ShardingStrategy | str | StrataMap[ShardingStrategy | str] = ShardingStrategy.chunk,
        chunk_batch_size: PositiveInt | StrataMap[PositiveInt] = 4096,
        file_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        observation_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        sample_rate: SampleRate | StrataMap[SampleRate] = 1.0,
    ):
        super().__init__()

        self.hyperparameters = hyperparameters
        self.dataset = dataset
        self.interprocess_encoding_context = interprocess_encoding_context
        self.batch_size = batch_size
        self.num_workers = _by_strata(num_workers, default=None)
        self.persistent_workers = _by_strata(persistent_workers, default=True)
        self.pin_memory = _by_strata(pin_memory, default=True)
        self.sharding = {
            strata: _normalize_sharding(strategy)
            for strata, strategy in _by_strata(sharding, default=ShardingStrategy.chunk).items()
        }
        self.chunk_batch_size = _by_strata(chunk_batch_size, default=4096)
        self.file_buffer_size = _by_strata(file_buffer_size, default=1)
        self.observation_buffer_size = _by_strata(observation_buffer_size, default=1)
        self.sample_rate = {
            strata: float(rate)
            for strata, rate in _by_strata(sample_rate, default=1.0).items()
        }

    @classmethod
    def from_model(
        cls,
        model: Model,
        dataset: Dataset,
        **kwargs: Any,
    ) -> "StreamingDataModule":
        """Construct a streaming data module from a model and dataset config."""
        datamodule = cls(
            hyperparameters=model.hyperparameters,
            dataset=dataset,
            interprocess_encoding_context=model.interprocess_encoding_context,
            batch_size=model.batch_size,
            **kwargs,
        )
        register = getattr(model, "_register_data_module", None)
        if callable(register):
            register(datamodule)

        return datamodule

    def _set_interprocess_encoding_context(self, context: InterprocessEncodingContext) -> None:
        self.interprocess_encoding_context = context

    def dataloader(self, strata: Strata) -> DataLoader:
        trainer = getattr(self, "trainer", None)
        global_rank = getattr(trainer, "global_rank", None)
        world_size = getattr(trainer, "world_size", None)

        return dataloader(
            hyperparameters=self.hyperparameters,
            dataset=self.dataset,
            interprocess_encoding_context=self.interprocess_encoding_context,
            batch_size=self.batch_size,
            strata=strata,
            num_workers=self.num_workers[strata],
            persistent_workers=self.persistent_workers[strata],
            pin_memory=self.pin_memory[strata],
            sharding=self.sharding[strata],
            chunk_batch_size=self.chunk_batch_size[strata],
            file_buffer_size=self.file_buffer_size[strata],
            observation_buffer_size=self.observation_buffer_size[strata],
            sample_rate=self.sample_rate[strata],
            global_rank=global_rank,
            world_size=world_size,
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.train)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.validate)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.test)

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader(strata=Strata.predict)

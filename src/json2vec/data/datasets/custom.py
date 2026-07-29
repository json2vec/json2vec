"""Custom iterable datasets and Lightning data modules."""

from __future__ import annotations

import os
import weakref
from collections.abc import Iterator, Mapping
from functools import partial, partialmethod
from typing import TYPE_CHECKING, TypeAlias, cast

import lightning.pytorch as lit
import torch
from beartype import beartype
from torch.utils.data import DataLoader, IterableDataset

from json2vec.data.datasets.base import (
    InterprocessEncodingContext,
    NonNegativeInt,
    Pipeline,
    PositiveInt,
    PreprocessorConfig,
    RawObservation,
    SampleRate,
    StrataMap,
    _worker_buffer_size,
    identity,
    share_interprocess_encoding_context,
)
from json2vec.data.iterables import (
    JMESPathResolutionMonitor,
    batch,
    mask,
    process,
    sample,
    shuffle,
    transform,
)
from json2vec.data.processors import Preprocessor
from json2vec.distributed import rank as distributed_rank
from json2vec.distributed import world_size as distributed_world_size
from json2vec.structs.enums import Strata
from json2vec.structs.experiment import Schema

if TYPE_CHECKING:
    from json2vec.architecture.root import Model
else:
    Model = "json2vec.architecture.root.Model"

DatasetMap: TypeAlias = Mapping[Strata | str, IterableDataset]


@beartype
def _validate_loader_configuration(
    num_workers: NonNegativeInt | None | StrataMap[NonNegativeInt | None],
    persistent_workers: bool | StrataMap[bool],
    pin_memory: bool | StrataMap[bool],
    observation_buffer_size: PositiveInt | StrataMap[PositiveInt],
    sample_rate: SampleRate | StrataMap[SampleRate],
) -> None:
    return None


def _datasets_by_strata(datasets: DatasetMap) -> dict[Strata, IterableDataset]:
    normalized: dict[Strata, IterableDataset] = {}
    for key, dataset in cast(DatasetMap, datasets).items():
        if not isinstance(dataset, IterableDataset):
            raise TypeError(f"dataset for strata '{key}' must be an IterableDataset")
        normalized[Strata.normalize(key)] = dataset

    if not normalized:
        raise ValueError("dataset mapping must include at least one strata")

    return normalized


def observe_dataset(dataset: IterableDataset) -> Iterator[RawObservation]:
    yield from dataset


class CustomBatchDataset(IterableDataset):
    def __init__(
        self,
        schema: Schema,
        dataset: IterableDataset,
        preprocessor: PreprocessorConfig.Value,
        interprocess_encoding_context: InterprocessEncodingContext,
        batch_size: int,
        strata: Strata,
        observation_buffer_size: int,
        sample_rate: float,
        global_rank: int | None = None,
        world_size: int | None = None,
    ):
        super().__init__()

        self.schema = schema
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.interprocess_encoding_context = interprocess_encoding_context
        self.global_rank = distributed_rank() if global_rank is None else global_rank
        self.world_size = distributed_world_size() if world_size is None else world_size
        self.batch_size = batch_size
        self.strata = strata
        self.observation_buffer_size = observation_buffer_size
        self.sample_rate = sample_rate

    def __iter__(self):
        for field_context in self.interprocess_encoding_context.values():
            if hasattr(field_context, "configure_distributed"):
                field_context.configure_distributed(global_rank=self.global_rank, world_size=self.world_size)

        observation_buffer_size = _worker_buffer_size(self.observation_buffer_size)
        yield from (
            Pipeline(
                schema=self.schema,
                dataset=self.dataset,
                preprocessor=self.preprocessor,
                strata=self.strata,
                interprocess_encoding_context=self.interprocess_encoding_context,
                jmespath_resolution_monitor=JMESPathResolutionMonitor(),
                sample_rate=self.sample_rate,
                batch_size=self.batch_size,
            )
            | observe_dataset
            | process
            | sample
            | partial(shuffle, size=observation_buffer_size)
            | batch
            | transform
            | mask
        )


def custom_dataloader(
    schema: Schema,
    dataset: IterableDataset,
    preprocessor: PreprocessorConfig.Value,
    interprocess_encoding_context: InterprocessEncodingContext,
    batch_size: int,
    strata: Strata,
    num_workers: int | None,
    persistent_workers: bool,
    pin_memory: bool,
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
        dataset=CustomBatchDataset(
            schema=schema,
            dataset=dataset,
            preprocessor=preprocessor,
            interprocess_encoding_context=interprocess_encoding_context,
            batch_size=batch_size,
            strata=strata,
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


class CustomDataModule(lit.LightningDataModule):
    """Lightning data module for user-provided iterable datasets."""

    def __init__(
        self,
        model: Model,
        train: IterableDataset | None = None,
        validate: IterableDataset | None = None,
        test: IterableDataset | None = None,
        predict: IterableDataset | None = None,
        preprocessor: Preprocessor | None = None,
        datasets: DatasetMap | None = None,
        num_workers: NonNegativeInt | None | StrataMap[NonNegativeInt | None] = None,
        persistent_workers: bool | StrataMap[bool] = True,
        pin_memory: bool | StrataMap[bool] = True,
        observation_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        sample_rate: SampleRate | StrataMap[SampleRate] = 1.0,
    ):
        super().__init__()

        _validate_loader_configuration(
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            observation_buffer_size=observation_buffer_size,
            sample_rate=sample_rate,
        )

        if datasets is not None and any(dataset is not None for dataset in (train, validate, test, predict)):
            raise ValueError("pass either datasets or named splits, not both")

        if datasets is None:
            split_datasets = {}
            for strata, dataset in {
                Strata.train: train,
                Strata.validate: validate,
                Strata.test: test,
                Strata.predict: predict,
            }.items():
                if dataset is None:
                    continue
                if not isinstance(dataset, IterableDataset):
                    raise TypeError(f"dataset for strata '{strata}' must be an IterableDataset")
                split_datasets[strata] = dataset
            if not split_datasets:
                raise ValueError("at least one dataset split is required")
        else:
            split_datasets = _datasets_by_strata(datasets)

        self.datasets = split_datasets
        self.preprocessor = PreprocessorConfig.normalize(preprocessor)
        try:
            self._model_ref = weakref.ref(model)
        except TypeError:
            self._model_ref = None
        self._schema = model.schema
        self._interprocess_encoding_context = model.interprocess_encoding_context
        self._batch_size = model.batch_size
        self.num_workers = Strata.expand(num_workers, default=None)
        self.persistent_workers = Strata.expand(persistent_workers, default=True)
        self.pin_memory = Strata.expand(pin_memory, default=True)
        self.observation_buffer_size = Strata.expand(observation_buffer_size, default=1)
        self.sample_rate = {strata: float(rate) for strata, rate in Strata.expand(sample_rate, default=1.0).items()}

    def _model(self) -> Model | None:
        if self._model_ref is None:
            return None

        return self._model_ref()

    @property
    def schema(self) -> Schema:
        model = self._model()
        if model is not None:
            return model.schema

        return self._schema

    @schema.setter
    def schema(self, schema: Schema) -> None:
        self._model_ref = None
        self._schema = schema

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
        if strata not in self.datasets:
            if not required:
                return None
            raise ValueError(f"no dataset configured for strata: {strata}")

        workers = self.num_workers[strata]
        if workers is None:
            workers = os.cpu_count() or 0

        interprocess_encoding_context = self.interprocess_encoding_context
        if strata == Strata.train and workers > 0:
            share_interprocess_encoding_context(interprocess_encoding_context)

        return custom_dataloader(
            schema=self.schema,
            dataset=self.datasets[strata],
            preprocessor=self.preprocessor,
            interprocess_encoding_context=interprocess_encoding_context,
            batch_size=self.batch_size,
            strata=strata,
            num_workers=workers,
            persistent_workers=self.persistent_workers[strata],
            pin_memory=self.pin_memory[strata],
            observation_buffer_size=self.observation_buffer_size[strata],
            sample_rate=self.sample_rate[strata],
            global_rank=global_rank,
            world_size=world_size,
        )

    train_dataloader = partialmethod(dataloader, strata=Strata.train, required=False)
    val_dataloader = partialmethod(dataloader, strata=Strata.validate, required=False)
    test_dataloader = partialmethod(dataloader, strata=Strata.test, required=False)
    predict_dataloader = partialmethod(dataloader, strata=Strata.predict, required=False)

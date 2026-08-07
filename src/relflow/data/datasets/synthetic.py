"""Lightning data modules for generated synthetic observations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, TypeAlias

from torch.utils.data import IterableDataset

from relflow.data.datasets.base import NonNegativeInt, PositiveInt, RawObservation, SampleRate, StrataMap
from relflow.data.datasets.custom import CustomDataModule
from relflow.data.processors import Preprocessor

if TYPE_CHECKING:
    from relflow.architecture.root import Model
else:
    Model = "relflow.architecture.root.Model"

Generator: TypeAlias = Callable[[], Iterator[RawObservation]]


class _SyntheticDataset(IterableDataset):
    """A restartable iterable dataset backed by a generator function.

    The generator is called whenever the dataset is iterated, including once in
    each data-loader worker. It may produce either a finite or infinite stream
    of raw observation dictionaries.
    """

    def __init__(self, generator: Generator):
        super().__init__()

        if not callable(generator):
            raise TypeError("generator must be callable")

        self.generator = generator

    def __iter__(self) -> Iterator[RawObservation]:
        yield from self.generator()


class SyntheticDataModule(CustomDataModule):
    """Lightning data module whose splits are defined by generator functions."""

    def __init__(
        self,
        model: Model,
        train: Generator | None = None,
        validate: Generator | None = None,
        test: Generator | None = None,
        predict: Generator | None = None,
        preprocessor: Preprocessor | None = None,
        num_workers: NonNegativeInt | None | StrataMap[NonNegativeInt | None] = None,
        persistent_workers: bool | StrataMap[bool] = True,
        pin_memory: bool | StrataMap[bool] = True,
        observation_buffer_size: PositiveInt | StrataMap[PositiveInt] = 1,
        sample_rate: SampleRate | StrataMap[SampleRate] = 1.0,
    ):
        super().__init__(
            model=model,
            train=_SyntheticDataset(train) if train is not None else None,
            validate=_SyntheticDataset(validate) if validate is not None else None,
            test=_SyntheticDataset(test) if test is not None else None,
            predict=_SyntheticDataset(predict) if predict is not None else None,
            preprocessor=preprocessor,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            observation_buffer_size=observation_buffer_size,
            sample_rate=sample_rate,
        )

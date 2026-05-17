from __future__ import annotations

import datetime
from collections import defaultdict
from functools import partialmethod
from typing import TYPE_CHECKING

from lightning import Callback, Trainer
from loguru import logger

from json2vec.structs.enums import Strata

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec


class ThroughputLogger(Callback):
    def __init__(self):
        super().__init__()

        self.timestamp: dict[Strata, datetime.datetime] = defaultdict(lambda: datetime.datetime.now())
        self.batches: dict[Strata, int] = defaultdict(int)

    def start(self, trainer: Trainer, pl_module: JSON2Vec, strata: Strata):
        self.timestamp[strata] = datetime.datetime.now()
        self.batches[strata] = 0

    def count(self, trainer: Trainer, pl_module: JSON2Vec, *args, strata: Strata, **kwargs):
        self.batches[strata] += 1

    def end(self, trainer: Trainer, pl_module: JSON2Vec, strata: Strata):
        now = datetime.datetime.now()
        then = self.timestamp[strata]
        elapsed = (now - then).total_seconds()
        observations = self.batches[strata] * pl_module.batch_size
        throughput = observations / elapsed if elapsed > 0.0 else 0.0

        logger.bind(
            component="throughput",
            strata=strata,
            batches=self.batches[strata],
            observations=observations,
            seconds=elapsed,
            throughput=throughput,
        ).info(f"{strata} epoch throughput: {throughput:.2f} observations/s")

    on_train_epoch_start = partialmethod(start, strata=Strata.train)
    on_validation_epoch_start = partialmethod(start, strata=Strata.validate)
    on_test_epoch_start = partialmethod(start, strata=Strata.test)
    on_predict_epoch_start = partialmethod(start, strata=Strata.predict)

    on_train_batch_end = partialmethod(count, strata=Strata.train)
    on_validation_batch_end = partialmethod(count, strata=Strata.validate)
    on_test_batch_end = partialmethod(count, strata=Strata.test)
    on_predict_batch_end = partialmethod(count, strata=Strata.predict)

    on_train_epoch_end = partialmethod(end, strata=Strata.train)
    on_validation_epoch_end = partialmethod(end, strata=Strata.validate)
    on_test_epoch_end = partialmethod(end, strata=Strata.test)
    on_predict_epoch_end = partialmethod(end, strata=Strata.predict)

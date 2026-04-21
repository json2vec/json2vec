from __future__ import annotations

import datetime
from collections import defaultdict
from functools import partialmethod
from typing import TYPE_CHECKING

import torch
from lightning import Callback, Trainer

from json2vec.structs.enums import Metric, Strata

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec


class ThroughputLogger(Callback):
    def __init__(self):
        super().__init__()

        self.timestamp: dict[Strata, datetime.datetime] = defaultdict(lambda: datetime.datetime.now())

    def start(self, trainer: Trainer, pl_module: JSON2Vec, batch, batch_idx, strata: Strata):
        self.timestamp[strata] = datetime.datetime.now()

    def end(self, trainer: Trainer, pl_module: JSON2Vec, outputs, batch, batch_idx, strata: Strata):
        now = datetime.datetime.now()
        then = self.timestamp[strata]
        throughput = pl_module.session.structure.batch_size / (now - then).total_seconds()

        pl_module.track((Metric.throughput, strata), torch.tensor(throughput))

    on_train_batch_start = partialmethod(start, strata=Strata.train)
    on_validation_batch_start = partialmethod(start, strata=Strata.validate)
    on_test_batch_start = partialmethod(start, strata=Strata.test)

    on_train_batch_end = partialmethod(end, strata=Strata.train)
    on_validation_batch_end = partialmethod(end, strata=Strata.validate)
    on_test_batch_end = partialmethod(end, strata=Strata.test)

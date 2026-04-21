from __future__ import annotations

from functools import partialmethod
from typing import TYPE_CHECKING, Literal

from lightning import Callback, Trainer
from loguru import logger

from json2vec.structs.enums import Strata

if TYPE_CHECKING:
    from json2vec.architecture.root import JSON2Vec


class EpochLifecycleLogger(Callback):
    def info(
        self,
        trainer: Trainer,
        pl_module: JSON2Vec,
        strata: Strata,
        hook: Literal["start", "end"],
    ):
        logger.bind(
            source="lightning",
            rank=pl_module.global_rank,
            epoch=pl_module.current_epoch,
            step=pl_module.global_step,
            hook=hook,
            strata=str(strata),
        ).info(f"{hook}ing {strata} epoch {pl_module.current_epoch}")

    on_train_epoch_start = partialmethod(info, strata=Strata.train, hook="start")
    on_train_epoch_end = partialmethod(info, strata=Strata.train, hook="end")

    on_validation_epoch_start = partialmethod(info, strata=Strata.validate, hook="start")
    on_validation_epoch_end = partialmethod(info, strata=Strata.validate, hook="end")

    on_test_epoch_start = partialmethod(info, strata=Strata.test, hook="start")
    on_test_epoch_end = partialmethod(info, strata=Strata.test, hook="end")

    on_predict_epoch_start = partialmethod(info, strata=Strata.predict, hook="start")
    on_predict_epoch_end = partialmethod(info, strata=Strata.predict, hook="end")

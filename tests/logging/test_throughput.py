import datetime
from types import SimpleNamespace

import torch

from json2vec.logging.throughput import ThroughputLogger
from json2vec.structs.enums import Metric, Strata


def test_throughput_logger_tracks_once_per_epoch():
    tracked: list[tuple[tuple[str, ...], torch.Tensor]] = []

    def track(names: tuple[str, ...], /, value: torch.Tensor) -> torch.Tensor:
        tracked.append((names, value))
        return value

    callback = ThroughputLogger()
    callback.timestamp[Strata.train] = datetime.datetime.now() - datetime.timedelta(seconds=2)
    module = SimpleNamespace(batch_size=10, track=track)

    callback.count(trainer=object(), pl_module=module, outputs=None, batch=None, batch_idx=0, strata=Strata.train)
    callback.count(trainer=object(), pl_module=module, outputs=None, batch=None, batch_idx=1, strata=Strata.train)
    assert tracked == []

    callback.end(trainer=object(), pl_module=module, strata=Strata.train)

    assert len(tracked) == 1
    names, value = tracked[0]
    assert names == (Metric.throughput, Strata.train)
    assert value.item() > 0.0


def test_throughput_logger_does_not_use_module_logging_during_predict():
    logged: list[tuple[dict[str, float], int | None]] = []

    class Logger:
        def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
            logged.append((metrics, step))

    def track(*args, **kwargs):
        raise AssertionError("prediction hooks must not call Model.track")

    callback = ThroughputLogger()
    callback.timestamp[Strata.predict] = datetime.datetime.now() - datetime.timedelta(seconds=2)
    trainer = SimpleNamespace(logger=Logger(), is_global_zero=True, global_step=7)
    module = SimpleNamespace(batch_size=10, track=track)

    callback.count(
        trainer=trainer,
        pl_module=module,
        outputs=None,
        batch=None,
        batch_idx=0,
        strata=Strata.predict,
    )
    callback.end(trainer=trainer, pl_module=module, strata=Strata.predict)

    assert callback.throughput[Strata.predict] > 0.0
    assert logged == [({"throughput/predict": callback.throughput[Strata.predict]}, 7)]

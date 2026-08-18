from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from tensordict import TensorDict
from torch.utils.data import DataLoader

import relflow.logging.throughput as throughput_module
from relflow.logging.throughput import ThroughputLogger
from relflow.structs.enums import Metric, Strata


def batch(size: int) -> TensorDict:
    field = TensorDict({"value": torch.zeros(size)}, batch_size=[size])
    return TensorDict({"field": field})


def identity(value):
    return value


def test_throughput_logger_counts_partial_batches_and_logs_once_per_epoch(monkeypatch):
    logged: list[tuple[str, torch.Tensor, dict[str, bool]]] = []

    def log(name: str, value: torch.Tensor, **kwargs: bool) -> None:
        logged.append((name, value, kwargs))

    def track(*args, **kwargs):
        raise AssertionError("globally reduced throughput must not use Model.track")

    callback = ThroughputLogger()
    callback.timestamp[Strata.train] = 10.0
    module = SimpleNamespace(device=torch.device("cpu"), log=log, track=track)
    trainer = SimpleNamespace(strategy=None)
    monkeypatch.setattr(throughput_module.time, "perf_counter", lambda: 12.0)

    callback.count(trainer=trainer, pl_module=module, outputs=None, batch=batch(10), batch_idx=0, strata=Strata.train)
    callback.count(trainer=trainer, pl_module=module, outputs=None, batch=batch(3), batch_idx=1, strata=Strata.train)
    assert logged == []

    callback.end(trainer=trainer, pl_module=module, strata=Strata.train)

    assert len(logged) == 1
    name, value, kwargs = logged[0]
    assert name == f"{Metric.throughput.value}/{Strata.train.value}"
    assert value.item() == pytest.approx(6.5)
    assert kwargs == {"on_step": False, "on_epoch": True, "sync_dist": False}


def test_throughput_remains_available_in_lightning_callback_metrics():
    class Module(LightningModule):
        def validation_step(self, batch, batch_idx):
            return None

    callback = ThroughputLogger()
    trainer = Trainer(
        accelerator="cpu",
        callbacks=[callback],
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
    )
    dataloader = DataLoader(
        [batch(3), batch(1)],
        batch_size=None,
        collate_fn=identity,
        num_workers=0,
    )

    trainer.validate(Module(), dataloaders=dataloader, verbose=False)

    metric = trainer.callback_metrics["throughput/validate"]
    assert metric.item() == pytest.approx(callback.throughput[Strata.validate])


def test_throughput_logger_resets_observation_count_at_epoch_start(monkeypatch):
    callback = ThroughputLogger()
    callback.observations[Strata.validate] = 12
    monkeypatch.setattr(throughput_module.time, "perf_counter", lambda: 31.0)

    callback.start(trainer=object(), pl_module=object(), strata=Strata.validate)

    assert callback.observations[Strata.validate] == 0
    assert callback.timestamp[Strata.validate] == 31.0


def test_prediction_throughput_uses_global_observations_and_slowest_rank(monkeypatch):
    logged: list[tuple[dict[str, float], int | None]] = []
    reductions: list[tuple[str, torch.dtype]] = []

    class Logger:
        def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
            logged.append((metrics, step))

    class Strategy:
        def reduce(self, value: torch.Tensor, reduce_op: str) -> torch.Tensor:
            reductions.append((reduce_op, value.dtype))
            if reduce_op == "sum":
                return value.new_tensor(9)
            return value.new_tensor(4.0)

    def track(*args, **kwargs):
        raise AssertionError("prediction hooks must not call Model.track")

    callback = ThroughputLogger()
    callback.timestamp[Strata.predict] = 10.0
    trainer = SimpleNamespace(
        logger=Logger(),
        is_global_zero=True,
        global_step=7,
        strategy=Strategy(),
    )
    module = SimpleNamespace(device=torch.device("cpu"), track=track)
    monkeypatch.setattr(throughput_module.time, "perf_counter", lambda: 12.0)

    callback.count(
        trainer=trainer,
        pl_module=module,
        outputs=None,
        batch=batch(3),
        batch_idx=0,
        strata=Strata.predict,
    )
    callback.end(trainer=trainer, pl_module=module, strata=Strata.predict)

    assert reductions == [("sum", torch.int64), ("max", torch.float32)]
    assert callback.throughput[Strata.predict] == pytest.approx(2.25)
    assert logged == [({"throughput/predict": callback.throughput[Strata.predict]}, 7)]

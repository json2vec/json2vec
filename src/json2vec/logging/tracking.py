from __future__ import annotations

import enum
from typing import Callable

from lightning.pytorch.loggers import Logger
from loguru import logger

from json2vec.structs.environment import TrackingEnvironment


class LoggingFramework(enum.StrEnum):
    wandb = "wandb"
    neptune = "neptune"
    comet = "comet"
    mlflow = "mlflow"
    tensorboard = "tensorboard"
    csv = "csv"


class LoggerFactory:
    AUTO_DETECTION_ORDER: tuple[LoggingFramework, ...] = (
        LoggingFramework.wandb,
        LoggingFramework.neptune,
        LoggingFramework.comet,
        LoggingFramework.mlflow,
        LoggingFramework.tensorboard,
        LoggingFramework.csv,
    )
    AUTO_LOGGER_FIELDS: dict[LoggingFramework, tuple[str, ...]] = {
        LoggingFramework.wandb: ("wandb_api_key",),
        LoggingFramework.neptune: ("neptune_api_token",),
        LoggingFramework.comet: ("comet_api_key",),
        LoggingFramework.mlflow: ("mlflow_tracking_uri",),
        LoggingFramework.tensorboard: ("tensorboard_log_dir",),
        LoggingFramework.csv: ("csv_log_dir",),
    }

    @staticmethod
    def wandb(project: str, run: str, notes: str) -> Logger:
        from lightning.pytorch.loggers import WandbLogger

        tracker = WandbLogger(project=project, name=run)
        if notes:
            try:
                tracker.experiment.notes = notes
            except Exception:
                logger.bind(component="tracking", backend=LoggingFramework.wandb.value).warning(
                    "failed to attach run notes"
                )
        return tracker

    @staticmethod
    def neptune(project: str, run: str, notes: str) -> Logger:
        from lightning.pytorch.loggers import NeptuneLogger

        tracker = NeptuneLogger(project=project, name=run)
        if notes:
            try:
                tracker.experiment["sys/notes"] = notes
            except Exception:
                logger.bind(component="tracking", backend=LoggingFramework.neptune.value).warning(
                    "failed to attach run notes"
                )
        return tracker

    @staticmethod
    def comet(project: str, run: str, notes: str) -> Logger:
        from lightning.pytorch.loggers import CometLogger

        tracker = CometLogger(project_name=project, experiment_name=run)
        if notes:
            try:
                tracker.experiment.log_other("notes", notes)
            except Exception:
                logger.bind(component="tracking", backend=LoggingFramework.comet.value).warning(
                    "failed to attach run notes"
                )
        return tracker

    @staticmethod
    def mlflow(project: str, run: str, notes: str) -> Logger:
        from lightning.pytorch.loggers import MLFlowLogger

        tags = {"notes": notes} if notes else None
        return MLFlowLogger(experiment_name=project, run_name=run, tags=tags)

    @staticmethod
    def tensorboard(project: str, run: str, _: str) -> Logger:
        from lightning.pytorch.loggers import TensorBoardLogger

        save_dir = TrackingEnvironment.from_env().resolved_tensorboard_log_dir
        return TensorBoardLogger(save_dir=save_dir, name=project, version=run)

    @staticmethod
    def csv(project: str, run: str, _: str) -> Logger:
        from lightning.pytorch.loggers import CSVLogger

        save_dir = TrackingEnvironment.from_env().resolved_csv_log_dir
        return CSVLogger(save_dir=save_dir, name=project, version=run)

    @staticmethod
    def _builders() -> dict[LoggingFramework, Callable[[str, str, str], Logger]]:
        return {
            LoggingFramework.wandb: LoggerFactory.wandb,
            LoggingFramework.neptune: LoggerFactory.neptune,
            LoggingFramework.comet: LoggerFactory.comet,
            LoggingFramework.mlflow: LoggerFactory.mlflow,
            LoggingFramework.tensorboard: LoggerFactory.tensorboard,
            LoggingFramework.csv: LoggerFactory.csv,
        }

    @staticmethod
    def _resolve_framework() -> LoggingFramework | None:
        settings = TrackingEnvironment.from_env()
        forced = settings.logger
        if forced is not None:
            forced = forced.lower()
            if forced in {"none", "false", "off", "disabled"}:
                return None

            try:
                return LoggingFramework(forced)
            except ValueError:
                logger.bind(component="tracking", backend=forced).warning("unsupported logger backend override")
                return None

        for backend in LoggerFactory.AUTO_DETECTION_ORDER:
            if any(getattr(settings, field) is not None for field in LoggerFactory.AUTO_LOGGER_FIELDS[backend]):
                return backend

        return None

    @staticmethod
    def create(project: str, run: str, notes: str) -> Logger | bool:
        backend = LoggerFactory._resolve_framework()
        if backend is None:
            return False

        builder = LoggerFactory._builders().get(backend)
        if builder is None:
            logger.bind(component="tracking", backend=backend.value).warning("unsupported logger backend")
            return False

        try:
            tracker = builder(project, run, notes)
        except Exception:
            logger.bind(component="tracking", backend=backend.value).exception("failed to initialize trainer logger")
            return False

        logger.bind(component="tracking", backend=backend.value, project=project, run=run).info("enabled trainer logger")
        return tracker

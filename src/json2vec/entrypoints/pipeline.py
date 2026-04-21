from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.trainer.trainer import Trainer
from loguru import logger

from json2vec.architecture.root import JSON2Vec
from json2vec.inference.callback import Writer
from json2vec.logging.epoch import EpochLifecycleLogger
from json2vec.logging.throughput import ThroughputLogger
from json2vec.logging.tracking import LoggerFactory
from json2vec.structs.enums import Metric, Stage, Strata
from json2vec.structs.experiment import Experiment, PatchOp, Session


def build(model: JSON2Vec, callbacks: list[Callback], names: list[str] | None = None) -> Trainer:
    active_callbacks: list[Callback] = list(callbacks)
    if not any(isinstance(callback, EpochLifecycleLogger) for callback in active_callbacks):
        active_callbacks.append(EpochLifecycleLogger())

    logger.bind(
        component="trainer",
        session=model.session.name,
        stage=model.session.task,
        callbacks=[type(callback).__name__ for callback in active_callbacks],
    ).info("building lightning trainer")

    return Trainer(
        accelerator="auto" if torch.cuda.is_available() else "cpu",
        precision="bf16-mixed" if torch.cuda.is_available() else None,
        logger=LoggerFactory.create(*names) if names is not None else False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=active_callbacks,
        **model.session.trainer,
    )


def fit(
    names: list[str],
    session: Session | None = None,
    checkpoint: str | os.PathLike[str] | None = None,
    patches: list[PatchOp] | None = None,
) -> Path:
    logger.bind(component="task", task="fit", session=session.name if session else None).info("starting fit task")

    checkpoint_path = str(checkpoint) if checkpoint is not None else None
    model: JSON2Vec = JSON2Vec.get_or_create(session=session, checkpoint=checkpoint_path)
    model.session = model.session.patch(patches)

    monitor = f"{Metric.loss}/{Strata.validate}"
    filename: str = f"{model.session.structure.name}-{model.session.name}-" + "{epoch}-{step}-{val_loss:.2f}"

    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpointer: ModelCheckpoint = ModelCheckpoint(dirpath=checkpoint_dir, filename=filename, monitor=monitor)
    callbacks: list[Callback] = [ThroughputLogger(), checkpointer]

    if (patience := model.session.patience) is not None:
        callbacks.append(EarlyStopping(patience=patience, monitor=monitor))

    trainer: Trainer = build(model=model, callbacks=callbacks, names=names)
    trainer.fit(model=model)

    best_path = Path(str(checkpointer.best_model_path))
    logger.bind(component="task", task="fit", session=model.session.name, checkpoint=str(best_path)).info(
        "finished fit task"
    )
    return best_path


def validate(
    names: list[str],
    checkpoint: str | os.PathLike[str],
    session: Session | None = None,
    patches: list[PatchOp] | None = None,
) -> None:
    logger.bind(component="task", task="validate", session=session.name if session else None).info(
        "starting validate task"
    )

    model: JSON2Vec = JSON2Vec.get_or_create(session=session, checkpoint=str(checkpoint))
    model.session = model.session.patch(patches)

    callbacks: list[Callback] = [ThroughputLogger()]
    trainer: Trainer = build(model=model, callbacks=callbacks, names=names)
    trainer.validate(model=model)
    logger.bind(component="task", task="validate", session=model.session.name).info("finished validate task")


def test(
    names: list[str],
    checkpoint: str | os.PathLike[str],
    session: Session | None = None,
    patches: list[PatchOp] | None = None,
) -> None:
    logger.bind(component="task", task="test", session=session.name if session else None).info("starting test task")

    model: JSON2Vec = JSON2Vec.get_or_create(session=session, checkpoint=str(checkpoint))
    model.session = model.session.patch(patches)

    callbacks: list[Callback] = [ThroughputLogger()]
    trainer: Trainer = build(model=model, callbacks=callbacks, names=names)
    trainer.test(model=model)
    logger.bind(component="task", task="test", session=model.session.name).info("finished test task")


def predict(
    session: Session | None,
    names: list[str] | None,
    checkpoint: str | os.PathLike[str],
    patches: list[PatchOp] | None = None,
) -> Path:
    logger.bind(component="task", task="predict", session=session.name if session else None).info("starting predict task")

    model: JSON2Vec = JSON2Vec.get_or_create(session=session, checkpoint=str(checkpoint))
    model.session = model.session.patch(patches)

    os.makedirs(name=(outpath := "tmp/predictions"), exist_ok=True)
    callbacks: list[Callback] = [Writer(outpath)]
    trainer: Trainer = build(model=model, callbacks=callbacks, names=names)
    trainer.predict(model=model, return_predictions=False)

    output_path = Path(outpath)
    logger.bind(component="task", task="predict", session=model.session.name, output=str(output_path)).info(
        "finished predict task"
    )
    return output_path


def execute(experiment: Experiment) -> dict[str, Any]:
    logger.bind(
        component="pipeline",
        project=experiment.project,
        run=experiment.name,
        sessions=len(experiment.sessions),
    ).info("starting experiment execution")

    checkpoint: str | os.PathLike[str] | None = experiment.checkpoint
    names: list[str] = [experiment.project, experiment.name, experiment.notes]

    tasks: dict[Stage, Any] = {
        Stage.fit: fit,
        Stage.validate: validate,
        Stage.test: test,
        Stage.predict: predict,
    }

    outputs: dict[str, Any] = {}

    for session in experiment.sessions:
        logger.bind(component="pipeline", session=session.name, stage=session.task).info("dispatching session")
        task = tasks[session.task]

        output = task(
            session=session,
            checkpoint=checkpoint,
            names=names,
        )
        outputs[session.name] = output

        if isinstance(output, (str, os.PathLike)) and session.task == Stage.fit:
            checkpoint = output

    logger.bind(component="pipeline", project=experiment.project, run=experiment.name).info(
        "finished experiment execution"
    )

    return outputs

"""Checkpoint serialization helpers for JSON2Vec models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from json2vec.architecture.graph import ModelGraph
from json2vec.structs.experiment import Hyperparameters

if TYPE_CHECKING:
    from json2vec.architecture.root import Model


class CheckpointState:
    """Save, load, and restore model state without owning the public facade."""

    required_fields = {"state_dict", "hyperparameters", "batch_size"}

    @staticmethod
    def dump(module: "Model", checkpoint: dict[str, Any]) -> None:
        checkpoint["hyperparameters"] = module.hyperparameters.model_dump(mode="python")
        checkpoint["batch_size"] = module.batch_size

    @staticmethod
    def save(module: "Model", pathname: str | Path) -> None:
        path = Path(pathname)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {"state_dict": module.state_dict()}
        CheckpointState.dump(module, checkpoint)
        torch.save(checkpoint, path)

    @staticmethod
    def restore(module: "Model", checkpoint: dict[str, Any]) -> None:
        missing = CheckpointState.required_fields - set(checkpoint)
        if missing:
            fields = ", ".join(sorted(missing))
            raise ValueError(f"missing checkpoint fields: {fields}")

        device = module.device
        was_training = module.training
        module.hyperparameters = Hyperparameters.model_validate(checkpoint["hyperparameters"])
        module.batch_size = checkpoint["batch_size"]
        ModelGraph.install(module)
        if isinstance(device, torch.device):
            module.to(device=device)
        module.load_state_dict(state_dict=checkpoint["state_dict"])
        module.train(was_training)

    @staticmethod
    def load(model_cls: type["Model"], checkpoint: str | Path) -> "Model":
        path = Path(checkpoint)
        logger.bind(component="model_factory", checkpoint=str(path)).info("loading Model from checkpoint")
        state = torch.load(path, weights_only=False, map_location="cpu")
        if "hyperparameters" not in state:
            raise ValueError("missing hyperparameters in checkpoint")

        model = model_cls(
            hyperparameters=Hyperparameters.model_validate(state["hyperparameters"]),
            batch_size=state["batch_size"],
        )
        model.restore_checkpoint_state(state)
        logger.bind(component="model_factory", checkpoint=str(path)).info("restored model state from checkpoint")

        return model

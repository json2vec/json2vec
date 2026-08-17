"""Checkpoint serialization helpers for `relflow` models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning.pytorch as lit
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from relflow.architecture.graph import ModelGraph
from relflow.rich import console, is_verbose, record_incident
from relflow.structs.experiment import Schema

if TYPE_CHECKING:
    from relflow.architecture.root import Model


class RollbackCheckpoint(ModelCheckpoint):
    """Checkpoint the best model during fit and restore it into the module at fit end."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.save_weights_only:
            raise ValueError("RollbackCheckpoint requires full checkpoints; set save_weights_only=False")
        if self.save_top_k == 0:
            raise ValueError("RollbackCheckpoint requires at least one saved checkpoint; set save_top_k != 0")

    def on_fit_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        from relflow.architecture.root import Model

        super().on_fit_end(trainer=trainer, pl_module=pl_module)
        if not isinstance(pl_module, Model):
            raise TypeError("RollbackCheckpoint can only restore relflow Model instances")

        best_model_path = self.best_model_path
        if not best_model_path:
            raise RuntimeError("RollbackCheckpoint did not find a best checkpoint to restore")

        strategy = getattr(trainer, "strategy", None)
        if strategy is not None:
            strategy.barrier("rollback_checkpoint_load")
            checkpoint = strategy.checkpoint_io.load_checkpoint(
                best_model_path,
                map_location=pl_module.device,
                weights_only=False,
            )
        else:
            checkpoint = torch.load(best_model_path, weights_only=False, map_location=pl_module.device)

        pl_module.restore_checkpoint_state(checkpoint)
        if (
            getattr(trainer, "is_global_zero", True)
            and record_incident("checkpoint-rollback", id(self), best_model_path).emit
        ):
            console.log(
                "[relflow.warning]restored the best checkpoint at fit end[/]",
                {"checkpoint": Path(best_model_path), "score": self.best_model_score},
            )


class CheckpointState:
    """Save, load, and restore model state without owning the public facade."""

    required_fields = {"state_dict", "schema", "batch_size"}

    @staticmethod
    def dump(module: "Model", checkpoint: dict[str, Any]) -> None:
        checkpoint["schema"] = module.schema.model_dump(mode="python")
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
        module.schema = Schema.model_validate(checkpoint["schema"])
        module.batch_size = checkpoint["batch_size"]
        ModelGraph.install(module)
        if isinstance(device, torch.device):
            module.to(device=device)
        module.load_state_dict(state_dict=checkpoint["state_dict"])
        module.train(was_training)

    @staticmethod
    def load(model_cls: type["Model"], checkpoint: str | Path) -> "Model":
        path = Path(checkpoint)
        if is_verbose():
            console.log("[relflow.info]loading model checkpoint[/]", path)
        state = torch.load(path, weights_only=False, map_location="cpu")
        if "schema" not in state:
            raise ValueError("missing schema in checkpoint")

        model = model_cls(
            schema=Schema.model_validate(state["schema"]),
            batch_size=state["batch_size"],
        )
        model.restore_checkpoint_state(state)

        return model

from __future__ import annotations

import argparse
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import optuna
import polars as pl
from lightning.pytorch.callbacks import BatchSizeFinder
from rich import print

import json2vec as j2v


def records() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"amount": 18.50, "merchant": "grocery", "channel": "card", "label": "normal"},
            {"amount": 21.20, "merchant": "grocery", "channel": "card", "label": "normal"},
            {"amount": 95.00, "merchant": "electronics", "channel": "card", "label": "review"},
            {"amount": 130.00, "merchant": "travel", "channel": "wire", "label": "review"},
            {"amount": 8.25, "merchant": "coffee", "channel": "card", "label": "normal"},
            {"amount": 160.00, "merchant": "travel", "channel": "wire", "label": "review"},
            {"amount": 11.75, "merchant": "coffee", "channel": "cash", "label": "normal"},
            {"amount": 88.00, "merchant": "electronics", "channel": "wire", "label": "review"},
        ]
    )


BASE_PARAMS = j2v.Hyperparameters.from_schema(
    j2v.Number(name="amount"),
    j2v.Category(name="merchant", max_vocab_size=16),
    j2v.Category(name="channel", max_vocab_size=8),
    j2v.Category(name="label", target=True, max_vocab_size=2),
    name="transaction",
    d_model=32,
    n_layers=1,
    n_heads=4,
)


def objective(trial: optuna.Trial) -> float:
    model = j2v.helpers.tune(BASE_PARAMS, trial=trial, batch_size=1)
    data = records()
    datamodule = j2v.PolarsDataModule(
        model=model,
        train=data,
        validate=data,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        observation_buffer_size=16,
        sample_rate=1.0,
    )

    with TemporaryDirectory() as root:
        trainer = lit.Trainer(
            accelerator="cpu",
            callbacks=[BatchSizeFinder()],
            default_root_dir=root,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            limit_train_batches=1,
            limit_val_batches=1,
        )
        trainer.fit(model=model, datamodule=datamodule)
        loss = trainer.callback_metrics["loss/validate"]

    trial.set_user_attr("batch_size", model.batch_size)
    return float(loss.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune a tiny json2vec model with Optuna.")
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()

    lit.seed_everything(7, workers=True)
    sampler = optuna.samplers.TPESampler(seed=7)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials)

    best_model = j2v.helpers.tune(
        BASE_PARAMS,
        trial=study.best_trial,
        batch_size=study.best_trial.user_attrs["batch_size"],
    )

    print(f"best validation loss: {study.best_value:.4f}")
    print(f"best batch size: {best_model.batch_size}")
    print("best parameters")
    print(study.best_trial.params)
    print(best_model)


if __name__ == "__main__":
    main()

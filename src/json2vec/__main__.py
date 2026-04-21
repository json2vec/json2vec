from __future__ import annotations

import argparse

from loguru import logger

from json2vec.entrypoints import execute
from json2vec.structs.experiment import Experiment


def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--notes", type=str, default=None)
    parser.add_argument("--experiments", type=str, default="experiments")
    args = parser.parse_args()

    experiment: Experiment = Experiment.from_config(
        args.experiments,
        experiment=args.experiment,
        name=args.name,
        notes=args.notes,
    )
    outputs = execute(experiment=experiment)

    for session_name, output in outputs.items():
        logger.info(f"session={session_name} output={output}")


if __name__ == "__main__":
    train()

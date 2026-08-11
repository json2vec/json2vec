"""Benchmark keyed cross-branch transfer with and without References.

Branch A contains a random categorical symbol keyed by a Hash ID. Branch B is
independently shuffled and must reconstruct the symbol for each of its IDs.
The benchmark crosses three routing strategies with three tasks:

``tree``
    Use only the ordinary tree path through the root Branch.
``reference``
    Add ``rf.Reference("record/a")`` to Branch B.
``reference-grafted``
    Add the same Reference with ``graft=True``, removing A from its native
    parent path while retaining it as Reference context for B.

``linked`` uses A's IDs in B, ``unlinked`` substitutes canonical decoy IDs,
and ``global`` uses decoy IDs but one shared symbol per observation. The
linked/unlinked pair isolates keyed transfer; global is a positive control for
aggregate transfer that does not require item correspondence.

Every routing strategy replays the same generated observations and model seed.
Training uses a fixed optimizer-step budget. For a quick smoke run:

    uv run python benchmarks/hash_transfer.py \
        --classes 2 --lengths 8 --seeds 17 --steps 20 \
        --jsonl /tmp/hash-transfer.jsonl

For a full run, use several paired seeds and sweep sequence length, for example:

    uv run python benchmarks/hash_transfer.py \
        --classes 2 4 8 --lengths 8 16 32 --seeds 17 29 43 59 71 \
        --steps 320 --jsonl hash-transfer.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any, Literal, TypeAlias, cast

import lightning.pytorch as lit
import torch

import relflow as rf

Task: TypeAlias = Literal["linked", "unlinked", "global"]
Routing: TypeAlias = Literal["tree", "reference", "reference-grafted"]


@dataclass(frozen=True)
class Trial:
    classes: int
    length: int
    seed: int
    routing: Routing
    task: Task


def record_generator(
    *,
    seed: int,
    observations: int,
    length: int,
    classes: int,
    task: Task,
) -> Callable[[], Iterator[dict[str, list[dict[str, str]]]]]:
    """Return a replayable stream whose latent draws are task-independent.

    Real IDs, decoy IDs, item labels, the global label, and both permutations
    are generated unconditionally and in a fixed order. Consequently, linked
    and unlinked observations differ only in the ID selected for Branch B;
    selecting the global task does not perturb IDs or ordering either.
    """

    def generate() -> Iterator[dict[str, list[dict[str, str]]]]:
        rng = random.Random(seed)
        balanced_labels = [label for label in range(classes) for _ in range(length // classes)]

        for observation in range(observations):
            item_labels = balanced_labels.copy()
            rng.shuffle(item_labels)
            global_label = rng.randrange(classes)

            identifiers = [f"real:{seed}:{observation}:{slot}:{rng.getrandbits(64)}" for slot in range(length)]
            decoys = [f"decoy:{seed}:{observation}:{slot}:{rng.getrandbits(64)}" for slot in range(length)]

            a_order = list(range(length))
            b_order = list(range(length))
            rng.shuffle(a_order)
            rng.shuffle(b_order)

            labels = [global_label] * length if task == "global" else item_labels
            branch_a = [{"id": identifiers[slot], "symbol": str(labels[slot])} for slot in a_order]
            branch_b = [
                {
                    "id": identifiers[slot] if task == "linked" else decoys[slot],
                    "symbol": str(labels[slot]),
                }
                for slot in b_order
            ]
            yield {"a": branch_a, "b": branch_b}

    return generate


def build_model(args: argparse.Namespace, trial: Trial) -> rf.Model:
    branch_b: dict[str, Any] = {
        "length": trial.length,
        "id": rf.Hash(n_hashes=args.hashes),
        "symbol": rf.Category(size=trial.classes, target=True, p_unavailable=0.0),
    }
    if trial.routing == "reference":
        branch_b["reference"] = rf.Reference("record/a")
    elif trial.routing == "reference-grafted":
        branch_b["reference"] = rf.Reference("record/a", graft=True)

    return rf.Model(
        name="record",
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        batch_size=args.batch_size,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=args.learning_rate),
        a=rf.Branch(
            length=trial.length,
            id=rf.Hash(n_hashes=args.hashes),
            symbol=rf.Category(size=trial.classes, p_unavailable=0.0),
        ),
        b=rf.Branch(**branch_b),
    )


def metric(metrics: dict[str, Any], suffix: str) -> float:
    matches = [float(value) for key, value in metrics.items() if "record:b:symbol" in key and key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one metric ending in {suffix!r}, found {len(matches)} in {tuple(metrics)}")
    return matches[0]


def run_trial(args: argparse.Namespace, trial: Trial) -> dict[str, Any]:
    lit.seed_everything(trial.seed, workers=True)
    model = build_model(args, trial)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    datamodule = rf.SyntheticDataModule(
        model=model,
        train=record_generator(
            seed=trial.seed,
            observations=args.train_observations,
            length=trial.length,
            classes=trial.classes,
            task=trial.task,
        ),
        test=record_generator(
            seed=trial.seed + args.test_seed_offset,
            observations=args.test_observations,
            length=trial.length,
            classes=trial.classes,
            task=trial.task,
        ),
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        observation_buffer_size=1,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        deterministic=True,
        max_epochs=-1,
        max_steps=args.steps,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=args.progress,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )

    started = time.perf_counter()
    trainer.fit(model=model, datamodule=datamodule)
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    test_metrics = trainer.test(model=model, datamodule=datamodule, verbose=False)[0]
    test_seconds = time.perf_counter() - started
    accuracy = metric(test_metrics, "/test:accuracy:content")
    loss_nats = metric(test_metrics, "/test:loss:content")
    loss_bits_per_item = loss_nats / math.log(2)
    decoded_information_bits_per_item = max(0.0, math.log2(trial.classes) - loss_bits_per_item)
    steps = int(trainer.global_step)
    reference_context_tokens = 0 if trial.routing == "tree" else 2 * trial.length

    return {
        "record": "trial",
        **asdict(trial),
        "hashes": args.hashes,
        "d_model": args.d_model,
        "layers": args.layers,
        "heads": args.heads,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "requested_train_steps": args.steps,
        "train_observations": args.train_observations,
        "test_observations": args.test_observations,
        "test_seed": trial.seed + args.test_seed_offset,
        "torch_version": torch.__version__,
        "lightning_version": lit.__version__,
        "accelerator": "cpu",
        "chance_accuracy": 1.0 / trial.classes,
        "payload_bits": (
            math.log2(trial.classes)
            if trial.task == "global"
            else (math.lgamma(trial.length + 1) - trial.classes * math.lgamma((trial.length // trial.classes) + 1))
            / math.log(2)
        ),
        "accuracy": accuracy,
        "loss_nats": loss_nats,
        "loss_bits_per_item": loss_bits_per_item,
        "decoded_information_bits_per_item": decoded_information_bits_per_item,
        "decoded_information_bits_per_observation": decoded_information_bits_per_item * trial.length,
        "b_native_context_tokens": trial.length,
        "b_reference_context_tokens": reference_context_tokens,
        "b_total_context_tokens": trial.length + reference_context_tokens,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "train_steps": steps,
        "train_seconds": train_seconds,
        "seconds_per_step": train_seconds / max(steps, 1),
        "test_seconds": test_seconds,
        "test_observations_per_second": args.test_observations / max(test_seconds, sys.float_info.min),
        "test_items_per_second": (args.test_observations * trial.length / max(test_seconds, sys.float_info.min)),
    }


def bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float | None, float | None]:
    mean = statistics.fmean(values)
    if len(values) < 2 or samples == 0:
        return mean, None, None

    rng = random.Random(seed)
    bootstrapped = sorted(statistics.fmean(values[rng.randrange(len(values))] for _ in values) for _ in range(samples))
    return (
        mean,
        bootstrapped[int(0.025 * (samples - 1))],
        bootstrapped[int(0.975 * (samples - 1))],
    )


def summarize(
    results: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    configurations = sorted({(row["length"], row["classes"]) for row in results})

    for length, classes in configurations:
        selected = [row for row in results if row["length"] == length and row["classes"] == classes]
        lookup = {(row["routing"], row["task"], row["seed"]): row for row in selected}

        for routing in sorted({cast(str, row["routing"]) for row in selected}):
            for task in sorted({cast(str, row["task"]) for row in selected}):
                rows = [row for row in selected if row["routing"] == routing and row["task"] == task]
                if not rows:
                    continue
                summary: dict[str, Any] = {
                    "record": "summary",
                    "summary": "condition",
                    "length": length,
                    "classes": classes,
                    "routing": routing,
                    "task": task,
                    "seeds": sorted(row["seed"] for row in rows),
                }
                for name in (
                    "accuracy",
                    "loss_nats",
                    "decoded_information_bits_per_item",
                    "decoded_information_bits_per_observation",
                    "train_seconds",
                    "seconds_per_step",
                    "test_seconds",
                ):
                    mean, low, high = bootstrap_mean(
                        [row[name] for row in rows],
                        samples=bootstrap_samples,
                        seed=bootstrap_seed,
                    )
                    summary[f"{name}_mean"] = mean
                    summary[f"{name}_ci95_low"] = low
                    summary[f"{name}_ci95_high"] = high
                summaries.append(summary)

            paired_seeds = sorted(
                seed
                for seed in {row["seed"] for row in selected}
                if (routing, "linked", seed) in lookup and (routing, "unlinked", seed) in lookup
            )
            if paired_seeds:
                accuracy_delta = [
                    lookup[routing, "linked", seed]["accuracy"] - lookup[routing, "unlinked", seed]["accuracy"]
                    for seed in paired_seeds
                ]
                bits_saved = [
                    (lookup[routing, "unlinked", seed]["loss_nats"] - lookup[routing, "linked", seed]["loss_nats"])
                    / math.log(2)
                    for seed in paired_seeds
                ]
                accuracy_mean, accuracy_low, accuracy_high = bootstrap_mean(
                    accuracy_delta,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                bits_mean, bits_low, bits_high = bootstrap_mean(
                    bits_saved,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed,
                )
                summaries.append(
                    {
                        "record": "summary",
                        "summary": "linked-minus-unlinked",
                        "length": length,
                        "classes": classes,
                        "routing": routing,
                        "seeds": paired_seeds,
                        "accuracy_delta_mean": accuracy_mean,
                        "accuracy_delta_ci95_low": accuracy_low,
                        "accuracy_delta_ci95_high": accuracy_high,
                        "predictive_bits_saved_per_item_mean": bits_mean,
                        "predictive_bits_saved_per_item_ci95_low": bits_low,
                        "predictive_bits_saved_per_item_ci95_high": bits_high,
                    }
                )

        for routing in ("reference", "reference-grafted"):
            paired_seeds = sorted(
                seed
                for seed in {row["seed"] for row in selected}
                if all(
                    (candidate, task, seed) in lookup
                    for candidate in ("tree", routing)
                    for task in ("linked", "unlinked")
                )
            )
            if not paired_seeds:
                continue

            accuracy_interaction = [
                (lookup[routing, "linked", seed]["accuracy"] - lookup[routing, "unlinked", seed]["accuracy"])
                - (lookup["tree", "linked", seed]["accuracy"] - lookup["tree", "unlinked", seed]["accuracy"])
                for seed in paired_seeds
            ]
            bits_interaction = [
                (
                    lookup[routing, "unlinked", seed]["loss_nats"]
                    - lookup[routing, "linked", seed]["loss_nats"]
                    - lookup["tree", "unlinked", seed]["loss_nats"]
                    + lookup["tree", "linked", seed]["loss_nats"]
                )
                / math.log(2)
                for seed in paired_seeds
            ]
            accuracy_mean, accuracy_low, accuracy_high = bootstrap_mean(
                accuracy_interaction,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            bits_mean, bits_low, bits_high = bootstrap_mean(
                bits_interaction,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
            summaries.append(
                {
                    "record": "summary",
                    "summary": "reference-interaction",
                    "length": length,
                    "classes": classes,
                    "routing": routing,
                    "baseline": "tree",
                    "seeds": paired_seeds,
                    "accuracy_delta_mean": accuracy_mean,
                    "accuracy_delta_ci95_low": accuracy_low,
                    "accuracy_delta_ci95_high": accuracy_high,
                    "predictive_bits_saved_per_item_mean": bits_mean,
                    "predictive_bits_saved_per_item_ci95_low": bits_low,
                    "predictive_bits_saved_per_item_ci95_high": bits_high,
                }
            )

    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--lengths", "--length", dest="lengths", type=int, nargs="+", default=[8])
    parser.add_argument("--seeds", "--seed", dest="seeds", type=int, nargs="+", default=[17])
    parser.add_argument(
        "--routings",
        type=str,
        nargs="+",
        choices=("tree", "reference", "reference-grafted"),
        default=["tree", "reference", "reference-grafted"],
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=("linked", "unlinked", "global"),
        default=["linked", "unlinked", "global"],
    )
    parser.add_argument("--hashes", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--train-observations", type=int, default=2048)
    parser.add_argument("--test-observations", type=int, default=1024)
    parser.add_argument("--test-seed-offset", type=int, default=1_000_003)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--jsonl",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="write trial and paired-summary JSON Lines; omit PATH for stdout",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if any(classes < 2 for classes in args.classes):
        parser.error("every --classes value must be at least 2")
    if any(length < 2 for length in args.lengths):
        parser.error("every --lengths value must be at least 2")
    invalid = [(length, classes) for length in args.lengths for classes in args.classes if length % classes]
    if invalid:
        parser.error(f"every class count must divide every length; invalid pairs: {invalid}")
    if args.hashes < 1 or args.d_model < 1 or args.layers < 1 or args.heads < 1:
        parser.error("--hashes, --d-model, --layers, and --heads must be positive")
    if args.d_model % args.heads:
        parser.error("--heads must divide --d-model")
    if args.batch_size < 1 or args.steps < 1:
        parser.error("--batch-size and --steps must be positive")
    if args.train_observations < 1 or args.test_observations < 1:
        parser.error("observation counts must be positive")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be nonnegative")
    return args


def display_trial(row: dict[str, Any], stream: IO[str]) -> None:
    print(
        f"length={row['length']:<3} classes={row['classes']:<3} seed={row['seed']:<5} "
        f"routing={row['routing']:<18} task={row['task']:<8} "
        f"accuracy={row['accuracy']:.1%} loss={row['loss_bits_per_item']:.3f} bits/item "
        f"train={row['train_seconds']:.1f}s ({row['seconds_per_step']:.3f}s/step) "
        f"params={row['trainable_parameters']:,}",
        file=stream,
        flush=True,
    )


def display_summaries(rows: Sequence[dict[str, Any]], stream: IO[str]) -> None:
    print("\nPaired transfer estimates", file=stream)
    for row in rows:
        if row["summary"] == "linked-minus-unlinked":
            accuracy_ci = (
                ""
                if row["accuracy_delta_ci95_low"] is None
                else f" [{row['accuracy_delta_ci95_low']:+.1%}, {row['accuracy_delta_ci95_high']:+.1%}]"
            )
            bits_ci = (
                ""
                if row["predictive_bits_saved_per_item_ci95_low"] is None
                else (
                    f" [{row['predictive_bits_saved_per_item_ci95_low']:+.3f}, "
                    f"{row['predictive_bits_saved_per_item_ci95_high']:+.3f}]"
                )
            )
            print(
                f"length={row['length']:<3} classes={row['classes']:<3} routing={row['routing']:<18} "
                f"accuracy_delta={row['accuracy_delta_mean']:+.1%}{accuracy_ci} "
                f"bits_saved/item={row['predictive_bits_saved_per_item_mean']:+.3f}{bits_ci}",
                file=stream,
            )
        elif row["summary"] == "reference-interaction":
            accuracy_ci = (
                ""
                if row["accuracy_delta_ci95_low"] is None
                else f" [{row['accuracy_delta_ci95_low']:+.1%}, {row['accuracy_delta_ci95_high']:+.1%}]"
            )
            bits_ci = (
                ""
                if row["predictive_bits_saved_per_item_ci95_low"] is None
                else (
                    f" [{row['predictive_bits_saved_per_item_ci95_low']:+.3f}, "
                    f"{row['predictive_bits_saved_per_item_ci95_high']:+.3f}]"
                )
            )
            print(
                f"length={row['length']:<3} classes={row['classes']:<3} routing={row['routing']:<18} "
                f"vs_tree_accuracy={row['accuracy_delta_mean']:+.1%}{accuracy_ci} "
                f"vs_tree_bits/item={row['predictive_bits_saved_per_item_mean']:+.3f}{bits_ci}",
                file=stream,
            )


def main() -> None:
    args = parse_args()
    trials = [
        Trial(
            classes=classes,
            length=length,
            seed=seed,
            routing=cast(Routing, routing),
            task=cast(Task, task),
        )
        for length in args.lengths
        for classes in args.classes
        for seed in args.seeds
        for routing in args.routings
        for task in args.tasks
    ]
    random.Random(args.order_seed).shuffle(trials)

    with ExitStack() as stack:
        jsonl: IO[str] | None = None
        if args.jsonl == "-":
            jsonl = sys.stdout
        elif args.jsonl is not None:
            path = Path(args.jsonl)
            path.parent.mkdir(parents=True, exist_ok=True)
            jsonl = stack.enter_context(path.open("w", encoding="utf-8"))
        human = sys.stderr if jsonl is sys.stdout else sys.stdout

        results: list[dict[str, Any]] = []
        for index, trial in enumerate(trials, start=1):
            print(f"[{index}/{len(trials)}]", end=" ", file=human, flush=True)
            row = run_trial(args, trial)
            results.append(row)
            display_trial(row, human)
            if jsonl is not None:
                print(json.dumps(row, sort_keys=True, allow_nan=False), file=jsonl, flush=True)

        summaries = summarize(
            results,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.order_seed,
        )
        display_summaries(summaries, human)
        if jsonl is not None:
            for row in summaries:
                print(json.dumps(row, sort_keys=True, allow_nan=False), file=jsonl)


if __name__ == "__main__":
    main()

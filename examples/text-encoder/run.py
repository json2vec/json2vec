from __future__ import annotations

import argparse
from typing import Any

from rich import print

import json2vec as jv


def records() -> list[dict[str, str]]:
    return [
        {"message": "Refund was approved after manual review."},
        {"message": "Customer asked for shipment tracking details."},
    ]


def build_model(*, text_model: str) -> jv.Model:
    return jv.Model.from_tree(
        d_model=32,
        n_layers=1,
        n_heads=4,
        batch_size=2,
        embed=True,
        message=jv.Text(
            model=text_model,
            max_length=32,
            encoder_batch_size=4,
            encoder_pooling="mean",
        ),
    )


def payload(predictions: dict[Any, dict[str, Any]], path: str) -> dict[str, Any]:
    for address, value in predictions.items():
        if str(address) == path:
            return value

    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny BERT text-encoder json2vec example.")
    parser.add_argument(
        "--model",
        default="google/bert_uncased_L-2_H-128_A-2",
        help="Hugging Face model name or local model path.",
    )
    args = parser.parse_args()

    model = build_model(text_model=args.model)
    predictions = model.predict(records())
    root = payload(predictions, "record")
    embedding = root["embedding"]

    print(f"model: {args.model}")
    print(f"root embedding rows: {len(embedding)}")
    print(f"root embedding width: {len(embedding[0]) if embedding else 0}")
    print("first embedding preview:")
    print([round(value, 4) for value in embedding[0][:8]])


if __name__ == "__main__":
    main()

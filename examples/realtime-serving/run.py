from __future__ import annotations

import argparse
import os
from typing import Any

import lightning.pytorch as lit
import polars as pl
import pydantic
import torch

import json2vec as jv


class TransactionRequest(pydantic.BaseModel):
    amount: float = pydantic.Field(ge=0)
    merchant: str = pydantic.Field(min_length=1)


class RiskCandidate(pydantic.BaseModel):
    label: str
    probability: float


class RiskPrediction(pydantic.BaseModel):
    risk: str | None
    probability: float | None
    topk: list[RiskCandidate]
    embedding: list[float]


class PredictResponse(pydantic.BaseModel):
    predictions: RiskPrediction


RiskPrediction.model_rebuild(_types_namespace={"RiskCandidate": RiskCandidate})
PredictResponse.model_rebuild(_types_namespace={"RiskPrediction": RiskPrediction})


def records() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"amount": 12.0, "merchant": "coffee", "risk": "low"},
            {"amount": 27.0, "merchant": "grocery", "risk": "low"},
            {"amount": 86.0, "merchant": "pharmacy", "risk": "low"},
            {"amount": 250.0, "merchant": "electronics", "risk": "high"},
            {"amount": 430.0, "merchant": "jewelry", "risk": "high"},
            {"amount": 700.0, "merchant": "travel", "risk": "high"},
        ]
    )


def build_model() -> jv.Model:
    model = jv.Model.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        batch_size=4,
        embed=True,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-2),
        amount=jv.Number,
        merchant=jv.Category(size=16),
        risk=jv.Category(target=True, size=4, topk=[2]),
    )

    datamodule = jv.PolarsDataModule(
        model=model,
        train=records(),
        validate=records(),
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=1,
        limit_val_batches=1,
    )
    trainer.fit(model=model, datamodule=datamodule)
    model.eval()
    return model


def _payload(predictions: dict[Any, dict[str, Any]], path: str) -> dict[str, Any]:
    for address, payload in predictions.items():
        if str(address) == path:
            return payload

    return {}


def risk_response(context: dict[str, Any], predictions: dict[Any, dict[str, Any]]) -> dict[str, Any]:
    del context

    risk = _payload(predictions, "record/risk").get("content", {})
    root = _payload(predictions, "record")

    return {
        "risk": risk.get("value"),
        "probability": risk.get("probability"),
        "topk": risk.get("topk", []),
        "embedding": root.get("embedding", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and serve a tiny json2vec realtime model.")
    parser.add_argument("--host", default=os.environ.get("JSON2VEC_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("JSON2VEC_PORT", "8000")))
    parser.add_argument("--log-level", default=os.environ.get("JSON2VEC_LOG_LEVEL", "info"))
    args = parser.parse_args()

    deployment = jv.Deployment(
        model=build_model(),
        accelerator="cpu",
        max_batch_size=8,
        batch_timeout=0.0,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    deployment.postprocess(risk_response).forge(request=TransactionRequest, response=PredictResponse)
    deployment.serve()


if __name__ == "__main__":
    main()

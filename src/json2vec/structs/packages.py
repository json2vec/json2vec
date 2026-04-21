from typing import Any

import numpy as np
import torch
from tensordict import TensorClass, TensorDict

from json2vec.structs.enums import TensorKey
from json2vec.structs.tree import Address


class Parcel(TensorClass):
    payload: torch.Tensor
    origin: Address
    destination: Address | None


# @jaxtyped(typechecker=beartype)
class Prediction(TensorClass):
    address: Address
    payload: TensorDict[TensorKey, torch.Tensor]

    @staticmethod
    def unbatch(predictions: list["Prediction"]) -> list[list["Prediction"]]:
        if len(predictions) == 0:
            return []

        batch_sizes: set[int] = {len(prediction.payload) for prediction in predictions}

        if batch_sizes == {0}:
            return [predictions]

        if 0 in batch_sizes:
            raise ValueError("cannot unbatch predictions with mixed batched and unbatched payloads")

        if len(batch_sizes) != 1:
            raise ValueError(f"cannot unbatch predictions with inconsistent batch sizes: {sorted(batch_sizes)}")

        batch_size = next(iter(batch_sizes))

        return [
            [
                type(prediction)(
                    address=prediction.address,
                    payload=prediction.payload[index:index + 1],
                )
                for prediction in predictions
            ]
            for index in range(batch_size)
        ]

    @staticmethod
    def serialize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: Prediction.serialize(item) for key, item in value.items()}

        if isinstance(value, list):
            return [Prediction.serialize(item) for item in value]

        if isinstance(value, tuple):
            return [Prediction.serialize(item) for item in value]

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()

        if isinstance(value, np.ndarray):
            return value.tolist()

        if isinstance(value, np.generic):
            return value.item()

        return value

    @staticmethod
    def denest(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: Prediction.denest(item) for key, item in value.items()}

        if isinstance(value, list):
            items = [Prediction.denest(item) for item in value]
            if len(items) == 1:
                return items[0]
            return items

        if isinstance(value, tuple):
            items = [Prediction.denest(item) for item in value]
            if len(items) == 1:
                return items[0]
            return items

        return value

class Embedding(Prediction):

    @classmethod
    def from_parcel(cls, parcel: Parcel) -> "Embedding":
        return cls(
            address=parcel.origin,
            payload=TensorDict(
                {TensorKey.embedding: parcel.payload},
                batch_size=parcel.payload.shape[0],
            )
        )

    @staticmethod
    def normalize(values: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        # L2-normalize each embedding vector for consistent similarity distance scales.
        return torch.nn.functional.normalize(values, p=2, dim=-1, eps=eps)

    @classmethod
    def write(cls, prediction: "Embedding") -> dict[str, Any]:
        return {
            TensorKey.embedding.name: cls.normalize(
                prediction.payload[TensorKey.embedding]
                .detach()
                .float()
            ).cpu().tolist()
        }

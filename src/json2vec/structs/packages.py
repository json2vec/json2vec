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
                    payload=prediction.payload[index : index + 1],
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
    def squeeze(value: Any, *, preserve_first_dimension: bool = False) -> Any:  # ty:ignore[invalid-method-override]
        if isinstance(value, dict):
            return {
                key: Prediction.squeeze(item, preserve_first_dimension=preserve_first_dimension)
                for key, item in value.items()
            }

        if isinstance(value, torch.Tensor):
            start = 1 if preserve_first_dimension else 0
            for dim in reversed(range(start, value.ndim)):
                if value.shape[dim] == 1:
                    value = value.squeeze(dim)
            return value

        if isinstance(value, np.ndarray):
            axes = tuple(
                dim for dim, size in enumerate(value.shape) if size == 1 and (dim > 0 or not preserve_first_dimension)
            )
            return np.squeeze(value, axis=axes) if axes else value

        if isinstance(value, list):
            if preserve_first_dimension:
                return [Prediction.squeeze(item) for item in value]

            items = [Prediction.squeeze(item) for item in value]
            if len(items) == 1 and not isinstance(items[0], dict):
                return items[0]
            return items

        if isinstance(value, tuple):
            if preserve_first_dimension:
                return [Prediction.squeeze(item) for item in value]

            items = [Prediction.squeeze(item) for item in value]
            if len(items) == 1 and not isinstance(items[0], dict):
                return items[0]
            return items

        return value

    @staticmethod
    def denest(value: Any) -> Any:
        return Prediction.squeeze(value)

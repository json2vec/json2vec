import torch
from tensordict import TensorClass, TensorDict

from relflow.structs.enums import TensorKey
from relflow.structs.tree import Address


class Parcel(TensorClass):
    payload: torch.Tensor
    origin: Address
    destination: Address | None


# @jaxtyped(typechecker=beartype)
class Prediction(TensorClass):
    address: Address
    payload: TensorDict[TensorKey, torch.Tensor]

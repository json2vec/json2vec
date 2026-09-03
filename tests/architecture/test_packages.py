import torch
from tensordict import TensorDict

from relflow.structs.enums import TensorKey
from relflow.structs.packages import Parcel, Prediction


def test_parcel():
    parcel = Parcel(
        payload=torch.randn(2, 3, 4),
        origin="input",
        destination="output",
        batch_size=[2],
    )

    assert isinstance(parcel.payload, torch.Tensor)
    assert parcel.payload.shape == (2, 3, 4)
    assert parcel.origin == "input"
    assert parcel.destination == "output"


def test_prediction():
    prediction = Prediction(
        address="output",
        payload=TensorDict(
            {
                TensorKey.content: torch.randn(2, 3),
                TensorKey.state: torch.randint(0, 2, (2, 3), dtype=torch.int8),
            },
            batch_size=[2],
        ),
    )

    assert prediction.address == "output"
    assert isinstance(prediction.payload, TensorDict)
    assert isinstance(prediction.payload[TensorKey.content], torch.Tensor)
    assert prediction.payload[TensorKey.content].shape == (2, 3)
    assert isinstance(prediction.payload[TensorKey.state], torch.Tensor)
    assert prediction.payload[TensorKey.state].shape == (2, 3)


def test_prediction_can_carry_embedding_payload():
    parcel = Parcel(
        payload=torch.randn(2, 3, 4),
        origin="source",
        destination="dest",
        batch_size=[2],
    )

    prediction = Prediction(
        address=parcel.origin,
        payload=TensorDict({TensorKey.embedding: parcel.payload}, batch_size=[2]),
        batch_size=[2],
    )
    assert prediction.address == "source"
    assert prediction.payload[TensorKey.embedding].shape == parcel.payload.shape

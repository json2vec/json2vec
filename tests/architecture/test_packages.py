import torch
from tensordict import TensorDict

from json2vec.structs.enums import TensorKey
from json2vec.structs.packages import Embedding, Parcel, Prediction


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


def test_embedding_from_parcel_copies_payload_and_origin():
    parcel = Parcel(
        payload=torch.randn(2, 3, 4),
        origin="source",
        destination="dest",
        batch_size=[2],
    )

    embedding = Embedding.from_parcel(parcel)
    assert embedding.address == "source"
    assert embedding.payload[TensorKey.embedding].shape == parcel.payload.shape


def test_embedding_normalize_l2_normalizes_last_dimension():
    values = torch.tensor([[[3.0, 4.0], [0.0, 5.0]]], dtype=torch.float32)
    normalized = Embedding.normalize(values)

    norms = torch.linalg.norm(normalized, ord=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms))


def test_embedding_normalize_preserves_zero_vectors():
    values = torch.zeros(2, 3, 4, dtype=torch.float32)
    normalized = Embedding.normalize(values)

    assert torch.equal(normalized, values)


def test_prediction_denest_collapses_only_singleton_lists():
    value = {
        "content": [["ALPHA"]],
        "probability": [[0.9]],
        "topk": [[[
            {"label": "ALPHA", "probability": 0.9},
            {"label": "BETA", "probability": 0.1},
        ]]],
        "keep_list": [[1, 2]],
    }

    output = Prediction.denest(value)

    assert output["content"] == "ALPHA"
    assert output["probability"] == 0.9
    assert output["topk"] == [
        {"label": "ALPHA", "probability": 0.9},
        {"label": "BETA", "probability": 0.1},
    ]
    assert output["keep_list"] == [1, 2]


def test_prediction_unbatch_preserves_prediction_types_and_singleton_batch_dims():
    outputs = [
        Prediction(
            address="record/brand",
            payload=TensorDict(
                {
                    TensorKey.content: torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                },
                batch_size=[2],
            ),
        ),
        Embedding(
            address="root/label",
            payload=TensorDict(
                {
                    TensorKey.embedding: torch.randn(2, 8),
                },
                batch_size=[2],
            ),
        ),
    ]

    unbatched = Prediction.unbatch(outputs)

    assert len(unbatched) == 2
    assert all(len(item) == 2 for item in unbatched)
    assert isinstance(unbatched[0][0], Prediction)
    assert isinstance(unbatched[0][1], Embedding)
    assert len(unbatched[0][0].payload) == 1
    assert len(unbatched[0][1].payload) == 1
    assert unbatched[0][0].payload[TensorKey.content].shape == (1, 2)
    assert unbatched[0][1].payload[TensorKey.embedding].shape == (1, 8)

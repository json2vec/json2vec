import torch

from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.pool import MeanPool
from relflow.architecture.root import Model
from relflow.structs.enums import TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Parcel


def _payload(*, attention: str = "mha", pooling: str = "query") -> dict:
    field: dict = {
        "name": "category",
        "type": "category",
        "mask": True,
        "pooling": pooling,
        "size": 8,
    }
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "attention": attention,
            "dropout": 0.0,
            "fields": [
                {
                    "name": "items",
                    "type": "branch",
                    "length": 2,
                    "fields": [field],
                }
            ],
        },
    }


def test_branch_encoder_uses_gqa_kv_head_count():
    schema = Schema.model_validate(_payload(attention="gqa"))
    encoder = BranchEncoder(schema=schema, address="root")

    assert len(encoder.encoder) == 1
    assert encoder.encoder[0].attention.n_kv_heads == 2


def test_branch_encoder_uses_mqa_kv_head_count():
    schema = Schema.model_validate(_payload(attention="mqa"))
    encoder = BranchEncoder(schema=schema, address="root")

    assert len(encoder.encoder) == 1
    assert encoder.encoder[0].attention.n_kv_heads == 1


def test_branch_encoder_none_skips_transformer_layers():
    schema = Schema.model_validate(_payload(attention="none"))
    encoder = BranchEncoder(schema=schema, address="root")

    assert len(encoder.encoder) == 0


def test_decoder_mean_pooling_repeats_heritage_mean_for_each_target_slot():
    schema = Schema.model_validate(_payload(pooling="mean"))
    model = Model(schema=schema, batch_size=2)
    decoder = model.nodes["root/items/category"].decoder
    parcel = Parcel(
        origin="root",
        destination="",
        payload=torch.randn(2, 3, 16),
        present=torch.ones(2, 3, dtype=torch.bool),
        batch_size=2,
    )

    prediction = decoder([parcel], batch_size=2, device=parcel.payload.device)

    assert isinstance(decoder.pool, MeanPool)
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 8)


def test_branch_encoder_propagates_presence_and_zeros_empty_rows():
    schema = Schema.model_validate(_payload())
    encoder = BranchEncoder(schema=schema, address="root")
    payload = torch.randn(2, 3, 16)
    parcel = Parcel(
        origin="root/items",
        destination="root",
        payload=payload,
        present=torch.tensor([[True, False, True], [False, False, False]]),
        batch_size=2,
    )

    encoded = encoder([parcel])

    assert encoded.payload.shape == (2, 16)
    assert torch.equal(encoded.present, torch.tensor([True, False]))
    assert torch.equal(encoded.payload[1], torch.zeros_like(encoded.payload[1]))


def test_nested_branch_encoder_preserves_repeated_parent_geometry():
    schema = Schema.model_validate(
        {
            "d_model": 16,
            "fields": {
                "name": "root",
                "type": "branch",
                "fields": [
                    {
                        "name": "parents",
                        "type": "branch",
                        "length": 3,
                        "fields": [
                            {
                                "name": "children",
                                "type": "branch",
                                "length": 2,
                                "fields": [
                                    {
                                        "name": "value",
                                        "type": "category",
                                        "size": 8,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        }
    )
    encoder = BranchEncoder(schema=schema, address="root/parents/children")
    present = torch.tensor(
        [
            [[[True, False], [False, False], [True, True]]],
            [[[False, False], [True, False], [False, True]]],
        ]
    )
    parcel = Parcel(
        origin="root/parents/children/value",
        destination="root/parents/children",
        payload=torch.randn(2, 1, 3, 2, 16),
        present=present,
        batch_size=2,
    )

    encoded = encoder([parcel])

    assert encoded.payload.shape == (2, 1, 3, 16)
    assert torch.equal(encoded.present, present.any(dim=-1))


def test_decoder_supports_zero_context():
    schema = Schema.model_validate(_payload(pooling="mean"))
    model = Model(schema=schema, batch_size=2)
    decoder = model.nodes["root/items/category"].decoder

    prediction = decoder([], batch_size=2, device=torch.device("cpu"))

    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 8)
    assert all(torch.isfinite(value).all() for value in prediction.payload.values())

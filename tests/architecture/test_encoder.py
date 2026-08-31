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
        batch_size=2,
    )

    prediction = decoder([parcel])

    assert isinstance(decoder.pool, MeanPool)
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 8)

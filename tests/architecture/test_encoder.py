import torch

from json2vec.architecture.encoder import ArrayEncoder
from json2vec.architecture.pool import MeanPool
from json2vec.architecture.root import Model
from json2vec.structs.enums import TensorKey, Tokens
from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.packages import Parcel


def _payload(*, attention: str = "mha", pooling: str = "query") -> dict:
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "array",
            "attention": attention,
            "dropout": 0.0,
            "max_length": 2,
            "fields": [
                {
                    "name": "category",
                    "type": "category",
                    "query": "[*].label",
                    "pooling": pooling,
                    "max_vocab_size": 8,
                }
            ],
        },
    }


def test_array_encoder_uses_gqa_kv_head_count():
    hyperparameters = Hyperparameters.model_validate(_payload(attention="gqa"))
    encoder = ArrayEncoder(hyperparameters=hyperparameters, address="root")

    assert len(encoder.encoder) == 1
    assert encoder.encoder[0].attention.n_kv_heads == 2


def test_array_encoder_uses_mqa_kv_head_count():
    hyperparameters = Hyperparameters.model_validate(_payload(attention="mqa"))
    encoder = ArrayEncoder(hyperparameters=hyperparameters, address="root")

    assert len(encoder.encoder) == 1
    assert encoder.encoder[0].attention.n_kv_heads == 1


def test_array_encoder_none_skips_transformer_layers():
    hyperparameters = Hyperparameters.model_validate(_payload(attention="none"))
    encoder = ArrayEncoder(hyperparameters=hyperparameters, address="root")

    assert len(encoder.encoder) == 0


def test_decoder_mean_pooling_repeats_heritage_mean_for_each_target_slot():
    hyperparameters = Hyperparameters.model_validate(_payload(pooling="mean"))
    model = Model(hyperparameters=hyperparameters, batch_size=2)
    decoder = model.nodes["root/category"].decoder
    parcel = Parcel(
        origin="root",
        destination="",
        payload=torch.randn(2, 3, 16),
        batch_size=2,
    )

    prediction = decoder([parcel])

    assert isinstance(decoder.pool, MeanPool)
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 9)

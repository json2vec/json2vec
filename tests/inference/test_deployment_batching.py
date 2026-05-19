from __future__ import annotations

import torch
from tensordict import TensorDict

from json2vec.inference.deployment import Deployment, ErrorItem
from json2vec.structs.enums import TensorKey
from json2vec.structs.packages import Prediction


def _input(value: int) -> TensorDict:
    return TensorDict({"dummy": torch.tensor([value])}, batch_size=[1])


class _DummyModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, data: TensorDict) -> list[Prediction]:
        self.calls += 1
        batch_size = int(data.batch_size[0])
        return [
            Prediction(
                address="root/label",
                payload=TensorDict(
                    {TensorKey.content: torch.zeros(batch_size, 1)},
                    batch_size=[batch_size],
                ),
            )
        ]

    def write(self, predictions: list[Prediction]):
        return (
            {"root/label": {"value": ["ok"]}},
            {},
        )


def _deployment(deployment_cls=Deployment) -> tuple[Deployment, _DummyModel]:
    model = _DummyModel()
    deployment = deployment_cls(checkpoint="unused")
    deployment.model = model
    deployment.device = "cpu"
    return deployment, model


def test_deployment_batches_only_valid_inputs_and_preserves_per_item_errors():
    deployment, model = _deployment()

    batch = deployment.batch(
        [
            _input(1),
            ErrorItem(status_code=422, message="cxr DM not present"),
            _input(2),
        ]
    )

    outputs = deployment.predict(batch)
    encoded = [deployment.encode_response(item) for item in deployment.unbatch(outputs)]

    assert model.calls == 1
    assert encoded[0]["predictions"]["root/label"]["value"] == "ok"
    assert encoded[1]["predictions"] == {}
    assert encoded[1]["error"] == {
        "status_code": 422,
        "message": "cxr DM not present",
    }
    assert encoded[2]["predictions"]["root/label"]["value"] == "ok"


def test_deployment_postprocess_can_rewrite_encoded_response():
    class PostprocessedDeployment(Deployment):
        pass

    seen = {}

    def processor(predictions, embeddings):
        seen["predictions"] = predictions
        seen["embeddings"] = embeddings
        return (
            {"root/label": {"value": ["rewritten"]}},
            {"root/vector": {"embedding": [[1.0, 2.0]]}},
        )

    deployment, _ = _deployment(PostprocessedDeployment)
    PostprocessedDeployment.postprocess(processor)

    encoded = deployment.encode_response([])

    assert seen["predictions"]["root/label"]["value"] == ["ok"]
    assert seen["embeddings"] == {}
    assert encoded["predictions"]["root/label"]["value"] == "rewritten"
    assert encoded["embeddings"]["root/vector"]["embedding"] == [1.0, 2.0]


def test_deployment_skips_model_when_every_item_in_batch_is_invalid():
    deployment, model = _deployment()

    batch = deployment.batch(
        [
            ErrorItem(status_code=422, message="cxr DM not present"),
            ErrorItem(status_code=422, message="cxr RJ not present"),
        ]
    )

    outputs = deployment.predict(batch)
    encoded = [deployment.encode_response(item) for item in deployment.unbatch(outputs)]

    assert model.calls == 0
    assert encoded == [
        {
            "predictions": {},
            "error": {"status_code": 422, "message": "cxr DM not present"},
        },
        {
            "predictions": {},
            "error": {"status_code": 422, "message": "cxr RJ not present"},
        },
    ]

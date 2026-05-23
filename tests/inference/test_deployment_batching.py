from __future__ import annotations

from types import SimpleNamespace

import pydantic
import pytest
import torch
from tensordict import TensorDict

import json2vec.inference.deployment as deployment_module
from json2vec import Model, Number, where
from json2vec.inference.deployment import API, Deployment, ErrorItem
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


def _api(**kwargs) -> tuple[API, _DummyModel]:
    model = _DummyModel()
    deployment = API(checkpoint="unused", **kwargs)
    deployment.model = model
    deployment.device = "cpu"
    return deployment, model


def test_deployment_batches_only_valid_inputs_and_preserves_per_item_errors():
    deployment, model = _api()

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
    seen = {}

    context = {"request": {"color": "r"}, "input": _input(7)}

    def processor(context, predictions, embeddings):
        seen["context"] = context
        seen["predictions"] = predictions
        seen["embeddings"] = embeddings
        return (
            {"root/label": {"value": ["rewritten"]}},
            {"root/vector": {"embedding": [[1.0, 2.0]]}},
        )

    deployment, _ = _api(postprocessor=processor)

    encoded = deployment.encode_response([], context=context)

    assert seen["context"] is context
    assert seen["context"]["request"] == {"color": "r"}
    assert seen["predictions"]["root/label"]["value"] == ["ok"]
    assert seen["embeddings"] == {}
    assert encoded["predictions"]["root/label"]["value"] == "rewritten"
    assert encoded["embeddings"]["root/vector"]["embedding"] == [1.0, 2.0]


def test_deployment_preprocesses_decode_request(monkeypatch):
    def __deployment_preprocess(observation: dict):
        return {"color": observation["hue"]}

    captured = {}

    def fake_encode(batch, hyperparameters, strata, interprocess_encoding_context, jmespath_resolution_monitor):
        captured["batch"] = batch
        return _input(1)

    monkeypatch.setattr(deployment_module, "encode", fake_encode)

    deployment = API(checkpoint="unused", preprocessor=__deployment_preprocess)
    deployment.model = SimpleNamespace(hyperparameters=object())
    deployment.interprocess_encoding_context = {}
    context = {}

    encoded = deployment.decode_request({"hue": "red"}, context=context)

    assert isinstance(encoded, TensorDict)
    assert captured["batch"] == [[{"color": "red"}]]
    assert context["observations"] == [[{"color": "red"}]]


def test_deployment_preprocess_generator_returns_error():
    def __deployment_generator(observation: dict):
        yield {"color": observation["hue"]}

    deployment = API(checkpoint="unused", preprocessor=__deployment_generator)
    deployment.interprocess_encoding_context = {}

    error = deployment.decode_request({"hue": "red"})

    assert isinstance(error, ErrorItem)
    assert error.status_code == 422
    assert "preprocessor must return a dict object" in error.message


def test_deployment_skips_model_when_every_item_in_batch_is_invalid():
    deployment, model = _api()

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


def test_deployment_launcher_configures_litserve_api(monkeypatch):
    class Request(pydantic.BaseModel):
        color: str

    class Response(pydantic.BaseModel):
        predictions: dict = {}

    captured = {}

    class FakeServer:
        def __init__(self, *, lit_api, accelerator, workers_per_device, track_requests):
            captured["lit_api"] = lit_api
            captured["accelerator"] = accelerator
            captured["workers_per_device"] = workers_per_device
            captured["track_requests"] = track_requests

        def run(self, *, generate_client_file):
            captured["generate_client_file"] = generate_client_file

    monkeypatch.setattr(deployment_module.ls, "LitServer", FakeServer)

    Deployment(
        checkpoint="unused",
        max_batch_size=16,
        batch_timeout=0.25,
        workers_per_device=2,
        accelerator="cpu",
        track_requests=True,
    ).set(where("name") == "label", target=False).forge(request=Request, response=Response).serve()

    assert isinstance(captured["lit_api"], API)
    assert captured["lit_api"].checkpoint == "unused"
    assert captured["lit_api"].preprocessor is None
    assert len(captured["lit_api"].set_operations) == 1
    assert captured["accelerator"] == "cpu"
    assert captured["workers_per_device"] == 2
    assert captured["track_requests"] is True
    assert captured["generate_client_file"] is False
    assert API.decode_request.__annotations__["request"] is Request
    assert API.encode_response.__annotations__["return"] is Response


def test_deployment_launcher_accepts_model_instance(monkeypatch):
    model = Model.from_schema(
        Number("amount"),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=1,
        embed=True,
    )
    captured = []

    class FakeServer:
        def __init__(self, *, lit_api, accelerator, workers_per_device, track_requests):
            captured.append({"lit_api": lit_api})

        def run(self, *, generate_client_file):
            captured[-1]["generate_client_file"] = generate_client_file

    monkeypatch.setattr(deployment_module.ls, "LitServer", FakeServer)

    Deployment(model=model, accelerator="cpu").serve()
    Deployment(checkpoint=model, accelerator="cpu").serve()

    assert len(captured) == 2
    for item in captured:
        assert isinstance(item["lit_api"], API)
        assert item["lit_api"].checkpoint is None
        assert item["lit_api"].model_source is model
        assert item["generate_client_file"] is False


def test_deployment_rejects_explicit_checkpoint_and_model():
    model = Model.from_schema(Number("amount"), d_model=8, n_layers=1, n_heads=2)

    with pytest.raises(ValueError, match="pass either checkpoint or model"):
        Deployment(checkpoint="model.ckpt", model=model)


def test_deployment_api_applies_queued_set_operations(monkeypatch):
    calls = []

    class Mutation:
        def model_dump(self, mode):
            return {"mode": mode, "updated": 1}

    class Hyperparameters:
        last_mutation = Mutation()

    class FakeModel:
        hyperparameters = Hyperparameters()
        interprocess_encoding_context = {}

        def to(self, device):
            calls.append(("to", device))
            return self

        def set(self, *predicates, **values):
            calls.append(("set", predicates, values))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    fake = FakeModel()
    monkeypatch.setattr(deployment_module.Model, "load", classmethod(lambda cls, checkpoint: fake))

    predicate = where("name") == "label"
    api = API(
        checkpoint="unused",
        set_operations=[
            ((predicate,), {"target": False}),
        ],
    )

    api.setup(device="cpu")

    assert calls[0] == ("to", "cpu")
    assert calls[1] == ("set", (predicate,), {"target": False})
    assert calls[2] == ("eval",)
    assert api.applied_set_operations == [{"mode": "python", "updated": 1}]


def test_deployment_api_setup_uses_model_instance(monkeypatch):
    model = Model.from_schema(Number("amount"), d_model=8, n_layers=1, n_heads=2)
    monkeypatch.setattr(
        deployment_module.Model,
        "load",
        classmethod(lambda cls, checkpoint: (_ for _ in ()).throw(AssertionError("checkpoint should not load"))),
    )

    api = API(checkpoint=model)

    api.setup(device="cpu")

    assert api.model is model
    assert api.checkpoint is None


def test_deployment_launcher_binds_preprocessor_kwargs(monkeypatch):
    def __deployment_preprocess(observation: dict, suffix: str):
        return {"color": observation["hue"] + suffix}

    captured = {}

    class FakeServer:
        def __init__(self, *, lit_api, accelerator, workers_per_device, track_requests):
            captured["lit_api"] = lit_api

        def run(self, *, generate_client_file):
            pass

    def fake_encode(batch, hyperparameters, strata, interprocess_encoding_context, jmespath_resolution_monitor):
        captured["batch"] = batch
        return _input(1)

    monkeypatch.setattr(deployment_module.ls, "LitServer", FakeServer)
    monkeypatch.setattr(deployment_module, "encode", fake_encode)

    Deployment(checkpoint="unused").preprocess(__deployment_preprocess, suffix="!").serve()
    api = captured["lit_api"]
    api.model = SimpleNamespace(hyperparameters=object())
    api.interprocess_encoding_context = {}
    context = {}

    encoded = api.decode_request({"hue": "red"}, context=context)

    assert isinstance(encoded, TensorDict)
    assert captured["batch"] == [[{"color": "red!"}]]
    assert context["observations"] == [[{"color": "red!"}]]

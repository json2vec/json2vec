from __future__ import annotations

import asyncio

import pydantic
import pytest
import torch
from tensordict import TensorDict

import json2vec as jv
import json2vec.inference.deployment as deployment_module
from json2vec import Model, Number, where
from json2vec.inference.deployment import Deployment, ErrorItem, RequestItem
from json2vec.structs.enums import Strata, TensorKey
from json2vec.structs.packages import Prediction


class _DummyModel:
    def __init__(self):
        self.calls = 0
        self.write_calls = 0
        self.schema = object()
        self.interprocess_encoding_context = {}

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, data: TensorDict, *, strata: Strata | str) -> list[Prediction]:
        assert Strata.normalize(strata) == Strata.predict
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
        self.write_calls += 1
        batch_size = int(predictions[0].payload.batch_size[0]) if predictions else 0
        return {"root/label": {"value": ["ok"] * batch_size}}


def _runtime(model=None, **kwargs) -> deployment_module.FastAPIRuntime:
    runtime = deployment_module.FastAPIRuntime(
        checkpoint="unused",
        accelerator=deployment_module.Accelerator.cpu,
        **kwargs,
    )
    runtime.model = _DummyModel() if model is None else model
    runtime.device = torch.device("cpu")
    runtime.interprocess_encoding_context = {}
    runtime.jmespath_resolution_monitor = None
    return runtime


def test_fastapi_batcher_submit_many_splits_large_payload_by_max_batch_size():
    class FakeRuntime:
        def __init__(self):
            self.started = False
            self.batches = []

        def setup(self):
            self.started = True

        def predict_payloads(self, payloads):
            self.batches.append([payload["id"] for payload in payloads])
            return [{"id": payload["id"]} for payload in payloads]

    async def run():
        runtime = FakeRuntime()
        batcher = deployment_module.FastAPIBatcher(runtime=runtime, max_batch_size=2, batch_timeout=0.0)
        await batcher.start()
        try:
            responses = await batcher.submit_many([{"id": 1}, {"id": 2}, {"id": 3}])
        finally:
            await batcher.stop()

        return runtime, responses

    runtime, responses = asyncio.run(run())

    assert runtime.started is True
    assert runtime.batches == [[1, 2], [3]]
    assert responses == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_fastapi_runtime_encodes_real_batched_requests_once(monkeypatch):
    model = Model.from_tree(
        Number(name="amount"),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
        embed=True,
    )
    runtime = deployment_module.FastAPIRuntime(
        checkpoint=model,
        accelerator=deployment_module.Accelerator.cpu,
    )
    runtime.setup()

    captured_batches = []
    captured_monitors = []
    real_encode = deployment_module.encode

    def spy_encode(batch, schema, strata, interprocess_encoding_context, jmespath_resolution_monitor):
        captured_batches.append(batch)
        captured_monitors.append(jmespath_resolution_monitor)
        return real_encode(
            batch=batch,
            schema=schema,
            strata=strata,
            interprocess_encoding_context=interprocess_encoding_context,
            jmespath_resolution_monitor=jmespath_resolution_monitor,
        )

    monkeypatch.setattr(deployment_module, "encode", spy_encode)

    outputs = runtime.predict_payloads([{"amount": 1.5}, {"amount": 2.5}])

    assert captured_batches == [[[{"amount": 1.5}], [{"amount": 2.5}]]]
    assert captured_monitors == [None]
    assert len(outputs) == 2
    assert all("predictions" in output for output in outputs)


def test_fastapi_runtime_preserves_per_item_errors_and_batches_valid_requests_once(monkeypatch):
    @jv.preprocess
    def __deployment_preprocess(observation: dict):
        if observation["hue"] == "bad":
            raise ValueError("bad hue")
        return jv.Observation({"color": observation["hue"]})

    captured = {"calls": 0}

    def fake_encode(batch, schema, strata, interprocess_encoding_context, jmespath_resolution_monitor):
        captured["calls"] += 1
        captured["batch"] = batch
        captured["strata"] = strata
        return TensorDict({"dummy": torch.tensor([1, 2])}, batch_size=[2])

    monkeypatch.setattr(deployment_module, "encode", fake_encode)

    model = _DummyModel()
    runtime = _runtime(model=model, preprocessor=__deployment_preprocess)

    outputs = runtime.predict_payloads(
        [
            {"hue": "red"},
            {"hue": "bad"},
            {"hue": "blue"},
        ]
    )

    assert captured["calls"] == 1
    assert captured["batch"] == [[{"color": "red"}], [{"color": "blue"}]]
    assert captured["strata"] == Strata.predict
    assert model.calls == 1
    assert model.write_calls == 1
    assert outputs[0]["predictions"]["root/label"]["value"] == "ok"
    assert outputs[1]["predictions"] == {}
    assert outputs[1]["error"]["status_code"] == 422
    assert "bad hue" in outputs[1]["error"]["message"]
    assert outputs[2]["predictions"]["root/label"]["value"] == "ok"


def test_fastapi_runtime_postprocess_can_rewrite_response(monkeypatch):
    seen = {}

    def fake_encode(batch, schema, strata, interprocess_encoding_context, jmespath_resolution_monitor):
        return TensorDict({"dummy": torch.tensor([1])}, batch_size=[1])

    @jv.postprocess
    def processor(predictions, *, request, observations, input):
        seen["request"] = request
        seen["observations"] = observations
        seen["input"] = input
        seen["predictions"] = predictions
        return {
            "root/label": {"value": ["rewritten"]},
            "root/vector": {"embedding": [[1.0, 2.0]]},
        }

    monkeypatch.setattr(deployment_module, "encode", fake_encode)

    runtime = _runtime(postprocessor=processor)
    output = runtime.predict_payloads([{"color": "r"}])[0]

    assert seen["request"] == {"color": "r"}
    assert seen["observations"] == [[{"color": "r"}]]
    assert seen["input"] is not None
    assert seen["predictions"]["root/label"]["value"] == ["ok"]
    assert output["predictions"]["root/label"]["value"] == "rewritten"
    assert output["predictions"]["root/vector"]["embedding"] == [1.0, 2.0]


def test_fastapi_runtime_postprocess_receives_device_moved_input(monkeypatch):
    seen = {}

    class DeviceCheckingModel(_DummyModel):
        def __call__(self, data: TensorDict, *, strata: Strata | str) -> list[Prediction]:
            assert data["dummy"].device == torch.device("meta")
            return super().__call__(data, strata=strata)

    def fake_encode(batch, schema, strata, interprocess_encoding_context, jmespath_resolution_monitor):
        return TensorDict({"dummy": torch.tensor([1])}, batch_size=[1])

    @jv.postprocess
    def processor(predictions, *, input):
        seen["input_device"] = input["dummy"].device
        return predictions

    monkeypatch.setattr(deployment_module, "encode", fake_encode)

    runtime = _runtime(model=DeviceCheckingModel(), postprocessor=processor)
    runtime.device = torch.device("meta")

    output = runtime.predict_payloads([{"color": "r"}])[0]

    assert seen["input_device"] == torch.device("meta")
    assert output["predictions"]["root/label"]["value"] == "ok"


def test_fastapi_runtime_with_no_predictions_returns_empty_response(monkeypatch):
    class EmptyModel(_DummyModel):
        def __call__(self, data: TensorDict, *, strata: Strata | str) -> list[Prediction]:
            assert Strata.normalize(strata) == Strata.predict
            self.calls += 1
            return []

        def write(self, predictions: list[Prediction]):
            assert predictions == []
            return {}

    def fake_encode(batch, schema, strata, interprocess_encoding_context, jmespath_resolution_monitor):
        return TensorDict({"dummy": torch.tensor([1])}, batch_size=[1])

    monkeypatch.setattr(deployment_module, "encode", fake_encode)

    model = EmptyModel()
    runtime = _runtime(model=model)
    output = runtime.predict_payloads([{"amount": 1}])[0]

    assert model.calls == 1
    assert output == {"predictions": {}}


def test_fastapi_runtime_decode_rejects_multiple_preprocessor_outputs():
    @jv.preprocess
    def __deployment_generator(observation: dict):
        yield jv.Observation({"color": observation["hue"]})
        yield jv.Observation({"color": observation["hue"] + "2"})

    runtime = _runtime(preprocessor=__deployment_generator)
    context = {}

    error = runtime.decode_payload({"hue": "red"}, context=context)

    assert isinstance(error, ErrorItem)
    assert error.status_code == 422
    assert "deployment requests must encode exactly one observation" in error.message


def test_fastapi_runtime_decode_validates_pydantic_request_model():
    class Request(pydantic.BaseModel):
        hue: str

    @jv.preprocess
    def __deployment_preprocess(observation: dict):
        return jv.Observation({"color": observation["hue"]})

    runtime = _runtime(
        request_signature=Request,
        preprocessor=__deployment_preprocess,
    )
    context = {}

    decoded = runtime.decode_payload({"hue": "red"}, context=context)

    assert isinstance(decoded, RequestItem)
    assert decoded.observations == [[{"color": "red"}]]
    assert context["request"] == {"hue": "red"}


def test_fastapi_runtime_setup_can_enable_query_monitor():
    runtime = deployment_module.FastAPIRuntime(
        checkpoint=Model.from_tree(Number(name="amount"), d_model=8, n_layers=1, n_heads=2),
        accelerator=deployment_module.Accelerator.cpu,
        monitor_queries=True,
        query_monitor_every=7,
    )

    runtime.setup()

    assert runtime.jmespath_resolution_monitor is not None
    assert runtime.jmespath_resolution_monitor.every == 7


def test_deployment_launcher_configures_fastapi_app(monkeypatch):
    class Request(pydantic.BaseModel):
        color: str

    class Response(pydantic.BaseModel):
        predictions: dict = {}

    captured = {}

    def fake_run(app, *, host, port, log_level):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr(deployment_module.uvicorn, "run", fake_run)

    Deployment(
        checkpoint="unused",
        max_batch_size=16,
        batch_timeout=0.25,
        accelerator="cpu",
        host="127.0.0.1",
        port=8765,
        log_level="error",
    ).update(where("name") == "label", target=False).forge(request=Request, response=Response).serve()

    assert isinstance(captured["app"], deployment_module.fastapi.FastAPI)
    assert {route.path for route in captured["app"].routes} >= {"/health", "/predict"}
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["log_level"] == "error"


def test_deployment_forge_registers_openapi_signatures():
    class Request(pydantic.BaseModel):
        amount: float

    class Candidate(pydantic.BaseModel):
        label: str
        probability: float

    class Response(pydantic.BaseModel):
        predictions: list[Candidate]

    app = Deployment(checkpoint="unused").forge(request=Request, response=Response).app()

    schema = app.openapi()
    components = schema["components"]["schemas"]
    operation = schema["paths"]["/predict"]["post"]

    assert {"Request", "Candidate", "Response"} <= set(components)

    request_schema = operation["requestBody"]["content"]["application/json"]["schema"]
    assert request_schema == {
        "anyOf": [
            {"$ref": "#/components/schemas/Request"},
            {"type": "array", "items": {"$ref": "#/components/schemas/Request"}},
        ]
    }

    response_schema = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema == {
        "anyOf": [
            {"$ref": "#/components/schemas/Response"},
            {"type": "array", "items": {"$ref": "#/components/schemas/Response"}},
        ]
    }
    assert components["Response"]["properties"]["predictions"]["items"] == {"$ref": "#/components/schemas/Candidate"}


def test_deployment_launcher_configures_worker_import_string(monkeypatch):
    captured = {}

    def fake_run(app, *, factory, workers, host, port, log_level):
        captured["app"] = app
        captured["factory"] = factory
        captured["workers"] = workers
        captured["host"] = host
        captured["port"] = port
        captured["log_level"] = log_level

    monkeypatch.setattr(deployment_module.uvicorn, "run", fake_run)

    Deployment(checkpoint="model.ckpt", workers=2, accelerator="cpu", port=8765).serve()

    assert captured == {
        "app": "json2vec.inference.deployment:create_app",
        "factory": True,
        "workers": 2,
        "host": "0.0.0.0",
        "port": 8765,
        "log_level": "info",
    }


def test_deployment_launcher_rejects_workers_with_model_instance():
    model = Model.from_tree(Number(name="amount"), d_model=8, n_layers=1, n_heads=2)

    with pytest.raises(ValueError, match="workers > 1"):
        Deployment(model=model, workers=2).serve()


def test_deployment_launcher_accepts_model_instance(monkeypatch):
    model = Model.from_tree(
        Number(name="amount"),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=1,
        embed=True,
    )
    captured = []

    def fake_run(app, *, host, port, log_level):
        captured.append({"app": app, "host": host, "port": port, "log_level": log_level})

    monkeypatch.setattr(deployment_module.uvicorn, "run", fake_run)

    Deployment(model=model, accelerator="cpu", port=9001).serve()
    Deployment(checkpoint=model, accelerator="cpu", port=9002).serve()

    assert len(captured) == 2
    for item in captured:
        assert isinstance(item["app"], deployment_module.fastapi.FastAPI)


def test_deployment_rejects_explicit_checkpoint_and_model():
    model = Model.from_tree(Number(name="amount"), d_model=8, n_layers=1, n_heads=2)

    with pytest.raises(ValueError, match="pass either checkpoint or model"):
        Deployment(checkpoint="model.ckpt", model=model)


def test_fastapi_runtime_setup_applies_queued_update_operations(monkeypatch):
    calls = []

    class FakeModel:
        interprocess_encoding_context = {}

        def __init__(self):
            self.placed = False

        def to(self, device):
            calls.append(("to", str(device)))
            self.placed = True
            return self

        def update(self, *predicates, **values):
            calls.append(("update", predicates, values))
            self.placed = False
            return self

        def eval(self):
            assert self.placed is True
            calls.append(("eval",))
            return self

    fake = FakeModel()
    monkeypatch.setattr(deployment_module.Model, "load", classmethod(lambda cls, checkpoint: fake))

    predicate = where("name") == "label"
    runtime = deployment_module.FastAPIRuntime(
        checkpoint="unused",
        accelerator=deployment_module.Accelerator.cpu,
        update_operations=[
            ((predicate,), {"target": False}),
        ],
    )

    runtime.setup()

    assert calls[0] == ("update", (predicate,), {"target": False})
    assert calls[1] == ("to", "cpu")
    assert calls[2] == ("eval",)


def test_fastapi_runtime_setup_uses_model_instance(monkeypatch):
    model = Model.from_tree(Number(name="amount"), d_model=8, n_layers=1, n_heads=2)
    monkeypatch.setattr(
        deployment_module.Model,
        "load",
        classmethod(lambda cls, checkpoint: (_ for _ in ()).throw(AssertionError("checkpoint should not load"))),
    )

    runtime = deployment_module.FastAPIRuntime(
        checkpoint=model,
        accelerator=deployment_module.Accelerator.cpu,
    )

    runtime.setup()

    assert runtime.model is model


def test_deployment_uses_bound_preprocessor_object():
    @jv.preprocess
    def __deployment_preprocess(observation: dict, *, suffix: str):
        return jv.Observation({"color": observation["hue"] + suffix})

    deployment = Deployment(checkpoint="unused").preprocess(__deployment_preprocess.partial(suffix="!"))
    runtime = deployment_module.FastAPIRuntime(
        checkpoint="unused",
        accelerator=deployment_module.Accelerator.cpu,
        preprocessor=deployment._preprocessor,
    )
    context = {}

    decoded = runtime.decode_payload({"hue": "red"}, context=context)

    assert isinstance(decoded, RequestItem)
    assert decoded.observations == [[{"color": "red!"}]]
    assert context["observations"] == [[{"color": "red!"}]]

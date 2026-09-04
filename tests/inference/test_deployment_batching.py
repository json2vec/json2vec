from __future__ import annotations

import asyncio

import pyarrow as pa
import pydantic
import pytest
import torch

import relflow as rf
import relflow.inference.deployment as deployment_module
from relflow import Model, Number, where
from relflow.data.datasets.arrow import identity
from relflow.inference.deployment import Deployment, ErrorItem


class PredictModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, source, *, preprocess, postprocess, retain):
        self.calls.append(
            {
                "source": source,
                "preprocess": preprocess,
                "postprocess": postprocess,
                "retain": retain,
            }
        )
        ids = source.data["id"].combine_chunks()
        inputs = pa.StructArray.from_arrays([ids], names=["id"])
        labels = pa.array([str(value) for value in ids.to_pylist()], type=pa.large_string())
        predictions = pa.StructArray.from_arrays([labels], names=["label"])
        result = rf.Batch(
            data=pa.table({"inputs": inputs, "predictions": predictions}),
            identity=source.identity,
        )
        for processor in rf.Postprocessor.normalize(postprocess):
            result = processor.run(result)
        return result.data


def runtime(model=None, **kwargs) -> deployment_module.FastAPIRuntime:
    server = deployment_module.FastAPIRuntime(
        checkpoint="unused",
        accelerator=deployment_module.Accelerator.cpu,
        **kwargs,
    )
    server.model = PredictModel() if model is None else model
    server.device = torch.device("cpu")
    return server


def test_fastapi_batcher_splits_ready_requests_by_max_batch_size():
    class Runtime:
        def __init__(self):
            self.started = False
            self.batches = []

        def setup(self):
            self.started = True

        def predict_payloads(self, payloads):
            self.batches.append([payload["id"] for payload in payloads])
            return [{"id": payload["id"]} for payload in payloads]

    async def run():
        server = Runtime()
        batcher = deployment_module.FastAPIBatcher(runtime=server, max_batch_size=2, batch_timeout=0.0)
        await batcher.start()
        try:
            responses = await batcher.submit_many([{"id": 1}, {"id": 2}, {"id": 3}])
        finally:
            await batcher.stop()
        return server, responses

    server, responses = asyncio.run(run())

    assert server.started is True
    assert server.batches == [[1, 2], [3]]
    assert responses == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_batcher_runs_processors_once_for_one_collated_arrow_batch():
    preprocessed = []
    postprocessed = []

    @rf.preprocess
    def prepare(batch: rf.Batch) -> rf.Batch:
        preprocessed.append(batch)
        return batch.take(pa.array([2, 0, 1], type=pa.int64()))

    @rf.postprocess(requires=("inputs",), produces=("value",))
    def compact(batch: rf.Batch) -> rf.Batch:
        postprocessed.append("compact")
        values = pa.compute.struct_field(batch.data["inputs"], "value")
        return batch.replace(pa.table({"value": values}))

    @rf.postprocess(requires=("value",), produces=("value",))
    def finish(batch: rf.Batch) -> rf.Batch:
        postprocessed.append("finish")
        return batch.replace(pa.table({"value": batch.data["value"]}))

    async def run():
        model = Model(
            value=Number,
            d_model=8,
            n_layers=1,
            n_heads=2,
            embed=True,
        )
        server = deployment_module.FastAPIRuntime(
            checkpoint=model,
            accelerator=deployment_module.Accelerator.cpu,
            preprocessor=prepare,
            postprocessor=[compact, finish],
            retain=("value",),
        )
        batcher = deployment_module.FastAPIBatcher(
            runtime=server,
            max_batch_size=3,
            batch_timeout=0.05,
        )
        await batcher.start()
        try:
            return await asyncio.gather(
                batcher.submit({"value": 3.0}),
                batcher.submit({"value": 1.0}),
                batcher.submit({"value": 2.0}),
            )
        finally:
            await batcher.stop()

    responses = asyncio.run(run())

    assert len(preprocessed) == 1
    assert postprocessed == ["compact", "finish"]
    assert isinstance(preprocessed[0], rf.Batch)
    assert preprocessed[0].data["value"].to_pylist() == [3.0, 1.0, 2.0]
    assert responses == [{"value": 3.0}, {"value": 1.0}, {"value": 2.0}]


def test_runtime_converts_valid_requests_to_one_arrow_prediction_call():
    model = PredictModel()

    @rf.preprocess
    def prepare(batch: rf.Batch) -> rf.Batch:
        return batch

    server = runtime(model=model, preprocessor=prepare, retain=("id",))
    outputs = server.predict_payloads([{"id": 3}, {"id": 1}])

    assert len(model.calls) == 1
    call = model.calls[0]
    assert isinstance(call["source"], rf.Batch)
    assert call["source"].data.to_pylist() == [{"id": 3}, {"id": 1}]
    assert call["preprocess"] == (prepare,)
    assert isinstance(call["postprocess"], tuple)
    assert len(call["postprocess"]) == 1
    assert isinstance(call["postprocess"][0], rf.Postprocessor)
    assert call["retain"] == ("id",)
    assert outputs == [
        {"predictions": {"label": "3"}},
        {"predictions": {"label": "1"}},
    ]


def test_runtime_preserves_request_fields_introduced_after_the_first_row():
    model = PredictModel()
    server = runtime(model=model)

    server.predict_payloads([{"id": 3}, {"id": 1, "context": "kept"}])

    assert model.calls[0]["source"].data.to_pylist() == [
        {"id": 3, "context": None},
        {"id": 1, "context": "kept"},
    ]


def test_runtime_preserves_request_order_around_validation_errors():
    class Request(pydantic.BaseModel):
        id: int = pydantic.Field(gt=0)

    model = PredictModel()
    server = runtime(model=model, request_signature=Request)

    outputs = server.predict_payloads([{"id": 2}, {"id": 0}, "wrong", {"id": 4}])

    assert len(model.calls) == 1
    assert model.calls[0]["source"].data.to_pylist() == [{"id": 2}, {"id": 4}]
    assert outputs[0] == {"predictions": {"label": "2"}}
    assert outputs[1]["error"]["status_code"] == 422
    assert "greater than 0" in outputs[1]["error"]["message"]
    assert outputs[2] == {
        "predictions": {},
        "error": {"status_code": 422, "message": "each request must be a JSON object, got str"},
    }
    assert outputs[3] == {"predictions": {"label": "4"}}


def test_runtime_restores_order_after_arrow_preprocessing():
    class ReorderingModel(PredictModel):
        def predict(self, source, *, preprocess, postprocess, retain):
            reversed_source = source.take(pa.array([1, 0], type=pa.int64()))
            return super().predict(
                reversed_source,
                preprocess=preprocess,
                postprocess=postprocess,
                retain=retain,
            )

    server = runtime(model=ReorderingModel())

    assert server.predict_payloads([{"id": 3}, {"id": 9}]) == [
        {"predictions": {"label": "3"}},
        {"predictions": {"label": "9"}},
    ]


def test_default_response_never_exposes_retained_inputs():
    server = runtime(retain=("id",))

    output = server.predict_payloads([{"id": 8}])[0]

    assert output == {"predictions": {"label": "8"}}
    assert "inputs" not in output
    assert "identity" not in output


def test_postprocessors_run_inside_the_single_model_prediction_call():
    seen = []

    @rf.postprocess(requires=("inputs", "predictions"), produces=("request_id", "label"))
    def compact(batch: rf.Batch) -> rf.Batch:
        seen.append("compact")
        return batch.replace(
            pa.table(
                {
                    "request_id": pa.compute.struct_field(batch.data["inputs"], "id"),
                    "label": pa.compute.struct_field(batch.data["predictions"], "label"),
                }
            )
        )

    @rf.postprocess(requires=("request_id", "label"), produces=("request_id", "label"))
    def finish(batch: rf.Batch) -> rf.Batch:
        seen.append("finish")
        return batch.replace(batch.data.select(["request_id", "label"]))

    model = PredictModel()
    server = runtime(model=model, postprocessor=(compact, finish), retain=("id",))

    output = server.predict_payloads([{"id": 5}])[0]

    assert len(model.calls) == 1
    pipeline = model.calls[0]["postprocess"]
    assert isinstance(pipeline, tuple)
    assert pipeline[:2] == (compact, finish)
    assert len(pipeline) == 3
    assert isinstance(pipeline[-1], rf.Postprocessor)
    assert seen == ["compact", "finish"]
    assert output == {"request_id": 5, "label": "5"}


def test_postprocessor_cannot_claim_the_deployment_identity_column():
    @rf.postprocess
    def collide(batch: rf.Batch) -> rf.Batch:
        return batch.replace(
            batch.data.append_column(
                deployment_module.TRANSPORT_IDENTITY,
                pa.array(["mine"] * len(batch)),
            )
        )

    server = runtime(postprocessor=collide)

    with pytest.raises(ValueError, match="reserved column"):
        server.predict_payloads([{"id": 1}])


def test_runtime_rejects_replaced_request_identity():
    class ReplacingModel(PredictModel):
        def predict(self, source, *, preprocess, postprocess, retain):
            replaced = rf.Batch(
                data=source.data,
                identity=identity(len(source), namespace="replaced"),
            )
            return super().predict(
                replaced,
                preprocess=preprocess,
                postprocess=postprocess,
                retain=retain,
            )

    server = runtime(model=ReplacingModel())

    with pytest.raises(ValueError, match="exactly one output for each request"):
        server.predict_payloads([{"id": 1}])


def test_response_signature_validates_each_terminal_arrow_row():
    class Response(pydantic.BaseModel):
        predictions: dict[str, str]

    server = runtime(response_signature=Response)

    assert server.predict_payloads([{"id": 7}]) == [{"predictions": {"label": "7"}}]


def test_runtime_rejects_non_arrow_model_output():
    class BrokenModel:
        def predict(self, source, **kwargs):
            return [{"predictions": {}}]

    server = runtime(model=BrokenModel())

    with pytest.raises(TypeError, match="Model.predict must return a pyarrow.Table"):
        server.predict_payloads([{"id": 1}])


def test_runtime_rejects_prediction_row_count_drift():
    class BrokenModel:
        def predict(self, source, **kwargs):
            return pa.table({"predictions": pa.array([], type=pa.null())})

    server = runtime(model=BrokenModel())

    with pytest.raises(ValueError, match="returned 0 rows for 1 valid"):
        server.predict_payloads([{"id": 1}])


def test_runtime_reports_arrow_incompatible_requests_without_calling_model():
    model = PredictModel()
    server = runtime(model=model)
    opaque = object()

    output = server.predict_payloads([{"id": opaque}])[0]

    assert model.calls == []
    assert output["error"]["status_code"] == 422
    assert "Arrow-compatible" in output["error"]["message"]


def test_runtime_setup_applies_updates_before_device_placement(monkeypatch):
    calls = []

    class LoadedModel:
        def update(self, *predicates, **values):
            calls.append(("update", predicates, values))
            return self

        def to(self, device):
            calls.append(("to", str(device)))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    loaded = LoadedModel()
    monkeypatch.setattr(deployment_module.Model, "load", classmethod(lambda cls, checkpoint: loaded))
    predicate = where("name") == "label"
    server = deployment_module.FastAPIRuntime(
        checkpoint="unused",
        accelerator=deployment_module.Accelerator.cpu,
        update_operations=[((predicate,), {"target": False})],
    )

    server.setup()

    assert calls == [
        ("update", (predicate,), {"target": False}),
        ("to", "cpu"),
        ("eval",),
    ]


def test_runtime_setup_uses_an_in_memory_model(monkeypatch):
    model = Model(Number(name="amount"), d_model=8, n_layers=1, n_heads=2)
    monkeypatch.setattr(
        deployment_module.Model,
        "load",
        classmethod(lambda cls, checkpoint: (_ for _ in ()).throw(AssertionError("checkpoint should not load"))),
    )
    server = deployment_module.FastAPIRuntime(
        checkpoint=model,
        accelerator=deployment_module.Accelerator.cpu,
    )

    server.setup()

    assert server.model is model


def test_deployment_builder_keeps_arrow_processors_and_retention():
    @rf.preprocess
    def prepare(batch: rf.Batch) -> rf.Batch:
        return batch

    @rf.postprocess
    def compact(batch: rf.Batch) -> rf.Batch:
        return batch

    deployment = Deployment(checkpoint="unused", retain=("request_id",)).preprocess(prepare).postprocess(compact)

    assert deployment.retain == ("request_id",)
    assert deployment._preprocessors == (prepare,)
    assert deployment._postprocessors == (compact,)


def test_deployment_builder_replaces_processor_collections():
    @rf.preprocess
    def prepare(batch: rf.Batch) -> rf.Batch:
        return batch

    @rf.preprocess
    def replace(batch: rf.Batch) -> rf.Batch:
        return batch

    @rf.postprocess
    def compact(batch: rf.Batch) -> rf.Batch:
        return batch

    @rf.postprocess
    def finish(batch: rf.Batch) -> rf.Batch:
        return batch

    deployment = (
        Deployment(checkpoint="unused")
        .preprocess([prepare])
        .preprocess((replace,))
        .postprocess([compact])
        .postprocess((finish,))
    )

    assert deployment._preprocessors == (replace,)
    assert deployment._postprocessors == (finish,)


@pytest.mark.parametrize("retain", [("id", "id"), ("",), ["id"]])
def test_deployment_rejects_invalid_retention(retain):
    with pytest.raises((TypeError, ValueError, pydantic.ValidationError)):
        Deployment(checkpoint="unused", retain=retain)


def test_deployment_forge_registers_openapi_signatures():
    class Request(pydantic.BaseModel):
        amount: float

    class Response(pydantic.BaseModel):
        predictions: dict[str, object]

    app = Deployment(checkpoint="unused").forge(request=Request, response=Response).app()
    schema = app.openapi()
    operation = schema["paths"]["/predict"]["post"]

    assert {"Request", "Response"} <= set(schema["components"]["schemas"])
    assert operation["requestBody"]["content"]["application/json"]["schema"] == {
        "anyOf": [
            {"$ref": "#/components/schemas/Request"},
            {"type": "array", "items": {"$ref": "#/components/schemas/Request"}},
        ]
    }


def test_deployment_serve_configures_uvicorn(monkeypatch):
    captured = {}

    def run(app, *, host, port, log_level):
        captured.update(app=app, host=host, port=port, log_level=log_level)

    monkeypatch.setattr(deployment_module.uvicorn, "run", run)

    Deployment(
        checkpoint="unused",
        accelerator="cpu",
        host="127.0.0.1",
        port=8765,
        log_level="error",
    ).serve()

    assert isinstance(captured["app"], deployment_module.fastapi.FastAPI)
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert captured["log_level"] == "error"


def test_multiworker_deployment_rejects_in_process_builder_configuration():
    @rf.postprocess
    def compact(batch: rf.Batch) -> rf.Batch:
        return batch

    with pytest.raises(ValueError, match="cannot serialize in-process"):
        Deployment(checkpoint="model.ckpt", workers=2).postprocess(compact).serve()


def test_deployment_rejects_explicit_checkpoint_and_model():
    model = Model(Number(name="amount"), d_model=8, n_layers=1, n_heads=2)

    with pytest.raises(ValueError, match="pass either checkpoint or model"):
        Deployment(checkpoint="model.ckpt", model=model)


def test_error_item_is_stable():
    error = ErrorItem(status_code=422, message="bad input")

    assert runtime().failure(error) == {
        "predictions": {},
        "error": {"status_code": 422, "message": "bad input"},
    }

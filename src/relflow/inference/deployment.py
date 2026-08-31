"""Realtime deployment wrappers for `relflow` checkpoints."""

import asyncio
import os
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast

import fastapi
import orjson
import pydantic
import torch
import uvicorn
from beartype import beartype
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tensordict import TensorDict

from relflow.architecture.root import Model
from relflow.data.iterables import encode
from relflow.data.processors import Postprocessor, Preprocessor
from relflow.structs.enums import Strata, TensorKey
from relflow.structs.experiment import NodeAttribute, NodePredicate
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address, Node
from relflow.tensorfields.base import TensorFieldBase

Input: TypeAlias = TensorDict[Address, TensorFieldBase]
ModelSource: TypeAlias = str | Path | Model
UpdateOperation: TypeAlias = tuple[tuple[NodePredicate | NodeAttribute | Callable[[Node], bool], ...], dict[str, Any]]


class Accelerator(StrEnum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"

    @classmethod
    def _missing_(cls, value: object) -> "Accelerator | None":
        if not isinstance(value, str):
            return None

        normalized = value.strip().lower()
        if normalized == "":
            raise ValueError("accelerator must not be blank")

        return cast(Accelerator | None, cls._value2member_map_.get(normalized))


class JSONBackend(StrEnum):
    orjson = "orjson"
    stdlib = "stdlib"

    @classmethod
    def _missing_(cls, value: object) -> "JSONBackend | None":
        if not isinstance(value, str):
            return None

        normalized = value.strip().lower()
        if normalized == "":
            raise ValueError("json_backend must not be blank")

        return cast(JSONBackend | None, cls._value2member_map_.get(normalized))


class ErrorItem(pydantic.BaseModel):
    status_code: int
    message: str


@dataclass
class RequestItem:
    observations: list[Any]


@dataclass
class ResponseItem:
    predictions: list[Prediction]
    input: Input | None = None
    observations: list[Any] | None = None


class FastAPIRuntime:
    """Load a relflow model and execute batched request prediction."""

    def __init__(
        self,
        *,
        checkpoint: ModelSource,
        accelerator: Accelerator,
        preprocessor: Preprocessor | None = None,
        postprocessor: Postprocessor | None = None,
        update_operations: list[UpdateOperation] | None = None,
        request_signature: type[pydantic.BaseModel] | None = None,
        response_signature: type[pydantic.BaseModel] | None = None,
    ) -> None:
        self.model_source: ModelSource = checkpoint
        self.accelerator = accelerator
        self.preprocessor: Preprocessor | None = Preprocessor.normalize(preprocessor)
        self.postprocessor: Postprocessor | None = Postprocessor.normalize(postprocessor)
        self.update_operations = list(update_operations or [])
        self.request_signature = request_signature
        self.response_signature = response_signature
        self.device: torch.device = torch.device("cpu")

    def setup(self) -> None:
        if self.accelerator == Accelerator.auto:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                mps = getattr(torch.backends, "mps", None)
                self.device = torch.device("mps" if mps is not None and mps.is_available() else "cpu")
        else:
            self.device = torch.device(self.accelerator.value)

        model = self.model_source if isinstance(self.model_source, Model) else Model.load(self.model_source)
        for predicates, values in self.update_operations:
            model.update(*predicates, **values)

        self.model: Model = model.to(self.device)
        self.model.eval()
        self.interprocess_encoding_context = self.model.interprocess_encoding_context

    def decode_payload(self, payload: dict[str, Any], context: dict[str, Any]) -> RequestItem | ErrorItem:
        try:
            request: dict[str, Any] | pydantic.BaseModel
            if self.request_signature is None:
                request = payload
            else:
                request = self.request_signature.model_validate(payload)

            if isinstance(request, pydantic.BaseModel):
                request = request.model_dump()

            context["request"] = request

            if self.preprocessor is None:
                observations: list[Any] = [[request]]
            else:
                model = getattr(self, "model", None)
                observations = list(
                    self.preprocessor.outputs(
                        request,
                        strata=Strata.predict,
                        schema=model.schema if model is not None else None,
                        encoding_context=getattr(self, "interprocess_encoding_context", {}),
                    )
                )

        except Exception as exception:
            return ErrorItem(status_code=422, message=str(exception))

        if len(observations) == 0 or any(x is None for x in observations):
            return ErrorItem(status_code=422, message="preprocessor returned no observations for request")
        if len(observations) != 1:
            return ErrorItem(status_code=422, message="deployment requests must encode exactly one observation")

        context["observations"] = observations
        return RequestItem(observations=observations)

    def encode_response(
        self,
        response: ResponseItem | ErrorItem,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(response, ErrorItem):
            return {
                "predictions": {},
                "error": {
                    "status_code": response.status_code,
                    "message": response.message,
                },
            }

        if response.input is not None:
            context["input"] = response.input
        if response.observations is not None:
            context["observations"] = response.observations

        predictions = self.model.write(predictions=response.predictions)
        if self.postprocessor is not None:
            available: dict[str, Any] = {}
            for name in ("request", "observations"):
                if name in context:
                    available[name] = context[name]
            if response.input is not None:
                available["input"] = response.input
                if TensorKey.metadata in response.input.keys():
                    available["metadata"] = response.input[TensorKey.metadata]

            processed = self.postprocessor.run(predictions, available=available)
            if processed is not None:
                predictions = dict(processed)

        encoded = Prediction.denest(dict(predictions=predictions))
        if self.response_signature is not None:
            return self.response_signature.model_validate(encoded).model_dump(mode="json")

        return cast(dict[str, Any], encoded)

    def encode_written_response(self, predictions: dict[Address, dict[str, Any]]) -> dict[str, Any]:
        encoded = Prediction.denest(dict(predictions=predictions))
        if self.response_signature is not None:
            return self.response_signature.model_validate(encoded).model_dump(mode="json")

        return cast(dict[str, Any], encoded)

    def split_written_responses(
        self,
        predictions: dict[Address, dict[str, Any]],
        batch_size: int,
    ) -> list[dict[str, Any]]:
        responses: list[dict[str, Any]] = []

        def select(value: Any, index: int) -> Any:
            if isinstance(value, dict):
                return {key: select(item, index) for key, item in value.items()}

            if isinstance(value, list):
                if len(value) != batch_size:
                    raise ValueError(
                        f"cannot split batched prediction payload with leading dimension {len(value)} "
                        f"for batch size {batch_size}"
                    )
                return [value[index]]

            return value

        for index in range(batch_size):
            item_predictions = {address: select(payload, index) for address, payload in predictions.items()}
            responses.append(self.encode_written_response(item_predictions))

        return responses

    def predict_payloads(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = [{} for _ in payloads]
        outputs: list[dict[str, Any] | None] = [None for _ in payloads]
        valid_indices: list[int] = []
        observations: list[Any] = []
        spans: list[tuple[int, int]] = []

        for index, payload in enumerate(payloads):
            decoded = self.decode_payload(payload, contexts[index])
            if isinstance(decoded, ErrorItem):
                outputs[index] = self.encode_response(decoded, contexts[index])
                continue

            start = len(observations)
            observations.extend(decoded.observations)
            spans.append((start, len(observations)))
            valid_indices.append(index)

        if observations:
            encoded = encode(
                batch=observations,
                schema=self.model.schema,
                strata=Strata.predict,
                interprocess_encoding_context=self.interprocess_encoding_context,
            )

            model_input = encoded
            if TensorKey.metadata in encoded.keys():
                model_input = cast(
                    Input,
                    TensorDict(
                        source=cast(
                            Any,
                            {Address(str(key)): value for key, value in encoded.items() if key != TensorKey.metadata},
                        ),
                        batch_size=encoded.batch_size,
                    ),
                )

            model_input = cast(Input, model_input.to(self.device))
            with torch.inference_mode():
                predictions = self.model(model_input, strata=Strata.predict)

            if self.postprocessor is None:
                written = self.model.write(predictions=predictions)
                for index, response in zip(valid_indices, self.split_written_responses(written, len(valid_indices))):
                    outputs[index] = response

                return [cast(dict[str, Any], output) for output in outputs]

            unbatched = Prediction.unbatch(predictions=predictions)
            for index, (start, stop) in zip(valid_indices, spans):
                if stop - start != 1:
                    raise ValueError("deployment requests must encode exactly one observation")

                input_slice: dict[Address, TensorFieldBase] = {}
                for key, value in model_input.items():
                    input_slice[Address(str(key))] = value[start:stop]

                sliced = cast(Input, TensorDict(source=cast(Any, input_slice), batch_size=[stop - start]))
                if TensorKey.metadata in encoded.keys():
                    sliced[TensorKey.metadata] = encoded[TensorKey.metadata][start:stop]

                response = ResponseItem(
                    predictions=unbatched[start] if unbatched else [],
                    input=sliced,
                    observations=observations[start:stop],
                )
                outputs[index] = self.encode_response(response, contexts[index])

        return [cast(dict[str, Any], output) for output in outputs]


@dataclass
class _QueuedRequest:
    payload: dict[str, Any]
    future: asyncio.Future[dict[str, Any]]


class FastAPIBatcher:
    """Single-model async request batcher for FastAPI endpoints."""

    def __init__(
        self,
        runtime: FastAPIRuntime,
        *,
        max_batch_size: int,
        batch_timeout: float,
    ) -> None:
        self.runtime = runtime
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.queue: asyncio.Queue[_QueuedRequest] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.runtime.setup()
        self.task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self.task is None:
            return

        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task

    async def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return (await self.submit_many([payload]))[0]

    async def submit_many(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not payloads:
            return []

        loop = asyncio.get_running_loop()
        items = [_QueuedRequest(payload=payload, future=loop.create_future()) for payload in payloads]
        for item in items:
            await self.queue.put(item)

        return list(await asyncio.gather(*(item.future for item in items)))

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            self._drain_ready(batch)

            if len(batch) < self.max_batch_size and self.batch_timeout > 0.0:
                deadline = asyncio.get_running_loop().time() + self.batch_timeout
                while len(batch) < self.max_batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0.0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
                    except TimeoutError:
                        break
                    self._drain_ready(batch)

            try:
                payloads = [item.payload for item in batch]
                responses = await asyncio.to_thread(self.runtime.predict_payloads, payloads)
            except Exception as exception:
                for item in batch:
                    if not item.future.cancelled():
                        item.future.set_exception(exception)
                continue

            for item, response in zip(batch, responses):
                if not item.future.cancelled():
                    item.future.set_result(response)

    def _drain_ready(self, batch: list[_QueuedRequest]) -> None:
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                return


class Deployment(BaseSettings):
    """Serving configuration for a `relflow` checkpoint or model instance.

    `Deployment` queues request/response schemas, optional preprocessors,
    optional postprocessors, and `update(...)` mutations before the model is
    loaded by FastAPI application startup.
    """

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
        validate_by_name=True,
        validate_by_alias=True,
        arbitrary_types_allowed=True,
    )

    checkpoint: ModelSource = Field(
        default="model.ckpt",
        validation_alias=AliasChoices("RELFLOW_CHECKPOINT", "CHECKPOINT"),
    )
    model: Model | None = Field(default=None, exclude=True)
    max_batch_size: int = Field(
        default=128,
        ge=1,
        validation_alias=AliasChoices("RELFLOW_MAX_BATCH_SIZE", "MAX_BATCH_SIZE"),
    )
    batch_timeout: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("RELFLOW_BATCH_TIMEOUT", "BATCH_TIMEOUT"),
    )
    workers: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("RELFLOW_WORKERS", "WORKERS"),
    )
    accelerator: Accelerator = Field(
        default=Accelerator.auto,
        validation_alias=AliasChoices("RELFLOW_ACCELERATOR", "ACCELERATOR"),
    )
    host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("RELFLOW_HOST", "HOST"),
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("RELFLOW_PORT", "PORT"),
    )
    log_level: str = Field(
        default="info",
        validation_alias=AliasChoices("RELFLOW_LOG_LEVEL", "LOG_LEVEL"),
    )
    json_backend: JSONBackend = Field(
        default=JSONBackend.orjson,
        validation_alias=AliasChoices("RELFLOW_JSON_BACKEND", "JSON_BACKEND"),
    )

    _request_signature: type[pydantic.BaseModel] | None = pydantic.PrivateAttr(default=None)
    _response_signature: type[pydantic.BaseModel] | None = pydantic.PrivateAttr(default=None)
    _preprocessor = pydantic.PrivateAttr(default=None)
    _postprocessor = pydantic.PrivateAttr(default=None)
    _update_operations: list[UpdateOperation] = pydantic.PrivateAttr(default_factory=list)

    @field_validator("checkpoint", mode="before")
    @classmethod
    def strip_checkpoint(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                raise ValueError("checkpoint must not be blank")
            return stripped

        return value

    @model_validator(mode="after")
    def check_model_source(self) -> "Deployment":
        if self.model is not None and "checkpoint" in self.model_fields_set:
            raise ValueError("pass either checkpoint or model, not both")

        return self

    @beartype
    def forge(
        self,
        request: type[pydantic.BaseModel] | None = None,
        response: type[pydantic.BaseModel] | None = None,
    ) -> "Deployment":
        """Attach optional Pydantic request and response signatures."""
        self._request_signature = request
        self._response_signature = response

        return self

    @beartype
    def preprocess(self, preprocessor: Preprocessor) -> "Deployment":
        """Attach an optional request preprocessor.

        If this method is not called, request objects are encoded unchanged.
        """
        self._preprocessor = Preprocessor.normalize(preprocessor)

        return self

    @beartype
    def postprocess(self, postprocessor: Postprocessor) -> "Deployment":
        """Attach an optional response postprocessor."""
        self._postprocessor = Postprocessor.normalize(postprocessor)

        return self

    @beartype
    def update(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        strict: bool = True,
        allow_extra: bool = False,
        include_root: bool = True,
        validate: bool = True,
        **values: Any,
    ) -> "Deployment":
        """Queue a model schema mutation to apply during server startup.

        This mirrors `Model.update(...)` and is useful for serving-time changes
        such as `target=False`.
        """
        self._update_operations.append(
            (
                tuple(predicates),
                {
                    "strict": strict,
                    "allow_extra": allow_extra,
                    "include_root": include_root,
                    "validate": validate,
                    **values,
                },
            )
        )

        return self

    def app(self) -> fastapi.FastAPI:
        """Build a FastAPI app for the configured checkpoint or model."""
        runtime = FastAPIRuntime(
            checkpoint=self.model if self.model is not None else self.checkpoint,
            accelerator=self.accelerator,
            preprocessor=self._preprocessor,
            postprocessor=self._postprocessor,
            update_operations=self._update_operations,
            request_signature=self._request_signature,
            response_signature=self._response_signature,
        )
        batcher = FastAPIBatcher(
            runtime=runtime,
            max_batch_size=self.max_batch_size,
            batch_timeout=self.batch_timeout,
        )

        @asynccontextmanager
        async def lifespan(app: fastapi.FastAPI):
            await batcher.start()
            app.state.relflow_runtime = runtime
            app.state.relflow_batcher = batcher
            try:
                yield
            finally:
                await batcher.stop()

        app = fastapi.FastAPI(lifespan=lifespan)

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        def json_response(content: Any, status_code: int = 200) -> fastapi.Response:
            if self.json_backend == JSONBackend.orjson:
                return fastapi.Response(
                    content=orjson.dumps(content, option=orjson.OPT_NON_STR_KEYS),
                    status_code=status_code,
                    media_type="application/json",
                )

            return JSONResponse(content=content, status_code=status_code)

        @app.post("/predict")
        async def predict(request: fastapi.Request) -> fastapi.Response:
            try:
                if self.json_backend == JSONBackend.orjson:
                    payload = orjson.loads(await request.body())
                else:
                    payload = await request.json()
            except Exception as exception:
                return json_response(
                    status_code=400,
                    content={
                        "predictions": {},
                        "error": {
                            "status_code": 400,
                            "message": str(exception),
                        },
                    },
                )

            if isinstance(payload, dict):
                response = await batcher.submit(cast(dict[str, Any], payload))
                return json_response(content=response)

            if isinstance(payload, list):
                for index, item in enumerate(payload):
                    if not isinstance(item, dict):
                        return json_response(
                            status_code=422,
                            content={
                                "predictions": {},
                                "error": {
                                    "status_code": 422,
                                    "message": (
                                        "request body must be a JSON object or an array of JSON objects; "
                                        f"item {index} is {type(item).__name__}"
                                    ),
                                },
                            },
                        )

                responses = await batcher.submit_many(cast(list[dict[str, Any]], payload))
                return json_response(content=responses)

            return json_response(
                status_code=422,
                content={
                    "predictions": {},
                    "error": {
                        "status_code": 422,
                        "message": f"request body must be a JSON object or an array of JSON objects, got {type(payload).__name__}",
                    },
                },
            )

        self._register_openapi_signatures(app)
        return app

    def _register_openapi_signatures(self, app: fastapi.FastAPI) -> None:
        if self._request_signature is None and self._response_signature is None:
            return

        def register_model(openapi_schema: dict[str, Any], model: type[pydantic.BaseModel]) -> str:
            schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
            definitions = schema.pop("$defs", {})

            components = openapi_schema.setdefault("components", {}).setdefault("schemas", {})
            for name, definition in definitions.items():
                components.setdefault(name, definition)

            name = str(schema.get("title") or model.__name__)
            components.setdefault(name, schema)
            return name

        def single_or_batch_schema(name: str) -> dict[str, Any]:
            item = {"$ref": f"#/components/schemas/{name}"}
            return {
                "anyOf": [
                    item,
                    {
                        "type": "array",
                        "items": item,
                    },
                ]
            }

        def custom_openapi() -> dict[str, Any]:
            if app.openapi_schema is not None:
                return app.openapi_schema

            openapi_schema = get_openapi(
                title=app.title,
                version=app.version,
                openapi_version=app.openapi_version,
                summary=app.summary,
                description=app.description,
                routes=app.routes,
            )
            operation = openapi_schema["paths"]["/predict"]["post"]

            if self._request_signature is not None:
                request_name = register_model(openapi_schema, self._request_signature)
                operation["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": single_or_batch_schema(request_name),
                        }
                    },
                }

            if self._response_signature is not None:
                response_name = register_model(openapi_schema, self._response_signature)
                operation.setdefault("responses", {}).setdefault("200", {}).setdefault("content", {})[
                    "application/json"
                ] = {
                    "schema": single_or_batch_schema(response_name),
                }

            app.openapi_schema = openapi_schema
            return app.openapi_schema

        app.openapi = custom_openapi  # type: ignore[method-assign]

    def serve(self) -> None:
        """Start the FastAPI server for the configured checkpoint or model."""
        if self.workers > 1:
            if self.model is not None or isinstance(self.checkpoint, Model):
                raise ValueError("workers > 1 requires a checkpoint path, not an in-memory model")

            env_updates = {
                "RELFLOW_CHECKPOINT": str(self.checkpoint),
                "RELFLOW_MAX_BATCH_SIZE": str(self.max_batch_size),
                "RELFLOW_BATCH_TIMEOUT": str(self.batch_timeout),
                "RELFLOW_ACCELERATOR": self.accelerator.value,
                "RELFLOW_JSON_BACKEND": self.json_backend.value,
            }
            previous = {key: os.environ.get(key) for key in env_updates}
            try:
                os.environ.update(env_updates)
                uvicorn.run(
                    "relflow.inference.deployment:create_app",
                    factory=True,
                    workers=self.workers,
                    host=self.host,
                    port=self.port,
                    log_level=self.log_level,
                )
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            return

        uvicorn.run(
            self.app(),
            host=self.host,
            port=self.port,
            log_level=self.log_level,
        )


def create_app() -> fastapi.FastAPI:
    """Build a FastAPI app from deployment environment variables."""
    return Deployment().app()

"""Realtime deployment over RelFlow's Arrow prediction boundary."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import fastapi
import orjson
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
import uvicorn
from beartype import beartype
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from relflow.architecture.root import Model
from relflow.data.arrow import Batch, mappings
from relflow.data.datasets.arrow import identity
from relflow.data.processors import Postprocessor, Preprocessor, equal
from relflow.structs.experiment import NodeAttribute, NodePredicate
from relflow.structs.tree import Node

Input: TypeAlias = dict[str, Any]
ModelSource: TypeAlias = str | Path | Model
Retain: TypeAlias = tuple[str, ...] | Literal["*"]
UpdateOperation: TypeAlias = tuple[tuple[NodePredicate | NodeAttribute | Callable[[Node], bool], ...], dict[str, Any]]
TRANSPORT_IDENTITY = "__relflow_identity__"


class Accelerator(StrEnum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"

    @classmethod
    def _missing_(cls, value: object) -> Accelerator | None:
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
    def _missing_(cls, value: object) -> JSONBackend | None:
        if not isinstance(value, str):
            return None

        normalized = value.strip().lower()
        if normalized == "":
            raise ValueError("json_backend must not be blank")
        return cast(JSONBackend | None, cls._value2member_map_.get(normalized))


class ErrorItem(pydantic.BaseModel):
    """One request-local validation error."""

    status_code: int
    message: str


def retention(value: Retain) -> Retain:
    """Validate the deployment's processed-input retention projection."""

    if value == "*":
        return value
    if not isinstance(value, tuple):
        raise TypeError("retain must be a tuple of top-level column names or '*'")
    if any(not isinstance(name, str) or not name for name in value):
        raise TypeError("retain entries must be non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError("retain entries must be unique")
    return value


class FastAPIRuntime:
    """Load one model and predict one valid Arrow microbatch at a time."""

    def __init__(
        self,
        *,
        checkpoint: ModelSource,
        accelerator: Accelerator,
        preprocessor: Preprocessor | None = None,
        postprocessor: Postprocessor | None = None,
        retain: Retain = (),
        update_operations: list[UpdateOperation] | None = None,
        request_signature: type[pydantic.BaseModel] | None = None,
        response_signature: type[pydantic.BaseModel] | None = None,
    ) -> None:
        self.model_source = checkpoint
        self.accelerator = accelerator
        self.preprocessor = Preprocessor.normalize(preprocessor)
        self.postprocessor = Postprocessor.normalize(postprocessor)
        self.retain = retention(retain)
        self.update_operations = list(update_operations or [])
        self.request_signature = request_signature
        self.response_signature = response_signature
        self.device = torch.device("cpu")

    def setup(self) -> None:
        """Load, mutate, place, and freeze the configured model."""

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

        self.model = model.to(self.device)
        self.model.eval()

    def validate(self, payload: Any) -> Input | ErrorItem:
        """Validate one JSON object without changing its request position."""

        if not isinstance(payload, dict):
            return ErrorItem(
                status_code=422,
                message=f"each request must be a JSON object, got {type(payload).__name__}",
            )

        try:
            if self.request_signature is None:
                return cast(Input, payload)
            request = self.request_signature.model_validate(payload)
            return cast(Input, request.model_dump(mode="python"))
        except Exception as exception:
            return ErrorItem(status_code=422, message=str(exception))

    def failure(self, error: ErrorItem) -> dict[str, Any]:
        """Encode one request-local error using the stable response envelope."""

        return {
            "predictions": {},
            "error": {
                "status_code": error.status_code,
                "message": error.message,
            },
        }

    def predict_payloads(self, payloads: list[Any]) -> list[dict[str, Any]]:
        """Validate, predict, serialize, and scatter one request microbatch.

        Valid requests cross the Python-to-Arrow boundary together, call
        ``Model.predict`` once, and cross back to Python with one terminal
        ``Table.to_pylist``. Invalid request rows retain their original slots.
        """

        outputs: list[dict[str, Any] | None] = [None] * len(payloads)
        valid: list[Input] = []
        request_positions: list[int] = []

        for position, payload in enumerate(payloads):
            item = self.validate(payload)
            if isinstance(item, ErrorItem):
                outputs[position] = self.failure(item)
            else:
                valid.append(item)
                request_positions.append(position)

        if valid:
            try:
                data = pa.Table.from_pylist(mappings(valid, context="deployment requests"))
            except (pa.ArrowException, TypeError, ValueError) as exception:
                error = self.failure(
                    ErrorItem(status_code=422, message=f"request is not Arrow-compatible: {exception}")
                )
                for position in request_positions:
                    outputs[position] = error
                return [cast(dict[str, Any], output) for output in outputs]

            source = Batch(
                data=data,
                identity=identity(len(valid), namespace="deployment"),
            )

            def transport(batch: Batch) -> Batch:
                result = self.postprocessor.run(batch) if self.postprocessor is not None else batch
                if TRANSPORT_IDENTITY in result.data.column_names:
                    raise ValueError(f"postprocessor output cannot use reserved column {TRANSPORT_IDENTITY!r}")
                return result.replace(result.data.append_column(TRANSPORT_IDENTITY, result.identity))

            result = self.model.predict(
                source,
                preprocess=self.preprocessor,
                postprocess=Postprocessor(func=transport),
                retain=self.retain,
            )
            if not isinstance(result, pa.Table):
                raise TypeError(f"Model.predict must return a pyarrow.Table, got {type(result).__name__}")
            if result.num_rows != len(valid):
                raise ValueError(
                    f"Model.predict returned {result.num_rows} rows for {len(valid)} valid deployment requests"
                )
            if TRANSPORT_IDENTITY not in result.column_names:
                raise ValueError("Model.predict did not preserve deployment identity through postprocessing")

            source_instances = pc.struct_field(source.identity, "instance")
            result_instances = pc.struct_field(result[TRANSPORT_IDENTITY], "instance")
            if pc.count_distinct(result_instances).as_py() != len(result_instances):
                raise ValueError("deployment preprocessing produced duplicate request identities")
            alignment = pc.index_in(source_instances, value_set=result_instances)
            if alignment.null_count:
                raise ValueError("deployment preprocessing must produce exactly one output for each request")
            result = result.take(alignment)
            aligned = result[TRANSPORT_IDENTITY]
            if not equal(pc.struct_field(aligned, "logical"), pc.struct_field(source.identity, "logical")) or not equal(
                pc.struct_field(aligned, "instance"), source_instances
            ):
                raise ValueError("deployment preprocessing must produce exactly one output for each request")
            result = result.drop([TRANSPORT_IDENTITY])

            if self.postprocessor is None:
                if "predictions" not in result.column_names:
                    raise ValueError("canonical model output is missing its 'predictions' column")
                result = result.select(["predictions"])

            rows = result.to_pylist()
            for position, row in zip(request_positions, rows, strict=True):
                if self.response_signature is not None:
                    response = self.response_signature.model_validate(row)
                    row = response.model_dump(mode="json")
                outputs[position] = cast(dict[str, Any], row)

        return [cast(dict[str, Any], output) for output in outputs]


@dataclass(slots=True)
class QueuedRequest:
    payload: Any
    future: asyncio.Future[dict[str, Any]]


class FastAPIBatcher:
    """Gather concurrent HTTP requests into bounded model microbatches."""

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
        self.queue: asyncio.Queue[QueuedRequest] = asyncio.Queue()
        self.task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self.runtime.setup()
        self.task = asyncio.create_task(self.work())

    async def stop(self) -> None:
        if self.task is None:
            return

        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task
        self.task = None

    async def submit(self, payload: Any) -> dict[str, Any]:
        return (await self.submit_many([payload]))[0]

    async def submit_many(self, payloads: list[Any]) -> list[dict[str, Any]]:
        if not payloads:
            return []

        loop = asyncio.get_running_loop()
        items = [QueuedRequest(payload=payload, future=loop.create_future()) for payload in payloads]
        for item in items:
            await self.queue.put(item)
        return list(await asyncio.gather(*(item.future for item in items)))

    async def work(self) -> None:
        while True:
            first = await self.queue.get()
            batch = [first]
            self.drain(batch)

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
                    self.drain(batch)

            try:
                responses = await asyncio.to_thread(
                    self.runtime.predict_payloads,
                    [item.payload for item in batch],
                )
            except Exception as exception:
                for item in batch:
                    if not item.future.cancelled():
                        item.future.set_exception(exception)
                continue

            if len(responses) != len(batch):
                exception = RuntimeError(
                    f"deployment runtime returned {len(responses)} responses for {len(batch)} requests"
                )
                for item in batch:
                    if not item.future.cancelled():
                        item.future.set_exception(exception)
                continue

            for item, response in zip(batch, responses, strict=True):
                if not item.future.cancelled():
                    item.future.set_result(response)

    def drain(self, batch: list[QueuedRequest]) -> None:
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                return


class Deployment(BaseSettings):
    """Serving configuration for a checkpoint or an in-memory model.

    Pydantic request signatures validate each item before the valid rows are
    converted into one Arrow table. ``retain`` controls which processed input
    columns are available to an Arrow postprocessor; the default JSON response
    still exposes predictions only.
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
    retain: Retain = Field(
        default=(),
        validation_alias=AliasChoices("RELFLOW_RETAIN", "RETAIN"),
    )

    _request_signature: type[pydantic.BaseModel] | None = pydantic.PrivateAttr(default=None)
    _response_signature: type[pydantic.BaseModel] | None = pydantic.PrivateAttr(default=None)
    _preprocessor: Preprocessor | None = pydantic.PrivateAttr(default=None)
    _postprocessor: Postprocessor | None = pydantic.PrivateAttr(default=None)
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

    @field_validator("retain", mode="before")
    @classmethod
    def check_retain(cls, value: Any) -> Retain:
        return retention(cast(Retain, value))

    @model_validator(mode="after")
    def check_model_source(self) -> Deployment:
        if self.model is not None and "checkpoint" in self.model_fields_set:
            raise ValueError("pass either checkpoint or model, not both")
        return self

    @beartype
    def forge(
        self,
        request: type[pydantic.BaseModel] | None = None,
        response: type[pydantic.BaseModel] | None = None,
    ) -> Deployment:
        """Attach optional Pydantic request and response signatures."""

        self._request_signature = request
        self._response_signature = response
        return self

    @beartype
    def preprocess(self, preprocessor: Preprocessor) -> Deployment:
        """Attach one Arrow batch preprocessor."""

        self._preprocessor = Preprocessor.normalize(preprocessor)
        return self

    @beartype
    def postprocess(self, postprocessor: Postprocessor) -> Deployment:
        """Attach one Arrow output postprocessor."""

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
    ) -> Deployment:
        """Queue a model schema mutation to apply during server startup."""

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
        """Build a FastAPI app for the configured model."""

        runtime = FastAPIRuntime(
            checkpoint=self.model if self.model is not None else self.checkpoint,
            accelerator=self.accelerator,
            preprocessor=self._preprocessor,
            postprocessor=self._postprocessor,
            retain=self.retain,
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
                payload = (
                    orjson.loads(await request.body())
                    if self.json_backend == JSONBackend.orjson
                    else await request.json()
                )
            except Exception as exception:
                return json_response(
                    self.error(status_code=400, message=str(exception)),
                    status_code=400,
                )

            if isinstance(payload, dict):
                return json_response(await batcher.submit(payload))
            if isinstance(payload, list):
                return json_response(await batcher.submit_many(payload))
            return json_response(
                self.error(
                    status_code=422,
                    message=(
                        f"request body must be a JSON object or an array of JSON objects, got {type(payload).__name__}"
                    ),
                ),
                status_code=422,
            )

        self.register(app)
        return app

    def error(self, *, status_code: int, message: str) -> dict[str, Any]:
        """Build a transport-level error response."""

        return {
            "predictions": {},
            "error": {
                "status_code": status_code,
                "message": message,
            },
        }

    def register(self, app: fastapi.FastAPI) -> None:
        """Register optional Pydantic signatures in the OpenAPI schema."""

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

        def cardinality(name: str) -> dict[str, Any]:
            item = {"$ref": f"#/components/schemas/{name}"}
            return {"anyOf": [item, {"type": "array", "items": item}]}

        def schema() -> dict[str, Any]:
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
                    "content": {"application/json": {"schema": cardinality(request_name)}},
                }
            if self._response_signature is not None:
                response_name = register_model(openapi_schema, self._response_signature)
                operation.setdefault("responses", {}).setdefault("200", {}).setdefault("content", {})[
                    "application/json"
                ] = {"schema": cardinality(response_name)}

            app.openapi_schema = openapi_schema
            return openapi_schema

        app.openapi = schema  # type: ignore[method-assign]

    def serve(self) -> None:
        """Start the configured FastAPI server."""

        if self.workers > 1:
            if self.model is not None or isinstance(self.checkpoint, Model):
                raise ValueError("workers > 1 requires a checkpoint path, not an in-memory model")
            if any(
                (
                    self._request_signature is not None,
                    self._response_signature is not None,
                    self._preprocessor is not None,
                    self._postprocessor is not None,
                    bool(self._update_operations),
                )
            ):
                raise ValueError(
                    "workers > 1 cannot serialize in-process forge, preprocess, postprocess, or update configuration"
                )

            if self.retain not in ((), "*"):
                raise ValueError("workers > 1 does not support a tuple retain projection; use '*' or ()")

            env_updates = {
                "RELFLOW_CHECKPOINT": str(self.checkpoint),
                "RELFLOW_MAX_BATCH_SIZE": str(self.max_batch_size),
                "RELFLOW_BATCH_TIMEOUT": str(self.batch_timeout),
                "RELFLOW_ACCELERATOR": self.accelerator.value,
                "RELFLOW_JSON_BACKEND": self.json_backend.value,
            }
            if self.retain == "*":
                env_updates["RELFLOW_RETAIN"] = orjson.dumps(self.retain).decode()
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

        uvicorn.run(self.app(), host=self.host, port=self.port, log_level=self.log_level)


def create_app() -> fastapi.FastAPI:
    """Build a FastAPI app from deployment environment variables."""

    return Deployment().app()


__all__ = [
    "Accelerator",
    "Deployment",
    "FastAPIBatcher",
    "FastAPIRuntime",
    "Input",
    "JSONBackend",
    "ModelSource",
    "Retain",
    "UpdateOperation",
    "create_app",
]

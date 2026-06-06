"""Realtime deployment wrappers for `json2vec` checkpoints."""

import asyncio
import functools
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast

import fastapi
import pydantic
import torch
import uvicorn
from beartype import beartype
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tensordict import TensorDict

from json2vec.architecture.root import Model
from json2vec.data.iterables import JMESPathResolutionMonitor, encode
from json2vec.structs.enums import Strata, TensorKey
from json2vec.structs.experiment import NodeAttribute, NodePredicate
from json2vec.structs.packages import Prediction
from json2vec.structs.tree import Address, Node
from json2vec.tensorfields.base import TensorFieldBase

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
    """Load a json2vec model and execute batched request prediction."""

    def __init__(
        self,
        *,
        checkpoint: ModelSource,
        accelerator: Accelerator,
        preprocessor=None,
        postprocessor=None,
        update_operations: list[UpdateOperation] | None = None,
        request_signature: type[pydantic.BaseModel] | None = None,
        response_signature: type[pydantic.BaseModel] | None = None,
    ) -> None:
        self.model_source: ModelSource = checkpoint
        self.accelerator = accelerator
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
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
        self.model: Model = model.to(self.device)
        for predicates, values in self.update_operations:
            self.model.update(*predicates, **values)

        self.model.eval()
        self.interprocess_encoding_context = self.model.interprocess_encoding_context
        self.jmespath_resolution_monitor = JMESPathResolutionMonitor()

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
                observation = self.preprocessor(request)
                if not isinstance(observation, dict):
                    raise TypeError(f"preprocessor must return a dict object, got {type(observation).__name__}")

                observations = [[observation]]

        except Exception as exception:
            return ErrorItem(status_code=422, message=str(exception))

        if len(observations) == 0 or any(x is None for x in observations):
            return ErrorItem(status_code=422, message="preprocessor returned no observations for request")

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
            processed = self.postprocessor(context, predictions)
            if processed is not None:
                predictions = processed

        encoded = Prediction.denest(dict(predictions=predictions))
        if self.response_signature is not None:
            return self.response_signature.model_validate(encoded).model_dump(mode="json")

        return cast(dict[str, Any], encoded)

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
                hyperparameters=self.model.hyperparameters,
                strata=Strata.predict,
                interprocess_encoding_context=self.interprocess_encoding_context,
                jmespath_resolution_monitor=getattr(self, "jmespath_resolution_monitor", None),
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

            with torch.inference_mode():
                predictions = self.model(model_input.to(self.device), strata=Strata.predict)

            unbatched = Prediction.unbatch(predictions=predictions)
            for index, (start, stop) in zip(valid_indices, spans):
                if stop - start != 1:
                    raise ValueError("deployment requests must encode exactly one observation")

                input_slice: dict[Address, TensorFieldBase] = {}
                for key, value in encoded.items():
                    if key == TensorKey.metadata:
                        continue

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
    """Serving configuration for a `json2vec` checkpoint or model instance.

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
        validation_alias=AliasChoices("JSON2VEC_CHECKPOINT", "CHECKPOINT"),
    )
    model: Model | None = Field(default=None, exclude=True)
    max_batch_size: int = Field(
        default=128,
        ge=1,
        validation_alias=AliasChoices("JSON2VEC_MAX_BATCH_SIZE", "MAX_BATCH_SIZE"),
    )
    batch_timeout: float = Field(
        default=0.0,
        ge=0.0,
        validation_alias=AliasChoices("JSON2VEC_BATCH_TIMEOUT", "BATCH_TIMEOUT"),
    )
    accelerator: Accelerator = Field(
        default=Accelerator.auto,
        validation_alias=AliasChoices("JSON2VEC_ACCELERATOR", "ACCELERATOR"),
    )
    host: str = Field(
        default="0.0.0.0",
        validation_alias=AliasChoices("JSON2VEC_HOST", "HOST"),
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias=AliasChoices("JSON2VEC_PORT", "PORT"),
    )
    log_level: str = Field(
        default="info",
        validation_alias=AliasChoices("JSON2VEC_LOG_LEVEL", "LOG_LEVEL"),
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
    def preprocess(self, preprocessor, **kwargs: Any) -> "Deployment":
        """Attach an optional request preprocessor.

        If this method is not called, request objects are encoded unchanged.
        """
        self._preprocessor = functools.partial(preprocessor, **kwargs) if kwargs else preprocessor

        return self

    @beartype
    def postprocess(self, postprocessor, **kwargs: Any) -> "Deployment":
        """Attach an optional response postprocessor."""
        self._postprocessor = functools.partial(postprocessor, **kwargs) if kwargs else postprocessor

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
            app.state.json2vec_runtime = runtime
            app.state.json2vec_batcher = batcher
            try:
                yield
            finally:
                await batcher.stop()

        app = fastapi.FastAPI(lifespan=lifespan)

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/predict")
        async def predict(request: fastapi.Request) -> JSONResponse:
            try:
                payload = await request.json()
            except Exception as exception:
                return JSONResponse(
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
                return JSONResponse(content=response)

            if isinstance(payload, list):
                for index, item in enumerate(payload):
                    if not isinstance(item, dict):
                        return JSONResponse(
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
                return JSONResponse(content=responses)

            return JSONResponse(
                status_code=422,
                content={
                    "predictions": {},
                    "error": {
                        "status_code": 422,
                        "message": f"request body must be a JSON object or an array of JSON objects, got {type(payload).__name__}",
                    },
                },
            )

        return app

    def serve(self) -> None:
        """Start the FastAPI server for the configured checkpoint or model."""
        uvicorn.run(
            self.app(),
            host=self.host,
            port=self.port,
            log_level=self.log_level,
        )

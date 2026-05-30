"""LitServe deployment wrappers for JSON2Vec checkpoints."""

import functools
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, cast

import litserve as ls
import pydantic
import torch
from beartype import beartype
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tensordict import TensorDict

from json2vec.architecture.root import Model
from json2vec.data.iterables import JMESPathResolutionMonitor, encode
from json2vec.structs.enums import Strata
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


class BatchItem(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    data: Input | None
    valid_indices: list[int]
    items: list[Input | ErrorItem]


class API(ls.LitAPI):
    def __init__(
        self,
        checkpoint: ModelSource | None = None,
        model: Model | None = None,
        preprocessor=None,
        postprocessor=None,
        update_operations: list[UpdateOperation] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(checkpoint, Model):
            if model is not None:
                raise ValueError("pass either checkpoint or model, not both")
            self._model_source: ModelSource = checkpoint
            self.checkpoint: str | None = None
        elif model is not None:
            if checkpoint is not None:
                raise ValueError("pass either checkpoint or model, not both")
            self._model_source = model
            self.checkpoint = None
        else:
            if checkpoint is None:
                raise ValueError("checkpoint or model is required")
            self._model_source = checkpoint
            self.checkpoint = str(checkpoint)

        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.update_operations = list(update_operations or [])

    @property
    def model_source(self) -> ModelSource:
        return self._model_source

    def _load_model(self) -> Model:
        if isinstance(self._model_source, Model):
            return self._model_source

        return Model.load(self._model_source)

    def setup(self, device: str) -> None:
        self.model: Model = self._load_model().to(device)
        for predicates, values in self.update_operations:
            self.model.update(*predicates, **values)

        self.model.eval()
        self.interprocess_encoding_context = self.model.interprocess_encoding_context
        self.jmespath_resolution_monitor = JMESPathResolutionMonitor()

    @beartype
    def decode_request(
        self,
        request: dict[str, Any] | pydantic.BaseModel,
        context: dict[str, Any] | None = None,
    ) -> Input | ErrorItem:  # ty:ignore[invalid-method-override]
        if isinstance(request, pydantic.BaseModel):
            request = request.model_dump()

        if context is not None:
            context["request"] = request

        try:
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

        if context is not None:
            context["observations"] = observations

        encoded = encode(
            batch=observations,
            hyperparameters=self.model.hyperparameters,
            strata=Strata.predict,
            interprocess_encoding_context=self.interprocess_encoding_context,
            jmespath_resolution_monitor=getattr(self, "jmespath_resolution_monitor", None),
        )

        if encoded is None:
            return ErrorItem(status_code=422, message="preprocessor eliminated observation (filter)")

        if context is not None:
            context["input"] = encoded

        return encoded

    @beartype
    def batch(self, inputs: list[Input | ErrorItem]) -> BatchItem:
        valid_indices: list[int] = []
        valid_inputs: list[Input] = []

        for index, item in enumerate(inputs):
            if isinstance(item, ErrorItem):
                continue

            valid_indices.append(index)
            valid_inputs.append(item)

        data = torch.stack(valid_inputs, dim=0) if valid_inputs else None
        return BatchItem(data=data, valid_indices=valid_indices, items=inputs)

    @beartype
    def unbatch(self, outputs: list[Any]) -> list[Any]:  # ty:ignore[invalid-method-override]
        return list(outputs)

    @beartype
    def predict(
        self, data: BatchItem | Input | ErrorItem
    ) -> list[list[Prediction] | ErrorItem] | list[Prediction] | ErrorItem:  # ty:ignore[invalid-method-override]
        if isinstance(data, ErrorItem):
            return data

        if isinstance(data, TensorDict):
            with torch.inference_mode():
                return self.model(data.to(self.device), strata=Strata.predict)

        outputs: list[Any] = list(data.items)

        if data.data is None:
            return outputs

        with torch.inference_mode():
            predictions = self.model(data.data.to(self.device), strata=Strata.predict)

        unbatched = Prediction.unbatch(predictions=predictions)

        for index, item_predictions in zip(data.valid_indices, unbatched):
            outputs[index] = item_predictions

        return outputs

    @beartype
    def encode_response(
        self,
        response: list[Prediction] | ErrorItem,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | pydantic.BaseModel:  # ty:ignore[invalid-method-override]
        if isinstance(response, ErrorItem):
            return {
                "predictions": {},
                "error": {
                    "status_code": response.status_code,
                    "message": response.message,
                },
            }

        predictions = self.model.write(predictions=response)
        postprocessor = self.postprocessor

        if postprocessor is not None:
            processed = postprocessor({} if context is None else context, predictions)

            if processed is not None:
                predictions = processed

        return Prediction.denest(dict(predictions=predictions))


_DEFAULT_DECODE_REQUEST_ANNOTATIONS = dict(API.decode_request.__annotations__)
_DEFAULT_ENCODE_RESPONSE_ANNOTATIONS = dict(API.encode_response.__annotations__)


class Deployment(BaseSettings):
    """Serving configuration for a JSON2Vec checkpoint or model instance.

    `Deployment` queues request/response schemas, optional preprocessors,
    optional postprocessors, and `update(...)` mutations before the model is
    loaded by LitServe workers.
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
    workers_per_device: int = Field(
        default=1,
        ge=1,
        validation_alias=AliasChoices("JSON2VEC_WORKERS_PER_DEVICE", "JSON2VEC_N_WORKERS", "N_WORKERS"),
    )
    accelerator: Accelerator = Field(
        default=Accelerator.auto,
        validation_alias=AliasChoices("JSON2VEC_ACCELERATOR", "ACCELERATOR"),
    )
    track_requests: bool = Field(
        default=False,
        validation_alias=AliasChoices("JSON2VEC_TRACK_REQUESTS", "TRACK_REQUESTS"),
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

    def serve(self) -> None:
        """Start the LitServe server for the configured checkpoint or model."""
        API.decode_request.__annotations__ = dict(_DEFAULT_DECODE_REQUEST_ANNOTATIONS)
        API.encode_response.__annotations__ = dict(_DEFAULT_ENCODE_RESPONSE_ANNOTATIONS)

        if self._request_signature is not None:
            API.decode_request.__annotations__["request"] = self._request_signature

        if self._response_signature is not None:
            API.encode_response.__annotations__["return"] = self._response_signature

        server: ls.LitServer = ls.LitServer(
            lit_api=API(
                checkpoint=self.model if self.model is not None else self.checkpoint,
                max_batch_size=self.max_batch_size,
                batch_timeout=self.batch_timeout,
                preprocessor=self._preprocessor,
                postprocessor=self._postprocessor,
                update_operations=self._update_operations,
            ),
            accelerator=self.accelerator.value,
            workers_per_device=self.workers_per_device,
            track_requests=self.track_requests,
        )

        server.run(generate_client_file=False)

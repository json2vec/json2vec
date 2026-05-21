import functools
from typing import Any, Literal, TypeAlias

import litserve as ls
import pydantic
import torch
from beartype import beartype
from pydantic import AliasChoices, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tensordict import TensorDict

from json2vec.architecture.root import JSON2Vec
from json2vec.data.datasets import encode
from json2vec.structs.enums import Strata
from json2vec.structs.packages import Prediction
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TensorFieldBase

Input: TypeAlias = TensorDict[Address, TensorFieldBase]


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
        checkpoint: str,
        preprocessor=None,
        postprocessor=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoint = checkpoint
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def setup(self, device: str) -> None:
        self.model: JSON2Vec = JSON2Vec.get_or_create(checkpoint=self.checkpoint).to(device)
        self.model.eval()
        self.state = self.model.state

    @beartype
    def decode_request(
        self,
        request: dict[str, Any] | pydantic.BaseModel,
        context: dict[str, Any] | None = None,
    ) -> Input | ErrorItem:

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
            return ErrorItem(status_code=422, message="processor returned no observations for request")

        if context is not None:
            context["observations"] = observations

        encoded = encode(
            batch=observations,
            hyperparameters=self.model.hyperparameters,
            strata=Strata.predict,
            state=self.state,
        )

        if encoded is None:
            return ErrorItem(status_code=422, message="processor eliminated observation (filter)")

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

        data = torch.stack(valid_inputs, dim=0) if len(valid_inputs) > 0 else None
        return BatchItem(data=data, valid_indices=valid_indices, items=inputs)

    @beartype
    def unbatch(self, outputs: list[Any]) -> list[Any]:
        return list(outputs)

    @beartype
    def predict(
        self, data: BatchItem | Input | ErrorItem
    ) -> list[list[Prediction] | ErrorItem] | list[Prediction] | ErrorItem:
        if isinstance(data, ErrorItem):
            return data

        if isinstance(data, TensorDict):
            with torch.inference_mode():
                return self.model(data.to(self.device))

        outputs: list[Any] = list(data.items)

        if data.data is None:
            return outputs

        with torch.inference_mode():
            predictions = self.model(data.data.to(self.device))

        unbatched = Prediction.unbatch(predictions=predictions)

        for index, item_predictions in zip(data.valid_indices, unbatched):
            outputs[index] = item_predictions

        return outputs

    @beartype
    def encode_response(
        self,
        response: list[Prediction] | ErrorItem,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | pydantic.BaseModel:
        if isinstance(response, ErrorItem):
            return {
                "predictions": {},
                "error": {
                    "status_code": response.status_code,
                    "message": response.message,
                },
            }

        predictions, embeddings = self.model.write(predictions=response)
        postprocessor = self.postprocessor

        if postprocessor is not None:
            processed = postprocessor({} if context is None else context, predictions, embeddings)

            if processed is not None:
                predictions, embeddings = processed

        payload = dict(predictions=predictions)

        if len(embeddings) > 0:
            payload["embeddings"] = embeddings

        return Prediction.denest(payload)


_DEFAULT_DECODE_REQUEST_ANNOTATIONS = dict(API.decode_request.__annotations__)
_DEFAULT_ENCODE_RESPONSE_ANNOTATIONS = dict(API.encode_response.__annotations__)


class Deployment(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
        validate_by_name=True,
        validate_by_alias=True,
    )

    checkpoint: str = Field(
        default="model.ckpt",
        validation_alias=AliasChoices("JSON2VEC_CHECKPOINT", "CHECKPOINT"),
    )
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
    accelerator: Literal["auto", "cpu", "cuda", "mps"] = Field(
        default="auto",
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

    @field_validator("checkpoint", "accelerator", mode="before")
    @classmethod
    def strip_required_strings(cls, value: str | None, info: ValidationInfo) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                raise ValueError(f"{info.field_name} must not be blank")
            return stripped

        return value

    @beartype
    def forge(
        self,
        request: type[pydantic.BaseModel] | None = None,
        response: type[pydantic.BaseModel] | None = None,
    ) -> "Deployment":
        self._request_signature = request
        self._response_signature = response

        return self

    @beartype
    def preprocess(self, processor, **kwargs: Any) -> "Deployment":
        self._preprocessor = functools.partial(processor, **kwargs) if len(kwargs) > 0 else processor

        return self

    @beartype
    def postprocess(self, processor, **kwargs: Any) -> "Deployment":
        self._postprocessor = functools.partial(processor, **kwargs) if len(kwargs) > 0 else processor

        return self

    def serve(self) -> None:
        API.decode_request.__annotations__ = dict(_DEFAULT_DECODE_REQUEST_ANNOTATIONS)
        API.encode_response.__annotations__ = dict(_DEFAULT_ENCODE_RESPONSE_ANNOTATIONS)

        if self._request_signature is not None:
            API.decode_request.__annotations__["request"] = self._request_signature

        if self._response_signature is not None:
            API.encode_response.__annotations__["return"] = self._response_signature

        server: ls.LitServer = ls.LitServer(
            lit_api=API(
                checkpoint=self.checkpoint,
                max_batch_size=self.max_batch_size,
                batch_timeout=self.batch_timeout,
                preprocessor=self._preprocessor,
                postprocessor=self._postprocessor,
            ),
            accelerator=self.accelerator,
            workers_per_device=self.workers_per_device,
            track_requests=self.track_requests,
        )

        server.run(generate_client_file=False)

from collections.abc import Callable
from typing import Any, ClassVar, Literal, Type, TypeAlias

import litserve as ls
import pydantic
import torch
from beartype import beartype
from pydantic import AliasChoices, Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tensordict import TensorDict

from json2vec.architecture.root import JSON2Vec
from json2vec.data.datasets import Dataset, encode, process
from json2vec.structs.enums import Strata
from json2vec.structs.packages import Prediction
from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TensorFieldBase

Input: TypeAlias = TensorDict[Address, TensorFieldBase]
Preprocessor: TypeAlias = str | Callable[..., Any]
Postprocessor: TypeAlias = Callable[
    [dict[str, Any], dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]],
    tuple[dict[Address, dict[str, Any]], dict[Address, dict[str, Any]]] | None,
]


def default_dataset() -> Dataset:
    return Dataset(
        root=None,
        processor="default",
    )


class DeploymentEnvironment(BaseSettings):
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

    @field_validator("checkpoint", "accelerator", mode="before")
    @classmethod
    def strip_required_strings(cls, value: str | None, info: ValidationInfo) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                raise ValueError(f"{info.field_name} must not be blank")
            return stripped

        return value


class ErrorItem(pydantic.BaseModel):
    status_code: int
    message: str


class BatchItem(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    data: Input | None
    valid_indices: list[int]
    items: list[Input | ErrorItem]


class Deployment(ls.LitAPI):
    _preprocessor: ClassVar[str | None] = None
    _preprocessor_kwargs: ClassVar[dict[str, Any]] = {}
    _postprocessor: ClassVar[Postprocessor | None] = None

    def __init__(self, checkpoint: str, dataset: Dataset | None = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoint = checkpoint
        self.dataset = default_dataset() if dataset is None else dataset
        preprocessor = type(self)._preprocessor

        if preprocessor is not None:
            payload = self.dataset.model_dump(mode="python")
            payload["processor"] = preprocessor
            payload["kwargs"] = dict(type(self)._preprocessor_kwargs)
            self.dataset = Dataset.model_validate(payload)

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
            observations: list[Any] = list(
                process(
                    pipe=[request],
                    dataset=self.dataset,
                    strata=Strata.predict,
                    state=self.state,
                )
            )

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
    def predict(self, data: BatchItem | Input | ErrorItem) -> list[list[Prediction] | ErrorItem] | list[Prediction] | ErrorItem:
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
        postprocessor = type(self)._postprocessor

        if postprocessor is not None:
            processed = postprocessor({} if context is None else context, predictions, embeddings)

            if processed is not None:
                predictions, embeddings = processed

        payload = dict(predictions = predictions)

        if len(embeddings) > 0:
            payload["embeddings"] = embeddings

        return Prediction.denest(payload)

    @classmethod
    @beartype
    def forge(
        cls,
        request: Type[pydantic.BaseModel]|None=None,
        response: Type[pydantic.BaseModel]|None=None,
    ) -> Type["Deployment"]:

        if request is not None:
            cls.decode_request.__annotations__["request"] = request

        if response is not None:
            cls.encode_response.__annotations__["return"] = response

        return cls

    @classmethod
    @beartype
    def preprocess(cls, processor: Preprocessor, **kwargs: Any) -> Type["Deployment"]:
        dataset = Dataset(root=None, processor=processor, kwargs=kwargs)
        cls._preprocessor = dataset.processor
        cls._preprocessor_kwargs = dict(dataset.kwargs)

        return cls

    @classmethod
    @beartype
    def postprocess(cls, processor: Postprocessor) -> Type["Deployment"]:
        cls._postprocessor = processor

        return cls

    @classmethod
    def serve(cls, environment: DeploymentEnvironment | None = None):

        if environment is None:
            environment = DeploymentEnvironment()

        server: ls.LitServer = ls.LitServer(
            lit_api=cls(
                checkpoint=environment.checkpoint,
                max_batch_size=environment.max_batch_size,
                batch_timeout=environment.batch_timeout,
            ),
            accelerator=environment.accelerator,
            track_requests=environment.track_requests,
            workers_per_device=environment.workers_per_device,
        )

        server.run(generate_client_file=False)

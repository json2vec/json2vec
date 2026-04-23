from typing import Any, Type, TypeAlias

import litserve as ls
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict

from json2vec.architecture.root import JSON2Vec
from json2vec.data.datasets import encode, process
from json2vec.structs.enums import Strata
from json2vec.structs.environment import DeploymentEnvironment
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


class Deployment(ls.LitAPI):
    def __init__(self, checkpoint: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checkpoint = checkpoint

    def setup(self, device: str) -> None:
        self.model: JSON2Vec = JSON2Vec.get_or_create(checkpoint=self.checkpoint).to(device)
        self.model.eval()
        self.state = self.model.state

    @beartype
    def decode_request(self, request: dict[str, Any] | pydantic.BaseModel) -> Input | ErrorItem:

        if isinstance(request, pydantic.BaseModel):
            request = request.model_dump()

        try:
            observations: list[Any] = list(
                process(
                    pipe=[request],
                    session=self.model.session,
                    strata=Strata.predict,
                    state=self.state,
                )
            )

        except Exception as exception:
            return ErrorItem(status_code=422, message=str(exception))


        if len(observations) == 0 or any(x is None for x in observations):
            return ErrorItem(status_code=422, message="processor returned no observations for request")
    

        encoded = encode(
            batch=observations,
            session=self.model.session,
            strata=Strata.predict,
            state=self.state,
        )

        if encoded is None:
            return ErrorItem(status_code=422, message="processor eliminated observation (filter)")

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
    def encode_response(self, response: list[Prediction] | ErrorItem) -> dict[str, Any] | pydantic.BaseModel:
        if isinstance(response, ErrorItem):
            return {
                "predictions": {},
                "error": {
                    "status_code": response.status_code,
                    "message": response.message,
                },
            }

        predictions, embeddings = self.model.write(predictions=response)

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
    def serve(cls):

        environment = DeploymentEnvironment()

        server: ls.LitServer = ls.LitServer(
            lit_api=Deployment(
                checkpoint=environment.checkpoint,
                max_batch_size=environment.max_batch_size,
                batch_timeout=environment.batch_timeout,
            ),
            accelerator=environment.accelerator,
            track_requests=environment.track_requests,
            workers_per_device=environment.workers_per_device,
        )

        server.run(generate_client_file=False)

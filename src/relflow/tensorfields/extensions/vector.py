# ty: ignore[unknown-argument]
from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    Context,
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
    TensorInput,
)
from relflow.tensorfields.output import array, fixed, struct

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema


vector: Plugin = Plugin(name="vector", types=(int | float,))


class Objective(enum.StrEnum):
    l1 = "l1"
    l2 = "l2"

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self == Objective.l1:
            return torch.nn.functional.l1_loss(input=inputs, target=targets, reduction="none").mean(dim=1)

        return torch.nn.functional.mse_loss(input=inputs, target=targets, reduction="none").mean(dim=1)


@vector.register
class Request(RequestBase):
    """Fixed-width numeric vector tensorfield request."""

    type: Literal["vector"] = "vector"
    n_dim: Annotated[int, pydantic.Field(gt=0)]
    objective: Objective = Objective.l2


@vector.register
@tensorclass
class TensorField(TensorFieldBase):
    content: torch.Tensor
    state: torch.Tensor
    present: torch.Tensor
    trainable: torch.Tensor
    inferred: torch.Tensor
    targets: TensorDict[TensorKey, torch.Tensor]

    @classmethod
    def new(
        cls,
        input: RaggedField,
        target: RaggedField,
        present: torch.Tensor,
        trainable: torch.Tensor,
        inferred: torch.Tensor,
        address: Address,
        schema: Schema,
        strata: Strata,
        context: Context,
    ) -> TensorFieldBase:
        request: Request = schema.requests[address]

        def encode(field: RaggedField) -> torch.Tensor:
            values = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
            if not len(values):
                encoded = np.empty((0, request.n_dim), dtype=np.float32)
            else:
                if not (
                    pa.types.is_list(values.type)
                    or pa.types.is_large_list(values.type)
                    or pa.types.is_fixed_size_list(values.type)
                ):
                    raise ValueError(f"vector field at '{address}' expects an Arrow list, got {values.type}")
                lengths = pc.list_value_length(values)
                if pc.any(pc.not_equal(lengths, request.n_dim)).as_py():
                    raise ValueError(f"vector field at '{address}' expects every value to have length {request.n_dim}")
                flattened = pc.list_flatten(values)
                if flattened.null_count:
                    raise ValueError(f"vector field at '{address}' expects non-null numeric elements")
                try:
                    flattened = pc.cast(flattened, pa.float32(), safe=True)
                except pa.ArrowException as error:
                    raise ValueError(f"vector field at '{address}' could not be converted to float32") from error
                encoded = flattened.to_numpy(zero_copy_only=False).reshape(-1, request.n_dim)
            return torch.from_numpy(field.place(encoded, fill=0.0, value_shape=(request.n_dim,)))

        state = torch.from_numpy(input.dense)

        return cls(
            content=encode(input),
            state=state,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: torch.from_numpy(target.dense),
                    TensorKey.content: encode(target),
                },
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
        )


@vector.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address

        self.embeddings = torch.nn.Embedding(
            num_embeddings=len(Tokens),
            embedding_dim=schema.d_model,
        )
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=request.n_dim, out_features=schema.d_model),
            torch.nn.GELU(),
        )

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N, *dims = inputs.state.shape
        D = math.prod((N, *dims))

        state = inputs.state.reshape(D)
        content = inputs.content.reshape(D, -1)

        projected = self.linear(content)
        embeddings = self.embeddings(state)

        return Parcel(
            payload=(projected + embeddings).reshape(N, *dims, -1),
            present=torch.ones(N, dtype=torch.bool, device=projected.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@vector.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]

        self.classification = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=len(Tokens),
        )
        self.regression = torch.nn.Linear(
            in_features=schema.d_model,
            out_features=request.n_dim,
        )

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            source={
                TensorKey.state: self.classification(pooled),
                TensorKey.content: self.regression(pooled),
            }
        )


@vector.register
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    address: Address = prediction.address
    request: Request = module.schema.requests[address]

    trainable = batch.trainable.reshape(-1)
    state_targets = batch.targets[TensorKey.state].reshape(-1)
    state_inputs = prediction.payload[TensorKey.state].reshape(-1, len(Tokens))

    output: torch.Tensor = module.track(
        (address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=state_inputs,
                target=state_targets,
                reduction="none",
            )
            .masked_select(trainable)
            .mean()
        ),
    )

    module.track(
        (address, strata, Metric.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )

    valued = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        return output

    inputs = prediction.payload[TensorKey.content].reshape(-1, request.n_dim)
    targets = batch.targets[TensorKey.content].reshape(-1, request.n_dim)
    diff = inputs.subtract(targets)

    output += module.track(
        (address, strata, Metric.loss, TensorKey.content),
        value=request.objective.loss(inputs=inputs, targets=targets).masked_select(valued).mean(),
    )

    module.track(
        (address, strata, Metric.mae, TensorKey.content),
        value=diff.absolute().mean(dim=1).masked_select(valued).mean(),
    )

    module.track(
        (address, strata, Metric.rmse, TensorKey.content),
        value=diff.square().mean(dim=1).sqrt().masked_select(valued).mean(),
    )

    return output


@vector.register
def output(module: Model, address: Address) -> pa.StructType:
    request: Request = module.schema.requests[address]
    content = pa.list_(pa.float32(), request.n_dim)
    return pa.struct([pa.field(TensorKey.content.name, content, nullable=False)])


@vector.register
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    request: Request = module.schema.requests[prediction.address]
    content = prediction.payload[TensorKey.content].detach().float().clone()
    non_valued = prediction.payload[TensorKey.state].argmax(dim=-1).ne(Tokens.valued.value)
    content[non_valued] = 0.0
    values = fixed(array(content, pa.float32()), request.n_dim)
    return struct({TensorKey.content.name: values}, datatype)

"""Arrow batch encoding into policy-resolved tensorfields."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import torch
from tensordict import TensorDict

from relflow.data.arrow import Batch, Encoded
from relflow.data.datasets.base import EncodedInput, InterprocessEncodingContext
from relflow.data.ragged import RaggedField, boolean, coalesce
from relflow.structs.enums import Component, Strata, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS, Context, TensorFieldBase


def encode(
    batch: Batch,
    schema: Schema,
    strata: Strata,
    interprocess_encoding_context: InterprocessEncodingContext,
    seed: int = 0,
    epoch: int = 0,
    retain: tuple[str, ...] | Literal["*"] = (),
) -> Encoded:
    out: dict[Address, TensorFieldBase] = {}
    observations: dict[Address, TensorDict] = {}

    salt = random.getrandbits(64) if strata in {Strata.train, Strata.validate} else 0

    projections = coalesce(batch, schema=schema, strata=strata, seed=seed, epoch=epoch)

    for address, request in schema.active_requests.items():
        extension = TENSORFIELDS[request.type]
        TensorField = extension.TensorField
        projection = projections.pop(address)
        pristine = projection.pristine
        nulls = pristine.state.to_numpy(zero_copy_only=False) == Tokens.null.value
        if not request.nullable and nulls.any():
            raise ValueError(
                f"request '{address}' has nullable=False but input contains {int(nulls.sum())} null value(s)"
            )

        values = pa.nulls(0) if projection.vacant else extension.prepare(pristine.values, address=address)
        canonical = replace(pristine, values=values)
        observation = None
        if not projection.vacant:
            observation = extension.observe(
                field=canonical,
                address=address,
                schema=schema,
                state=interprocess_encoding_context.get(address),
                learn=strata == Strata.train,
            )

        learner = extension.components.get(Component.learn)
        if observation is not None and not isinstance(observation, TensorDict):
            raise TypeError(
                f"extension '{extension.name}' observer at '{address}' must return a TensorDict or None, "
                f"got {type(observation).__name__}"
            )
        if observation is not None and learner is None:
            raise RuntimeError(
                f"extension '{extension.name}' observer at '{address}' returned an observation without a learner"
            )
        if strata == Strata.train and learner is not None and observation is None:
            raise RuntimeError(f"extension '{extension.name}' learner at '{address}' requires a training observation")
        if strata != Strata.train and observation is not None:
            raise RuntimeError(
                f"extension '{extension.name}' observer at '{address}' returned an observation with learn=False"
            )
        if observation is not None:
            observations[address] = observation

        input_field, target_field = projection.split(values)
        out[address] = TensorField.new(
            input=input_field,
            target=target_field,
            present=torch.from_numpy(boolean(projection.present).copy()).reshape(input_field.shape),
            trainable=torch.from_numpy(boolean(projection.trainable).copy()).reshape(input_field.shape),
            inferred=torch.from_numpy(boolean(projection.inferred).copy()).reshape(input_field.shape),
            address=address,
            schema=schema,
            strata=strata,
            context=Context(
                state=interprocess_encoding_context.get(address),
                salt=salt,
            ),
        )

    inputs = cast(EncodedInput, TensorDict(source=cast(Any, out)))
    return Encoded(tensors=inputs, source=batch, retain=retain, observations=observations)


def mock(schema: Schema, batch_size: int) -> EncodedInput:
    out: dict[Address, TensorFieldBase] = {}

    for address, request in schema.active_requests.items():
        TensorField = TENSORFIELDS[request.type].TensorField
        shape = (batch_size, *schema.shapes[address])
        state = pa.array(np.full(np.prod(shape), Tokens.padded.value, dtype=np.int8))
        field = RaggedField(
            values=pa.nulls(0),
            state=state,
            placement=pa.array([], type=pa.int64()),
            shape=shape,
        )
        routing = torch.zeros(shape, dtype=torch.bool)
        out[address] = TensorField.new(
            input=field,
            target=field,
            present=routing,
            trainable=routing,
            inferred=routing,
            address=address,
            schema=schema,
            strata=Strata.predict,
            context=Context(),
        )

    return cast(EncodedInput, TensorDict(source=cast(Any, out), batch_size=batch_size))

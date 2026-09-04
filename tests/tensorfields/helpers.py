"""Shared tensorfield construction for focused plugin tests."""

from dataclasses import replace

import torch

from relflow.data.ragged import Projection, boolean
from relflow.structs.enums import Strata
from relflow.structs.experiment import Schema
from relflow.structs.tree import Address
from relflow.tensorfields.base import Context, Plugin, TensorFieldBase


def tensorize(
    tensorfield: type[TensorFieldBase],
    projection: Projection,
    plugin: Plugin,
    *,
    address: Address | str,
    schema: Schema,
    strata: Strata,
    context: Context = Context(),
) -> TensorFieldBase:
    """Prepare once, split by policy, and construct one tensorfield."""

    address = Address(address)
    values = plugin.prepare(projection.pristine.values, address=address)
    plugin.observe(
        field=replace(projection.pristine, values=values),
        address=address,
        schema=schema,
        state=context.state,
        learn=strata == Strata.train,
    )
    input, target = projection.split(values)
    shape = input.shape
    return tensorfield.new(
        input=input,
        target=target,
        present=torch.from_numpy(boolean(projection.present).copy()).reshape(shape),
        trainable=torch.from_numpy(boolean(projection.trainable).copy()).reshape(shape),
        inferred=torch.from_numpy(boolean(projection.inferred).copy()).reshape(shape),
        address=address,
        schema=schema,
        strata=strata,
        context=context,
    )


__all__ = ["tensorize"]

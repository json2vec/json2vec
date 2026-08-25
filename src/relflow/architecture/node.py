from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch

from relflow.architecture.encoder import BranchEncoder
from relflow.structs.tree import Address, Node
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Plugin,
)

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema


class NodeModule(torch.nn.Module):
    embedder: EmbedderBase
    decoder: DecoderBase
    encoder: BranchEncoder

    def __init__(
        self,
        schema: Schema,
        address: Address,
        **kwargs: Any,
    ):
        super().__init__()

        if address in schema.requests:
            request: Node = schema.requests[address]
            plugin: Plugin = TENSORFIELDS[request.type]
            component_context = dict(kwargs, schema=schema, address=address)

            for component in [plugin.Embedder, plugin.Decoder]:
                parameters = inspect.signature(component.__init__).parameters
                component_kwargs = {name: value for name, value in component_context.items() if name in parameters}
                setattr(self, component.__name__.lower(), component(**component_kwargs))

        elif address in schema.branches:
            self.encoder = BranchEncoder(schema=schema, address=address)

        else:
            raise ValueError("how did we get here?")

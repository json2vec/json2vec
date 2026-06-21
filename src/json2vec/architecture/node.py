from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch

from json2vec.architecture.encoder import BranchEncoder
from json2vec.structs.tree import Address, Node
from json2vec.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Plugin,
)

if TYPE_CHECKING:
    from json2vec.structs.experiment import Schema


class NodeModule(torch.nn.Module):
    def __init__(self, schema: Schema, address: Address, batch_size: int):
        super().__init__()

        if address in schema.requests:
            request: Node = schema.requests[address]
            plugin: Plugin = TENSORFIELDS[request.type]
            embedder_kwargs: dict[str, Any] = dict(schema=schema, address=address)
            if "batch_size" in inspect.signature(plugin.Embedder.__init__).parameters:
                embedder_kwargs["batch_size"] = batch_size

            self.embedder: EmbedderBase = plugin.Embedder(**embedder_kwargs)
            self.decoder: DecoderBase = plugin.Decoder(schema=schema, address=address)

        elif address in schema.branches:
            self.encoder: BranchEncoder = BranchEncoder(schema=schema, address=address)

        else:
            raise ValueError("how did we get here?")

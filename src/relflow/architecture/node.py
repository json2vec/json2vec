from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from relflow.architecture.encoder import BranchEncoder
from relflow.structs.tree import Address, Node
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Extension,
)

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema


class NodeModule(torch.nn.Module):
    def __init__(self, schema: Schema, address: Address):
        super().__init__()

        if address in schema.requests:
            request: Node = schema.requests[address]
            extension: Extension = TENSORFIELDS[request.type]
            self.embedder: EmbedderBase = extension.Embedder(schema=schema, address=address)
            if address in schema.objectives or address in schema.embed:
                self.decoder: DecoderBase = extension.Decoder(schema=schema, address=address)
            if address in schema.objectives:
                loss = extension.loss
                if not callable(loss):
                    raise TypeError(f"extension '{extension.name}' loss must be callable for objective '{address}'")

        elif address in schema.branches:
            self.encoder: BranchEncoder = BranchEncoder(schema=schema, address=address)

        else:
            raise ValueError("how did we get here?")

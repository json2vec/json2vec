from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from json2vec.architecture.encoder import ContextEncoder
from json2vec.structs.tree import Address, Node
from json2vec.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Plugin,
)

if TYPE_CHECKING:
    from json2vec.structs.config import Structure


class NodeModule(torch.nn.Module):
    def __init__(self, structure: Structure, address: Address):
        super().__init__()

        if address in structure.requests:
            request: Node = structure.requests[address]
            plugin: Plugin = TENSORFIELDS[request.type]
            self.embedder: EmbedderBase = plugin.Embedder(structure=structure, address=address)
            self.decoder: DecoderBase = plugin.Decoder(structure=structure, address=address)

        elif address in structure.contexts:
            self.encoder: ContextEncoder = ContextEncoder(structure=structure, address=address)

        else:
            raise ValueError("how did we get here?")

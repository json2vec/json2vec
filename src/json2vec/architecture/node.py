from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import torch

from json2vec.architecture.encoder import ArrayEncoder
from json2vec.structs.tree import Address, Node
from json2vec.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Plugin,
)

if TYPE_CHECKING:
    from json2vec.structs.experiment import Hyperparameters


class NodeModule(torch.nn.Module):
    def __init__(self, hyperparameters: Hyperparameters, address: Address, batch_size: int):
        super().__init__()

        if address in hyperparameters.requests:
            request: Node = hyperparameters.requests[address]
            plugin: Plugin = TENSORFIELDS[request.type]
            embedder_kwargs = dict(hyperparameters=hyperparameters, address=address)
            if "batch_size" in inspect.signature(plugin.Embedder.__init__).parameters:
                embedder_kwargs["batch_size"] = batch_size

            self.embedder: EmbedderBase = plugin.Embedder(**embedder_kwargs)
            self.decoder: DecoderBase = plugin.Decoder(hyperparameters=hyperparameters, address=address)

        elif address in hyperparameters.arrays:
            self.encoder: ArrayEncoder = ArrayEncoder(hyperparameters=hyperparameters, address=address)

        else:
            raise ValueError("how did we get here?")

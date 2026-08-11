from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import torch

from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.pool import ConvolutionPool, LearnedQueryCrossAttention
from relflow.structs.pooling import Attention, Convolution
from relflow.structs.reference import Reference
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
            self.reference_reducers = torch.nn.ModuleDict()
            branch = schema.branches[address]
            references = (branch.reference,) if isinstance(branch.reference, Reference) else branch.reference
            for index, reference in enumerate(references):
                if reference.reduce is None:
                    continue
                reducer = reference.reduce.reducer
                match reducer:
                    case Attention():
                        self.reference_reducers[str(index)] = LearnedQueryCrossAttention(
                            n_context=1,
                            d_model=schema.d_model,
                            nhead=reducer.n_heads or branch.n_heads,
                            dropout=float(reducer.dropout if reducer.dropout is not None else branch.dropout or 0.0),
                            n_layers=reducer.n_layers,
                        )
                    case Convolution():
                        self.reference_reducers[str(index)] = ConvolutionPool(
                            width=1,
                            d_model=schema.d_model,
                            kernel_size=reducer.kernel_size,
                            n_layers=reducer.n_layers,
                            dropout=float(reducer.dropout if reducer.dropout is not None else branch.dropout or 0.0),
                        )

        else:
            raise ValueError("how did we get here?")

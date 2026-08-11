from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from einops import pack, rearrange, unpack

from relflow.architecture.attention import RotaryMultiheadAttention
from relflow.architecture.pool import ConvolutionPool, LearnedQueryCrossAttention
from relflow.structs.enums import AttentionMode
from relflow.structs.packages import Parcel
from relflow.structs.pooling import Attention, Convolution, Mean
from relflow.structs.tree import Address

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema


@dataclass(frozen=True)
class BranchEncoding:
    """Full Branch memory plus its pooled structural summary."""

    memory: torch.Tensor
    summary: torch.Tensor


class RotaryTransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        n_kv_heads: int,
        dropout: float,
        ffn_multiplier: int = 4,
    ):
        super().__init__()

        self.attention_norm = torch.nn.LayerNorm(normalized_shape=d_model)
        self.ffn_norm = torch.nn.LayerNorm(normalized_shape=d_model)

        self.attention = RotaryMultiheadAttention(
            d_model=d_model,
            nhead=nhead,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
        )

        hidden = d_model * ffn_multiplier
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model, out_features=hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=hidden, out_features=d_model),
            torch.nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normed = self.attention_norm(inputs)
        inputs = inputs + self.attention(normed, normed, normed)
        return inputs + self.ffn(self.ffn_norm(inputs))


class BranchEncoder(torch.nn.Module):
    def __init__(self, schema: Schema, address: Address):
        super().__init__()

        branch = schema.branches[address]
        dropout = float(branch.dropout or 0.0)

        self.origin: Address = address
        self.destination: Address = branch.parent.address

        layers: list[RotaryTransformerEncoderLayer] = []
        attention = AttentionMode.normalize(branch.attention)
        if attention != AttentionMode.none:
            for _ in range(branch.n_layers):
                layers.append(
                    RotaryTransformerEncoderLayer(
                        d_model=schema.d_model,
                        nhead=branch.n_heads,
                        n_kv_heads=attention.kv_heads(branch.n_heads),
                        dropout=dropout,
                    )
                )

        self.encoder = torch.nn.ModuleList(layers)

        self.pool_width = branch.pooling.width or 1
        match branch.pooling:
            case Mean():
                self.pool: torch.nn.Module | None = None
            case Attention():
                self.pool = LearnedQueryCrossAttention(
                    n_context=self.pool_width,
                    d_model=schema.d_model,
                    nhead=branch.pooling.n_heads or branch.n_heads,
                    dropout=float(branch.pooling.dropout if branch.pooling.dropout is not None else dropout),
                    n_layers=branch.pooling.n_layers,
                )
            case Convolution():
                self.pool = ConvolutionPool(
                    width=self.pool_width,
                    d_model=schema.d_model,
                    kernel_size=branch.pooling.kernel_size,
                    n_layers=branch.pooling.n_layers,
                    dropout=float(branch.pooling.dropout if branch.pooling.dropout is not None else dropout),
                )
            case _:
                raise ValueError(f"unsupported branch pooling: {branch.pooling}")

    def forward(self, inputs: list[Parcel] | list[torch.Tensor]) -> BranchEncoding:
        payloads = [item.payload if isinstance(item, Parcel) else item for item in inputs]
        if not payloads:
            raise ValueError("branch encoder requires at least one input")

        concatenated = torch.cat(payloads, dim=-2)
        encoded, leading_shape = pack([concatenated], "* token channel")

        for layer in self.encoder:
            encoded = layer(encoded)

        [memory] = unpack(encoded, leading_shape, "* token channel")
        pooled_flat = encoded.mean(dim=-2, keepdim=True) if self.pool is None else self.pool(encoded)
        [pooled] = unpack(pooled_flat, leading_shape, "* width channel")
        summary = (
            rearrange(
                pooled,
                "batch ... coordinate width channel -> batch ... (coordinate width) channel",
            )
            if pooled.ndim > 3
            else pooled
        )

        return BranchEncoding(memory=memory, summary=summary)

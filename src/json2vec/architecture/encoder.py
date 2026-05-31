from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from json2vec.architecture.attention import RotaryMultiheadAttention
from json2vec.architecture.pool import LearnedQueryCrossAttention
from json2vec.structs.enums import AttentionMode
from json2vec.structs.packages import Parcel
from json2vec.structs.tree import Address

if TYPE_CHECKING:
    from json2vec.structs.experiment import Hyperparameters


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


class ArrayEncoder(torch.nn.Module):
    def __init__(self, hyperparameters: Hyperparameters, address: Address):
        super().__init__()

        array = hyperparameters.arrays[address]
        dropout = float(array.dropout or 0.0)

        self.origin: Address = address
        self.destination: Address = array.parent.address

        layers: list[RotaryTransformerEncoderLayer] = []
        attention = AttentionMode.normalize(array.attention)
        if attention != AttentionMode.none:
            for _ in range(array.n_layers):
                layers.append(
                    RotaryTransformerEncoderLayer(
                        d_model=hyperparameters.d_model,
                        nhead=array.n_heads,
                        n_kv_heads=attention.kv_heads(array.n_heads),
                        dropout=dropout,
                    )
                )

        self.encoder = torch.nn.ModuleList(layers)

        self.pool = LearnedQueryCrossAttention(
            n_context=1,
            d_model=hyperparameters.d_model,
            nhead=array.n_heads,
            dropout=dropout,
            n_linear=array.n_linear,
        )

    def forward(self, parcels: list[Parcel]) -> Parcel:
        payloads: list[torch.Tensor] = []
        for parcel in parcels:
            payloads.append(parcel.payload)

        concatenated: torch.Tensor = torch.cat(payloads, dim=-2)
        N, *dims, L, C = concatenated.shape
        encoded: torch.Tensor = concatenated.reshape(-1, L, C)

        for layer in self.encoder:
            encoded = layer(encoded)

        pooled: torch.Tensor = self.pool(encoded).reshape(N, *dims[:-1], -1, C)

        return Parcel(
            payload=pooled,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )

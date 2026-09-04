from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from relflow.architecture.attention import RotaryMultiheadAttention
from relflow.architecture.pool import LearnedQueryCrossAttention
from relflow.structs.enums import AttentionMode
from relflow.structs.packages import Parcel
from relflow.structs.tree import Address

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema


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

    def forward(self, inputs: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        if present.dtype != torch.bool or tuple(present.shape) != tuple(inputs.shape[:2]):
            raise ValueError(
                f"encoder presence must have bool shape {tuple(inputs.shape[:2])}, "
                f"got {tuple(present.shape)} with dtype {present.dtype}"
            )

        inputs = inputs.masked_fill(~present.unsqueeze(-1), 0.0)
        normed = self.attention_norm(inputs)
        inputs = inputs + self.attention(normed, normed, normed, key_padding_mask=~present)
        inputs = inputs + self.ffn(self.ffn_norm(inputs))
        return inputs.masked_fill(~present.unsqueeze(-1), 0.0)


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

        self.pool = LearnedQueryCrossAttention(
            n_context=1,
            d_model=schema.d_model,
            nhead=branch.n_heads,
            dropout=dropout,
            n_linear=branch.n_linear,
        )

    def forward(self, parcels: list[Parcel]) -> Parcel:
        if not parcels:
            raise ValueError(f"branch encoder '{self.origin}' requires at least one child parcel")

        payloads = [parcel.payload for parcel in parcels]
        presence = [parcel.present for parcel in parcels]
        for parcel in parcels:
            if parcel.present.dtype != torch.bool or tuple(parcel.present.shape) != tuple(parcel.payload.shape[:-1]):
                raise ValueError(
                    f"parcel from '{parcel.origin}' presence must have bool shape "
                    f"{tuple(parcel.payload.shape[:-1])}, got {tuple(parcel.present.shape)} "
                    f"with dtype {parcel.present.dtype}"
                )

        concatenated: torch.Tensor = torch.cat(payloads, dim=-2)
        present = torch.cat(presence, dim=-1)
        N, *dims, L, C = concatenated.shape
        encoded: torch.Tensor = concatenated.reshape(-1, L, C)
        present = present.reshape(-1, L)
        active = present.any(dim=1)
        indices = active.nonzero(as_tuple=False).reshape(-1)

        pooled = encoded.new_zeros((encoded.shape[0], 1, C))
        if indices.numel():
            selected = encoded.index_select(0, indices)
            selected_present = present.index_select(0, indices)

            for layer in self.encoder:
                selected = layer(selected, present=selected_present)

            selected = self.pool(selected, present=selected_present)
            pooled = pooled.index_copy(0, indices, selected)
        else:
            for parameter in self.parameters():
                pooled = pooled + parameter.sum() * 0.0

        pooled = pooled.reshape(N, *dims, C)
        present = active.reshape(N, *dims)

        return Parcel(
            payload=pooled,
            present=present,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )

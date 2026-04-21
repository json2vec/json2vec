import math

import torch

from json2vec.architecture.rotary import RotaryEmbedding


class RotaryMultiheadAttention(torch.nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

        self.rotary = RotaryEmbedding(d_model=self.head_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def splitheads(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = inputs.shape
        return inputs.reshape(batch, seq_len, self.nhead, self.head_dim).transpose(1, 2)

    def rotate(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, nhead, seq_len, head_dim = inputs.shape
        rotated = self.rotary(inputs.reshape(batch * nhead, seq_len, head_dim))
        return rotated.reshape(batch, nhead, seq_len, head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.rotate(self.splitheads(self.q_proj(query)))
        k = self.rotate(self.splitheads(self.k_proj(key)))
        v = self.splitheads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            mask = key_padding_mask
            all_masked = mask.all(dim=1)
            if all_masked.any():
                mask = mask.clone()
                mask[all_masked, 0] = False
            scores = scores.masked_fill(mask[:, None, None, :], torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.d_model)

        return self.out_proj(context)

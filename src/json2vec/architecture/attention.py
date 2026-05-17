import torch
import torch.nn.functional as F

from json2vec.architecture.rotary import RotaryEmbedding


class RotaryMultiheadAttention(torch.nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, n_kv_heads: int | None = None):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        n_kv_heads = nhead if n_kv_heads is None else n_kv_heads
        if n_kv_heads < 1:
            raise ValueError("n_kv_heads must be >= 1")
        if nhead % n_kv_heads != 0:
            raise ValueError("nhead must be divisible by n_kv_heads")

        self.d_model = d_model
        self.nhead = nhead
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // nhead

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.v_proj = torch.nn.Linear(d_model, n_kv_heads * self.head_dim)
        self.out_proj = torch.nn.Linear(d_model, d_model)

        self.rotary = RotaryEmbedding(d_model=self.head_dim)
        self.dropout_p = dropout

    def splitheads(self, inputs: torch.Tensor, nhead: int) -> torch.Tensor:
        batch, seq_len, _ = inputs.shape
        return inputs.reshape(batch, seq_len, nhead, self.head_dim).transpose(1, 2)

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
        q = self.rotate(self.splitheads(self.q_proj(query), nhead=self.nhead))
        k = self.rotate(self.splitheads(self.k_proj(key), nhead=self.n_kv_heads))
        v = self.splitheads(self.v_proj(value), nhead=self.n_kv_heads)

        attn_mask: torch.Tensor | None = None

        if key_padding_mask is not None:
            mask = key_padding_mask
            all_masked = mask.all(dim=1)
            if all_masked.any():
                mask = mask.clone()
                mask[all_masked, 0] = False

            # SDPA boolean masks use True for positions that may participate in attention.
            attn_mask = ~mask[:, None, None, :]

        context = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            enable_gqa=self.n_kv_heads != self.nhead,
        )
        context = context.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.d_model)

        return self.out_proj(context)

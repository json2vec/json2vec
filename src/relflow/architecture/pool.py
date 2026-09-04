import torch

from relflow.architecture.attention import RotaryMultiheadAttention


class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float, ffn_multiplier: int):
        super().__init__()

        self.attention_norm = torch.nn.LayerNorm(normalized_shape=d_model)
        self.ffn_norm = torch.nn.LayerNorm(normalized_shape=d_model)
        self.attention = RotaryMultiheadAttention(d_model=d_model, nhead=nhead, dropout=dropout)

        hidden = d_model * ffn_multiplier
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model, out_features=hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=hidden, out_features=d_model),
            torch.nn.Dropout(p=dropout),
        )

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        padding = None if present is None else ~present
        attended = self.attention(
            self.attention_norm(queries),
            memory,
            memory,
            key_padding_mask=padding,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class LearnedQueryCrossAttention(torch.nn.Module):
    def __init__(
        self,
        n_context: int,
        d_model: int,
        nhead: int,
        dropout: float,
        n_linear: int = 1,
        ffn_multiplier: int = 4,
    ):
        super().__init__()

        self.queries = torch.nn.Parameter(torch.normal(mean=0.0, std=1e-2, size=(n_context, d_model)))
        self.blocks = torch.nn.ModuleList()
        for _ in range(n_linear):
            self.blocks.append(
                CrossAttentionBlock(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    ffn_multiplier=ffn_multiplier,
                )
            )
        self.norm = torch.nn.LayerNorm(normalized_shape=d_model)

    def forward(self, memory: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        N, _, _ = memory.shape
        if present is None:
            present = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        if present.dtype != torch.bool or tuple(present.shape) != tuple(memory.shape[:2]):
            raise ValueError(
                f"pool presence must have bool shape {tuple(memory.shape[:2])}, "
                f"got {tuple(present.shape)} with dtype {present.dtype}"
            )

        active = present.any(dim=1)
        pooled = memory.new_zeros((N, self.queries.shape[0], self.queries.shape[1]))
        if not active.any():
            for parameter in self.parameters():
                pooled = pooled + parameter.sum() * 0.0
            return pooled

        indices = active.nonzero(as_tuple=False).reshape(-1)
        memory = memory.index_select(0, indices)
        present = present.index_select(0, indices)
        memory = memory.masked_fill(~present.unsqueeze(-1), 0.0)
        queries = self.queries

        if not torch.is_grad_enabled():
            queries = queries.detach()
            memory = memory.detach()

        queries = queries.unsqueeze(0).expand(indices.numel(), -1, -1)

        for block in self.blocks:
            queries = block(queries=queries, memory=memory, present=present)

        return pooled.index_copy(0, indices, self.norm(queries))


class MeanPool(torch.nn.Module):
    def __init__(self, n_context: int):
        super().__init__()
        self.n_context = n_context

    def forward(self, memory: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        if present is None:
            present = torch.ones(memory.shape[:2], dtype=torch.bool, device=memory.device)
        if present.dtype != torch.bool or tuple(present.shape) != tuple(memory.shape[:2]):
            raise ValueError(
                f"pool presence must have bool shape {tuple(memory.shape[:2])}, "
                f"got {tuple(present.shape)} with dtype {present.dtype}"
            )

        weights = present.unsqueeze(-1).to(dtype=memory.dtype)
        pooled = memory.masked_fill(~present.unsqueeze(-1), 0.0).sum(dim=1, keepdim=True)
        pooled = pooled / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return pooled.expand(-1, self.n_context, -1)

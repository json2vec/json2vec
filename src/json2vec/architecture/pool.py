import torch

from json2vec.architecture.attention import RotaryMultiheadAttention


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

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended = self.attention(self.attention_norm(queries), memory, memory)
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

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        N, _, _ = memory.shape
        queries = self.queries

        if not torch.is_grad_enabled():
            queries = queries.detach()
            memory = memory.detach()

        queries = queries.unsqueeze(0).expand(N, -1, -1)

        for block in self.blocks:
            queries = block(queries=queries, memory=memory)

        return self.norm(queries)


class MeanPool(torch.nn.Module):
    def __init__(self, n_context: int):
        super().__init__()
        self.n_context = n_context

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        pooled = memory.mean(dim=1, keepdim=True)
        return pooled.expand(-1, self.n_context, -1)

import torch
import torch.nn.functional as F
from einops import rearrange

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
        n_layers: int = 1,
        ffn_multiplier: int = 4,
    ):
        super().__init__()

        if n_context < 1:
            raise ValueError("n_context must be >= 1")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.queries = torch.nn.Parameter(torch.normal(mean=0.0, std=1e-2, size=(n_context, d_model)))
        self.blocks = torch.nn.ModuleList()
        for _ in range(n_layers):
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


class ConvolutionPool(torch.nn.Module):
    def __init__(
        self,
        width: int,
        d_model: int,
        kernel_size: int = 3,
        n_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        if width < 1:
            raise ValueError("width must be >= 1")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        self.width = width
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=1,
                    bias=True,
                ),
                torch.nn.GELU(),
                torch.nn.Dropout(p=dropout),
            )
            for _ in range(n_layers)
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        memory = rearrange(memory, "batch token channel -> batch channel token")

        for block in self.blocks:
            memory = memory + block(memory)

        memory = F.adaptive_avg_pool1d(memory, output_size=self.width)
        return rearrange(memory, "batch channel width -> batch width channel")

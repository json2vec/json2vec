import torch


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, base: float = 10_000.0):
        super().__init__()

        if d_model < 2:
            raise ValueError("d_model must be at least 2 for rotary embeddings")

        self.d_model = d_model
        self.rotary_dim = d_model - (d_model % 2)
        self.base = base

        index = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
        self.register_buffer("inv_freq", base ** (-index / self.rotary_dim), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = inputs.shape

        positions = torch.arange(seq_len, device=inputs.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        cos = freqs.cos().to(dtype=inputs.dtype).unsqueeze(0)
        sin = freqs.sin().to(dtype=inputs.dtype).unsqueeze(0)

        rotated = inputs[..., : self.rotary_dim]
        passthrough = inputs[..., self.rotary_dim :]

        even = rotated[..., ::2]
        odd = rotated[..., 1::2]

        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)

        if passthrough.shape[-1] == 0:
            return rotated

        return torch.cat((rotated, passthrough), dim=-1)

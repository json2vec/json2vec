import torch

from relflow.architecture.rotary import RotaryEmbedding


def test_rotary_embedding_preserves_arbitrary_leading_dimensions():
    rotary = RotaryEmbedding(d_model=5)
    inputs = torch.randn(2, 3, 7, 5)

    output = rotary(inputs)
    flattened = rotary(inputs.reshape(-1, 7, 5)).reshape_as(inputs)

    assert output.shape == inputs.shape
    assert torch.equal(output, flattened)


def test_rotary_embedding_preserves_odd_passthrough_channel():
    rotary = RotaryEmbedding(d_model=5)
    inputs = torch.randn(2, 3, 5)

    output = rotary(inputs)

    assert torch.equal(output[..., -1], inputs[..., -1])

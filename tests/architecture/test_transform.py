import torch

from json2vec.tensorfields.extensions.number import jitter


def test_jitter():
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    jitter_amount = torch.tensor(0.1)

    jittered = jitter(inputs, jitter_amount)

    assert isinstance(jittered, torch.Tensor)
    assert jittered.shape == inputs.shape
    assert torch.all(jittered <= inputs + jitter_amount)
    assert torch.all(jittered >= inputs - jitter_amount)
    assert not torch.allclose(jittered, inputs)


def test_zero_jitter_is_identity():
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    jittered = jitter(inputs, torch.tensor(0.0))
    assert torch.allclose(jittered, inputs)

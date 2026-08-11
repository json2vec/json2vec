import pytest
import torch
import torch.nn.functional as F
from torch.utils.module_tracker import ModuleTracker

from relflow.architecture.pool import ConvolutionPool, LearnedQueryCrossAttention


def test_pool():
    n_context = 5
    d_model = 16
    nhead = 4
    dropout = 0.1
    batch_size = 2
    seq_length = 10

    model = LearnedQueryCrossAttention(n_context, d_model, nhead, dropout, n_layers=2)
    memory = torch.randn(batch_size, seq_length, d_model)

    output = model(memory)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, n_context, d_model)


def test_learned_query_pooling_supports_module_tracker_with_grad_disabled():
    model = LearnedQueryCrossAttention(n_context=2, d_model=8, nhead=2, dropout=0.0)
    memory = torch.randn(3, 4, 8)

    with ModuleTracker():
        with torch.no_grad():
            output = model(memory)

    assert output.shape == (3, 2, 8)
    assert not output.requires_grad


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_context": 0}, "n_context must be >= 1"),
        ({"n_layers": 0}, "n_layers must be >= 1"),
    ],
)
def test_learned_query_pooling_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LearnedQueryCrossAttention(d_model=8, nhead=2, dropout=0.0, **{"n_context": 1, **kwargs})


def test_convolution_pool_resizes_memory_and_preserves_gradients():
    model = ConvolutionPool(width=3, d_model=8, kernel_size=3, n_layers=2, dropout=0.0)
    memory = torch.randn(2, 5, 8, requires_grad=True)

    output = model(memory)
    output.sum().backward()

    assert output.shape == (2, 3, 8)
    assert memory.grad is not None
    assert len(model.blocks) == 2
    assert model.blocks[0][0] is not model.blocks[1][0]


def test_convolution_pool_supports_width_larger_than_memory():
    model = ConvolutionPool(width=5, d_model=4, kernel_size=1, dropout=0.0)
    memory = torch.randn(2, 2, 4)

    output = model(memory)

    assert output.shape == (2, 5, 4)


def test_convolution_pool_residual_then_adaptive_average():
    model = ConvolutionPool(width=2, d_model=2, kernel_size=3, dropout=0.0)
    memory = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)

    with torch.no_grad():
        model.blocks[0][0].weight.zero_()
        model.blocks[0][0].bias.zero_()

    output = model(memory)
    expected = F.adaptive_avg_pool1d(memory.transpose(1, 2), output_size=2).transpose(1, 2)

    assert torch.equal(output, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width": 0}, "width must be >= 1"),
        ({"kernel_size": 2}, "kernel_size must be a positive odd integer"),
        ({"n_layers": 0}, "n_layers must be >= 1"),
    ],
)
def test_convolution_pool_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ConvolutionPool(d_model=4, **{"width": 1, **kwargs})

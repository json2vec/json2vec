import torch
from torch.utils.module_tracker import ModuleTracker

from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool


def test_pool():
    n_context = 5
    d_model = 16
    nhead = 4
    dropout = 0.1
    batch_size = 2
    seq_length = 10

    model = LearnedQueryCrossAttention(n_context, d_model, nhead, dropout, n_linear=2)
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


def test_learned_query_pooling_ignores_absent_memory_and_zeros_empty_rows():
    model = LearnedQueryCrossAttention(n_context=2, d_model=8, nhead=2, dropout=0.0).eval()
    memory = torch.randn(2, 4, 8)
    present = torch.tensor([[True, False, True, False], [False, False, False, False]])

    first = model(memory, present=present)
    changed = memory.clone()
    changed[0, ~present[0]] = torch.nan
    changed[1] = torch.nan
    second = model(changed, present=present)

    assert torch.allclose(first[0], second[0])
    assert torch.equal(first[1], torch.zeros_like(first[1]))
    assert torch.equal(second[1], torch.zeros_like(second[1]))


def test_mean_pooling_uses_only_present_memory():
    model = MeanPool(n_context=2)
    memory = torch.tensor(
        [
            [[1.0, 2.0], [torch.nan, torch.nan], [3.0, 4.0]],
            [[torch.nan, torch.nan], [torch.nan, torch.nan], [torch.nan, torch.nan]],
        ]
    )
    present = torch.tensor([[True, False, True], [False, False, False]])

    output = model(memory, present=present)

    assert torch.equal(output[0], torch.tensor([[2.0, 3.0], [2.0, 3.0]]))
    assert torch.equal(output[1], torch.zeros_like(output[1]))

import torch
from torch.utils.module_tracker import ModuleTracker

from json2vec.architecture.pool import LearnedQueryCrossAttention


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

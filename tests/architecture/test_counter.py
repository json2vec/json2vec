import torch

from json2vec.structs.tree import Address
from json2vec.tensorfields.shared.counter import Counter


def test_counter():
    counter = Counter(address=Address("test"), size=5)
    data = torch.tensor([0, 1, 2, 2, 3, 4, 4, 4])
    counter(data)

    counts = torch.tensor([1, 1, 2, 1, 3]).add(1)
    assert torch.all(counter.counts == counts)

    weight = counter.weight
    assert weight.shape[0] == 5
    expected = counts.to(dtype=torch.float32).rsqrt()
    expected *= counts.sum() / (expected * counts).sum()
    assert torch.allclose(weight, expected)


def test_counter_stops_counting_when_about_to_overflow():
    counter = Counter(address=Address("overflow"), size=2)
    max_value = torch.iinfo(counter.counts.dtype).max
    counter.counts.fill_(max_value - 1)

    before = counter.counts.clone()
    counter(torch.tensor([0, 1], dtype=torch.int64))

    assert counter.is_full is True
    assert torch.equal(counter.counts, before)


def test_counter_str_exposes_plot_details():
    counter = Counter(address=Address("details"), size=3)
    counter.counts.copy_(torch.tensor([4, 2, 1], dtype=torch.int64))

    rendered = str(counter)

    assert rendered == "\n".join(
        (
            "size: 3",
            "is_full: False",
            "counts: [4, 2, 1]",
        )
    )

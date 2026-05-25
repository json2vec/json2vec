from types import SimpleNamespace

import torch

from json2vec.structs.tree import Address
from json2vec.tensorfields.base import TENSORFIELDS
from json2vec.tensorfields.shared.counter import Counter, CounterUpdateCallback


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
    assert torch.equal(counter._pending_counts, torch.zeros_like(counter._pending_counts))  # noqa: SLF001


def test_counter_forward_does_not_call_distributed_collective(monkeypatch):
    def fail_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Counter.forward must not call all_reduce_sum")

    monkeypatch.setattr("json2vec.tensorfields.shared.counter.all_reduce_sum", fail_all_reduce)
    counter = Counter(address=Address("local"), size=3)

    counter(torch.tensor([0, 1, 1], dtype=torch.int64))

    assert torch.equal(counter.counts, torch.tensor([2, 3, 1], dtype=torch.int64))
    assert torch.equal(counter._pending_counts, torch.tensor([1, 2, 0], dtype=torch.int64))  # noqa: SLF001


def test_counter_sync_reduces_pending_delta_only(monkeypatch):
    reduced = []

    def fake_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        reduced.append(tensor.clone())
        return torch.tensor([2, 0, 3], dtype=tensor.dtype)

    monkeypatch.setattr("json2vec.tensorfields.shared.counter.all_reduce_sum", fake_all_reduce)
    counter = Counter(address=Address("sync"), size=3)
    counter(torch.tensor([0, 2, 2], dtype=torch.int64))

    counter.sync()

    assert len(reduced) == 1
    assert torch.equal(reduced[0], torch.tensor([1, 0, 2], dtype=torch.int64))
    assert torch.equal(counter.counts, torch.tensor([3, 1, 4], dtype=torch.int64))
    assert torch.equal(counter._pending_counts, torch.zeros(3, dtype=torch.int64))  # noqa: SLF001
    assert "_pending_counts" not in counter.state_dict()


def test_counter_sync_still_calls_collective_for_empty_pending(monkeypatch):
    reduced = []

    def fake_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        reduced.append(tensor.clone())
        return tensor

    monkeypatch.setattr("json2vec.tensorfields.shared.counter.all_reduce_sum", fake_all_reduce)
    counter = Counter(address=Address("empty"), size=2)

    counter.sync()

    assert len(reduced) == 1
    assert torch.equal(reduced[0], torch.zeros(2, dtype=torch.int64))
    assert torch.equal(counter.counts, torch.ones(2, dtype=torch.int64))


def test_counter_update_callback_syncs_counters_in_deterministic_order(monkeypatch):
    names = {}
    calls = []

    def named(name: str) -> Counter:
        counter = Counter(address=Address(name), size=2)
        names[id(counter)] = name
        return counter

    def sync(counter: Counter) -> None:
        calls.append(names[id(counter)])

    module = SimpleNamespace(
        nodes={
            Address("root", "z"): SimpleNamespace(
                embedder=SimpleNamespace(counter=named("root/z/counter")),
            ),
            Address("root", "a"): SimpleNamespace(
                embedder=SimpleNamespace(
                    counters=torch.nn.ModuleDict(
                        {
                            "state": named("root/a/state"),
                            "content": named("root/a/content"),
                        }
                    )
                ),
            ),
        }
    )
    monkeypatch.setattr(Counter, "sync", sync)

    CounterUpdateCallback().on_train_epoch_end(trainer=None, pl_module=module)

    assert calls == ["root/a/content", "root/a/state", "root/z/counter"]


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


def test_counter_callback_is_registered_for_counter_extensions():
    for extension in ("category", "number", "set", "text"):
        assert CounterUpdateCallback in TENSORFIELDS[extension].callback_factories

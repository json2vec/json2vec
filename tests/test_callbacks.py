from types import SimpleNamespace

from json2vec.tensorfields.shared.vocabulary import VocabularySyncCallback


class _Vocab:
    max_vocab_size = 8

    def __init__(self):
        self.values = ["ALPHA"]
        self.loaded = None

    def drain_proposals(self):
        return ["BETA"]

    def extend(self, proposals):
        accepted = 0
        for proposal in proposals:
            if proposal not in self.values:
                self.values.append(proposal)
                accepted += 1
        return accepted, 0

    def snapshot(self):
        return list(self.values)

    def load_snapshot(self, snapshot):
        self.loaded = list(snapshot)
        self.values = list(snapshot)


def test_vocabulary_sync_callback_gathers_rank_proposals(monkeypatch):
    vocab = _Vocab()
    trainer = SimpleNamespace(strategy=SimpleNamespace(barriers=[]))
    trainer.strategy.barrier = lambda name: trainer.strategy.barriers.append(name)
    module = SimpleNamespace(
        nodes={
            "root/category": SimpleNamespace(
                embedder=SimpleNamespace(vocab=vocab),
            ),
        },
    )

    monkeypatch.setattr("json2vec.tensorfields.shared.vocabulary.is_distributed", lambda: True)
    monkeypatch.setattr("json2vec.tensorfields.shared.vocabulary.is_rank_zero", lambda: True)
    monkeypatch.setattr(
        "json2vec.tensorfields.shared.vocabulary.all_gather_object",
        lambda local: [local, {"root/category": ["GAMMA"]}],
    )
    monkeypatch.setattr("json2vec.tensorfields.shared.vocabulary.broadcast_object", lambda payload, src: payload)

    VocabularySyncCallback().on_train_epoch_end(trainer=trainer, pl_module=module)

    assert vocab.values == ["ALPHA", "BETA", "GAMMA"]
    assert vocab.loaded == ["ALPHA", "BETA", "GAMMA"]
    assert trainer.strategy.barriers == ["vocabulary-sync-train_epoch_end"]

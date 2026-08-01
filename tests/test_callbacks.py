from types import SimpleNamespace

from json2vec.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularySyncCallback


def test_vocabulary_sync_callback_gathers_rank_proposals(monkeypatch):
    events = []
    vocab = OnlineVocabularyModel(size=8)
    vocab.load_snapshot(["ALPHA"])
    vocab.proposals.append("BETA")

    class TrainerStub:
        strategy = SimpleNamespace(barriers=[])

        @property
        def callback_metrics(self):
            events.append("metrics")
            return {}

    trainer = TrainerStub()
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
        lambda local: events.append("vocabulary") or [local, {"root/category": ["GAMMA"]}],
    )
    monkeypatch.setattr("json2vec.tensorfields.shared.vocabulary.broadcast_object", lambda payload, src: payload)

    VocabularySyncCallback().on_train_epoch_end(trainer=trainer, pl_module=module)

    assert vocab.snapshot() == ["ALPHA", "BETA", "GAMMA"]
    assert list(vocab.proposals) == []
    assert events == ["metrics", "vocabulary"]
    assert trainer.strategy.barriers == ["vocabulary-sync-train_epoch_end"]

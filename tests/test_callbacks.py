from types import SimpleNamespace

from relflow.rich import console, incidents
from relflow.tensorfields.shared.vocabulary import (
    OnlineVocabularyModel,
    VocabularySyncCallback,
    _emit_vocabulary_diagnostics,
)


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

    monkeypatch.setattr("relflow.tensorfields.shared.vocabulary.is_distributed", lambda: True)
    monkeypatch.setattr("relflow.tensorfields.shared.vocabulary.is_rank_zero", lambda: True)
    monkeypatch.setattr(
        "relflow.tensorfields.shared.vocabulary.all_gather_object",
        lambda local: events.append("vocabulary") or [local, {"root/category": ["GAMMA"]}],
    )
    monkeypatch.setattr("relflow.tensorfields.shared.vocabulary.broadcast_object", lambda payload, src: payload)

    VocabularySyncCallback().on_train_epoch_end(trainer=trainer, pl_module=module)

    assert vocab.snapshot() == ["ALPHA", "BETA", "GAMMA"]
    assert list(vocab.proposals) == []
    assert events == ["metrics", "vocabulary"]
    assert trainer.strategy.barriers == ["vocabulary-sync-train_epoch_end"]


def test_near_capacity_diagnostics_are_bounded_across_a_wide_schema() -> None:
    class SchemaScope:
        pass

    schema = SchemaScope()
    module = SimpleNamespace(schema=schema)
    resources = {f"root/field-{index}": OnlineVocabularyModel(size=100) for index in range(40)}
    stats = {address: {"rejected_full": 0, "size": 95, "max": 100} for address in resources}
    incidents.reset(scopes=(schema,))

    with console.capture() as captured:
        _emit_vocabulary_diagnostics(module, resources, stats)

    rendered = captured.get()
    assert rendered.count("vocabulary is near capacity") == 32
    assert rendered.count("additional vocabulary-capacity diagnostics are suppressed") == 1
    assert incidents.summary(scopes=(schema,))[0].suppressed == 7
    incidents.reset(scopes=(schema,))


def test_full_vocabulary_does_not_emit_a_later_near_capacity_warning() -> None:
    class SchemaScope:
        pass

    schema = SchemaScope()
    module = SimpleNamespace(schema=schema)
    vocabulary = OnlineVocabularyModel(size=2)
    vocabulary.load_snapshot(["ALPHA", "BETA"])
    resources = {"root/category": vocabulary}
    incidents.reset(scopes=(schema,))

    with console.capture() as captured:
        _emit_vocabulary_diagnostics(
            module,
            resources,
            {"root/category": {"rejected_full": 1, "size": 2, "max": 2}},
        )
        _emit_vocabulary_diagnostics(
            module,
            resources,
            {"root/category": {"rejected_full": 0, "size": 2, "max": 2}},
        )

    rendered = captured.get()
    assert rendered.count("vocabulary reached configured capacity") == 1
    assert "vocabulary is near capacity" not in rendered
    incidents.reset(scopes=(schema,))

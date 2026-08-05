from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

import relflow as rf
from relflow.architecture.root import Model, MutationLockCallback, RollbackCheckpoint, RuntimePlacementCallback
from relflow.data.iterables import encode
from relflow.logging.throughput import ThroughputLogger
from relflow.structs.enums import AttentionMode, Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.tree import Address
from relflow.tensorfields.shared.counter import CounterUpdateCallback
from relflow.tensorfields.shared.vocabulary import (
    OnlineVocabularyModel,
    VocabularySyncCallback,
)


def _schema() -> Schema:
    return Schema.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "branch",
                "dropout": 0.1,
                "length": 1,
                "fields": [
                    {
                        "name": "label",
                        "type": "category",
                        "query": "[*].label",
                        "size": 32,
                    }
                ],
            },
        }
    )


def test_model_accepts_schema_positionally_and_by_keyword() -> None:
    schema = _schema()

    positional = Model(schema)
    keyword = Model(schema=schema)

    assert positional.schema is schema
    assert keyword.schema is schema


def test_model_rejects_schema_combined_with_tree_configuration() -> None:
    with pytest.raises(TypeError, match="schema cannot be combined"):
        Model(schema=_schema(), d_model=8)


def test_model_tree_constructor_requires_architecture_options() -> None:
    with pytest.raises(TypeError, match="requires n_layers, n_heads"):
        Model(rf.Number("amount"), d_model=8)


def test_on_save_checkpoint_serializes_schema() -> None:
    schema = _schema()
    model = Model(schema=schema, batch_size=2)
    checkpoint = {}

    model.on_save_checkpoint(checkpoint)

    restored = Schema.model_validate(checkpoint["schema"])
    assert restored.model_dump(mode="python") == schema.model_dump(mode="python")
    assert checkpoint["batch_size"] == 2


def test_save_writes_loadable_checkpoint(tmp_path: Path) -> None:
    schema = _schema()
    model = Model(schema=schema, batch_size=2)
    pathname = tmp_path / "nested" / "model.ckpt"

    model.save(pathname=pathname)

    restored = Model.load(pathname)

    assert pathname.exists()
    assert restored.batch_size == 2
    assert restored.schema.model_dump(mode="python") == schema.model_dump(mode="python")

    restored_state = restored.state_dict()
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(restored_state[key], value)
        else:
            assert restored_state[key] == value


def _prediction_schema() -> Schema:
    return Schema(
        d_model=8,
        fields={
            "name": "root",
            "type": "branch",
            "embed": True,
            "length": 1,
            "attention": "none",
            "fields": [
                {
                    "name": "color",
                    "type": "category",
                    "query": "[*].color",
                    "embed": False,
                    "size": 16,
                },
                {
                    "name": "label",
                    "type": "category",
                    "query": "[*].label",
                    "embed": False,
                    "p_prune": 1.0,
                    "size": 16,
                    "topk": [2],
                },
            ],
        },
    )


def _primed_prediction_model() -> Model:
    schema = _prediction_schema()
    model = Model(schema=schema, batch_size=2)
    inputs = encode(
        batch=[
            [{"color": "red", "label": "warm"}],
            [{"color": "blue", "label": "cool"}],
        ],
        schema=schema,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    model(inputs, strata=Strata.train)
    return model


def _build_checkpoint(tmp_path: Path) -> tuple[Path, Schema]:
    schema = _schema()
    model = Model(schema=schema, batch_size=2)
    checkpoint_path = tmp_path / "model.ckpt"
    model.save(checkpoint_path)

    return checkpoint_path, schema


def test_load_restores_local_checkpoint(tmp_path: Path) -> None:
    checkpoint_path, schema = _build_checkpoint(tmp_path)

    model = Model.load(checkpoint_path)

    assert model.batch_size == 2
    assert model.schema.model_dump(mode="python") == schema.model_dump(mode="python")


def test_rollback_checkpoint_restores_best_model_from_disk(tmp_path: Path) -> None:
    model = Model(schema=_schema(), batch_size=2)
    best_path = tmp_path / "best.ckpt"
    model.save(best_path)
    best_state = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
        for key, value in model.state_dict().items()
    }
    best_schema = model.schema.model_dump(mode="python")
    address = Address("root", "label")

    model.update(lambda node: node.address == address, weight=3.0)
    mutated_node = model.nodes[address]
    with torch.no_grad():
        next(model.parameters()).add_(1.0)

    class CheckpointIOStub:
        def __init__(self) -> None:
            self.loaded: list[tuple[str, torch.device, bool | None]] = []

        def load_checkpoint(self, path: str, map_location: torch.device, weights_only: bool | None = None):
            self.loaded.append((path, map_location, weights_only))
            return torch.load(path, weights_only=weights_only, map_location=map_location)

    class StrategyStub:
        def __init__(self) -> None:
            self.checkpoint_io = CheckpointIOStub()
            self.barriers: list[str] = []

        def barrier(self, name: str) -> None:
            self.barriers.append(name)

    strategy = StrategyStub()
    trainer = type("TrainerStub", (), {"strategy": strategy})()
    callback = RollbackCheckpoint(dirpath=tmp_path)
    callback.best_model_path = str(best_path)
    callback.best_model_score = torch.tensor(0.25)

    callback.on_fit_end(trainer=trainer, pl_module=model)

    assert strategy.barriers == ["rollback_checkpoint_load"]
    assert strategy.checkpoint_io.loaded == [(str(best_path), torch.device("cpu"), False)]
    assert model.batch_size == 2
    assert model.schema.model_dump(mode="python") == best_schema
    assert model.nodes[address] is not mutated_node
    for key, value in best_state.items():
        restored = model.state_dict()[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(restored, value)
        else:
            assert restored == value


def test_rollback_checkpoint_loads_schema_metadata_with_weights_only_disabled(tmp_path: Path) -> None:
    model = Model(schema=_schema(), batch_size=2)
    best_path = tmp_path / "best.ckpt"
    model.save(best_path)
    checkpoint = torch.load(best_path, weights_only=False, map_location="cpu")
    assert checkpoint["schema"]["fields"]["attention"] == AttentionMode.mha

    with torch.no_grad():
        next(model.parameters()).add_(1.0)

    class CheckpointIOStub:
        def __init__(self) -> None:
            self.weights_only: bool | None = None

        def load_checkpoint(self, path: str, map_location: torch.device, weights_only: bool | None = None):
            self.weights_only = weights_only
            return torch.load(path, weights_only=weights_only, map_location=map_location)

    class StrategyStub:
        def __init__(self) -> None:
            self.checkpoint_io = CheckpointIOStub()

        def barrier(self, name: str) -> None:
            pass

    strategy = StrategyStub()
    trainer = type("TrainerStub", (), {"strategy": strategy})()
    callback = RollbackCheckpoint(dirpath=tmp_path)
    callback.best_model_path = str(best_path)

    callback.on_fit_end(trainer=trainer, pl_module=model)

    assert strategy.checkpoint_io.weights_only is False


def test_rollback_checkpoint_requires_full_saved_checkpoint() -> None:
    with pytest.raises(ValueError, match="full checkpoints"):
        RollbackCheckpoint(save_weights_only=True)


def test_rollback_checkpoint_requires_a_saved_checkpoint() -> None:
    with pytest.raises(ValueError, match="at least one saved checkpoint"):
        RollbackCheckpoint(save_top_k=0)


def test_configure_optimizers_uses_user_supplied_optimizer(tmp_path: Path) -> None:
    _, schema = _build_checkpoint(tmp_path)
    model = Model(
        schema=schema,
        batch_size=2,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-3),
    )
    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)


def test_configure_optimizers_uses_user_supplied_scheduler(tmp_path: Path) -> None:
    _, schema = _build_checkpoint(tmp_path)
    model = Model(
        schema=schema,
        batch_size=2,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-3),
        scheduler=lambda _module, optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
    )

    configured = model.configure_optimizers()

    assert isinstance(configured["optimizer"], torch.optim.AdamW)
    assert isinstance(configured["lr_scheduler"], torch.optim.lr_scheduler.StepLR)


def test_configure_callbacks_collects_active_extension_callbacks() -> None:
    model = Model(schema=_schema(), batch_size=2)

    callbacks = model.configure_callbacks()
    callback_types = [type(callback) for callback in callbacks]

    assert any(isinstance(callback, RuntimePlacementCallback) for callback in callbacks)
    assert any(isinstance(callback, MutationLockCallback) for callback in callbacks)
    assert any(isinstance(callback, ThroughputLogger) for callback in callbacks)
    assert any(isinstance(callback, VocabularySyncCallback) for callback in callbacks)
    assert any(isinstance(callback, CounterUpdateCallback) for callback in callbacks)
    assert callback_types == sorted(
        callback_types,
        key=lambda callback_type: (
            callback_type.__module__,
            callback_type.__qualname__,
        ),
    )


def test_configure_callbacks_deduplicates_shared_extension_callbacks() -> None:
    schema = Schema.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "branch",
                "length": 1,
                "fields": [
                    {
                        "name": "label",
                        "type": "category",
                        "query": "[*].label",
                        "size": 16,
                    },
                    {
                        "name": "tags",
                        "type": "set",
                        "query": "[*].tags",
                        "size": 16,
                    },
                ],
            },
        }
    )
    model = Model(schema=schema, batch_size=2)

    vocabulary_callbacks = [
        callback for callback in model.configure_callbacks() if isinstance(callback, VocabularySyncCallback)
    ]
    counter_callbacks = [
        callback for callback in model.configure_callbacks() if isinstance(callback, CounterUpdateCallback)
    ]

    mutation_lock_callbacks = [
        callback for callback in model.configure_callbacks() if isinstance(callback, MutationLockCallback)
    ]
    runtime_placement_callbacks = [
        callback for callback in model.configure_callbacks() if isinstance(callback, RuntimePlacementCallback)
    ]
    throughput_callbacks = [
        callback for callback in model.configure_callbacks() if isinstance(callback, ThroughputLogger)
    ]

    assert len(runtime_placement_callbacks) == 1
    assert len(mutation_lock_callbacks) == 1
    assert len(throughput_callbacks) == 1
    assert len(vocabulary_callbacks) == 1
    assert len(counter_callbacks) == 1

    callback_types = [type(callback) for callback in model.configure_callbacks()]
    assert callback_types.index(CounterUpdateCallback) < callback_types.index(VocabularySyncCallback)


def test_configure_callbacks_skips_callbacks_already_attached_to_trainer() -> None:
    model = Model(schema=_schema(), batch_size=2)
    model._trainer = type(  # noqa: SLF001
        "TrainerStub",
        (),
        {
            "callbacks": [
                RuntimePlacementCallback(),
                MutationLockCallback(),
                ThroughputLogger(),
                VocabularySyncCallback(),
                CounterUpdateCallback(),
            ]
        },
    )()

    assert model.configure_callbacks() == []


def test_builtin_resources_are_attached_to_extension_modules() -> None:
    model = Model(schema=_schema(), batch_size=2)
    address = Address("root", "label")

    assert isinstance(model.nodes[address].embedder.vocab, OnlineVocabularyModel)
    assert TensorKey.state.name in model.nodes[address].embedder.counters
    assert TensorKey.content.name in model.nodes[address].embedder.counters


def test_online_vocabulary_model_uses_local_storage_until_shared():
    vocab = OnlineVocabularyModel(size=8)

    assert vocab.manager is None
    assert vocab.is_shared is False

    local_state = vocab.state
    assert isinstance(local_state.master, list)
    local_state.reserve("ALPHA", learn=True)
    assert local_state.encode("ALPHA") == 0
    assert vocab.snapshot() == ["ALPHA"]

    vocab.share()

    assert vocab.manager is not None
    assert vocab.is_shared is True
    assert not isinstance(vocab.state.master, list)

    shared_state = vocab.state
    shared_state.reserve("BETA", learn=True)
    assert shared_state.encode("BETA") == 1
    assert vocab.snapshot() == ["ALPHA", "BETA"]

    vocab.freeze()

    assert vocab.manager is None
    assert vocab.is_shared is False
    assert isinstance(vocab.state.master, list)
    assert vocab.snapshot() == ["ALPHA", "BETA"]


def test_vocabulary_callback_freezes_model_vocabularies_on_fit_end():
    model = Model(schema=_schema(), batch_size=2)
    address = Address("root", "label")
    vocab = model.nodes[address].embedder.vocab

    vocab.share()

    assert vocab.is_shared is True

    VocabularySyncCallback().on_fit_end(trainer=None, pl_module=model)

    assert vocab.is_shared is False
    assert isinstance(vocab.state.master, list)


def test_runtime_placement_callback_moves_module_to_root_device() -> None:
    class ModuleStub(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.device = torch.device("cpu")
            self.calls: list[torch.device] = []

        def to(self, *args, **kwargs):
            self.calls.append(kwargs["device"])
            return self

    module = ModuleStub()

    RuntimePlacementCallback().on_train_start(trainer=None, pl_module=module)

    assert module.calls == [torch.device("cpu")]


def test_training_counters_observe_all_encoded_fields() -> None:
    schema = _prediction_schema()
    model = Model(schema=schema, batch_size=2)
    inputs = encode(
        batch=[
            [{"color": "red", "label": "warm"}],
            [{"color": "blue", "label": "cool"}],
        ],
        schema=schema,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    CounterUpdateCallback().on_train_batch_start(trainer=None, pl_module=model, batch=inputs, batch_idx=0)

    address = Address("root", "color")
    field = inputs[address]
    embedder = model.nodes[address].embedder

    expected_state_counts = torch.ones(len(Tokens), dtype=torch.int64)
    expected_state_counts += torch.bincount(field.state.reshape(-1), minlength=len(Tokens))
    assert torch.equal(embedder.counters[TensorKey.state.name].counts.cpu(), expected_state_counts)

    valued = field.state.eq(Tokens.valued.value)
    expected_content_counts = torch.ones(
        schema.requests[address].size,
        dtype=torch.int64,
    )
    expected_content_counts += torch.bincount(
        field.content.masked_select(valued).reshape(-1),
        minlength=schema.requests[address].size,
    )
    assert torch.equal(embedder.counters[TensorKey.content.name].counts.cpu(), expected_content_counts)

    target_address = Address("root", "label")
    target_field = inputs[target_address]
    target_embedder = model.nodes[target_address].embedder

    expected_target_counts = torch.ones(len(Tokens), dtype=torch.int64)
    expected_target_counts += torch.bincount(
        target_field.targets[TensorKey.state].reshape(-1),
        minlength=len(Tokens),
    )
    assert torch.equal(target_embedder.counters[TensorKey.state.name].counts.cpu(), expected_target_counts)

    target_valued = target_field.targets[TensorKey.state].eq(Tokens.valued.value)
    expected_target_content_counts = torch.ones(
        schema.requests[target_address].size,
        dtype=torch.int64,
    )
    expected_target_content_counts += torch.bincount(
        target_field.targets[TensorKey.content].masked_select(target_valued).reshape(-1),
        minlength=schema.requests[target_address].size,
    )
    assert torch.equal(
        target_embedder.counters[TensorKey.content.name].counts.cpu(),
        expected_target_content_counts,
    )


def test_training_counters_call_content_counter_for_empty_updates() -> None:
    class SpyCounter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[torch.Tensor] = []

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            self.calls.append(values.detach().cpu())
            return values

    schema = _prediction_schema()
    model = Model(schema=schema, batch_size=2)
    inputs = encode(
        batch=[
            [{"color": "red", "label": "warm"}],
            [{"color": "blue", "label": "cool"}],
        ],
        schema=schema,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    address = Address("root", "color")
    field = inputs[address]
    field.state.fill_(Tokens.null.value)
    spy = SpyCounter()
    model.nodes[address].embedder.counters[TensorKey.content.name] = spy

    CounterUpdateCallback().on_train_batch_start(trainer=None, pl_module=model, batch=inputs, batch_idx=0)

    assert len(spy.calls) == 1
    assert spy.calls[0].numel() == 0


def test_track_marks_metric_sync_handled_without_collective(monkeypatch) -> None:
    calls = []

    def log(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(Model, "log", log)
    model = Model(schema=_schema(), batch_size=2)
    value = torch.tensor(1.0, requires_grad=True)

    assert model.track(("loss", "train"), value=value) is value

    assert len(calls) == 1
    assert calls[0]["value"] is not value
    assert calls[0]["value"].requires_grad is False
    assert calls[0]["sync_dist"] is True
    assert calls[0]["rank_zero_only"] is True


def test_training_step_returns_only_loss_to_avoid_retaining_prediction_graphs(monkeypatch) -> None:
    monkeypatch.setattr(Model, "log", lambda self, **kwargs: None)
    schema = _prediction_schema()
    model = Model(schema=schema, batch_size=2)
    inputs = encode(
        batch=[
            [{"color": "red", "label": "warm"}],
            [{"color": "blue", "label": "cool"}],
        ],
        schema=schema,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    output = model.training_step(inputs, 0)

    assert set(output) == {"loss"}
    assert output["loss"].requires_grad


def test_inactive_leaf_nodes_are_ignored_by_encoding_and_forward() -> None:
    model = Model(
        schema=Schema(
            d_model=8,
            fields={
                "name": "root",
                "type": "branch",
                "embed": True,
                "length": 1,
                "attention": "none",
                "fields": [
                    {
                        "name": "color",
                        "type": "category",
                        "query": "[*].color",
                        "size": 16,
                    },
                    {
                        "name": "ignored",
                        "type": "category",
                        "query": "[*].ignored",
                        "active": False,
                        "embed": True,
                        "p_prune": 1.0,
                        "size": 16,
                    },
                ],
            },
        ),
        batch_size=2,
    )

    inputs = encode(
        batch=[
            [{"color": "red"}],
            [{"color": "blue"}],
        ],
        schema=model.schema,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )
    predictions = model(inputs, strata=Strata.train)

    assert Address("root", "ignored") not in inputs.keys()
    assert Address("root", "ignored") in model.nodes
    assert Address("root", "ignored") not in model.schema.active_requests
    assert Address("root", "ignored") not in model.schema.target
    assert Address("root", "ignored") not in model.schema.embed
    assert all(prediction.address != Address("root", "ignored") for prediction in predictions)


def test_predict_encodes_batch_and_returns_supervised_outputs() -> None:
    model = _primed_prediction_model()
    model.train()

    supervised = model.predict(
        batch=[
            [{"color": "red"}],
            [{"color": "blue"}],
        ]
    )

    assert model.training
    assert Address("root", "label") in supervised
    content = supervised[Address("root", "label")][TensorKey.content.name]
    state = supervised[Address("root", "label")][TensorKey.state.name]

    assert len(content[TensorKey.value.name]) == 2
    assert all(not isinstance(value, list) for value in content[TensorKey.value.name])
    assert all(not isinstance(probability, list) for probability in content[TensorKey.probability.name])
    assert len(content[TensorKey.topk.name]) == 2
    assert all(row and isinstance(row[0], dict) for row in content[TensorKey.topk.name])
    assert all(
        len(probabilities) == 2 and all(not isinstance(probability, list) for probability in probabilities)
        for probabilities in state.values()
    )
    assert supervised[Address("root", "label")][TensorKey.inferred.name] == [True, True]


def test_encode_returns_tensorfield_inputs_for_raw_batch() -> None:
    model = _primed_prediction_model()

    inputs = model.encode(
        batch=[
            {"color": "red"},
            {"color": "blue"},
        ]
    )

    color = inputs[Address("root", "color")]
    label = inputs[Address("root", "label")]

    assert TensorKey.metadata in inputs.keys()
    assert inputs[TensorKey.metadata] == [[{"color": "red"}], [{"color": "blue"}]]
    assert torch.equal(
        color.state,
        torch.tensor([[Tokens.valued.value], [Tokens.valued.value]], dtype=torch.int64),
    )
    assert torch.equal(label.state, torch.full((2, 1), Tokens.masked.value))


def test_encode_branch_tail_overflow_keeps_last_values() -> None:
    model = rf.Model(
        rf.Branch(
            rf.Number("amount"),
            name="events",
            length=2,
            overflow="tail",
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    inputs = model.encode(
        batch=[
            {
                "events": [
                    {"amount": 1.0},
                    {"amount": 2.0},
                    {"amount": 3.0},
                ]
            }
        ]
    )

    amount = inputs[Address("record", "events", "amount")]
    assert torch.equal(amount.content, torch.tensor([[[2.0, 3.0]]]))


def test_encode_branch_error_overflow_raises() -> None:
    model = rf.Model(
        rf.Branch(
            rf.Number("amount"),
            name="events",
            length=2,
            overflow="error",
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    with pytest.raises(ValueError, match="branch overflow at dimension 2 for record/events/amount"):
        model.encode(
            batch=[
                {
                    "events": [
                        {"amount": 1.0},
                        {"amount": 2.0},
                        {"amount": 3.0},
                    ]
                }
            ]
        )


def test_encode_allows_null_inputs_by_default() -> None:
    model = rf.Model(
        amount=rf.Number,
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    inputs = model.encode(batch=[{"amount": None}])
    amount = inputs[Address("record", "amount")]

    assert model.schema.requests[Address("record", "amount")].nullable is True
    assert torch.equal(amount.state, torch.tensor([[Tokens.null.value]], dtype=torch.int64))


def test_encode_nullable_false_rejects_null_inputs() -> None:
    model = rf.Model(
        amount=rf.Number(nullable=False),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    with pytest.raises(ValueError, match="record/amount.*nullable=False.*1 null"):
        model.encode(batch=[{"amount": None}])


def test_encode_accepts_preprocess() -> None:
    @rf.preprocess
    def __root_helper_preprocess(observation: dict):
        return rf.Observation({"color": observation["hue"]})

    model = _primed_prediction_model()

    inputs = model.encode(
        batch=[
            {"hue": "red"},
            {"hue": "blue"},
        ],
        preprocess=__root_helper_preprocess,
    )

    assert inputs[TensorKey.metadata] == [[{"color": "red"}], [{"color": "blue"}]]
    assert torch.equal(
        inputs[Address("root", "color")].state,
        torch.tensor([[Tokens.valued.value], [Tokens.valued.value]], dtype=torch.int64),
    )


def test_encode_accepts_strata_for_testing_training_inputs() -> None:
    model = _primed_prediction_model()

    inputs = model.encode(
        batch=[
            {"color": "red", "label": "warm"},
            {"color": "blue", "label": "cool"},
        ],
        strata=Strata.train,
    )

    assert TensorKey.metadata not in inputs.keys()
    assert torch.equal(
        inputs[Address("root", "label")].targets[TensorKey.state],
        torch.tensor([[Tokens.valued.value], [Tokens.valued.value]], dtype=torch.int64),
    )
    assert torch.equal(
        inputs[Address("root", "label")].state,
        torch.full((2, 1), Tokens.masked.value),
    )


def test_encode_mask_false_skips_training_target_masking() -> None:
    model = _primed_prediction_model()

    inputs = model.encode(
        batch=[
            {"color": "red", "label": "warm"},
            {"color": "blue", "label": "cool"},
        ],
        strata=Strata.train,
        mask=False,
    )
    label = inputs[Address("root", "label")]

    assert TensorKey.metadata not in inputs.keys()
    assert list(label.targets.keys()) == []
    assert not label.trainable.any()
    assert torch.equal(
        label.state,
        torch.tensor([[Tokens.valued.value], [Tokens.valued.value]], dtype=torch.int64),
    )


def test_predict_encodes_batch_and_returns_embedding_outputs() -> None:
    model = _primed_prediction_model()

    predictions = model.predict(
        batch=[
            [{"color": "red"}],
            [{"color": "blue"}],
        ]
    )

    assert Address("root") in predictions
    embedding = predictions[Address("root")][TensorKey.embedding.name]
    assert len(embedding) == 2
    assert all(not isinstance(row[0], list) for row in embedding)


def test_leaf_embed_uses_decoder_pooled_embedding() -> None:
    class ConstantPool(torch.nn.Module):
        def forward(self, memory: torch.Tensor) -> torch.Tensor:
            return torch.ones(memory.shape[0], 1, memory.shape[-1], device=memory.device)

    schema = Schema.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "branch",
                "length": 1,
                "fields": [
                    {
                        "name": "color",
                        "type": "category",
                        "query": "[*].color",
                        "embed": True,
                        "size": 16,
                    }
                ],
            },
        }
    )
    model = Model(schema=schema, batch_size=2)
    model.nodes[Address("root", "color")].decoder.pool = ConstantPool()

    inputs = model.encode(
        batch=[
            [{"color": "red"}],
            [{"color": "blue"}],
        ]
    )
    predictions = model(inputs, strata=Strata.predict)
    prediction = next(item for item in predictions if item.address == Address("root", "color"))

    assert TensorKey.embedding in prediction.payload.keys()
    assert torch.equal(prediction.payload[TensorKey.embedding], torch.ones(2, 1, 8))


def test_inference_helpers_accept_postprocess() -> None:
    model = _primed_prediction_model()
    calls = []

    @rf.postprocess
    def postprocess(predictions, *, batch, input, metadata):
        calls.append((batch, input, metadata, predictions))
        return {
            Address("root", "label"): {"value": ["postprocessed"]},
            Address("root"): {"embedding": [[1.0, 2.0]]},
        }

    batch = [
        [{"color": "red"}],
        [{"color": "blue"}],
    ]

    predictions = model.predict(batch=batch, postprocess=postprocess)

    assert len(calls) == 1
    assert calls[0][0] is batch
    assert TensorKey.metadata in calls[0][1].keys()
    assert list(calls[0][2]) == batch
    assert Address("root", "label") in calls[0][3]
    assert predictions[Address("root", "label")][TensorKey.value.name] == ["postprocessed"]
    assert predictions[Address("root")][TensorKey.embedding.name] == [[1.0, 2.0]]


def test_inference_helpers_accept_preprocess() -> None:
    @rf.preprocess
    def __root_helper_preprocess(observation: dict):
        return rf.Observation({"color": observation["hue"]})

    model = _primed_prediction_model()

    supervised = model.predict(
        batch=[
            {"hue": "red"},
            {"hue": "blue"},
        ],
        preprocess=__root_helper_preprocess,
    )

    assert Address("root", "label") in supervised

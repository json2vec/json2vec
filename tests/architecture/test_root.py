from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import lightning.pytorch as lit
import lightning.pytorch.overrides.distributed as lit_overrides_distributed
import polars as pl
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy

import json2vec as jv
import json2vec as j2v
from json2vec import distributed as j2v_distributed
from json2vec.architecture.root import (
    Model,
    MutationLockCallback,
    RollbackCheckpoint,
    RuntimePlacementCallback,
    _BypassDDPWrappingCallback,
)
from json2vec.data.iterables import encode
from json2vec.logging.throughput import ThroughputLogger
from json2vec.structs.enums import AttentionMode, Strata, TensorKey, Tokens
from json2vec.structs.experiment import Schema
from json2vec.structs.tree import Address
from json2vec.tensorfields.shared.counter import CounterUpdateCallback
from json2vec.tensorfields.shared.vocabulary import (
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
        Model(jv.Number("amount"), d_model=8)


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
                _BypassDDPWrappingCallback(),
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
    class OptimizerStub:
        def zero_grad(self) -> None:
            pass

        def step(self) -> None:
            pass

    def optimizers(self):
        return OptimizerStub()

    def lr_schedulers(self):
        return None

    def manual_backward(self, loss, gradient=None, *args, **kwargs):
        loss.backward(gradient=gradient)

    monkeypatch.setattr(Model, "log", lambda self, **kwargs: None)
    monkeypatch.setattr(Model, "optimizers", optimizers)
    monkeypatch.setattr(Model, "lr_schedulers", lr_schedulers)
    monkeypatch.setattr(Model, "manual_backward", manual_backward)

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
    model = jv.Model(
        jv.Branch(
            jv.Number("amount"),
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
    model = jv.Model(
        jv.Branch(
            jv.Number("amount"),
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
    model = jv.Model(
        amount=jv.Number,
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    inputs = model.encode(batch=[{"amount": None}])
    amount = inputs[Address("record", "amount")]

    assert model.schema.requests[Address("record", "amount")].nullable is True
    assert torch.equal(amount.state, torch.tensor([[Tokens.null.value]], dtype=torch.int64))


def test_encode_nullable_false_rejects_null_inputs() -> None:
    model = jv.Model(
        amount=jv.Number(nullable=False),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    with pytest.raises(ValueError, match="record/amount.*nullable=False.*1 null"):
        model.encode(batch=[{"amount": None}])


def test_encode_accepts_preprocess() -> None:
    @jv.preprocess
    def __root_helper_preprocess(observation: dict):
        return jv.Observation({"color": observation["hue"]})

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

    @jv.postprocess
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
    @jv.preprocess
    def __root_helper_preprocess(observation: dict):
        return jv.Observation({"color": observation["hue"]})

    model = _primed_prediction_model()

    supervised = model.predict(
        batch=[
            {"hue": "red"},
            {"hue": "blue"},
        ],
        preprocess=__root_helper_preprocess,
    )

    assert Address("root", "label") in supervised


def _multi_loss_hyperparameters() -> Hyperparameters:
    return Hyperparameters(
        d_model=8,
        fields={
            "name": "root",
            "type": "array",
            "embed": True,
            "max_length": 1,
            "attention": "none",
            "fields": [
                {
                    "name": "amount",
                    "type": "number",
                    "query": "[*].amount",
                },
                {
                    "name": "color",
                    "type": "category",
                    "query": "[*].color",
                    "embed": False,
                    "p_prune": 1.0,
                    "max_vocab_size": 16,
                },
                {
                    "name": "label",
                    "type": "category",
                    "query": "[*].label",
                    "embed": False,
                    "p_prune": 1.0,
                    "max_vocab_size": 16,
                    "topk": [2],
                },
            ],
        },
    )


def _multi_loss_batch(model: Model) -> dict:
    return encode(
        batch=[
            [{"amount": 1.5, "color": "red", "label": "warm"}],
            [{"amount": 2.0, "color": "blue", "label": "cool"}],
        ],
        hyperparameters=model.hyperparameters,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )


def _patch_manual_optimization(
    monkeypatch: pytest.MonkeyPatch,
    *,
    optimizer: object,
    scheduler: object | None = None,
) -> None:
    def optimizers(self):
        return optimizer

    def lr_schedulers(self):
        return scheduler

    def log(self, **kwargs):
        return None

    def manual_backward(self, loss, gradient=None, *args, **kwargs):
        loss.backward(gradient=gradient)

    monkeypatch.setattr(Model, "optimizers", optimizers)
    monkeypatch.setattr(Model, "lr_schedulers", lr_schedulers)
    monkeypatch.setattr(Model, "log", log)
    monkeypatch.setattr(Model, "manual_backward", manual_backward)


def test_training_step_handles_batch_without_trainable_fields(monkeypatch) -> None:
    class OptimizerStub:
        def zero_grad(self) -> None:
            pass

        def step(self) -> None:
            pass

    _patch_manual_optimization(monkeypatch, optimizer=OptimizerStub())
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    inputs = model.encode(
        [{"amount": 1.0}, {"amount": 2.0}],
        strata=Strata.train,
    )

    output = model.training_step(inputs, 0)

    loss = output["loss"]
    assert loss.shape == (1,)
    assert loss.requires_grad
    assert torch.equal(loss.detach(), torch.zeros(1))


def test_training_step_with_multi_loss_produces_finite_weighted_gradients(monkeypatch) -> None:
    class OptimizerStub:
        def zero_grad(self) -> None:
            pass

        def step(self) -> None:
            pass

    _patch_manual_optimization(monkeypatch, optimizer=OptimizerStub())
    model = Model(hyperparameters=_multi_loss_hyperparameters(), batch_size=2, distributed_jd="manual_allreduce")

    captured_jacobians: list[torch.Tensor] = []
    captured_weights: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured_jacobians.append(inputs[0].detach().clone())
        captured_weights.append(output.detach().clone())

    handle = model._jd_aggregation.weighting.register_forward_hook(capture)
    try:
        output = model.training_step(_multi_loss_batch(model), 0)
    finally:
        handle.remove()

    loss_vec = output["loss"]
    assert loss_vec.ndim == 1
    assert loss_vec.shape[0] == 2
    assert torch.isfinite(loss_vec).all()

    assert len(captured_weights) == 1
    jacobian = captured_jacobians[0]
    weights = captured_weights[0]

    assert jacobian.ndim == 2
    assert jacobian.shape[0] == loss_vec.shape[0]

    assert weights.shape == loss_vec.shape
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()

    grad_count = sum(
        1
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None and torch.any(parameter.grad != 0)
    )
    assert grad_count > 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_training_step_steps_optimizer_then_zeros_grads(monkeypatch) -> None:
    class RecordingOptimizer:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def zero_grad(self) -> None:
            self.calls.append("zero_grad")

        def step(self) -> None:
            self.calls.append("step")

    optimizer = RecordingOptimizer()
    _patch_manual_optimization(monkeypatch, optimizer=optimizer)
    model = Model(hyperparameters=_multi_loss_hyperparameters(), batch_size=2)

    model.training_step(_multi_loss_batch(model), 0)

    assert optimizer.calls == ["step", "zero_grad"]


def test_training_step_steps_scheduler_after_optimizer(monkeypatch) -> None:
    class RecordingOptimizer:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def zero_grad(self) -> None:
            self.calls.append("zero_grad")

        def step(self) -> None:
            self.calls.append("step")

    class RecordingScheduler:
        def __init__(self, optimizer: RecordingOptimizer) -> None:
            self.optimizer = optimizer
            self.optimizer_calls_at_step: list[str] = []

        def step(self) -> None:
            self.optimizer_calls_at_step = list(self.optimizer.calls)

    optimizer = RecordingOptimizer()
    scheduler = RecordingScheduler(optimizer)
    _patch_manual_optimization(monkeypatch, optimizer=optimizer, scheduler=scheduler)
    model = Model(hyperparameters=_multi_loss_hyperparameters(), batch_size=2)

    model.training_step(_multi_loss_batch(model), 0)

    assert optimizer.calls == ["step", "zero_grad"]
    assert scheduler.optimizer_calls_at_step == ["step", "zero_grad"]


def test_training_step_accumulates_grads_across_micro_batches(monkeypatch) -> None:
    class RecordingOptimizer:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def zero_grad(self) -> None:
            self.calls.append("zero_grad")

        def step(self) -> None:
            self.calls.append("step")

    optimizer = RecordingOptimizer()
    all_reduce_calls: list[torch.nn.Module] = []

    def fake_all_reduce(module: torch.nn.Module) -> None:
        all_reduce_calls.append(module)

    monkeypatch.setattr("json2vec.architecture.root.mean_all_reduce_grads", fake_all_reduce)
    _patch_manual_optimization(monkeypatch, optimizer=optimizer)
    model = Model(hyperparameters=_multi_loss_hyperparameters(), batch_size=2, distributed_jd="manual_allreduce")
    model._trainer = type("TrainerStub", (), {"accumulate_grad_batches": 3})()  # noqa: SLF001

    model.training_step(_multi_loss_batch(model), 0)
    assert optimizer.calls == []
    assert all_reduce_calls == []

    model.training_step(_multi_loss_batch(model), 1)
    assert optimizer.calls == []
    assert all_reduce_calls == []

    # Third micro-batch closes the accumulation window.
    model.training_step(_multi_loss_batch(model), 2)
    assert optimizer.calls == ["step", "zero_grad"]
    assert all_reduce_calls == [model]


def test_training_step_accumulates_grads_across_micro_batches_without_distributed_jd(monkeypatch) -> None:
    class RecordingOptimizer:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def zero_grad(self) -> None:
            self.calls.append("zero_grad")

        def step(self) -> None:
            self.calls.append("step")

    optimizer = RecordingOptimizer()
    all_reduce_calls: list[torch.nn.Module] = []

    def fake_all_reduce(module: torch.nn.Module) -> None:
        all_reduce_calls.append(module)

    monkeypatch.setattr("json2vec.architecture.root.mean_all_reduce_grads", fake_all_reduce)
    _patch_manual_optimization(monkeypatch, optimizer=optimizer)
    model = Model(
        hyperparameters=_multi_loss_hyperparameters(),
        batch_size=2,
        distributed_jd="off",
    )
    model._trainer = type("TrainerStub", (), {"accumulate_grad_batches": 2})()  # noqa: SLF001

    model.training_step(_multi_loss_batch(model), 0)
    assert optimizer.calls == []

    model.training_step(_multi_loss_batch(model), 1)
    assert optimizer.calls == ["step", "zero_grad"]
    # With JD disabled, gradient sync is the user's responsibility (stock DDP
    # would have synced via its reducer); no manual all-reduce runs.
    assert all_reduce_calls == []


def test_training_step_after_schema_extend_includes_new_field_loss(monkeypatch) -> None:
    class OptimizerStub:
        def zero_grad(self) -> None:
            pass

        def step(self) -> None:
            pass

    _patch_manual_optimization(monkeypatch, optimizer=OptimizerStub())
    model = Model(hyperparameters=_multi_loss_hyperparameters(), batch_size=2)
    model.on_fit_start()
    baseline = model.training_step(_multi_loss_batch(model), 0)
    assert baseline["loss"].shape[0] == 2

    model.extend(
        j2v.where("name") == "root",
        j2v.Category("vehicle", p_prune=1.0, max_vocab_size=8),
    )
    inputs = encode(
        batch=[
            [{"amount": 1.5, "color": "red", "label": "warm", "vehicle": "car"}],
            [{"amount": 2.0, "color": "blue", "label": "cool", "vehicle": "boat"}],
        ],
        hyperparameters=model.hyperparameters,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    model.on_fit_start()
    output = model.training_step(inputs, 0)

    assert output["loss"].shape[0] == baseline["loss"].shape[0] + 1
    assert torch.isfinite(output["loss"]).all()

    vehicle_grads = [
        parameter.grad
        for parameter in model.nodes["root/vehicle"].parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert any(torch.any(grad != 0) for grad in vehicle_grads)


def test_trainer_fit_advances_parameters_under_jd_manual_optimization() -> None:
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        j2v.Category("color", target=True, max_vocab_size=16),
        j2v.Category("label", target=True, max_vocab_size=16),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
        optimizer=lambda module: torch.optim.SGD(module.parameters(), lr=1e-2),
    )
    frame = pl.DataFrame(
        {
            "amount": [1.5, 2.0, 0.75],
            "color": ["red", "blue", "green"],
            "label": ["warm", "cool", "warm"],
        }
    )
    datamodule = j2v.PolarsDataModule(
        model=model,
        train=frame,
        validate=frame,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    parameters_before_fit = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    trainer = lit.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(model=model, datamodule=datamodule)

    assert model.automatic_optimization is False
    changed = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(parameter.detach(), parameters_before_fit[name])
    ]
    assert len(changed) > 0


def test_configure_callbacks_attaches_bypass_callback_when_jd_enabled() -> None:
    model = Model(hyperparameters=_hyperparameters(), batch_size=2, distributed_jd="manual_allreduce")

    callbacks = model.configure_callbacks()

    assert sum(isinstance(callback, _BypassDDPWrappingCallback) for callback in callbacks) == 1


def test_configure_callbacks_omits_bypass_callback_when_distributed_jd_off() -> None:
    model = Model(hyperparameters=_hyperparameters(), batch_size=2, distributed_jd="off")

    callbacks = model.configure_callbacks()

    assert all(not isinstance(callback, _BypassDDPWrappingCallback) for callback in callbacks)


def test_configure_callbacks_skips_bypass_callback_already_attached_to_trainer() -> None:
    model = Model(hyperparameters=_hyperparameters(), batch_size=2)
    model._trainer = type(  # noqa: SLF001
        "TrainerStub",
        (),
        {"callbacks": [_BypassDDPWrappingCallback()]},
    )()

    callbacks = model.configure_callbacks()

    assert all(not isinstance(callback, _BypassDDPWrappingCallback) for callback in callbacks)


def test_bypass_ddp_wrapping_callback_patches_configure_ddp() -> None:
    strategy = DDPStrategy()
    original_configure_ddp = strategy.configure_ddp
    trainer = type("TrainerStub", (), {"strategy": strategy})()
    module = torch.nn.Linear(1, 1)
    callback = _BypassDDPWrappingCallback()

    callback.setup(trainer=trainer, pl_module=module, stage="fit")

    assert strategy.configure_ddp is not original_configure_ddp
    # Calling the patched ``configure_ddp`` must not wrap the model or register
    # comm hooks; both would re-introduce the BatchedTensor crash under JD.
    strategy.model = module  # type: ignore[assignment]
    strategy.configure_ddp()
    assert strategy.model is module

    callback.teardown(trainer=trainer, pl_module=module, stage="fit")

    assert strategy.configure_ddp == original_configure_ddp


def test_bypass_ddp_wrapping_callback_skips_setup_model_and_register_hooks(monkeypatch) -> None:
    strategy = DDPStrategy()
    trainer = type("TrainerStub", (), {"strategy": strategy})()
    module = torch.nn.Linear(1, 1)
    strategy.model = module  # type: ignore[assignment]

    setup_model_calls: list[torch.nn.Module] = []
    register_hooks_calls: list[int] = []

    def fake_setup_model(self, model):  # noqa: ARG001
        setup_model_calls.append(model)
        return model

    def fake_register_hooks(self):  # noqa: ARG001
        register_hooks_calls.append(1)

    monkeypatch.setattr(DDPStrategy, "_setup_model", fake_setup_model)
    monkeypatch.setattr(DDPStrategy, "_register_ddp_hooks", fake_register_hooks)

    callback = _BypassDDPWrappingCallback()
    callback.setup(trainer=trainer, pl_module=module, stage="fit")
    strategy.configure_ddp()

    assert setup_model_calls == []
    assert register_hooks_calls == []


def test_bypass_ddp_wrapping_callback_syncs_module_states(monkeypatch) -> None:
    strategy = DDPStrategy()
    trainer = type("TrainerStub", (), {"strategy": strategy})()
    module = torch.nn.Linear(1, 1)
    strategy.model = module  # type: ignore[assignment]

    sync_calls: list[torch.nn.Module] = []

    monkeypatch.setattr("json2vec.architecture.root.is_distributed", lambda: True)
    monkeypatch.setattr(lit_overrides_distributed, "_sync_module_states", sync_calls.append)

    callback = _BypassDDPWrappingCallback()
    callback.setup(trainer=trainer, pl_module=module, stage="fit")
    strategy.configure_ddp()

    assert sync_calls == [module]


def test_bypass_ddp_wrapping_callback_skips_sync_outside_process_group(monkeypatch) -> None:
    strategy = DDPStrategy()
    trainer = type("TrainerStub", (), {"strategy": strategy})()
    module = torch.nn.Linear(1, 1)
    strategy.model = module  # type: ignore[assignment]

    sync_calls: list[torch.nn.Module] = []

    monkeypatch.setattr("json2vec.architecture.root.is_distributed", lambda: False)
    monkeypatch.setattr(lit_overrides_distributed, "_sync_module_states", sync_calls.append)

    callback = _BypassDDPWrappingCallback()
    callback.setup(trainer=trainer, pl_module=module, stage="fit")
    strategy.configure_ddp()

    assert sync_calls == []


def test_bypass_ddp_wrapping_callback_is_noop_for_non_ddp_strategy() -> None:
    class StrategyStub:
        pass

    strategy = StrategyStub()
    trainer = type("TrainerStub", (), {"strategy": strategy})()
    module = torch.nn.Linear(1, 1)
    callback = _BypassDDPWrappingCallback()

    callback.setup(trainer=trainer, pl_module=module, stage="fit")
    callback.teardown(trainer=trainer, pl_module=module, stage="fit")

    assert not hasattr(strategy, "configure_ddp")


def test_bypass_ddp_wrapping_callback_rejects_sharded_strategies() -> None:
    trainer = type("TrainerStub", (), {"strategy": FSDPStrategy()})()
    module = torch.nn.Linear(1, 1)

    with pytest.raises(RuntimeError, match="FSDPStrategy"):
        _BypassDDPWrappingCallback().setup(trainer=trainer, pl_module=module, stage="fit")


def test_bypass_ddp_wrapping_callback_rejection_uses_isinstance_not_name() -> None:
    # A local class whose ``__name__`` collides with a real incompatible
    # strategy must not trigger rejection; matching is by ``isinstance``.
    class FSDPStrategy:  # noqa: N801
        pass

    strategy = FSDPStrategy()
    trainer = type("TrainerStub", (), {"strategy": strategy})()
    module = torch.nn.Linear(1, 1)
    callback = _BypassDDPWrappingCallback()

    callback.setup(trainer=trainer, pl_module=module, stage="fit")

    assert not hasattr(strategy, "configure_ddp")


def test_bypass_ddp_wrapping_callback_rejects_ddp_with_comm_hook() -> None:
    strategy = DDPStrategy()
    strategy._ddp_comm_hook = lambda state, bucket: None  # noqa: SLF001
    trainer = type("TrainerStub", (), {"strategy": strategy})()
    module = torch.nn.Linear(1, 1)

    with pytest.raises(RuntimeError, match="ddp_comm_hook"):
        _BypassDDPWrappingCallback().setup(trainer=trainer, pl_module=module, stage="fit")


def test_training_step_calls_mean_all_reduce_grads(monkeypatch) -> None:
    class OptimizerStub:
        def zero_grad(self) -> None:
            pass

        def step(self) -> None:
            pass

    calls: list[torch.nn.Module] = []

    def fake_all_reduce(module: torch.nn.Module) -> None:
        calls.append(module)

    monkeypatch.setattr("json2vec.architecture.root.mean_all_reduce_grads", fake_all_reduce)
    _patch_manual_optimization(monkeypatch, optimizer=OptimizerStub())
    model = Model(hyperparameters=_multi_loss_hyperparameters(), batch_size=2, distributed_jd="manual_allreduce")

    model.training_step(_multi_loss_batch(model), 0)

    assert calls == [model]


def test_training_step_skips_all_reduce_when_distributed_jd_off(monkeypatch) -> None:
    class OptimizerStub:
        def zero_grad(self) -> None:
            pass

        def step(self) -> None:
            pass

    calls: list[torch.nn.Module] = []

    def fake_all_reduce(module: torch.nn.Module) -> None:
        calls.append(module)

    monkeypatch.setattr("json2vec.architecture.root.mean_all_reduce_grads", fake_all_reduce)
    _patch_manual_optimization(monkeypatch, optimizer=OptimizerStub())
    model = Model(
        hyperparameters=_multi_loss_hyperparameters(),
        batch_size=2,
        distributed_jd="off",
    )

    output = model.training_step(_multi_loss_batch(model), 0)

    assert calls == []
    assert set(output) == {"loss"}
    assert output["loss"].requires_grad


def test_mean_all_reduce_grads_is_noop_without_process_group() -> None:
    module = torch.nn.Linear(2, 3)
    for parameter in module.parameters():
        parameter.grad = torch.ones_like(parameter)

    j2v_distributed.mean_all_reduce_grads(module)

    for parameter in module.parameters():
        assert torch.equal(parameter.grad, torch.ones_like(parameter))


def test_mean_all_reduce_grads_averages_across_ranks(monkeypatch) -> None:
    module = torch.nn.Linear(2, 3)
    for parameter in module.parameters():
        parameter.grad = torch.full_like(parameter, 4.0)

    all_reduce_calls: list[tuple[torch.Tensor, bool]] = []

    class _WorkStub:
        def __init__(self) -> None:
            self.waited = False

        def wait(self) -> None:
            self.waited = True

    works: list[_WorkStub] = []

    def fake_all_reduce(tensor: torch.Tensor, op=None, async_op: bool = False):
        all_reduce_calls.append((tensor, async_op))
        tensor.mul_(2.0)
        work = _WorkStub()
        works.append(work)
        return work

    monkeypatch.setattr(j2v_distributed, "is_distributed", lambda: True)
    monkeypatch.setattr(j2v_distributed, "world_size", lambda: 2)
    monkeypatch.setattr(j2v_distributed, "_backend_supports_avg", lambda: False)
    monkeypatch.setattr(j2v_distributed.dist, "all_reduce", fake_all_reduce)

    j2v_distributed.mean_all_reduce_grads(module)

    # All parameters share (float32, cpu) and fit under the default cap, so
    # coalescing collapses the per-parameter sequence into one collective.
    assert len(all_reduce_calls) == 1
    tensor, async_op = all_reduce_calls[0]
    assert async_op is True
    assert all(work.waited for work in works)
    # fake_all_reduce doubles the buffer; we then divide by world_size and copy
    # back, giving 4.0 * 2.0 / 2 == 4.0 per element.
    for parameter in module.parameters():
        assert torch.equal(parameter.grad, torch.full_like(parameter, 4.0))


def test_mean_all_reduce_grads_buckets_by_dtype(monkeypatch) -> None:
    class MixedModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.f32 = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
            self.f64 = torch.nn.Parameter(torch.zeros(4, dtype=torch.float64))

    module = MixedModule()
    for parameter in module.parameters():
        parameter.grad = torch.full_like(parameter, 3.0)

    bucket_sizes: list[int] = []

    def fake_all_reduce(tensor: torch.Tensor, op=None, async_op: bool = False):
        bucket_sizes.append(tensor.numel())
        return None

    monkeypatch.setattr(j2v_distributed, "is_distributed", lambda: True)
    monkeypatch.setattr(j2v_distributed, "world_size", lambda: 2)
    monkeypatch.setattr(j2v_distributed, "_backend_supports_avg", lambda: False)
    monkeypatch.setattr(j2v_distributed.dist, "all_reduce", fake_all_reduce)

    j2v_distributed.mean_all_reduce_grads(module)

    # One bucket per dtype: collective ops require a uniform dtype.
    assert len(bucket_sizes) == 2
    assert sorted(bucket_sizes) == [4, 4]


def test_mean_all_reduce_grads_skips_parameters_without_grad(monkeypatch) -> None:
    module = torch.nn.Linear(2, 3)
    module.weight.grad = torch.full_like(module.weight, 1.0)
    assert module.bias.grad is None

    all_reduce_calls: list[torch.Tensor] = []

    def fake_all_reduce(tensor: torch.Tensor, op=None, async_op: bool = False):
        all_reduce_calls.append(tensor)
        return None

    monkeypatch.setattr(j2v_distributed, "is_distributed", lambda: True)
    monkeypatch.setattr(j2v_distributed, "world_size", lambda: 1)
    monkeypatch.setattr(j2v_distributed, "_backend_supports_avg", lambda: False)
    monkeypatch.setattr(j2v_distributed.dist, "all_reduce", fake_all_reduce)

    j2v_distributed.mean_all_reduce_grads(module)

    assert len(all_reduce_calls) == 1
    assert all_reduce_calls[0].numel() == module.weight.numel()
    assert module.bias.grad is None


def test_mean_all_reduce_grads_splits_large_dtype_group_by_bucket_bytes(monkeypatch) -> None:
    class WideModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Parameter(torch.zeros(64, dtype=torch.float32))
            self.b = torch.nn.Parameter(torch.zeros(64, dtype=torch.float32))
            self.c = torch.nn.Parameter(torch.zeros(64, dtype=torch.float32))

    module = WideModule()
    for parameter in module.parameters():
        parameter.grad = torch.full_like(parameter, 1.0)

    bucket_sizes: list[int] = []

    def fake_all_reduce(tensor: torch.Tensor, op=None, async_op: bool = False):
        bucket_sizes.append(tensor.numel())
        return None

    monkeypatch.setattr(j2v_distributed, "is_distributed", lambda: True)
    monkeypatch.setattr(j2v_distributed, "world_size", lambda: 1)
    monkeypatch.setattr(j2v_distributed, "_backend_supports_avg", lambda: False)
    monkeypatch.setattr(j2v_distributed.dist, "all_reduce", fake_all_reduce)

    # Each float32 tensor is 64 * 4 = 256 bytes; a 300-byte cap forces one
    # grad per bucket.
    j2v_distributed.mean_all_reduce_grads(module, bucket_bytes=300)

    assert bucket_sizes == [64, 64, 64]
    for parameter in module.parameters():
        assert torch.equal(parameter.grad, torch.full_like(parameter, 1.0))


def test_mean_all_reduce_grads_uses_avg_op_when_backend_supports_it(monkeypatch) -> None:
    module = torch.nn.Linear(2, 3)
    for parameter in module.parameters():
        parameter.grad = torch.full_like(parameter, 6.0)

    captured_ops: list[Any] = []

    def fake_all_reduce(tensor: torch.Tensor, op=None, async_op: bool = False):
        captured_ops.append(op)
        # AVG semantics: simulate division by world_size inside the collective.
        tensor.div_(2.0)
        return None

    monkeypatch.setattr(j2v_distributed, "is_distributed", lambda: True)
    monkeypatch.setattr(j2v_distributed, "world_size", lambda: 2)
    monkeypatch.setattr(j2v_distributed, "_backend_supports_avg", lambda: True)
    monkeypatch.setattr(j2v_distributed.dist, "all_reduce", fake_all_reduce)

    j2v_distributed.mean_all_reduce_grads(module)

    assert captured_ops == [j2v_distributed.dist.ReduceOp.AVG]
    # Divided exactly once (inside the collective), not twice.
    for parameter in module.parameters():
        assert torch.equal(parameter.grad, torch.full_like(parameter, 3.0))


def test_mean_all_reduce_grads_submits_all_buckets_async_before_waiting(monkeypatch) -> None:
    class WideModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = torch.nn.Parameter(torch.zeros(64, dtype=torch.float32))
            self.b = torch.nn.Parameter(torch.zeros(64, dtype=torch.float32))
            self.c = torch.nn.Parameter(torch.zeros(64, dtype=torch.float32))

    module = WideModule()
    for parameter in module.parameters():
        parameter.grad = torch.full_like(parameter, 1.0)

    events: list[str] = []

    class _WorkStub:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def wait(self) -> None:
            events.append(f"wait:{self.tag}")

    counter = {"n": 0}

    def fake_all_reduce(tensor: torch.Tensor, op=None, async_op: bool = False):
        tag = str(counter["n"])
        counter["n"] += 1
        events.append(f"submit:{tag}")
        return _WorkStub(tag)

    monkeypatch.setattr(j2v_distributed, "is_distributed", lambda: True)
    monkeypatch.setattr(j2v_distributed, "world_size", lambda: 1)
    monkeypatch.setattr(j2v_distributed, "_backend_supports_avg", lambda: False)
    monkeypatch.setattr(j2v_distributed.dist, "all_reduce", fake_all_reduce)

    j2v_distributed.mean_all_reduce_grads(module, bucket_bytes=300)

    submits = [event for event in events if event.startswith("submit:")]
    waits = [event for event in events if event.startswith("wait:")]
    assert submits == ["submit:0", "submit:1", "submit:2"]
    assert waits == ["wait:0", "wait:1", "wait:2"]
    assert events.index("submit:2") < events.index("wait:0")

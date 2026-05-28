from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from json2vec.architecture.root import Model, MutationLockCallback, RollbackCheckpoint, RuntimePlacementCallback
from json2vec.data.iterables import encode
from json2vec.structs.enums import AttentionMode, Strata, TensorKey, Tokens
from json2vec.structs.experiment import Hyperparameters
from json2vec.structs.tree import Address
from json2vec.tensorfields.shared.counter import CounterUpdateCallback
from json2vec.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularySyncCallback


def _hyperparameters() -> Hyperparameters:
    return Hyperparameters.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "array",
                "dropout": 0.1,
                "max_length": 1,
                "n_outputs": 1,
                "fields": [
                    {
                        "name": "label",
                        "type": "category",
                        "query": "[*].label",
                        "max_vocab_size": 32,
                    }
                ],
            },
        }
    )


def test_on_save_checkpoint_serializes_hyperparameters() -> None:
    hyperparameters = _hyperparameters()
    model = Model(hyperparameters=hyperparameters, batch_size=2)
    checkpoint = {}

    model.on_save_checkpoint(checkpoint)

    restored = Hyperparameters.model_validate(checkpoint["hyperparameters"])
    assert restored.model_dump(mode="python") == hyperparameters.model_dump(mode="python")
    assert checkpoint["batch_size"] == 2


def test_save_writes_loadable_checkpoint(tmp_path: Path) -> None:
    hyperparameters = _hyperparameters()
    model = Model(hyperparameters=hyperparameters, batch_size=2)
    pathname = tmp_path / "nested" / "model.ckpt"

    model.save(pathname=pathname)

    restored = Model.load(pathname)

    assert pathname.exists()
    assert restored.batch_size == 2
    assert restored.hyperparameters.model_dump(mode="python") == hyperparameters.model_dump(mode="python")

    restored_state = restored.state_dict()
    for key, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(restored_state[key], value)
        else:
            assert restored_state[key] == value


def _prediction_hyperparameters() -> Hyperparameters:
    return Hyperparameters(
        d_model=8,
        fields={
            "name": "root",
            "type": "array",
            "embed": True,
            "max_length": 1,
            "n_outputs": 1,
            "attention": "none",
            "fields": [
                {
                    "name": "color",
                    "type": "category",
                    "query": "[*].color",
                    "embed": False,
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


def _primed_prediction_model() -> Model:
    hyperparameters = _prediction_hyperparameters()
    model = Model(hyperparameters=hyperparameters, batch_size=2)
    inputs = encode(
        batch=[
            [{"color": "red", "label": "warm"}],
            [{"color": "blue", "label": "cool"}],
        ],
        hyperparameters=hyperparameters,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    model(inputs, strata=Strata.train)
    return model


def _build_checkpoint(tmp_path: Path) -> tuple[Path, Hyperparameters]:
    hyperparameters = _hyperparameters()
    model = Model(hyperparameters=hyperparameters, batch_size=2)
    checkpoint_path = tmp_path / "model.ckpt"
    model.save(checkpoint_path)

    return checkpoint_path, hyperparameters


def test_load_restores_local_checkpoint(tmp_path: Path) -> None:
    checkpoint_path, hyperparameters = _build_checkpoint(tmp_path)

    model = Model.load(checkpoint_path)

    assert model.batch_size == 2
    assert model.hyperparameters.model_dump(mode="python") == hyperparameters.model_dump(mode="python")


def test_rollback_checkpoint_restores_best_model_from_disk(tmp_path: Path) -> None:
    model = Model(hyperparameters=_hyperparameters(), batch_size=2)
    best_path = tmp_path / "best.ckpt"
    model.save(best_path)
    best_state = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
        for key, value in model.state_dict().items()
    }
    best_hyperparameters = model.hyperparameters.model_dump(mode="python")
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
    assert model.hyperparameters.model_dump(mode="python") == best_hyperparameters
    assert model.nodes[address] is not mutated_node
    for key, value in best_state.items():
        restored = model.state_dict()[key]
        if isinstance(value, torch.Tensor):
            assert torch.equal(restored, value)
        else:
            assert restored == value


def test_rollback_checkpoint_loads_schema_metadata_with_weights_only_disabled(tmp_path: Path) -> None:
    model = Model(hyperparameters=_hyperparameters(), batch_size=2)
    best_path = tmp_path / "best.ckpt"
    model.save(best_path)
    checkpoint = torch.load(best_path, weights_only=False, map_location="cpu")
    assert checkpoint["hyperparameters"]["fields"]["attention"] == AttentionMode.mha

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
    _, hyperparameters = _build_checkpoint(tmp_path)
    model = Model(
        hyperparameters=hyperparameters,
        batch_size=2,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-3),
    )
    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)


def test_configure_optimizers_uses_user_supplied_scheduler(tmp_path: Path) -> None:
    _, hyperparameters = _build_checkpoint(tmp_path)
    model = Model(
        hyperparameters=hyperparameters,
        batch_size=2,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-3),
        scheduler=lambda _module, optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
    )

    configured = model.configure_optimizers()

    assert isinstance(configured["optimizer"], torch.optim.AdamW)
    assert isinstance(configured["lr_scheduler"], torch.optim.lr_scheduler.StepLR)


def test_configure_callbacks_collects_active_extension_callbacks() -> None:
    model = Model(hyperparameters=_hyperparameters(), batch_size=2)

    callbacks = model.configure_callbacks()
    callback_types = [type(callback) for callback in callbacks]

    assert any(isinstance(callback, RuntimePlacementCallback) for callback in callbacks)
    assert any(isinstance(callback, MutationLockCallback) for callback in callbacks)
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
    hyperparameters = Hyperparameters.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "array",
                "max_length": 1,
                "n_outputs": 1,
                "fields": [
                    {
                        "name": "label",
                        "type": "category",
                        "query": "[*].label",
                        "max_vocab_size": 16,
                    },
                    {
                        "name": "tags",
                        "type": "set",
                        "query": "[*].tags",
                        "max_vocab_size": 16,
                    },
                ],
            },
        }
    )
    model = Model(hyperparameters=hyperparameters, batch_size=2)

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

    assert len(runtime_placement_callbacks) == 1
    assert len(mutation_lock_callbacks) == 1
    assert len(vocabulary_callbacks) == 1
    assert len(counter_callbacks) == 1

    callback_types = [type(callback) for callback in model.configure_callbacks()]
    assert callback_types.index(CounterUpdateCallback) < callback_types.index(VocabularySyncCallback)


def test_configure_callbacks_skips_callbacks_already_attached_to_trainer() -> None:
    model = Model(hyperparameters=_hyperparameters(), batch_size=2)
    model._trainer = type(  # noqa: SLF001
        "TrainerStub",
        (),
        {
            "callbacks": [
                RuntimePlacementCallback(),
                MutationLockCallback(),
                VocabularySyncCallback(),
                CounterUpdateCallback(),
            ]
        },
    )()

    assert model.configure_callbacks() == []


def test_builtin_resources_are_attached_to_extension_modules() -> None:
    model = Model(hyperparameters=_hyperparameters(), batch_size=2)
    address = Address("root", "label")

    assert isinstance(model.nodes[address].embedder.vocab, OnlineVocabularyModel)
    assert TensorKey.state.name in model.nodes[address].embedder.counters
    assert TensorKey.content.name in model.nodes[address].embedder.counters


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
    hyperparameters = _prediction_hyperparameters()
    model = Model(hyperparameters=hyperparameters, batch_size=2)
    inputs = encode(
        batch=[
            [{"color": "red", "label": "warm"}],
            [{"color": "blue", "label": "cool"}],
        ],
        hyperparameters=hyperparameters,
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
        hyperparameters.requests[address].max_vocab_size + 1,
        dtype=torch.int64,
    )
    expected_content_counts += torch.bincount(
        field.content.masked_select(valued).reshape(-1),
        minlength=hyperparameters.requests[address].max_vocab_size + 1,
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
        hyperparameters.requests[target_address].max_vocab_size + 1,
        dtype=torch.int64,
    )
    expected_target_content_counts += torch.bincount(
        target_field.targets[TensorKey.content].masked_select(target_valued).reshape(-1),
        minlength=hyperparameters.requests[target_address].max_vocab_size + 1,
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

    hyperparameters = _prediction_hyperparameters()
    model = Model(hyperparameters=hyperparameters, batch_size=2)
    inputs = encode(
        batch=[
            [{"color": "red", "label": "warm"}],
            [{"color": "blue", "label": "cool"}],
        ],
        hyperparameters=hyperparameters,
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
    model = Model(hyperparameters=_hyperparameters(), batch_size=2)
    value = torch.tensor(1.0, requires_grad=True)

    assert model.track(("loss", "train"), value=value) is value

    assert len(calls) == 1
    assert calls[0]["value"] is not value
    assert calls[0]["value"].requires_grad is False
    assert calls[0]["sync_dist"] is True
    assert calls[0]["rank_zero_only"] is True


def test_training_step_returns_only_loss_to_avoid_retaining_prediction_graphs(monkeypatch) -> None:
    monkeypatch.setattr(Model, "log", lambda self, **kwargs: None)
    hyperparameters = _prediction_hyperparameters()
    model = Model(hyperparameters=hyperparameters, batch_size=2)
    inputs = encode(
        batch=[
            [{"color": "red", "label": "warm"}],
            [{"color": "blue", "label": "cool"}],
        ],
        hyperparameters=hyperparameters,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    output = model.training_step(inputs, 0)

    assert set(output) == {"loss"}
    assert output["loss"].requires_grad


def test_inactive_leaf_nodes_are_ignored_by_encoding_and_forward() -> None:
    model = Model(
        hyperparameters=Hyperparameters(
            d_model=8,
            fields={
                "name": "root",
                "type": "array",
                "embed": True,
                "max_length": 1,
                "n_outputs": 1,
                "attention": "none",
                "fields": [
                    {
                        "name": "color",
                        "type": "category",
                        "query": "[*].color",
                        "max_vocab_size": 16,
                    },
                    {
                        "name": "ignored",
                        "type": "category",
                        "query": "[*].ignored",
                        "active": False,
                        "embed": True,
                        "p_prune": 1.0,
                        "max_vocab_size": 16,
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
        hyperparameters=model.hyperparameters,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )
    predictions = model(inputs, strata=Strata.train)

    assert Address("root", "ignored") not in inputs.keys()
    assert Address("root", "ignored") in model.nodes
    assert Address("root", "ignored") not in model.hyperparameters.active_requests
    assert Address("root", "ignored") not in model.hyperparameters.target
    assert Address("root", "ignored") not in model.hyperparameters.embed
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


def test_embed_encodes_batch_and_returns_embedding_outputs() -> None:
    model = _primed_prediction_model()

    embeddings = model.embed(
        batch=[
            [{"color": "red"}],
            [{"color": "blue"}],
        ]
    )

    assert Address("root") in embeddings
    embedding = embeddings[Address("root")][TensorKey.embedding.name]
    assert len(embedding) == 2
    assert all(not isinstance(row[0], list) for row in embedding)


def test_inference_helpers_accept_postprocess() -> None:
    model = _primed_prediction_model()
    calls = []

    def postprocess(context, supervised, embeddings):
        calls.append((context, supervised, embeddings))
        return (
            {Address("root", "label"): {"value": ["postprocessed"]}},
            {Address("root"): {"embedding": [[1.0, 2.0]]}},
        )

    batch = [
        [{"color": "red"}],
        [{"color": "blue"}],
    ]

    supervised = model.predict(batch=batch, postprocess=postprocess)
    embeddings = model.embed(batch=batch, postprocess=postprocess)
    evaluated_supervised, evaluated_embeddings = model.evaluate(batch=batch, postprocess=postprocess)

    assert len(calls) == 3
    assert calls[0][0]["batch"] is batch
    assert TensorKey.metadata in calls[0][0]["input"].keys()
    assert list(calls[0][0][TensorKey.metadata]) == batch
    assert Address("root", "label") in calls[0][1]
    assert Address("root") in calls[1][2]
    assert supervised[Address("root", "label")][TensorKey.value.name] == ["postprocessed"]
    assert embeddings[Address("root")][TensorKey.embedding.name] == [[1.0, 2.0]]
    assert evaluated_supervised == supervised
    assert evaluated_embeddings == embeddings


def test_inference_helpers_accept_preprocess() -> None:
    def __root_helper_preprocess(observation: dict):
        return {"color": observation["hue"]}

    model = _primed_prediction_model()

    supervised = model.predict(
        batch=[
            {"hue": "red"},
            {"hue": "blue"},
        ],
        preprocess=__root_helper_preprocess,
    )

    assert Address("root", "label") in supervised

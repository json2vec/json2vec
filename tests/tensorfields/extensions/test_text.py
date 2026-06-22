import builtins
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.experiment import Schema
from json2vec.structs.packages import Prediction
from json2vec.tensorfields.extensions.text import (
    ATTENTION_MASK,
    DEFAULT_TEXT_MODEL,
    INPUT_IDS,
    CachedModel,
    Decoder,
    Embedder,
    TensorField,
    loss,
    write,
)

ADDRESS = "root/items/body"


def _structure_payload(*, objective: str = "l2", encoder_pooling: str = "cls", encoder_batch_size: int = 2) -> dict:
    field: dict = {
        "name": "body",
        "type": "text",
        "query": "[*].items[*].body",
        "model": "bert-base-uncased",
        "max_length": 4,
        "encoder_batch_size": encoder_batch_size,
        "encoder_pooling": encoder_pooling,
        "objective": objective,
    }
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "dropout": 0.1,
            "fields": [
                {
                    "name": "items",
                    "type": "branch",
                    "length": 2,
                    "fields": [field],
                }
            ],
        },
    }


def _values() -> list:
    return [
        [["alpha", "beta"]],
        [["gamma", "delta"]],
    ]


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "[EOS]"
    sep_token = "[SEP]"

    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"

        token_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []

        for text in texts:
            used = min(max_length, max(1, min(2, len(text))))
            tokens = list(range(len(text), len(text) + used))
            token_rows.append(tokens + [0] * (max_length - used))
            mask_rows.append([1] * used + [0] * (max_length - used))

        return {
            INPUT_IDS: torch.tensor(token_rows, dtype=torch.int64),
            ATTENTION_MASK: torch.tensor(mask_rows, dtype=torch.int64),
        }


class FakeHFModel:
    def __init__(self, hidden_size: int = 4):
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.device = torch.device("cpu")
        self.calls = 0

    def eval(self):
        return self

    def requires_grad_(self, flag: bool):
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.calls += 1
        input_ids = input_ids.to(dtype=torch.float32)
        attention_mask = attention_mask.to(dtype=torch.float32)
        hidden = torch.stack(
            [
                input_ids,
                input_ids + 1.0,
                attention_mask,
                input_ids * attention_mask,
            ],
            dim=-1,
        )
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=hidden[:, 0],
        )


def _patch_hf(monkeypatch: pytest.MonkeyPatch, *, hidden_size: int = 4) -> FakeHFModel:
    fake_model = FakeHFModel(hidden_size=hidden_size)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model: str):
            return fake_model

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model: str):
            return FakeTokenizer()

    monkeypatch.setattr(CachedModel, "_models", {})
    monkeypatch.setattr(CachedModel, "_tokenizers", {})
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=FakeAutoModel, AutoTokenizer=FakeAutoTokenizer),
    )
    return fake_model


def test_text_request_is_available_in_structure():
    structure = Schema.model_validate(_structure_payload())
    request = structure.requests[ADDRESS]
    assert request.type == "text"
    assert request.model == "bert-base-uncased"
    assert request.max_length == 4


def test_text_request_uses_default_model():
    payload = _structure_payload()
    del payload["fields"]["fields"][0]["fields"][0]["model"]

    request = Schema.model_validate(payload).requests[ADDRESS]

    assert request.model == DEFAULT_TEXT_MODEL


def test_text_request_rejects_blank_model():
    payload = _structure_payload()
    payload["fields"]["fields"][0]["fields"][0]["model"] = "   "

    with pytest.raises(ValueError, match="at least 1 character"):
        Schema.model_validate(payload)


def test_text_request_strips_model():
    payload = _structure_payload()
    field = payload["fields"]["fields"][0]["fields"][0]
    field["model"] = "  bert-base-uncased  "

    request = Schema.model_validate(payload).requests[ADDRESS]

    assert request.model == "bert-base-uncased"


def test_text_raises_when_transformers_is_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(CachedModel, "_models", {})
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("missing transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="json2vec\\[text\\]"):
        CachedModel.get_model("bert-base-uncased")


def test_text_cached_model_reuses_same_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(CachedModel, "_models", {})
    loaded: list[str] = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model: str):
            loaded.append(model)
            return FakeHFModel()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=FakeAutoModel))

    first = CachedModel.get_model("bert-base-uncased")
    second = CachedModel.get_model("bert-base-uncased")

    assert first is second
    assert first.key == "bert-base-uncased"
    assert first.model is second.model
    assert loaded == ["bert-base-uncased"]


def test_text_cached_model_reuses_same_tokenizer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(CachedModel, "_tokenizers", {})
    loaded: list[str] = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model: str):
            loaded.append(model)
            return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))

    first = CachedModel.get_tokenizer("bert-base-uncased")
    second = CachedModel.get_tokenizer("bert-base-uncased")

    assert first is second
    assert loaded == ["bert-base-uncased"]


def test_text_embedders_share_frozen_model_resource(monkeypatch: pytest.MonkeyPatch):
    _patch_hf(monkeypatch)

    structure = Schema.model_validate(_structure_payload())
    first = Embedder(schema=structure, address=ADDRESS)
    second = Embedder(schema=structure, address=ADDRESS)

    assert first.__dict__["_cached_model"] is second.__dict__["_cached_model"]
    assert first.__dict__["_cached_model"].model is second.__dict__["_cached_model"].model


def test_text_shared_model_moves_with_shared_device_state(monkeypatch: pytest.MonkeyPatch):
    fake_model = _patch_hf(monkeypatch)

    structure = Schema.model_validate(_structure_payload())
    first = Embedder(schema=structure, address=ADDRESS)
    second = Embedder(schema=structure, address=ADDRESS)
    cached_model = first.__dict__["_cached_model"]

    assert cached_model.module_for(torch.device("cpu")) is fake_model
    assert cached_model.device == torch.device("cpu")
    assert second.__dict__["_cached_model"].device == torch.device("cpu")
    assert fake_model.device == torch.device("cpu")


def test_text_tensorfield_tokenizes_strings(monkeypatch: pytest.MonkeyPatch):
    _patch_hf(monkeypatch)

    structure = Schema.model_validate(_structure_payload())
    schema = structure
    field = TensorField.new(
        values=_values(),
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )

    assert field.content[INPUT_IDS].shape == (2, 1, 2, 4)
    assert field.content[ATTENTION_MASK].shape == (2, 1, 2, 4)
    assert field.content[INPUT_IDS][0, 0, 0].tolist() == [5, 6, 0, 0]
    assert field.content[ATTENTION_MASK][0, 0, 0].tolist() == [1, 1, 0, 0]


def test_text_embedder_and_decoder_shapes(monkeypatch: pytest.MonkeyPatch):
    fake_model = _patch_hf(monkeypatch)

    structure = Schema.model_validate(_structure_payload(encoder_batch_size=1))
    schema = structure
    field = TensorField.new(
        values=_values(),
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )

    embedder = Embedder(schema=structure, address=ADDRESS)
    parcel = embedder(field)
    assert parcel.payload.shape == (2, 1, 2, 16)
    assert fake_model.calls == 4

    decoder = Decoder(schema=structure, address=ADDRESS)
    prediction = decoder([parcel])
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 4)


def test_text_frozen_hf_model_is_not_registered_in_state_dict(monkeypatch: pytest.MonkeyPatch):
    _patch_hf(monkeypatch)

    structure = Schema.model_validate(_structure_payload())
    embedder = Embedder(schema=structure, address=ADDRESS)

    state_keys = set(embedder.state_dict())
    assert not any("_cached_model" in key or "hf_model" in key for key in state_keys)
    assert "linear.0.weight" in state_keys


class _DummyModule:
    def __init__(self, structure: Schema, embedder: Embedder, decoder: Decoder | None):
        self.schema = structure
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder, decoder=decoder)}
        self.logged: list[tuple[tuple[str, ...], float]] = []

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        self.logged.append((names, float(value.detach().cpu())))
        return value


@pytest.mark.parametrize(("objective", "expected"), [("l1", 2.0), ("l2", 4.0)])
def test_text_loss_reconstructs_frozen_embedding(monkeypatch: pytest.MonkeyPatch, objective: str, expected: float):
    _patch_hf(monkeypatch)

    structure = Schema.model_validate(_structure_payload(objective=objective))
    schema = structure
    field = TensorField.new(
        values=_values(),
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )
    field.mask(1.0)

    embedder = Embedder(schema=structure, address=ADDRESS)
    decoder = Decoder(schema=structure, address=ADDRESS)
    targets = embedder.target_embeddings(field)
    state_logits = torch.full((*field.targets[TensorKey.state].shape, len(Tokens)), -50.0)
    state_logits[..., Tokens.valued.value] = 50.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: targets + 2.0,
            },
            batch_size=[2],
        ),
    )

    module = _DummyModule(structure, embedder, decoder)
    output = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
    assert torch.isclose(output, torch.tensor(expected, dtype=output.dtype), atol=1e-3)


def test_text_write_returns_no_payload(monkeypatch: pytest.MonkeyPatch):
    _patch_hf(monkeypatch)

    structure = Schema.model_validate(_structure_payload())
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(2, 2, len(Tokens)),
                TensorKey.content: torch.zeros(2, 2, 4),
            },
            batch_size=[2],
        ),
    )

    output = write(module=_DummyModule(structure, None, None), prediction=prediction)
    assert output is None

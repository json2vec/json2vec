from types import SimpleNamespace
import builtins

import pytest
import torch
from tensordict import TensorDict

from json2vec.structs.enums import Strata, TensorKey, Tokens
from json2vec.structs.packages import Prediction
from json2vec.structs.structure import Structure
from json2vec.tensorfields.extensions import text as text_extension
from json2vec.tensorfields.extensions.text import (
    ATTENTION_MASK,
    INPUT_IDS,
    Decoder,
    Embedder,
    TensorField,
    loss,
    write,
)


def _structure_payload(*, objective: str = "l2", pooling: str = "cls", encoder_batch_size: int = 2) -> dict:
    return {
        "name": "demo",
        "type": "structure",
        "batch_size": 2,
        "dropout": 0.1,
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "context",
            "context_size": 2,
            "n_outputs": 1,
            "fields": [
                {
                    "name": "body",
                    "type": "text",
                    "query": "[*].body",
                    "model_name": "bert-base-uncased",
                    "max_length": 4,
                    "encoder_batch_size": encoder_batch_size,
                    "pooling": pooling,
                    "objective": objective,
                }
            ],
        },
    }


def _values() -> list:
    return [
        ["alpha", "beta"],
        ["gamma", "delta"],
    ]


def _session(structure: Structure):
    return SimpleNamespace(structure=structure)


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
    monkeypatch.setattr(text_extension, "_get_tokenizer", lambda *args, **kwargs: FakeTokenizer())
    monkeypatch.setattr(text_extension, "_get_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(text_extension, "_hidden_size", lambda *args, **kwargs: hidden_size)
    return fake_model


def test_text_request_is_available_in_structure():
    structure = Structure.model_validate(_structure_payload())
    request = structure.requests["root/body"]
    assert request.type == "text"
    assert request.model_name == "bert-base-uncased"
    assert request.max_length == 4


def test_text_request_rejects_blank_model_name():
    payload = _structure_payload()
    payload["fields"]["fields"][0]["model_name"] = "   "

    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        Structure.model_validate(payload)


def test_text_warns_when_transformers_is_missing(monkeypatch: pytest.MonkeyPatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("missing transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.warns(RuntimeWarning, match="not installed by default"), pytest.raises(ImportError, match="not installed by default"):
        text_extension._import_transformers()


def test_text_tensorfield_tokenizes_strings(monkeypatch: pytest.MonkeyPatch):
    _patch_hf(monkeypatch)

    structure = Structure.model_validate(_structure_payload())
    session = _session(structure)
    field = TensorField.new(
        values=_values(),
        address="root/body",
        session=session,
        strata=Strata.train,
        state=None,
    )

    assert field.content[INPUT_IDS].shape == (2, 2, 4)
    assert field.content[ATTENTION_MASK].shape == (2, 2, 4)
    assert field.content[INPUT_IDS][0, 0].tolist() == [5, 6, 0, 0]
    assert field.content[ATTENTION_MASK][0, 0].tolist() == [1, 1, 0, 0]


def test_text_embedder_and_decoder_shapes(monkeypatch: pytest.MonkeyPatch):
    fake_model = _patch_hf(monkeypatch)

    structure = Structure.model_validate(_structure_payload(encoder_batch_size=1))
    session = _session(structure)
    field = TensorField.new(
        values=_values(),
        address="root/body",
        session=session,
        strata=Strata.train,
        state=None,
    )

    embedder = Embedder(structure=structure, address="root/body")
    parcel = embedder(field)
    assert parcel.payload.shape == (2, 2, 16)
    assert fake_model.calls == 4

    decoder = Decoder(structure=structure, address="root/body")
    prediction = decoder([parcel])
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 4)


class _DummyModule:
    def __init__(self, structure: Structure, embedder: Embedder, decoder: Decoder | None):
        self.session = SimpleNamespace(structure=structure)
        self.nodes = {"root/body": SimpleNamespace(embedder=embedder, decoder=decoder)}
        self.logged: list[tuple[tuple[str, ...], float]] = []

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        self.logged.append((names, float(value.detach().cpu())))
        return value


@pytest.mark.parametrize(("objective", "expected"), [("l1", 2.0), ("l2", 4.0)])
def test_text_loss_reconstructs_frozen_embedding(monkeypatch: pytest.MonkeyPatch, objective: str, expected: float):
    _patch_hf(monkeypatch)

    structure = Structure.model_validate(_structure_payload(objective=objective))
    session = _session(structure)
    field = TensorField.new(
        values=_values(),
        address="root/body",
        session=session,
        strata=Strata.train,
        state=None,
    )
    field.mask(1.0)

    embedder = Embedder(structure=structure, address="root/body")
    decoder = Decoder(structure=structure, address="root/body")
    targets = embedder.target_embeddings(field)
    state_logits = torch.full((*field.targets[TensorKey.state].shape, len(Tokens)), -50.0)
    state_logits[..., Tokens.valued.value] = 50.0
    prediction = Prediction(
        address="root/body",
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

    structure = Structure.model_validate(_structure_payload())
    prediction = Prediction(
        address="root/body",
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

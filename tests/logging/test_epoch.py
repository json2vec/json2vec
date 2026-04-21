from types import SimpleNamespace

import pytest

import json2vec.logging.epoch as epoch
from json2vec.logging.epoch import EpochLifecycleLogger
from json2vec.structs.enums import Strata


class _BoundLogger:
    def __init__(self, parent: "_StubLogger", payload: dict[str, object]):
        self.parent = parent
        self.payload = payload

    def info(self, message: str):
        self.parent.messages.append((self.payload, message))


class _StubLogger:
    def __init__(self):
        self.binds: list[dict[str, object]] = []
        self.messages: list[tuple[dict[str, object], str]] = []

    def bind(self, **kwargs):
        self.binds.append(kwargs)
        return _BoundLogger(parent=self, payload=kwargs)


@pytest.mark.parametrize(
    ("method_name", "strata", "hook"),
    [
        ("on_train_epoch_start", Strata.train, "start"),
        ("on_train_epoch_end", Strata.train, "end"),
        ("on_validation_epoch_start", Strata.validate, "start"),
        ("on_validation_epoch_end", Strata.validate, "end"),
        ("on_test_epoch_start", Strata.test, "start"),
        ("on_test_epoch_end", Strata.test, "end"),
        ("on_predict_epoch_start", Strata.predict, "start"),
        ("on_predict_epoch_end", Strata.predict, "end"),
    ],
)
def test_epoch_lifecycle_logger_binds_expected_payload(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    strata: Strata,
    hook: str,
):
    stub = _StubLogger()
    monkeypatch.setattr(epoch, "logger", stub)

    callback = EpochLifecycleLogger()
    module = SimpleNamespace(global_rank=3, current_epoch=7, global_step=101)

    getattr(callback, method_name)(trainer=object(), pl_module=module)

    payload = stub.binds[-1]
    assert payload["source"] == "lightning"
    assert payload["rank"] == 3
    assert payload["epoch"] == 7
    assert payload["step"] == 101
    assert payload["hook"] == hook
    assert payload["strata"] == str(strata)
    assert stub.messages[-1][1] == f"{hook}ing {strata} epoch"

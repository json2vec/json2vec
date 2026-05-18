import pytest
import torch
from pydantic import ValidationError

from json2vec.inference.deployment import DeploymentEnvironment, resolve_accelerator

ENV_VARS = (
    "JSON2VEC_CHECKPOINT",
    "CHECKPOINT",
    "JSON2VEC_MAX_BATCH_SIZE",
    "MAX_BATCH_SIZE",
    "JSON2VEC_BATCH_TIMEOUT",
    "BATCH_TIMEOUT",
    "JSON2VEC_WORKERS_PER_DEVICE",
    "JSON2VEC_N_WORKERS",
    "N_WORKERS",
    "JSON2VEC_ACCELERATOR",
    "ACCELERATOR",
    "JSON2VEC_TRACK_REQUESTS",
    "TRACK_REQUESTS",
)


@pytest.fixture(autouse=True)
def clear_data_env(monkeypatch: pytest.MonkeyPatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_deployment_environment_from_env_accepts_s3_checkpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_CHECKPOINT", "s3://bucket/models/model.ckpt")

    env = DeploymentEnvironment()
    assert env.checkpoint == "s3://bucket/models/model.ckpt"


def test_deployment_environment_invalid_accelerator_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_CHECKPOINT", "s3://bucket/models/model.ckpt")
    monkeypatch.setenv("JSON2VEC_ACCELERATOR", "tpu")

    with pytest.raises(ValidationError, match="JSON2VEC_ACCELERATOR"):
        DeploymentEnvironment()


def test_resolve_accelerator_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_accelerator("auto") == "cuda"


def test_resolve_accelerator_auto_uses_mps_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert resolve_accelerator("auto") == "mps"


def test_resolve_accelerator_auto_uses_cpu_when_no_accelerator_is_available(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_accelerator("auto") == "cpu"


def test_resolve_accelerator_falls_back_from_unavailable_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_accelerator("cuda") == "cpu"


def test_resolve_accelerator_falls_back_from_unavailable_mps(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_accelerator("mps") == "cpu"

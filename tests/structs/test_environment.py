import pytest
from pydantic import ValidationError

from json2vec.inference.deployment import Accelerator, Deployment

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

    env = Deployment()
    assert env.checkpoint == "s3://bucket/models/model.ckpt"


def test_deployment_environment_invalid_accelerator_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_CHECKPOINT", "s3://bucket/models/model.ckpt")
    monkeypatch.setenv("JSON2VEC_ACCELERATOR", "tpu")

    with pytest.raises(ValidationError, match="JSON2VEC_ACCELERATOR"):
        Deployment()


def test_deployment_environment_normalizes_accelerator(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_CHECKPOINT", "s3://bucket/models/model.ckpt")
    monkeypatch.setenv("JSON2VEC_ACCELERATOR", " CPU ")

    assert Deployment().accelerator is Accelerator.cpu

import pytest
from pydantic import ValidationError

from relflow.inference.deployment import Accelerator, Deployment, JSONBackend

ENV_VARS = (
    "RELFLOW_CHECKPOINT",
    "CHECKPOINT",
    "RELFLOW_MAX_BATCH_SIZE",
    "MAX_BATCH_SIZE",
    "RELFLOW_BATCH_TIMEOUT",
    "BATCH_TIMEOUT",
    "RELFLOW_WORKERS",
    "WORKERS",
    "RELFLOW_ACCELERATOR",
    "ACCELERATOR",
    "RELFLOW_HOST",
    "HOST",
    "RELFLOW_PORT",
    "PORT",
    "RELFLOW_LOG_LEVEL",
    "LOG_LEVEL",
    "RELFLOW_JSON_BACKEND",
    "JSON_BACKEND",
    "RELFLOW_RETAIN",
    "RETAIN",
)


@pytest.fixture(autouse=True)
def clear_data_env(monkeypatch: pytest.MonkeyPatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_deployment_environment_from_env_accepts_s3_checkpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RELFLOW_CHECKPOINT", "s3://bucket/models/model.ckpt")

    env = Deployment()
    assert env.checkpoint == "s3://bucket/models/model.ckpt"


def test_deployment_environment_invalid_accelerator_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RELFLOW_CHECKPOINT", "s3://bucket/models/model.ckpt")
    monkeypatch.setenv("RELFLOW_ACCELERATOR", "tpu")

    with pytest.raises(ValidationError, match="RELFLOW_ACCELERATOR"):
        Deployment()


def test_deployment_environment_normalizes_accelerator(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RELFLOW_CHECKPOINT", "s3://bucket/models/model.ckpt")
    monkeypatch.setenv("RELFLOW_ACCELERATOR", " CPU ")

    assert Deployment().accelerator is Accelerator.cpu


def test_deployment_environment_accepts_realtime_serving_knobs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RELFLOW_CHECKPOINT", "s3://bucket/models/model.ckpt")
    monkeypatch.setenv("RELFLOW_WORKERS", "2")
    monkeypatch.setenv("RELFLOW_JSON_BACKEND", "stdlib")

    env = Deployment()

    assert env.workers == 2
    assert env.json_backend is JSONBackend.stdlib


def test_deployment_environment_accepts_all_column_retention(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RELFLOW_RETAIN", '"*"')

    assert Deployment().retain == "*"

import pytest
from pydantic import ValidationError

from json2vec.inference.deployment import Accelerator, Deployment, JSONBackend

ENV_VARS = (
    "JSON2VEC_CHECKPOINT",
    "CHECKPOINT",
    "JSON2VEC_MAX_BATCH_SIZE",
    "MAX_BATCH_SIZE",
    "JSON2VEC_BATCH_TIMEOUT",
    "BATCH_TIMEOUT",
    "JSON2VEC_WORKERS",
    "WORKERS",
    "JSON2VEC_ACCELERATOR",
    "ACCELERATOR",
    "JSON2VEC_HOST",
    "HOST",
    "JSON2VEC_PORT",
    "PORT",
    "JSON2VEC_LOG_LEVEL",
    "LOG_LEVEL",
    "JSON2VEC_MONITOR_QUERIES",
    "MONITOR_QUERIES",
    "JSON2VEC_QUERY_MONITOR_EVERY",
    "QUERY_MONITOR_EVERY",
    "JSON2VEC_JSON_BACKEND",
    "JSON_BACKEND",
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


def test_deployment_environment_accepts_realtime_serving_knobs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_CHECKPOINT", "s3://bucket/models/model.ckpt")
    monkeypatch.setenv("JSON2VEC_WORKERS", "2")
    monkeypatch.setenv("JSON2VEC_MONITOR_QUERIES", "true")
    monkeypatch.setenv("JSON2VEC_QUERY_MONITOR_EVERY", "7")
    monkeypatch.setenv("JSON2VEC_JSON_BACKEND", "stdlib")

    env = Deployment()

    assert env.workers == 2
    assert env.monitor_queries is True
    assert env.query_monitor_every == 7
    assert env.json_backend is JSONBackend.stdlib

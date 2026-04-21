import pytest
from pydantic import ValidationError

from json2vec.structs.enums import ShardingStrategy
from json2vec.structs.environment import DataLoaderEnvironment, DeploymentEnvironment

ENV_VARS = (
    "JSON2VEC_NUM_WORKERS",
    "NUM_WORKERS",
    "JSON2VEC_PERSISTENT_WORKERS",
    "PERSISTENT_WORKERS",
    "JSON2VEC_PIN_MEMORY",
    "PIN_MEMORY",
    "JSON2VEC_SHARDING",
    "JSON2VEC_SHARDING_STRATEGY",
    "SHARDING_STRATEGY",
    "JSON2VEC_CHUNK_BATCH_SIZE",
    "JSON2VEC_PYARROW_BATCH_SIZE",
    "CHUNK_BATCH_SIZE",
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


def test_dataloader_environment_defaults():
    env = DataLoaderEnvironment.from_env()
    assert env.sharding == ShardingStrategy.chunk
    assert env.chunk_batch_size == 4096
    assert env.num_workers is None
    assert env.persistent_workers is True
    assert env.pin_memory is True


def test_dataloader_environment_enum_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_SHARDING", "record")
    monkeypatch.setenv("JSON2VEC_CHUNK_BATCH_SIZE", "2048")

    env = DataLoaderEnvironment.from_env()
    assert env.sharding == ShardingStrategy.record
    assert env.chunk_batch_size == 2048


def test_dataloader_environment_observation_alias_is_rejected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_SHARDING", "observation")
    with pytest.raises(ValidationError, match="JSON2VEC_SHARDING"):
        DataLoaderEnvironment.from_env()


def test_dataloader_environment_invalid_sharding_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_SHARDING", "garbage")
    with pytest.raises(ValidationError, match="JSON2VEC_SHARDING"):
        DataLoaderEnvironment.from_env()


def test_dataloader_environment_invalid_chunk_batch_size_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_CHUNK_BATCH_SIZE", "0")
    with pytest.raises(ValidationError, match="JSON2VEC_CHUNK_BATCH_SIZE"):
        DataLoaderEnvironment.from_env()


def test_dataloader_environment_negative_num_workers_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_NUM_WORKERS", "-1")
    with pytest.raises(ValidationError, match="JSON2VEC_NUM_WORKERS"):
        DataLoaderEnvironment.from_env()


def test_dataloader_environment_bool_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_PERSISTENT_WORKERS", "off")
    monkeypatch.setenv("JSON2VEC_PIN_MEMORY", "0")
    env = DataLoaderEnvironment.from_env()
    assert env.persistent_workers is False
    assert env.pin_memory is False


def test_dataloader_environment_invalid_integer_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_CHUNK_BATCH_SIZE", "not-an-int")

    with pytest.raises(ValidationError, match="JSON2VEC_CHUNK_BATCH_SIZE"):
        DataLoaderEnvironment.from_env()


def test_dataloader_environment_invalid_bool_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_PIN_MEMORY", "sometimes")

    with pytest.raises(ValidationError, match="JSON2VEC_PIN_MEMORY"):
        DataLoaderEnvironment.from_env()


def test_deployment_environment_from_env_accepts_s3_checkpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_CHECKPOINT", "s3://bucket/models/model.ckpt")

    env = DeploymentEnvironment.from_env()
    assert env.checkpoint == "s3://bucket/models/model.ckpt"


def test_deployment_environment_invalid_accelerator_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_CHECKPOINT", "s3://bucket/models/model.ckpt")
    monkeypatch.setenv("JSON2VEC_ACCELERATOR", "tpu")

    with pytest.raises(ValidationError, match="JSON2VEC_ACCELERATOR"):
        DeploymentEnvironment.from_env()

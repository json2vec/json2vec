import pytest

from json2vec.logging import tracking
from json2vec.logging.tracking import LoggingFramework

LOGGER_ENV_VARS = (
    "JSON2VEC_LOGGER",
    "WANDB_API_KEY",
    "NEPTUNE_API_TOKEN",
    "COMET_API_KEY",
    "MLFLOW_TRACKING_URI",
    "JSON2VEC_TENSORBOARD_LOG_DIR",
    "TENSORBOARD_LOG_DIR",
    "JSON2VEC_CSV_LOG_DIR",
    "CSV_LOG_DIR",
)


@pytest.fixture(autouse=True)
def clear_logger_env(monkeypatch: pytest.MonkeyPatch):
    for name in LOGGER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_resolve_framework_honors_forced_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_LOGGER", "mlflow")
    monkeypatch.setenv("WANDB_API_KEY", "token")

    backend = tracking.LoggerFactory._resolve_framework()
    assert backend == LoggingFramework.mlflow


@pytest.mark.parametrize("value", ["none", "false", "off", "disabled"])
def test_resolve_framework_supports_disable_tokens(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("JSON2VEC_LOGGER", value)
    assert tracking.LoggerFactory._resolve_framework() is None


def test_resolve_framework_autodetects_wandb(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WANDB_API_KEY", "token")
    assert tracking.LoggerFactory._resolve_framework() == LoggingFramework.wandb


def test_resolve_framework_prefers_first_autodetection_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("COMET_API_KEY", "comet-token")
    monkeypatch.setenv("WANDB_API_KEY", "wandb-token")
    assert tracking.LoggerFactory._resolve_framework() == LoggingFramework.wandb


def test_resolve_framework_invalid_forced_value_disables_logger(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_LOGGER", "invalid")
    assert tracking.LoggerFactory._resolve_framework() is None


def test_create_returns_false_when_no_backend():
    assert tracking.LoggerFactory.create("project", "run", "notes") is False


def test_create_invokes_selected_builder(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_LOGGER", "csv")
    sentinel = object()
    monkeypatch.setattr(tracking.LoggerFactory, "csv", lambda project, run, notes: sentinel)

    logger_obj = tracking.LoggerFactory.create("project", "run", "notes")
    assert logger_obj is sentinel


def test_create_returns_false_when_builder_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JSON2VEC_LOGGER", "csv")

    def broken_builder(project: str, run: str, notes: str):
        raise RuntimeError("boom")

    monkeypatch.setattr(tracking.LoggerFactory, "csv", broken_builder)
    assert tracking.LoggerFactory.create("project", "run", "notes") is False

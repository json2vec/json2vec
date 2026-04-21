import json
from pathlib import Path

import pytest
import yaml

from json2vec.processors.base import PROCESSORS, register
from json2vec.structs.enums import Strata
from json2vec.structs.experiment import Experiment, _jsonnet


def _processor_name() -> str:
    if PROCESSORS:
        return next(iter(PROCESSORS))

    def _experiment_loading_processor(observation: dict):
        return observation

    _experiment_loading_processor.__name__ = "__experiment_loading_processor"
    register(_experiment_loading_processor)
    return _experiment_loading_processor.__name__


def _experiment_payload() -> dict:
    return {
        "project": "demo",
        "sessions": [
            {
                "name": "train",
                "task": "fit",
                "learning_rate": 1e-3,
                "dataset": {
                    "root": "/tmp/dataset",
                    "sample_rate": 1.0,
                    "file_buffer_size": 16,
                    "observation_buffer_size": 16,
                    "processor": _processor_name(),
                    "kwargs": {},
                    "suffix": "ndjson",
                    "patterns": {strata.value: ".*" for strata in Strata},
                },
                "structure": {
                    "name": "demo-structure",
                    "type": "structure",
                    "batch_size": 2,
                    "dropout": 0.1,
                    "d_model": 16,
                    "fields": {
                        "name": "root",
                        "type": "context",
                        "context_size": 1,
                        "n_outputs": 1,
                        "fields": [
                            {
                                "name": "identifier",
                                "type": "category",
                                "max_vocab_size": 1024,
                                "query": "[*].id",
                            }
                        ],
                    },
                },
            }
        ],
    }


def test_experiment_from_json(tmp_path: Path):
    payload = _experiment_payload()
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    experiment = Experiment.from_json(path, name="run-json", notes="json-notes")

    assert experiment.project == "demo"
    assert experiment.name == "run-json"
    assert experiment.notes == "json-notes"
    assert experiment.sessions[0].name == "train"


def test_experiment_from_yaml(tmp_path: Path):
    payload = _experiment_payload()
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    experiment = Experiment.from_yaml(path, name="run-yaml", notes="yaml-notes")

    assert experiment.project == "demo"
    assert experiment.name == "run-yaml"
    assert experiment.notes == "yaml-notes"


def test_experiment_from_toml(tmp_path: Path):
    processor = _processor_name()
    path = tmp_path / "experiment.toml"
    path.write_text(
        f"""
project = "demo"

[[sessions]]
name = "train"
task = "fit"
learning_rate = 0.001

[sessions.dataset]
root = "/tmp/dataset"
sample_rate = 1.0
file_buffer_size = 16
observation_buffer_size = 16
processor = "{processor}"
suffix = "ndjson"

[sessions.dataset.patterns]
train = ".*"
validate = ".*"
test = ".*"
predict = ".*"

[sessions.structure]
name = "demo-structure"
type = "structure"
batch_size = 2
dropout = 0.1
d_model = 16

[sessions.structure.fields]
name = "root"
type = "context"
context_size = 1
n_outputs = 1

[[sessions.structure.fields.fields]]
name = "identifier"
type = "category"
max_vocab_size = 1024
query = "[*].id"
""".strip(),
        encoding="utf-8",
    )

    experiment = Experiment.from_toml(path, name="run-toml", notes="toml-notes")

    assert experiment.project == "demo"
    assert experiment.name == "run-toml"
    assert experiment.notes == "toml-notes"


def test_experiment_from_jsonnet_with_python_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    payload = _experiment_payload()
    path = tmp_path / "experiment.jsonnet"
    path.write_text("{ project: 'ignored' }", encoding="utf-8")

    def fake_evaluate_file(pathname: str, ext_vars: dict[str, str], tla_vars: dict[str, str]) -> str:
        assert pathname.endswith("experiment.jsonnet")
        assert ext_vars == {"region": "us-east-1"}
        assert tla_vars == {"stage": "fit"}
        return json.dumps(payload)

    monkeypatch.setattr(_jsonnet, "evaluate_file", fake_evaluate_file)

    experiment = Experiment.from_jsonnet(
        path,
        name="run-jsonnet",
        notes="jsonnet-notes",
        ext_vars={"region": "us-east-1"},
        tla_vars={"stage": "fit"},
    )

    assert experiment.project == "demo"
    assert experiment.name == "run-jsonnet"
    assert experiment.notes == "jsonnet-notes"


def test_experiment_from_path_dispatches_to_suffix_loader(tmp_path: Path):
    payload = _experiment_payload()
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    experiment = Experiment.from_path(path, name="from-path", notes="notes")

    assert experiment.project == "demo"
    assert experiment.name == "from-path"


def test_experiment_from_path_rejects_unsupported_suffix(tmp_path: Path):
    path = tmp_path / "experiment.ini"
    path.write_text("project=demo", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported config suffix"):
        Experiment.from_path(path, name="unused", notes="unused")


def test_experiment_from_config_resolves_stem(tmp_path: Path):
    payload = _experiment_payload()
    path = tmp_path / "demo.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    experiment = Experiment.from_config(tmp_path, experiment="demo", name="run-config", notes="config-notes")

    assert experiment.project == "demo"
    assert experiment.name == "run-config"
    assert experiment.notes == "config-notes"


def test_experiment_from_config_uses_only_available_config_when_unspecified(tmp_path: Path):
    payload = _experiment_payload()
    path = tmp_path / "only.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    experiment = Experiment.from_config(tmp_path, name="run-config", notes="config-notes")

    assert experiment.project == "demo"
    assert experiment.name == "run-config"
    assert experiment.notes == "config-notes"


def test_experiment_from_config_requires_name_when_multiple_configs_exist(tmp_path: Path):
    payload = _experiment_payload()
    (tmp_path / "first.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")
    (tmp_path / "second.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Experiment name is required"):
        Experiment.from_config(tmp_path, name="run-config", notes="config-notes")

import re

import polars as pl
import pytest
from torch.utils.data import IterableDataset

import relflow as rf
from relflow.data.datasets import streaming
from relflow.rich import render_text


@pytest.fixture
def model():
    return rf.Model(
        rf.Category(name="id", size=16),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
    )


def renderings(module) -> tuple[str, ...]:
    bundle = module._repr_mimebundle_()
    return (
        str(module),
        repr(module),
        render_text(module),
        render_text({"module": module}),
        bundle["text/plain"],
        bundle["text/html"],
    )


def forbid_loaders(module) -> None:
    def fail():
        raise AssertionError("rendering built a data loader")

    module.train_dataloader = fail
    module.val_dataloader = fail
    module.test_dataloader = fail
    module.predict_dataloader = fail


def test_polars_datamodule_rendering_handles_missing_splits_without_exposing_rows(model):
    module = rf.PolarsDataModule(
        model=model,
        train=pl.DataFrame(
            {
                "id": [1, 2],
                "private": ["SECRET_OBSERVATION_ONE", "SECRET_OBSERVATION_TWO"],
            }
        ),
        num_workers=0,
    )
    forbid_loaders(module)

    outputs = renderings(module)

    for output in outputs:
        assert "PolarsDataModule" in output
        assert "train" in output
        assert "rows" in output
        assert "columns" in output
        assert "SECRET_OBSERVATION" not in output
        assert "NoneType" not in output


def test_custom_datamodule_rendering_does_not_measure_or_iterate_datasets(model):
    class GuardedDataset(IterableDataset):
        raw_observation = "SECRET_CUSTOM_DATASET_VALUE"

        def __iter__(self):
            raise AssertionError("rendering iterated the dataset")

        def __len__(self):
            raise AssertionError("rendering measured the dataset")

    module = rf.CustomDataModule(model=model, train=GuardedDataset(), num_workers=0)
    forbid_loaders(module)

    outputs = renderings(module)

    for output in outputs:
        assert "CustomDataModule" in output
        assert "GuardedDataset" in output
        assert "SECRET_CUSTOM_DATASET_VALUE" not in output


def test_synthetic_datamodule_rendering_does_not_call_generator(model):
    calls = 0

    def generate_records():
        nonlocal calls
        calls += 1
        yield {"id": "SECRET_GENERATED_VALUE"}

    module = rf.SyntheticDataModule(model=model, train=generate_records, num_workers=0)
    forbid_loaders(module)

    outputs = renderings(module)

    assert calls == 0
    for output in outputs:
        assert "SyntheticDataModule" in output
        assert "generate_records" in output
        assert "SECRET_GENERATED_VALUE" not in output


def test_synthetic_datamodule_rendering_does_not_inspect_callable_objects(model):
    class GuardedGenerator:
        def __getattribute__(self, name):
            if name in {"__name__", "__qualname__"}:
                raise AssertionError("rendering inspected the generator object")
            return super().__getattribute__(name)

        def __call__(self):
            raise AssertionError("rendering called the generator object")
            yield

    module = rf.SyntheticDataModule(model=model, train=GuardedGenerator(), num_workers=0)

    rendered = str(module)

    assert "SyntheticDataModule" in rendered
    assert "GuardedGenerator" in rendered


def test_streaming_datamodule_rendering_sanitizes_source_without_opening_it(
    model,
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("rendering touched the streaming source")

    monkeypatch.setattr(streaming, "fetch", fail)
    monkeypatch.setattr(streaming, "dataloader", fail)
    module = rf.StreamingDataModule(
        model=model,
        root=f"s3://reader:password@bucket/private/{'x' * 300}\x1b[31m?token=SECRET_TOKEN#fragment",
        suffix="ndjson",
        train=re.compile(r"SECRET_PATTERN.*\.ndjson$"),
        num_workers=0,
    )
    forbid_loaders(module)

    outputs = renderings(module)

    assert calls == 0
    for output in outputs:
        assert "StreamingDataModule" in output
        assert "s3://bucket/private/" in output
        assert "ndjson" in output
        assert "reader" not in output
        assert "password" not in output
        assert "SECRET_TOKEN" not in output
        assert "fragment" not in output
        assert "SECRET_PATTERN" not in output
        assert "x" * 120 not in output
        assert "\x1b" not in output


def test_streaming_datamodule_rendering_handles_no_configured_splits(model):
    module = rf.StreamingDataModule(model=model, root="records", suffix="ndjson")
    forbid_loaders(module)

    rendered = str(module)

    assert "StreamingDataModule" in rendered
    assert "splits=()" in rendered


def test_datamodule_mime_bundle_respects_include_and_exclude(model):
    module = rf.PolarsDataModule(model=model, train=pl.DataFrame({"id": [1]}), num_workers=0)

    assert set(module._repr_mimebundle_(include={"text/plain"})) == {"text/plain"}
    assert set(module._repr_mimebundle_(include={"text/html"})) == {"text/html"}
    assert module._repr_mimebundle_(exclude={"text/plain", "text/html"}) == {}

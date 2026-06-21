import enum
import json
import random
import re
from collections import Counter
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from torch.utils.data import IterableDataset

import json2vec as jv
from json2vec.data import iterables
from json2vec.data.datasets import base, custom, polars, streaming
from json2vec.data.datasets.base import _is_assigned_to_worker, _worker_identity, sha256
from json2vec.data.datasets.custom import CustomBatchDataset, CustomDataModule
from json2vec.data.datasets.polars import PolarsBatchDataset, PolarsDataModule
from json2vec.data.datasets.streaming import BatchDataset, StreamingDataModule
from json2vec.preprocessors.base import Preprocessor, PreprocessorMode
from json2vec.structs.enums import ShardingStrategy, Strata, Suffix
from json2vec.structs.experiment import Schema
from json2vec.structs.tree import Address


def _datamodule_schema():
    return Schema.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "record",
                "type": "branch",
                "length": 1,
                "fields": [],
            },
        }
    )


def _datamodule_model(batch_size: int = 2):
    return jv.Model.from_tree(
        jv.Category("id", size=16),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=batch_size,
    )


def _checkpoint_payload(model: jv.Model) -> dict:
    return {
        "state_dict": model.state_dict(),
        "schema": model.schema.model_dump(mode="python"),
        "batch_size": model.batch_size,
    }


class RecordsDataset(IterableDataset):
    def __init__(self, records):
        self.records = records

    def __iter__(self):
        yield from self.records


def test_sha256():
    assert sha256("test", 32) == 2676412545
    assert sha256("test", 64) == 11495104353665842533
    assert sha256("test", 128) == 212047248112658246449511647784264716309


def test_is_assigned_to_worker_partitions_shards():
    key = "chunk:s3://bucket/path/file.parquet:7"
    owners = [worker_id for worker_id in range(4) if _is_assigned_to_worker(key, worker_id=worker_id, num_workers=4)]
    assert len(owners) == 1


def test_is_assigned_to_worker_single_worker():
    assert _is_assigned_to_worker("record:file:42", worker_id=0, num_workers=1)


def test_worker_identity_combines_rank_and_dataloader_worker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(base, "get_worker_info", lambda: SimpleNamespace(id=2, num_workers=4))

    assert _worker_identity(global_rank=1, world_size=3) == (6, 12)


def test_query_prepends_outer_batch_selector():
    expr = iterables.query("[*].foo.bar")
    assert expr.expression == "[*][*].foo.bar"

    result = expr.search([[{"foo": {"bar": 42}}]])
    assert result == [[42]]

    over_nested = iterables.query("[*][*].foo.bar")
    assert over_nested.expression == "[*][*][*].foo.bar"
    assert over_nested.search([[{"foo": {"bar": 42}}]]) == [[]]


def test_simple_query_extractor_matches_jmespath_for_supported_expressions():
    cases = [
        (
            "[*].amount",
            [[{"amount": 1.0}], [{"missing": 2.0}]],
        ),
        (
            "[*].transactions[*].amount",
            [[{"transactions": [{"amount": 1.0}, {"missing": 2.0}]}], [{"transactions": []}]],
        ),
        (
            "[*].transactions[*].amount",
            [[{"transactions": None}], [{"transactions": [None, {"amount": 3.0}]}]],
        ),
    ]

    for expression, batch in cases:
        extractor = iterables.compile_query_extractor(expression)

        assert extractor is not None
        assert extractor(batch) == iterables.query(expression).search(batch)


def test_simple_query_extractor_falls_back_for_unsupported_expressions():
    assert iterables.compile_query_extractor("[*].items[?amount > `1`].amount") is None
    assert iterables.compile_query_extractor("[*].items[0].amount") is None
    assert iterables.compile_query_extractor('[*]."hyphen-name"') is None


def test_read_ndjson_chunk_sharding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "records.ndjson"
    records = [{"id": i} for i in range(5)]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    monkeypatch.setattr(streaming, "_worker_identity", lambda **_: (0, 2))

    def assign_first_chunk_only(shard_key: str, worker_id: int, num_workers: int) -> bool:
        return int(shard_key.rsplit(":", 1)[1]) == 0

    monkeypatch.setattr(streaming, "_is_assigned_to_worker", assign_first_chunk_only)

    output = list(
        streaming.read.__wrapped__(
            [str(path)],
            suffix=Suffix.ndjson,
            sharding=ShardingStrategy.chunk,
            chunk_batch_size=2,
        )
    )
    assert output == records[:2]


def test_read_ndjson_chunk_sharding_partitions_records_across_ranks(tmp_path: Path):
    path = tmp_path / "records.ndjson"
    records = [{"id": i} for i in range(12)]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    outputs_by_rank = [
        list(
            streaming.read.__wrapped__(
                [str(path)],
                suffix=Suffix.ndjson,
                sharding=ShardingStrategy.chunk,
                chunk_batch_size=3,
                global_rank=rank,
                world_size=2,
            )
        )
        for rank in range(2)
    ]

    first = {record["id"] for record in outputs_by_rank[0]}
    second = {record["id"] for record in outputs_by_rank[1]}

    assert first.isdisjoint(second)
    assert first | second == {record["id"] for record in records}


def test_read_ndjson_record_sharding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "records.ndjson"
    records = [{"id": i} for i in range(6)]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    monkeypatch.setattr(streaming, "_worker_identity", lambda **_: (0, 2))

    def assign_even_records(shard_key: str, worker_id: int, num_workers: int) -> bool:
        return int(shard_key.rsplit(":", 1)[1]) % 2 == 0

    monkeypatch.setattr(streaming, "_is_assigned_to_worker", assign_even_records)

    output = list(
        streaming.read.__wrapped__(
            [str(path)],
            suffix=Suffix.ndjson,
            sharding=ShardingStrategy.record,
            chunk_batch_size=3,
        )
    )
    assert output == [records[index] for index in (0, 2, 4)]


def test_fetch_file_sharding_filters_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "keep.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "skip.ndjson").write_text("", encoding="utf-8")

    monkeypatch.setattr(streaming, "_worker_identity", lambda **_: (0, 2))

    def assign_keep_only(shard_key: str, worker_id: int, num_workers: int) -> bool:
        return "keep.ndjson" in shard_key

    monkeypatch.setattr(streaming, "_is_assigned_to_worker", assign_keep_only)

    files = list(
        streaming.fetch.__wrapped__(
            root=tmp_path,
            pattern=re.compile(r".*\.ndjson$"),
            sharding=ShardingStrategy.file,
        )
    )
    assert {Path(path).name for path in files} == {"keep.ndjson"}


def test_fetch_without_file_sharding_returns_all_matching_files(tmp_path: Path):
    (tmp_path / "first.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "second.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("", encoding="utf-8")

    files = list(
        streaming.fetch.__wrapped__(
            root=tmp_path,
            pattern=re.compile(r".*\.ndjson$"),
            sharding=ShardingStrategy.chunk,
        )
    )
    assert {Path(path).name for path in files} == {"first.ndjson", "second.ndjson"}


def test_fetch_accepts_raw_regex_string(tmp_path: Path):
    (tmp_path / "first.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("", encoding="utf-8")

    files = list(
        streaming.fetch(
            root=tmp_path,
            pattern=r".*\.ndjson$",
            sharding=ShardingStrategy.chunk,
        )
    )
    assert {Path(path).name for path in files} == {"first.ndjson"}


def test_fetch_all_pattern_returns_all_files(tmp_path: Path):
    (tmp_path / "first.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "second.csv").write_text("", encoding="utf-8")

    files = list(
        streaming.fetch.__wrapped__(
            root=tmp_path,
            pattern=re.compile(r".*"),
            sharding=ShardingStrategy.chunk,
        )
    )
    assert {Path(path).name for path in files} == {"first.ndjson", "second.csv"}


def test_observe_replacement_allows_workers_to_share_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first = tmp_path / "first.ndjson"
    second = tmp_path / "second.ndjson"
    first.write_text(json.dumps({"id": 1}), encoding="utf-8")
    second.write_text(json.dumps({"id": 2}), encoding="utf-8")

    def assign_first_file_only(shard_key: str, worker_id: int, num_workers: int) -> bool:
        return "first.ndjson" in shard_key

    monkeypatch.setattr(streaming, "_is_assigned_to_worker", assign_first_file_only)
    monkeypatch.setattr(streaming.random, "choice", lambda paths: str(second))

    output = list(
        islice(
            streaming.observe.__wrapped__(
                root=tmp_path,
                suffix=Suffix.ndjson,
                pattern=re.compile(r".*\.ndjson$"),
                strata=Strata.train,
                sharding=ShardingStrategy.file,
                chunk_batch_size=1,
                file_buffer_size=1,
                replacement=True,
                global_rank=0,
                world_size=1,
            ),
            3,
        )
    )

    assert output == [{"id": 2}, {"id": 2}, {"id": 2}]


def test_observe_polars_yields_dataframe_rows():
    frame = pl.DataFrame({"id": [1, 2], "name": ["alpha", "beta"]})

    output = list(
        polars.observe_polars.__wrapped__(
            dataframe=frame,
            strata=Strata.train,
            sharding=ShardingStrategy.chunk,
            chunk_batch_size=1,
            global_rank=0,
            world_size=1,
        )
    )

    assert output == [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]


def test_observe_polars_record_sharding_partitions_rows():
    frame = pl.DataFrame({"id": list(range(8))})
    rows_by_rank = [
        list(
            polars.observe_polars.__wrapped__(
                dataframe=frame,
                strata=Strata.train,
                sharding=ShardingStrategy.record,
                chunk_batch_size=2,
                global_rank=rank,
                world_size=2,
            )
        )
        for rank in range(2)
    ]

    first = {row["id"] for row in rows_by_rank[0]}
    second = {row["id"] for row in rows_by_rank[1]}

    assert first.isdisjoint(second)
    assert first | second == set(range(8))


def test_observe_polars_replacement_samples_full_dataframe_without_sharding(monkeypatch: pytest.MonkeyPatch):
    frame = pl.DataFrame({"id": [1, 2], "name": ["alpha", "beta"]})

    monkeypatch.setattr(polars, "_is_assigned_to_worker", lambda **_: False)
    monkeypatch.setattr(polars.random, "choice", lambda rows: rows[1])

    output = list(
        islice(
            polars.observe_polars.__wrapped__(
                dataframe=frame,
                strata=Strata.train,
                sharding=ShardingStrategy.file,
                chunk_batch_size=1,
                replacement=True,
                global_rank=0,
                world_size=2,
            ),
            3,
        )
    )

    assert output == [
        {"id": 2, "name": "beta"},
        {"id": 2, "name": "beta"},
        {"id": 2, "name": "beta"},
    ]


def test_observe_polars_replacement_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="no dataframe rows available"):
        next(
            polars.observe_polars.__wrapped__(
                dataframe=pl.DataFrame({"id": []}),
                strata=Strata.train,
                sharding=ShardingStrategy.chunk,
                chunk_batch_size=1,
                replacement=True,
                global_rank=0,
                world_size=1,
            )
        )


def test_read_unsupported_suffix_raises_value_error():
    class UnknownSuffix(enum.StrEnum):
        bad = "bad"

    with pytest.raises(ValueError, match="Unsupported suffix: bad"):
        list(
            streaming.read.__wrapped__(
                [],
                suffix=UnknownSuffix.bad,
                sharding=ShardingStrategy.chunk,
                chunk_batch_size=2,
            )
        )


def test_process_transformation_preprocessor_wraps_dict_output(monkeypatch: pytest.MonkeyPatch):
    def transformation(observation: dict):
        return {"id": observation["id"]}

    preprocessor = Preprocessor(name="__test_transformation", func=transformation, mode=PreprocessorMode.transformation)
    monkeypatch.setitem(iterables.PREPROCESSORS, "__test_transformation", preprocessor)

    output = list(
        iterables.process.__wrapped__(
            [{"id": 1}, {"id": 2}],
            preprocessor="__test_transformation",
            preprocessor_kwargs=None,
            strata=Strata.train,
            interprocess_encoding_context={},
        )
    )
    assert output == [[{"id": 1}], [{"id": 2}]]


def test_process_generator_preprocessor_wraps_list_outputs(monkeypatch: pytest.MonkeyPatch):
    def generator(observation: dict):
        return [{"id": observation["id"]}, {"id": observation["id"] + 100}]

    preprocessor = Preprocessor(name="__test_generator", func=generator, mode=PreprocessorMode.generator)
    monkeypatch.setitem(iterables.PREPROCESSORS, "__test_generator", preprocessor)

    output = list(
        iterables.process.__wrapped__(
            [{"id": 1}],
            preprocessor="__test_generator",
            preprocessor_kwargs=None,
            strata=Strata.train,
            interprocess_encoding_context={},
        )
    )
    assert output == [[{"id": 1}], [{"id": 101}]]


def test_process_generator_preprocessor_receives_strata_and_state(monkeypatch: pytest.MonkeyPatch):
    def generator(observation: dict, strata, interprocess_encoding_context):
        yield {"id": observation["id"], "strata": strata, "marker": interprocess_encoding_context["marker"]}

    preprocessor = Preprocessor(name="__test_generator_context", func=generator, mode=PreprocessorMode.generator)
    monkeypatch.setitem(iterables.PREPROCESSORS, "__test_generator_context", preprocessor)

    output = list(
        iterables.process.__wrapped__(
            [{"id": 1}],
            preprocessor="__test_generator_context",
            preprocessor_kwargs=None,
            strata=Strata.validate,
            interprocess_encoding_context={"marker": "seen"},
        )
    )
    assert output == [[{"id": 1, "strata": Strata.validate, "marker": "seen"}]]


def test_process_transformation_preprocessor_rejects_non_dict(monkeypatch: pytest.MonkeyPatch):
    def transformation(observation: dict):
        return observation["id"]

    preprocessor = Preprocessor(
        name="__test_invalid_transformation",
        func=transformation,
        mode=PreprocessorMode.transformation,
    )
    monkeypatch.setitem(iterables.PREPROCESSORS, "__test_invalid_transformation", preprocessor)

    with pytest.raises(TypeError, match="must produce dict objects"):
        list(
            iterables.process.__wrapped__(
                [{"id": 1}],
                preprocessor="__test_invalid_transformation",
                preprocessor_kwargs=None,
                strata=Strata.train,
                interprocess_encoding_context={},
            )
        )


def test_process_without_preprocessor_still_wraps_root_branch():
    output = list(
        iterables.process.__wrapped__(
            [{"id": 1}, {"id": 2}],
            preprocessor=None,
            preprocessor_kwargs=None,
            strata=Strata.train,
            interprocess_encoding_context={},
        )
    )
    assert output == [[{"id": 1}], [{"id": 2}]]


def test_batch_splits_and_preserves_tail():
    chunks = list(iterables.batch.__wrapped__([1, 2, 3, 4, 5], batch_size=2))
    assert chunks == [[1, 2], [3, 4], [5]]


def test_sample_predict_is_identity():
    data = list(range(8))
    output = list(iterables.sample.__wrapped__(data, sample_rate=0.1, strata=Strata.predict))
    assert output == data


def test_sample_filters_non_predict_with_sample_rate():
    random.seed(3)
    data = list(range(8))
    output = list(iterables.sample.__wrapped__(data, sample_rate=0.5, strata=Strata.train))
    assert output == [0, 2, 5, 6]


def test_shuffle_predict_is_identity():
    data = list(range(8))
    output = list(iterables.shuffle.__wrapped__(data, size=3, strata=Strata.predict))
    assert output == data


def test_shuffle_non_predict_preserves_elements():
    random.seed(7)
    data = list(range(12))
    output = list(iterables.shuffle.__wrapped__(data, size=4, strata=Strata.train))
    assert sorted(output) == data
    assert len(output) == len(data)


def test_shuffle_non_predict_preserves_duplicate_counts():
    random.seed(11)
    data = [1, 1, 1, 2, 2, 3, 4, 4]
    output = list(iterables.shuffle.__wrapped__(data, size=3, strata=Strata.train))
    assert Counter(output) == Counter(data)


def test_shuffle_stops_refilling_after_source_exhausted():
    class CountingIterator:
        def __init__(self, values):
            self.values = list(values)
            self.index = 0
            self.stop_count = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.values):
                self.stop_count += 1
                raise StopIteration

            value = self.values[self.index]
            self.index += 1
            return value

    random.seed(5)
    iterator = CountingIterator(range(5))
    output = list(iterables.shuffle.__wrapped__(iterator, size=3, strata=Strata.train))
    assert sorted(output) == [0, 1, 2, 3, 4]
    assert iterator.stop_count == 1


def test_jmespath_resolution_monitor_raises_for_empty_result():
    monitor = iterables.JMESPathResolutionMonitor(every=1)

    with pytest.raises(ValueError, match="JMESPath query returned empty result"):
        monitor.observe(address="root/id", expression="[*].id", result=[])


def test_jmespath_resolution_monitor_ignores_empty_result_until_threshold():
    monitor = iterables.JMESPathResolutionMonitor(every=3)

    monitor.observe(address="root/id", expression="[*].id", result=[])
    monitor.observe(address="root/id", expression="[*].id", result=[])


def test_jmespath_resolution_monitor_accepts_nested_observed_value():
    monitor = iterables.JMESPathResolutionMonitor(every=1)

    monitor.observe(address="root/id", expression="[*].id", result=[[None, {"id": 0}]])


def test_streaming_datamodule_accepts_named_loader_configuration_per_strata():
    module = StreamingDataModule(
        model=_datamodule_model(),
        root="/tmp/json2vec-test",
        suffix=Suffix.ndjson,
        train=re.compile(r".*\.ndjson$"),
        num_workers={Strata.train: 0},
        sharding={Strata.train: "record"},
        chunk_batch_size={Strata.train: 7},
        file_buffer_size={Strata.train: 11},
        observation_buffer_size={Strata.train: 13},
        sample_rate={Strata.train: 0.5},
    )

    assert module.num_workers[Strata.train] == 0
    assert module.num_workers[Strata.validate] is None
    assert module.sharding[Strata.train] == ShardingStrategy.record
    assert module.sharding[Strata.validate] == ShardingStrategy.file
    assert module.chunk_batch_size[Strata.train] == 7
    assert module.chunk_batch_size[Strata.validate] == 4096
    assert module.file_buffer_size[Strata.train] == 11
    assert module.file_buffer_size[Strata.validate] == 1
    assert module.observation_buffer_size[Strata.train] == 13
    assert module.observation_buffer_size[Strata.validate] == 1
    assert module.sample_rate[Strata.train] == 0.5
    assert module.sample_rate[Strata.validate] == 1.0
    assert module.replacement[Strata.train] is True
    assert module.replacement[Strata.validate] is False
    assert module.val_dataloader() is None


def test_streaming_datamodule_rejects_invalid_loader_configuration():
    kwargs = {
        "model": _datamodule_model(),
        "root": "/tmp/json2vec-test",
        "suffix": Suffix.ndjson,
        "train": re.compile(r".*\.ndjson$"),
    }

    with pytest.raises(BeartypeCallHintParamViolation):
        StreamingDataModule(**kwargs, num_workers={Strata.train: True})

    with pytest.raises(BeartypeCallHintParamViolation):
        StreamingDataModule(**kwargs, sample_rate={Strata.train: 0.0})


def test_streaming_datamodule_defaults_to_file_sharding():
    module = StreamingDataModule(
        model=_datamodule_model(),
        root="/tmp/json2vec-test",
        suffix=Suffix.ndjson,
        train=re.compile(r".*\.ndjson$"),
    )

    assert module.sharding == {strata: ShardingStrategy.file for strata in Strata}
    assert module.replacement == {strata: strata == Strata.train for strata in Strata}


def test_streaming_datamodule_accepts_raw_regex_string_patterns():
    module = StreamingDataModule(
        model=_datamodule_model(),
        root="/tmp/json2vec-test",
        suffix=Suffix.ndjson,
        train=r"/train/.*\.ndjson$",
        validate=r"/validate/.*\.ndjson$",
        predict=r"/predict/.*\.ndjson$",
    )

    assert module.train is not None
    assert module.validate is not None
    assert module.predict is not None
    assert module.train.pattern == r"/train/.*\.ndjson$"
    assert module.validate.pattern == r"/validate/.*\.ndjson$"
    assert module.predict.pattern == r"/predict/.*\.ndjson$"


def test_streaming_datamodule_regex_patterns_keep_strata_disjoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    for split in ["train", "validate", "test"]:
        split_dir = tmp_path / split
        split_dir.mkdir()
        for index in range(2):
            record = {"id": f"{split}-{index}"}
            (split_dir / f"part-{index}.ndjson").write_text(json.dumps(record), encoding="utf-8")

    def transform(pipe, schema, strata, interprocess_encoding_context):
        yield from pipe

    def mask(pipe, schema):
        yield from pipe

    monkeypatch.setattr(streaming, "transform", transform)
    monkeypatch.setattr(streaming, "mask", mask)

    module = StreamingDataModule(
        model=_datamodule_model(batch_size=4),
        root=tmp_path,
        suffix=Suffix.ndjson,
        train=r"(^|/)train/.*\.ndjson$",
        validate=r"(^|/)validate/.*\.ndjson$",
        test=r"(^|/)test/.*\.ndjson$",
        num_workers=0,
        replacement=False,
        file_buffer_size=1,
        observation_buffer_size=1,
    )

    def ids(loader):
        assert loader is not None
        return {observation[0]["id"] for batch in loader for observation in batch}

    assert ids(module.train_dataloader()) == {"train-0", "train-1"}
    assert ids(module.val_dataloader()) == {"validate-0", "validate-1"}
    assert ids(module.test_dataloader()) == {"test-0", "test-1"}


def test_streaming_datamodule_accepts_replacement_configuration_per_strata():
    module = StreamingDataModule(
        model=_datamodule_model(),
        root="/tmp/json2vec-test",
        suffix=Suffix.ndjson,
        train=re.compile(r".*\.ndjson$"),
        replacement={Strata.train: False, Strata.validate: True},
    )

    assert module.replacement[Strata.train] is False
    assert module.replacement[Strata.validate] is True
    assert module.replacement[Strata.test] is False


def test_streaming_datamodule_refreshes_model_state_after_checkpoint_restore(tmp_path: Path):
    model = jv.Model.from_tree(
        jv.Number("amount"),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
    )
    module = StreamingDataModule(
        model=model,
        root=tmp_path,
        suffix=Suffix.ndjson,
        train=re.compile(r".*\.ndjson$"),
        num_workers=0,
    )
    restored = jv.Model.from_tree(
        jv.Number("risk_score"),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=5,
    )

    model.restore_checkpoint_state(_checkpoint_payload(restored))

    loader = module.train_dataloader()
    assert loader is not None
    assert module.schema is model.schema
    assert module.batch_size == 5
    assert loader.dataset.schema is model.schema
    assert loader.dataset.batch_size == 5
    assert "record/risk_score" in loader.dataset.schema.requests
    assert "record/amount" not in loader.dataset.schema.requests


def test_custom_datamodule_accepts_named_datasets_and_loader_configuration_per_strata():
    train = RecordsDataset([{"id": 1}])
    predict = RecordsDataset([{"id": 2}])
    module = CustomDataModule(
        model=_datamodule_model(),
        train=train,
        predict=predict,
        preprocessor=None,
        num_workers={Strata.train: 0},
        observation_buffer_size={Strata.train: 13},
        sample_rate={Strata.train: 0.5},
    )

    assert module.datasets[Strata.train] is train
    assert module.datasets[Strata.predict] is predict
    assert set(module.datasets) == {Strata.train, Strata.predict}
    assert module.preprocessor is None
    assert module.preprocessor_kwargs == {}
    assert module.num_workers[Strata.train] == 0
    assert module.num_workers[Strata.validate] is None
    assert module.observation_buffer_size[Strata.train] == 13
    assert module.observation_buffer_size[Strata.validate] == 1
    assert module.sample_rate[Strata.train] == 0.5
    assert module.sample_rate[Strata.validate] == 1.0
    assert module.val_dataloader() is None


def test_custom_datamodule_accepts_dataset_mapping():
    train = RecordsDataset([{"id": 1}])
    validate = RecordsDataset([{"id": 2}])

    module = CustomDataModule(
        model=_datamodule_model(),
        datasets={"train": train, Strata.validate: validate},
        num_workers=0,
    )

    assert module.datasets[Strata.train] is train
    assert module.datasets[Strata.validate] is validate
    assert module.test_dataloader() is None


def test_custom_datamodule_rejects_invalid_loader_configuration():
    kwargs = {
        "model": _datamodule_model(),
        "train": RecordsDataset([{"id": 1}]),
    }

    with pytest.raises(BeartypeCallHintParamViolation):
        CustomDataModule(**kwargs, num_workers={Strata.train: True})

    with pytest.raises(BeartypeCallHintParamViolation):
        CustomDataModule(**kwargs, sample_rate={Strata.train: 0.0})


def test_custom_datamodule_requires_at_least_one_split():
    with pytest.raises(ValueError, match="at least one dataset split is required"):
        CustomDataModule(model=_datamodule_model())


def test_custom_datamodule_rejects_datasets_and_named_splits():
    with pytest.raises(ValueError, match="pass either datasets or named splits"):
        CustomDataModule(
            model=_datamodule_model(),
            train=RecordsDataset([{"id": 1}]),
            datasets={Strata.validate: RecordsDataset([{"id": 2}])},
        )


def test_custom_datamodule_accepts_partial_dataset_mapping_until_loader_requested():
    module = CustomDataModule(
        model=_datamodule_model(),
        datasets={Strata.train: RecordsDataset([{"id": 1}])},
        num_workers=0,
    )

    assert set(module.datasets) == {Strata.train}
    assert module.val_dataloader() is None

    with pytest.raises(ValueError, match="no dataset configured"):
        module.dataloader(strata=Strata.validate)


def test_custom_datamodule_refreshes_model_state_after_checkpoint_restore():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
    )
    module = CustomDataModule(
        model=model,
        train=RecordsDataset([{"amount": 1.0, "risk_score": 2.0}]),
        num_workers=0,
    )
    restored = jv.Model.from_tree(
        jv.Number("risk_score"),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=5,
    )

    model.restore_checkpoint_state(_checkpoint_payload(restored))

    loader = module.train_dataloader()
    assert loader is not None
    assert module.schema is model.schema
    assert module.batch_size == 5
    assert loader.dataset.schema is model.schema
    assert loader.dataset.batch_size == 5
    assert "record/risk_score" in loader.dataset.schema.requests
    assert "record/amount" not in loader.dataset.schema.requests


def test_custom_batch_dataset_reads_records_through_pipeline(monkeypatch: pytest.MonkeyPatch):
    def transform(pipe, schema, strata, interprocess_encoding_context):
        yield from pipe

    def mask(pipe, schema):
        yield from pipe

    monkeypatch.setattr(custom, "transform", transform)
    monkeypatch.setattr(custom, "mask", mask)

    batch_dataset = CustomBatchDataset(
        schema=SimpleNamespace(requests={}),
        dataset=RecordsDataset([{"id": 1}, {"id": 2}, {"id": 3}]),
        preprocessor=None,
        preprocessor_kwargs={},
        interprocess_encoding_context={},
        batch_size=2,
        strata=Strata.train,
        observation_buffer_size=1,
        sample_rate=1.0,
    )

    assert list(batch_dataset) == [
        [[{"id": 1}], [{"id": 2}]],
        [[{"id": 3}]],
    ]


def test_polars_datamodule_accepts_dataframe_and_loader_configuration_per_strata():
    frame = pl.DataFrame({"id": [1, 2]})
    module = PolarsDataModule(
        model=_datamodule_model(),
        dataframe=frame,
        num_workers={Strata.train: 0},
        sharding={Strata.train: "record"},
        chunk_batch_size={Strata.train: 7},
        observation_buffer_size={Strata.train: 13},
        sample_rate={Strata.train: 0.5},
    )

    assert module.dataframes[Strata.train] is frame
    assert module.dataframes[Strata.validate] is frame
    assert module.num_workers[Strata.train] == 0
    assert module.num_workers[Strata.validate] is None
    assert module.sharding[Strata.train] == ShardingStrategy.record
    assert module.sharding[Strata.validate] == ShardingStrategy.chunk
    assert module.chunk_batch_size[Strata.train] == 7
    assert module.chunk_batch_size[Strata.validate] == 4096
    assert module.observation_buffer_size[Strata.train] == 13
    assert module.observation_buffer_size[Strata.validate] == 1
    assert module.sample_rate[Strata.train] == 0.5
    assert module.sample_rate[Strata.validate] == 1.0
    assert module.replacement[Strata.train] is False
    assert module.replacement[Strata.validate] is False


def test_polars_datamodule_accepts_named_splits():
    train = pl.DataFrame({"id": [1]})
    predict = pl.DataFrame({"id": [2]})

    module = PolarsDataModule(
        model=_datamodule_model(),
        train=train,
        predict=predict,
        preprocessor=None,
        num_workers=0,
    )

    assert module.dataframes[Strata.train] is train
    assert module.dataframes[Strata.predict] is predict
    assert set(module.dataframes) == {Strata.train, Strata.predict}
    assert module.preprocessor is None
    assert module.preprocessor_kwargs == {}
    assert module.val_dataloader() is None


def test_polars_datamodule_refreshes_context_after_model_reset():
    model = jv.Model.from_tree(
        jv.Category("code", size=16),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
    )
    module = PolarsDataModule(
        model=model,
        train=pl.DataFrame({"code": ["a"]}),
        num_workers=0,
    )
    before = module.interprocess_encoding_context["record/code"]

    model.reset(jv.where("name") == "code")

    after = module.interprocess_encoding_context["record/code"]
    current = model.interprocess_encoding_context["record/code"]
    assert after.master is current.master
    assert after.master is not before.master
    assert after is not before


def test_polars_datamodule_shares_vocabulary_only_for_train_workers():
    model = jv.Model.from_tree(
        jv.Category("code", size=16),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
    )
    address = Address("record/code")
    vocab = model.nodes[address].embedder.vocab
    module = PolarsDataModule(
        model=model,
        train=pl.DataFrame({"code": ["a"]}),
        predict=pl.DataFrame({"code": ["b"]}),
        num_workers={Strata.train: 1, Strata.predict: 1},
        persistent_workers=False,
    )

    assert vocab.is_shared is False

    predict_loader = module.predict_dataloader()

    assert predict_loader is not None
    assert vocab.is_shared is False

    train_loader = module.train_dataloader()

    assert train_loader is not None
    assert vocab.is_shared is True

    vocab.freeze()
    assert vocab.is_shared is False


def test_polars_datamodule_refreshes_model_state_after_checkpoint_restore():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
    )
    module = PolarsDataModule(
        model=model,
        train=pl.DataFrame({"amount": [1.0], "risk_score": [2.0]}),
        num_workers=0,
    )
    restored = jv.Model.from_tree(
        jv.Number("risk_score"),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=5,
    )

    model.restore_checkpoint_state(_checkpoint_payload(restored))

    loader = module.train_dataloader()
    assert loader is not None
    assert module.schema is model.schema
    assert module.batch_size == 5
    assert loader.dataset.schema is model.schema
    assert loader.dataset.batch_size == 5
    assert "record/risk_score" in loader.dataset.schema.requests
    assert "record/amount" not in loader.dataset.schema.requests


def test_polars_datamodule_requires_at_least_one_split():
    with pytest.raises(ValueError, match="at least one dataframe split is required"):
        PolarsDataModule(model=_datamodule_model())


def test_polars_datamodule_accepts_partial_dataframe_mapping_until_loader_requested():
    module = PolarsDataModule(
        model=_datamodule_model(),
        dataframe={Strata.train: pl.DataFrame({"id": [1]})},
        num_workers=0,
    )

    assert set(module.dataframes) == {Strata.train}
    assert module.val_dataloader() is None

    with pytest.raises(ValueError, match="no dataframe configured"):
        module.dataloader(strata=Strata.validate)


def test_polars_datamodule_accepts_replacement_configuration_per_strata():
    module = PolarsDataModule(
        model=_datamodule_model(),
        train=pl.DataFrame({"id": [1]}),
        validate=pl.DataFrame({"id": [2]}),
        replacement={Strata.train: False, Strata.validate: True},
    )

    assert module.replacement[Strata.train] is False
    assert module.replacement[Strata.validate] is True
    assert module.replacement[Strata.test] is False


def test_polars_datamodule_passes_replacement_to_dataset():
    module = PolarsDataModule(
        model=_datamodule_model(),
        train=pl.DataFrame({"id": [1]}),
        num_workers=0,
        replacement=True,
    )

    loader = module.train_dataloader()

    assert loader is not None
    assert loader.dataset.replacement is True


def test_polars_batch_dataset_reads_dataframe_rows_through_pipeline(monkeypatch: pytest.MonkeyPatch):
    def transform(pipe, schema, strata, interprocess_encoding_context):
        yield from pipe

    def mask(pipe, schema):
        yield from pipe

    monkeypatch.setattr(polars, "transform", transform)
    monkeypatch.setattr(polars, "mask", mask)

    batch_dataset = PolarsBatchDataset(
        schema=SimpleNamespace(requests={}),
        dataframe=pl.DataFrame({"id": [1, 2, 3]}),
        preprocessor=None,
        preprocessor_kwargs={},
        interprocess_encoding_context={},
        batch_size=2,
        strata=Strata.train,
        sharding=ShardingStrategy.chunk,
        chunk_batch_size=2,
        observation_buffer_size=1,
        sample_rate=1.0,
    )

    assert list(batch_dataset) == [
        [[{"id": 1}], [{"id": 2}]],
        [[{"id": 3}]],
    ]


def test_batch_dataset_passes_sample_rate_into_pipeline(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def observe(root, suffix, pattern, strata, sharding, chunk_batch_size, file_buffer_size):
        yield {"id": 1}

    def process(pipe, preprocessor, preprocessor_kwargs, strata, interprocess_encoding_context):
        yield from ([item] for item in pipe)

    def sample(pipe, sample_rate, strata):
        seen["sample_rate"] = sample_rate
        yield from pipe

    def batch(pipe, batch_size):
        yield list(pipe)

    def transform(pipe, schema, strata, interprocess_encoding_context):
        yield from pipe

    def mask(pipe, schema):
        yield from pipe

    monkeypatch.setattr(streaming, "observe", observe)
    monkeypatch.setattr(streaming, "process", process)
    monkeypatch.setattr(streaming, "sample", sample)
    monkeypatch.setattr(streaming, "batch", batch)
    monkeypatch.setattr(streaming, "transform", transform)
    monkeypatch.setattr(streaming, "mask", mask)

    batch_dataset = BatchDataset(
        schema=SimpleNamespace(requests={}),
        root="/tmp/json2vec-test",
        suffix=Suffix.ndjson,
        pattern=re.compile(r".*\.ndjson$"),
        preprocessor=None,
        preprocessor_kwargs={},
        interprocess_encoding_context={},
        batch_size=2,
        strata=Strata.train,
        sharding=ShardingStrategy.chunk,
        chunk_batch_size=1,
        file_buffer_size=1,
        observation_buffer_size=1,
        sample_rate=0.25,
    )

    assert list(batch_dataset) == [[[{"id": 1}]]]
    assert seen["sample_rate"] == 0.25


def test_batch_dataset_configures_distributed_state(monkeypatch: pytest.MonkeyPatch):
    class DistributedState:
        def __init__(self):
            self.calls = []

        def configure_distributed(self, global_rank: int, world_size: int):
            self.calls.append((global_rank, world_size))

    state = DistributedState()

    def observe(root, suffix, pattern, strata, sharding, chunk_batch_size, file_buffer_size, global_rank, world_size):
        yield {"id": global_rank, "world_size": world_size}

    def process(pipe, preprocessor, preprocessor_kwargs, strata, interprocess_encoding_context):
        yield from ([item] for item in pipe)

    def sample(pipe, sample_rate, strata):
        yield from pipe

    def batch(pipe, batch_size):
        yield list(pipe)

    def transform(pipe, schema, strata, interprocess_encoding_context):
        yield from pipe

    def mask(pipe, schema):
        yield from pipe

    monkeypatch.setattr(streaming, "observe", observe)
    monkeypatch.setattr(streaming, "process", process)
    monkeypatch.setattr(streaming, "sample", sample)
    monkeypatch.setattr(streaming, "batch", batch)
    monkeypatch.setattr(streaming, "transform", transform)
    monkeypatch.setattr(streaming, "mask", mask)

    batch_dataset = BatchDataset(
        schema=SimpleNamespace(requests={}),
        root="/tmp/json2vec-test",
        suffix=Suffix.ndjson,
        pattern=re.compile(r".*\.ndjson$"),
        preprocessor=None,
        preprocessor_kwargs={},
        interprocess_encoding_context={"root/category": state},
        batch_size=2,
        strata=Strata.train,
        sharding=ShardingStrategy.chunk,
        chunk_batch_size=1,
        file_buffer_size=1,
        observation_buffer_size=1,
        sample_rate=1.0,
        global_rank=2,
        world_size=4,
    )

    assert list(batch_dataset) == [[[{"id": 2, "world_size": 4}]]]
    assert state.calls == [(2, 4)]


def test_mask_uses_direct_field_rates():
    class Field:
        def __init__(self):
            self.calls = []

        def mask(self, p_mask: float, **kwargs):
            self.calls.append((p_mask, kwargs))

    class Schema(SimpleNamespace):
        def branch_masks_for(self, address):
            return ()

    first = Field()
    second = Field()
    schema = Schema(
        active_requests={
            "root/first": SimpleNamespace(p_mask=0.25, p_prune=0.0),
            "root/second": SimpleNamespace(p_mask=0.0, p_prune=0.0),
        },
    )

    output = list(iterables.mask.__wrapped__([{"root/first": first, "root/second": second}], schema))

    assert output == [{"root/first": first, "root/second": second}]
    assert first.calls == [
        (
            0.25,
            {
                "p_prune": 0.0,
                "branch_masks": (),
                "address": "root/first",
                "schema": schema,
            },
        )
    ]
    assert second.calls == []


def test_target_uses_direct_field_rates():
    class Field:
        def __init__(self):
            self.calls = []

        def target(self, p_prune: float):
            self.calls.append(p_prune)

    first = Field()
    second = Field()
    schema = SimpleNamespace(
        active_requests={
            "root/first": SimpleNamespace(p_prune=0.0),
            "root/second": SimpleNamespace(p_prune=0.75),
        },
    )

    output = list(iterables.target.__wrapped__([{"root/first": first, "root/second": second}], schema))

    assert output == [{"root/first": first, "root/second": second}]
    assert first.calls == []
    assert second.calls == [0.75]

import enum
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from json2vec.data import datasets
from json2vec.processors.base import Processor, ProcessorMode
from json2vec.structs.enums import ShardingStrategy, Strata, Suffix


def _dataset_for_suffix(suffix: Suffix):
    return SimpleNamespace(suffix=suffix)


def _dataset_for_fetch(root: Path):
    return SimpleNamespace(
        root=str(root),
        patterns={strata: r".*\.ndjson$" for strata in Strata},
    )


def _datamodule_hyperparameters():
    return datasets.Hyperparameters.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "record",
                "type": "array",
                "max_length": 1,
                "fields": [],
            },
        }
    )


def _datamodule_dataset():
    return datasets.Dataset(root=None, processor=None)


def test_sha256():
    assert datasets.sha256("test", 32) == 2676412545
    assert datasets.sha256("test", 64) == 11495104353665842533
    assert datasets.sha256("test", 128) == 212047248112658246449511647784264716309


def test_is_assigned_to_worker_partitions_shards():
    key = "chunk:s3://bucket/path/file.parquet:7"
    owners = [
        worker_id
        for worker_id in range(4)
        if datasets._is_assigned_to_worker(key, worker_id=worker_id, num_workers=4)
    ]
    assert len(owners) == 1


def test_is_assigned_to_worker_single_worker():
    assert datasets._is_assigned_to_worker("record:file:42", worker_id=0, num_workers=1)


def test_worker_identity_combines_rank_and_dataloader_worker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(datasets, "get_worker_info", lambda: SimpleNamespace(id=2, num_workers=4))

    assert datasets._worker_identity(global_rank=1, world_size=3) == (6, 12)


def test_query():
    expr = datasets.query("[*].foo.bar")
    result = expr.search([[{"foo": {"bar": 42}}]])
    assert result == [[42]]


def test_read_ndjson_chunk_sharding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "records.ndjson"
    records = [{"id": i} for i in range(5)]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    monkeypatch.setattr(datasets, "_worker_identity", lambda **_: (0, 2))

    def assign_first_chunk_only(shard_key: str, worker_id: int, num_workers: int) -> bool:
        return int(shard_key.rsplit(":", 1)[1]) == 0

    monkeypatch.setattr(datasets, "_is_assigned_to_worker", assign_first_chunk_only)

    dataset = _dataset_for_suffix(Suffix.ndjson)
    output = list(
        datasets.read.__wrapped__(
            [str(path)],
            dataset=dataset,
            sharding=ShardingStrategy.chunk,
            chunk_batch_size=2,
        )
    )
    assert output == records[:2]


def test_read_ndjson_record_sharding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "records.ndjson"
    records = [{"id": i} for i in range(6)]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    monkeypatch.setattr(datasets, "_worker_identity", lambda **_: (0, 2))

    def assign_even_records(shard_key: str, worker_id: int, num_workers: int) -> bool:
        return int(shard_key.rsplit(":", 1)[1]) % 2 == 0

    monkeypatch.setattr(datasets, "_is_assigned_to_worker", assign_even_records)

    dataset = _dataset_for_suffix(Suffix.ndjson)
    output = list(
        datasets.read.__wrapped__(
            [str(path)],
            dataset=dataset,
            sharding=ShardingStrategy.record,
            chunk_batch_size=3,
        )
    )
    assert output == [records[index] for index in (0, 2, 4)]


def test_fetch_file_sharding_filters_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "keep.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "skip.ndjson").write_text("", encoding="utf-8")

    monkeypatch.setattr(datasets, "_worker_identity", lambda **_: (0, 2))

    def assign_keep_only(shard_key: str, worker_id: int, num_workers: int) -> bool:
        return "keep.ndjson" in shard_key

    monkeypatch.setattr(datasets, "_is_assigned_to_worker", assign_keep_only)

    dataset = _dataset_for_fetch(tmp_path)
    files = list(
        datasets.fetch.__wrapped__(
            dataset=dataset,
            strata=Strata.predict,
            sharding=ShardingStrategy.file,
        )
    )
    assert {Path(path).name for path in files} == {"keep.ndjson"}


def test_fetch_without_file_sharding_returns_all_matching_files(tmp_path: Path):
    (tmp_path / "first.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "second.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("", encoding="utf-8")

    dataset = _dataset_for_fetch(tmp_path)
    files = list(
        datasets.fetch.__wrapped__(
            dataset=dataset,
            strata=Strata.predict,
            sharding=ShardingStrategy.chunk,
        )
    )
    assert {Path(path).name for path in files} == {"first.ndjson", "second.ndjson"}


def test_fetch_without_patterns_returns_all_files(tmp_path: Path):
    (tmp_path / "first.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "second.csv").write_text("", encoding="utf-8")

    dataset = SimpleNamespace(root=str(tmp_path), patterns=None)
    files = list(
        datasets.fetch.__wrapped__(
            dataset=dataset,
            strata=Strata.validate,
            sharding=ShardingStrategy.chunk,
        )
    )
    assert {Path(path).name for path in files} == {"first.ndjson", "second.csv"}


def test_observe_with_none_root_seeds_single_worker(monkeypatch: pytest.MonkeyPatch):
    dataset = SimpleNamespace(
        root=None,
        suffix=Suffix.ndjson,
        patterns={strata: r".*" for strata in Strata},
    )

    monkeypatch.setattr(datasets, "_worker_identity", lambda **_: (0, 2))
    monkeypatch.setattr(
        datasets,
        "_is_assigned_to_worker",
        lambda shard_key, worker_id, num_workers: worker_id == 0,
    )

    output = list(
        datasets.observe.__wrapped__(
            dataset=dataset,
            strata=Strata.train,
            sharding=ShardingStrategy.chunk,
            chunk_batch_size=2,
            file_buffer_size=4,
        )
    )
    assert output == [{}]


def test_observe_with_none_root_yields_nothing_for_unassigned_worker(monkeypatch: pytest.MonkeyPatch):
    dataset = SimpleNamespace(
        root=None,
        suffix=Suffix.ndjson,
        patterns={strata: r".*" for strata in Strata},
    )

    monkeypatch.setattr(datasets, "_worker_identity", lambda **_: (1, 2))
    monkeypatch.setattr(
        datasets,
        "_is_assigned_to_worker",
        lambda shard_key, worker_id, num_workers: worker_id == 0,
    )

    output = list(
        datasets.observe.__wrapped__(
            dataset=dataset,
            strata=Strata.train,
            sharding=ShardingStrategy.chunk,
            chunk_batch_size=2,
            file_buffer_size=4,
        )
    )
    assert output == []


def test_observe_polars_yields_dataframe_rows():
    frame = pl.DataFrame({"id": [1, 2], "name": ["alpha", "beta"]})

    output = list(
        datasets.observe_polars.__wrapped__(
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
            datasets.observe_polars.__wrapped__(
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


def test_read_unsupported_suffix_raises_value_error():
    class UnknownSuffix(enum.StrEnum):
        bad = "bad"

    dataset = _dataset_for_suffix(UnknownSuffix.bad)
    with pytest.raises(ValueError, match="Unsupported suffix: bad"):
        list(
            datasets.read.__wrapped__(
                [],
                dataset=dataset,
                sharding=ShardingStrategy.chunk,
                chunk_batch_size=2,
            )
        )


def test_process_transformation_processor_wraps_dict_output(monkeypatch: pytest.MonkeyPatch):
    dataset = SimpleNamespace(processor="__test_transformation", kwargs={})

    def transformation(observation: dict):
        return {"id": observation["id"]}

    processor = Processor(name="__test_transformation", func=transformation, mode=ProcessorMode.transformation)
    monkeypatch.setitem(datasets.PROCESSORS, "__test_transformation", processor)

    output = list(
        datasets.process.__wrapped__(
            [{"id": 1}, {"id": 2}],
            dataset=dataset,
            strata=Strata.train,
            interprocess_encoding_context={},
        )
    )
    assert output == [[{"id": 1}], [{"id": 2}]]


def test_process_generator_processor_wraps_list_outputs(monkeypatch: pytest.MonkeyPatch):
    dataset = SimpleNamespace(processor="__test_generator", kwargs={})

    def generator(observation: dict):
        return [{"id": observation["id"]}, {"id": observation["id"] + 100}]

    processor = Processor(name="__test_generator", func=generator, mode=ProcessorMode.generator)
    monkeypatch.setitem(datasets.PROCESSORS, "__test_generator", processor)

    output = list(
        datasets.process.__wrapped__(
            [{"id": 1}],
            dataset=dataset,
            strata=Strata.train,
            interprocess_encoding_context={},
        )
    )
    assert output == [[{"id": 1}], [{"id": 101}]]


def test_process_generator_processor_receives_strata_and_state(monkeypatch: pytest.MonkeyPatch):
    dataset = SimpleNamespace(processor="__test_generator_context", kwargs={})

    def generator(observation: dict, strata, interprocess_encoding_context):
        yield {"id": observation["id"], "strata": strata, "marker": interprocess_encoding_context["marker"]}

    processor = Processor(name="__test_generator_context", func=generator, mode=ProcessorMode.generator)
    monkeypatch.setitem(datasets.PROCESSORS, "__test_generator_context", processor)

    output = list(
        datasets.process.__wrapped__(
            [{"id": 1}],
            dataset=dataset,
            strata=Strata.validate,
            interprocess_encoding_context={"marker": "seen"},
        )
    )
    assert output == [[{"id": 1, "strata": Strata.validate, "marker": "seen"}]]


def test_process_transformation_processor_rejects_non_dict(monkeypatch: pytest.MonkeyPatch):
    dataset = SimpleNamespace(processor="__test_invalid_transformation", kwargs={})

    def transformation(observation: dict):
        return observation["id"]

    processor = Processor(
        name="__test_invalid_transformation",
        func=transformation,
        mode=ProcessorMode.transformation,
    )
    monkeypatch.setitem(datasets.PROCESSORS, "__test_invalid_transformation", processor)

    with pytest.raises(TypeError, match="must produce dict objects"):
        list(
            datasets.process.__wrapped__(
                [{"id": 1}],
                dataset=dataset,
                strata=Strata.train,
                interprocess_encoding_context={},
            )
        )


def test_process_without_processor_still_wraps_root_array():
    dataset = SimpleNamespace(processor=None, kwargs={})

    output = list(
        datasets.process.__wrapped__(
            [{"id": 1}, {"id": 2}],
            dataset=dataset,
            strata=Strata.train,
            interprocess_encoding_context={},
        )
    )
    assert output == [[{"id": 1}], [{"id": 2}]]


def test_batch_splits_and_preserves_tail():
    chunks = list(datasets.batch.__wrapped__([1, 2, 3, 4, 5], batch_size=2))
    assert chunks == [[1, 2], [3, 4], [5]]


def test_sample_predict_is_identity():
    data = list(range(8))
    output = list(datasets.sample.__wrapped__(data, sample_rate=0.1, strata=Strata.predict))
    assert output == data


def test_sample_filters_non_predict_with_sample_rate():
    random.seed(3)
    data = list(range(8))
    output = list(datasets.sample.__wrapped__(data, sample_rate=0.5, strata=Strata.train))
    assert output == [0, 2, 5, 6]


def test_shuffle_predict_is_identity():
    data = list(range(8))
    output = list(datasets.shuffle.__wrapped__(data, size=3, strata=Strata.predict))
    assert output == data


def test_shuffle_non_predict_preserves_elements():
    random.seed(7)
    data = list(range(12))
    output = list(datasets.shuffle.__wrapped__(data, size=4, strata=Strata.train))
    assert sorted(output) == data
    assert len(output) == len(data)


def test_shuffle_non_predict_preserves_duplicate_counts():
    random.seed(11)
    data = [1, 1, 1, 2, 2, 3, 4, 4]
    output = list(datasets.shuffle.__wrapped__(data, size=3, strata=Strata.train))
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
    output = list(datasets.shuffle.__wrapped__(iterator, size=3, strata=Strata.train))
    assert sorted(output) == [0, 1, 2, 3, 4]
    assert iterator.stop_count == 1


def test_jmespath_resolution_monitor_raises_for_empty_result():
    monitor = datasets.JMESPathResolutionMonitor(every=1)

    with pytest.raises(ValueError, match="JMESPath query returned empty result"):
        monitor.observe(address="root/id", expression="[*].id", result=[])


def test_jmespath_resolution_monitor_ignores_empty_result_until_threshold():
    monitor = datasets.JMESPathResolutionMonitor(every=3)

    monitor.observe(address="root/id", expression="[*].id", result=[])
    monitor.observe(address="root/id", expression="[*].id", result=[])


def test_jmespath_resolution_monitor_accepts_nested_observed_value():
    monitor = datasets.JMESPathResolutionMonitor(every=1)

    monitor.observe(address="root/id", expression="[*].id", result=[[None, {"id": 0}]])


def test_streaming_datamodule_accepts_named_loader_configuration_per_strata():
    module = datasets.StreamingDataModule(
        hyperparameters=_datamodule_hyperparameters(),
        dataset=_datamodule_dataset(),
        interprocess_encoding_context={},
        batch_size=2,
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
    assert module.sharding[Strata.validate] == ShardingStrategy.chunk
    assert module.chunk_batch_size[Strata.train] == 7
    assert module.chunk_batch_size[Strata.validate] == 4096
    assert module.file_buffer_size[Strata.train] == 11
    assert module.file_buffer_size[Strata.validate] == 1
    assert module.observation_buffer_size[Strata.train] == 13
    assert module.observation_buffer_size[Strata.validate] == 1
    assert module.sample_rate[Strata.train] == 0.5
    assert module.sample_rate[Strata.validate] == 1.0


def test_streaming_datamodule_rejects_invalid_loader_configuration():
    kwargs = {
        "hyperparameters": _datamodule_hyperparameters(),
        "dataset": _datamodule_dataset(),
        "state": {},
    }

    with pytest.raises(BeartypeCallHintParamViolation):
        datasets.StreamingDataModule(**kwargs, batch_size=0)

    with pytest.raises(BeartypeCallHintParamViolation):
        datasets.StreamingDataModule(**kwargs, batch_size=2, num_workers={Strata.train: True})

    with pytest.raises(BeartypeCallHintParamViolation):
        datasets.StreamingDataModule(**kwargs, batch_size=2, sample_rate={Strata.train: 0.0})


def test_polars_datamodule_accepts_dataframe_and_loader_configuration_per_strata():
    frame = pl.DataFrame({"id": [1, 2]})
    module = datasets.PolarsDataModule(
        hyperparameters=_datamodule_hyperparameters(),
        dataframe=frame,
        dataset=_datamodule_dataset(),
        interprocess_encoding_context={},
        batch_size=2,
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


def test_polars_datamodule_from_model_accepts_named_splits():
    train = pl.DataFrame({"id": [1]})
    predict = pl.DataFrame({"id": [2]})
    model = SimpleNamespace(
        hyperparameters=_datamodule_hyperparameters(),
        interprocess_encoding_context={},
        batch_size=2,
    )

    module = datasets.PolarsDataModule.from_model(
        model,
        train=train,
        predict=predict,
        processor=None,
        num_workers=0,
    )

    assert module.dataframes[Strata.train] is train
    assert module.dataframes[Strata.predict] is predict
    assert set(module.dataframes) == {Strata.train, Strata.predict}
    assert module.dataset.root is None
    assert module.dataset.processor is None
    assert module.val_dataloader() is None


def test_polars_datamodule_from_model_requires_at_least_one_split():
    model = SimpleNamespace(
        hyperparameters=_datamodule_hyperparameters(),
        interprocess_encoding_context={},
        batch_size=2,
    )

    with pytest.raises(ValueError, match="at least one dataframe split is required"):
        datasets.PolarsDataModule.from_model(model)


def test_polars_datamodule_accepts_partial_dataframe_mapping_until_loader_requested():
    module = datasets.PolarsDataModule(
        hyperparameters=_datamodule_hyperparameters(),
        dataframe={Strata.train: pl.DataFrame({"id": [1]})},
        dataset=_datamodule_dataset(),
        interprocess_encoding_context={},
        batch_size=2,
        num_workers=0,
    )

    assert set(module.dataframes) == {Strata.train}
    assert module.val_dataloader() is None

    with pytest.raises(ValueError, match="no dataframe configured"):
        module.dataloader(strata=Strata.validate)


def test_polars_datamodule_rejects_dataset_with_file_root(tmp_path: Path):
    with pytest.raises(ValueError, match="must not define root"):
        datasets.PolarsDataModule(
            hyperparameters=_datamodule_hyperparameters(),
            dataframe=pl.DataFrame({"id": [1]}),
            dataset=datasets.Dataset(
                root=str(tmp_path),
                suffix=Suffix.ndjson,
                patterns={strata: r".*" for strata in Strata},
            ),
            interprocess_encoding_context={},
            batch_size=2,
        )


def test_polars_batch_dataset_reads_dataframe_rows_through_pipeline(monkeypatch: pytest.MonkeyPatch):
    def transform(pipe, hyperparameters, strata, interprocess_encoding_context):
        yield from pipe

    def mask(pipe, hyperparameters):
        yield from pipe

    def target(pipe, hyperparameters):
        yield from pipe

    monkeypatch.setattr(datasets, "transform", transform)
    monkeypatch.setattr(datasets, "mask", mask)
    monkeypatch.setattr(datasets, "target", target)

    batch_dataset = datasets.PolarsBatchDataset(
        hyperparameters=SimpleNamespace(requests={}),
        dataframe=pl.DataFrame({"id": [1, 2, 3]}),
        dataset=datasets.Dataset(root=None, processor=None),
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

    def observe(dataset, strata, sharding, chunk_batch_size, file_buffer_size):
        yield {"id": 1}

    def process(pipe, dataset, strata, interprocess_encoding_context):
        yield from ([item] for item in pipe)

    def sample(pipe, sample_rate, strata):
        seen["sample_rate"] = sample_rate
        yield from pipe

    def batch(pipe, batch_size):
        yield list(pipe)

    def transform(pipe, hyperparameters, strata, interprocess_encoding_context):
        yield from pipe

    def mask(pipe, hyperparameters):
        yield from pipe

    def target(pipe, hyperparameters):
        yield from pipe

    monkeypatch.setattr(datasets, "observe", observe)
    monkeypatch.setattr(datasets, "process", process)
    monkeypatch.setattr(datasets, "sample", sample)
    monkeypatch.setattr(datasets, "batch", batch)
    monkeypatch.setattr(datasets, "transform", transform)
    monkeypatch.setattr(datasets, "mask", mask)
    monkeypatch.setattr(datasets, "target", target)

    batch_dataset = datasets.BatchDataset(
        hyperparameters=SimpleNamespace(requests={}),
        dataset=SimpleNamespace(),
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

    def observe(dataset, strata, sharding, chunk_batch_size, file_buffer_size, global_rank, world_size):
        yield {"id": global_rank, "world_size": world_size}

    def process(pipe, dataset, strata, interprocess_encoding_context):
        yield from ([item] for item in pipe)

    def sample(pipe, sample_rate, strata):
        yield from pipe

    def batch(pipe, batch_size):
        yield list(pipe)

    def transform(pipe, hyperparameters, strata, interprocess_encoding_context):
        yield from pipe

    def mask(pipe, hyperparameters):
        yield from pipe

    def target(pipe, hyperparameters):
        yield from pipe

    monkeypatch.setattr(datasets, "observe", observe)
    monkeypatch.setattr(datasets, "process", process)
    monkeypatch.setattr(datasets, "sample", sample)
    monkeypatch.setattr(datasets, "batch", batch)
    monkeypatch.setattr(datasets, "transform", transform)
    monkeypatch.setattr(datasets, "mask", mask)
    monkeypatch.setattr(datasets, "target", target)

    batch_dataset = datasets.BatchDataset(
        hyperparameters=SimpleNamespace(requests={}),
        dataset=SimpleNamespace(),
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


def test_mask_uses_resolved_field_rates():
    class Field:
        def __init__(self):
            self.calls = []

        def mask(self, p_mask: float):
            self.calls.append(p_mask)

    first = Field()
    second = Field()
    hyperparameters = SimpleNamespace(
        requests={"root/first": object(), "root/second": object()},
        resolved_p_mask=lambda address: 0.25 if address == "root/first" else 0.0,
    )

    output = list(datasets.mask.__wrapped__([{"root/first": first, "root/second": second}], hyperparameters))

    assert output == [{"root/first": first, "root/second": second}]
    assert first.calls == [0.25]
    assert second.calls == []


def test_target_uses_resolved_field_rates():
    class Field:
        def __init__(self):
            self.calls = []

        def target(self, p_prune: float):
            self.calls.append(p_prune)

    first = Field()
    second = Field()
    hyperparameters = SimpleNamespace(
        requests={"root/first": object(), "root/second": object()},
        resolved_p_prune=lambda address: 0.0 if address == "root/first" else 0.75,
    )

    output = list(datasets.target.__wrapped__([{"root/first": first, "root/second": second}], hyperparameters))

    assert output == [{"root/first": first, "root/second": second}]
    assert first.calls == []
    assert second.calls == [0.75]

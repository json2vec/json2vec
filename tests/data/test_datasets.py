import enum
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from json2vec.data import datasets
from json2vec.processors.base import Processor, ProcessorMode
from json2vec.structs.enums import ShardingStrategy, Strata, Suffix


def _session_for_suffix(suffix: Suffix):
    return SimpleNamespace(dataset=SimpleNamespace(suffix=suffix))


def _session_for_fetch(root: Path):
    return SimpleNamespace(
        dataset=SimpleNamespace(
            root=str(root),
            patterns={strata: r".*\.ndjson$" for strata in Strata},
        )
    )


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


def test_query():
    expr = datasets.query("[*].foo.bar")
    result = expr.search([[{"foo": {"bar": 42}}]])
    assert result == [[42]]


def test_read_ndjson_chunk_sharding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "records.ndjson"
    records = [{"id": i} for i in range(5)]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    monkeypatch.setattr(datasets, "_worker_identity", lambda: (0, 2))

    def assign_first_chunk_only(shard_key: str, worker_id: int, num_workers: int) -> bool:
        return int(shard_key.rsplit(":", 1)[1]) == 0

    monkeypatch.setattr(datasets, "_is_assigned_to_worker", assign_first_chunk_only)

    session = _session_for_suffix(Suffix.ndjson)
    output = list(
        datasets.read.__wrapped__(
            [str(path)],
            session=session,
            sharding=ShardingStrategy.chunk,
            chunk_batch_size=2,
        )
    )
    assert output == records[:2]


def test_read_ndjson_record_sharding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "records.ndjson"
    records = [{"id": i} for i in range(6)]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    monkeypatch.setattr(datasets, "_worker_identity", lambda: (0, 2))

    def assign_even_records(shard_key: str, worker_id: int, num_workers: int) -> bool:
        return int(shard_key.rsplit(":", 1)[1]) % 2 == 0

    monkeypatch.setattr(datasets, "_is_assigned_to_worker", assign_even_records)

    session = _session_for_suffix(Suffix.ndjson)
    output = list(
        datasets.read.__wrapped__(
            [str(path)],
            session=session,
            sharding=ShardingStrategy.record,
            chunk_batch_size=3,
        )
    )
    assert output == [records[index] for index in (0, 2, 4)]


def test_fetch_file_sharding_filters_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "keep.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "skip.ndjson").write_text("", encoding="utf-8")

    monkeypatch.setattr(datasets, "_worker_identity", lambda: (0, 2))

    def assign_keep_only(shard_key: str, worker_id: int, num_workers: int) -> bool:
        return "keep.ndjson" in shard_key

    monkeypatch.setattr(datasets, "_is_assigned_to_worker", assign_keep_only)

    session = _session_for_fetch(tmp_path)
    files = list(
        datasets.fetch.__wrapped__(
            session=session,
            strata=Strata.predict,
            sharding=ShardingStrategy.file,
        )
    )
    assert {Path(path).name for path in files} == {"keep.ndjson"}


def test_fetch_without_file_sharding_returns_all_matching_files(tmp_path: Path):
    (tmp_path / "first.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "second.ndjson").write_text("", encoding="utf-8")
    (tmp_path / "ignore.csv").write_text("", encoding="utf-8")

    session = _session_for_fetch(tmp_path)
    files = list(
        datasets.fetch.__wrapped__(
            session=session,
            strata=Strata.predict,
            sharding=ShardingStrategy.chunk,
        )
    )
    assert {Path(path).name for path in files} == {"first.ndjson", "second.ndjson"}


def test_observe_with_none_root_seeds_single_worker(monkeypatch: pytest.MonkeyPatch):
    session = SimpleNamespace(
        dataset=SimpleNamespace(
            root=None,
            file_buffer_size=4,
            suffix=Suffix.ndjson,
            patterns={strata: r".*" for strata in Strata},
        )
    )

    monkeypatch.setattr(datasets, "_worker_identity", lambda: (0, 2))
    monkeypatch.setattr(
        datasets,
        "_is_assigned_to_worker",
        lambda shard_key, worker_id, num_workers: worker_id == 0,
    )

    output = list(
        datasets.observe.__wrapped__(
            session=session,
            strata=Strata.train,
            sharding=ShardingStrategy.chunk,
            chunk_batch_size=2,
        )
    )
    assert output == [{}]


def test_observe_with_none_root_yields_nothing_for_unassigned_worker(monkeypatch: pytest.MonkeyPatch):
    session = SimpleNamespace(
        dataset=SimpleNamespace(
            root=None,
            file_buffer_size=4,
            suffix=Suffix.ndjson,
            patterns={strata: r".*" for strata in Strata},
        )
    )

    monkeypatch.setattr(datasets, "_worker_identity", lambda: (1, 2))
    monkeypatch.setattr(
        datasets,
        "_is_assigned_to_worker",
        lambda shard_key, worker_id, num_workers: worker_id == 0,
    )

    output = list(
        datasets.observe.__wrapped__(
            session=session,
            strata=Strata.train,
            sharding=ShardingStrategy.chunk,
            chunk_batch_size=2,
        )
    )
    assert output == []


def test_read_unsupported_suffix_raises_value_error():
    class UnknownSuffix(enum.StrEnum):
        bad = "bad"

    session = _session_for_suffix(UnknownSuffix.bad)
    with pytest.raises(ValueError, match="Unsupported suffix: bad"):
        list(
            datasets.read.__wrapped__(
                [],
                session=session,
                sharding=ShardingStrategy.chunk,
                chunk_batch_size=2,
            )
        )


def test_process_transformation_processor_wraps_dict_output(monkeypatch: pytest.MonkeyPatch):
    session = SimpleNamespace(dataset=SimpleNamespace(processor="__test_transformation", kwargs={}))

    def transformation(observation: dict):
        return {"id": observation["id"]}

    processor = Processor(name="__test_transformation", func=transformation, mode=ProcessorMode.transformation)
    monkeypatch.setitem(datasets.PROCESSORS, "__test_transformation", processor)

    output = list(
        datasets.process.__wrapped__(
            [{"id": 1}, {"id": 2}],
            session=session,
            strata=Strata.train,
            state={},
        )
    )
    assert output == [[{"id": 1}], [{"id": 2}]]


def test_process_generator_processor_wraps_list_outputs(monkeypatch: pytest.MonkeyPatch):
    session = SimpleNamespace(dataset=SimpleNamespace(processor="__test_generator", kwargs={}))

    def generator(observation: dict):
        return [{"id": observation["id"]}, {"id": observation["id"] + 100}]

    processor = Processor(name="__test_generator", func=generator, mode=ProcessorMode.generator)
    monkeypatch.setitem(datasets.PROCESSORS, "__test_generator", processor)

    output = list(
        datasets.process.__wrapped__(
            [{"id": 1}],
            session=session,
            strata=Strata.train,
            state={},
        )
    )
    assert output == [[{"id": 1}], [{"id": 101}]]


def test_process_generator_processor_receives_strata_and_state(monkeypatch: pytest.MonkeyPatch):
    session = SimpleNamespace(dataset=SimpleNamespace(processor="__test_generator_context", kwargs={}))

    def generator(observation: dict, strata, state):
        yield {"id": observation["id"], "strata": strata, "marker": state["marker"]}

    processor = Processor(name="__test_generator_context", func=generator, mode=ProcessorMode.generator)
    monkeypatch.setitem(datasets.PROCESSORS, "__test_generator_context", processor)

    output = list(
        datasets.process.__wrapped__(
            [{"id": 1}],
            session=session,
            strata=Strata.validate,
            state={"marker": "seen"},
        )
    )
    assert output == [[{"id": 1, "strata": Strata.validate, "marker": "seen"}]]


def test_process_transformation_processor_rejects_non_dict(monkeypatch: pytest.MonkeyPatch):
    session = SimpleNamespace(dataset=SimpleNamespace(processor="__test_invalid_transformation", kwargs={}))

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
                session=session,
                strata=Strata.train,
                state={},
            )
        )


def test_process_without_processor_still_wraps_root_context():
    session = SimpleNamespace(dataset=SimpleNamespace(processor=None, kwargs={}))

    output = list(
        datasets.process.__wrapped__(
            [{"id": 1}, {"id": 2}],
            session=session,
            strata=Strata.train,
            state={},
        )
    )
    assert output == [[{"id": 1}], [{"id": 2}]]


def test_batch_splits_and_preserves_tail():
    session = SimpleNamespace(structure=SimpleNamespace(batch_size=2))
    chunks = list(datasets.batch.__wrapped__([1, 2, 3, 4, 5], session=session))
    assert chunks == [[1, 2], [3, 4], [5]]


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


def test_spotcheck_raises_for_repeated_empty_result(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(datasets, "_jmespath_counter", Counter())
    with pytest.raises(ValueError, match="JMESPath query returned empty result"):
        datasets.spotcheck([], "root/id", every=1)


def test_spotcheck_ignores_empty_result_until_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(datasets, "_jmespath_counter", Counter())
    datasets.spotcheck([], "root/id", every=3)
    datasets.spotcheck([], "root/id", every=3)

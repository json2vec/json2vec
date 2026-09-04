from __future__ import annotations

from collections.abc import Iterator

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from tensordict import TensorDict
from torch.utils.data import IterableDataset

import relflow as rf
from relflow.data.arrow import Batch, Encoded
from relflow.data.datasets import arrow
from relflow.data.datasets.custom import adapt
from relflow.structs.enums import Strata


def model(batch_size: int = 2) -> rf.Model:
    return rf.Model(
        id=rf.Number,
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=batch_size,
    )


class Records(IterableDataset):
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        yield from self.values


def collect(dataset: arrow.ArrowDataset, monkeypatch: pytest.MonkeyPatch) -> list[Batch]:
    monkeypatch.setattr(
        arrow,
        "encode",
        lambda *, batch, **kwargs: Encoded(
            tensors=TensorDict({}, batch_size=[len(batch)]),
            source=batch,
        ),
    )
    return [item.source for item in dataset]


def test_public_surface_has_four_modules_and_no_source_specific_datasets():
    import relflow.data.datasets as datasets

    assert rf.ArrowDataModule is arrow.ArrowDataModule
    assert datasets.__all__ == [
        "ArrowDataModule",
        "CustomDataModule",
        "PolarsDataModule",
        "SyntheticDataModule",
    ]
    assert not hasattr(rf, "StreamingDataModule")
    assert not hasattr(datasets, "PolarsBatchDataset")
    assert not hasattr(datasets, "CustomBatchDataset")


@pytest.mark.parametrize(
    "source",
    [
        pa.table({"id": [1, 2]}),
        pa.record_batch({"id": [1, 2]}),
    ],
)
def test_arrow_module_accepts_table_and_record_batch(source):
    module = rf.ArrowDataModule(model=model(), train=source, shuffle=False)

    assert module.sources == {Strata.train: source}
    assert isinstance(module.train_dataloader().dataset, arrow.ArrowDataset)
    assert module.val_dataloader() is None


def test_arrow_module_accepts_datasets_and_rejects_python_or_one_shot_sources():
    with pytest.raises(TypeError, match="CustomDataModule or SyntheticDataModule"):
        rf.ArrowDataModule(model=model(), train={"id": 1})

    reader = pa.RecordBatchReader.from_batches(pa.schema([("id", pa.int64())]), [])
    with pytest.raises(TypeError, match="one-shot RecordBatchReader"):
        rf.ArrowDataModule(model=model(), train=reader)

    dataset = ds.dataset(pa.table({"id": [1]}))
    module = rf.ArrowDataModule(model=model(), train=dataset)
    assert module.sources[Strata.train] is dataset
    with pytest.raises(TypeError, match="configured Scanner"):
        rf.ArrowDataModule(model=model(), train=dataset.scanner())


def test_empty_arrow_factory_must_declare_its_schema():
    module = rf.ArrowDataModule(model=model(), validate=lambda: iter(()), shuffle=False)

    with pytest.raises(ValueError, match="empty Arrow factory"):
        list(module.val_dataloader().dataset)


def test_zero_batch_reader_preserves_its_declared_schema(monkeypatch: pytest.MonkeyPatch):
    schema = pa.schema(
        [pa.field("id", pa.int64(), metadata={b"unit": b"stable"})],
        metadata={b"source": b"reader"},
    )

    def source():
        return pa.RecordBatchReader.from_batches(schema, [])

    module = rf.ArrowDataModule(model=model(), validate=source, shuffle=False)

    assert collect(module.val_dataloader().dataset, monkeypatch) == []
    assert module.schemas[Strata.validate].source.equals(schema, check_metadata=True)
    assert module.schemas[Strata.validate].processed.equals(schema, check_metadata=True)


def test_source_schema_lock_survives_repeated_dataloader_creation(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def source():
        nonlocal calls
        calls += 1
        dtype = pa.int64() if calls == 1 else pa.string()
        value = 1 if calls == 1 else "1"
        yield pa.record_batch({"id": pa.array([value], type=dtype)})

    module = rf.ArrowDataModule(model=model(), validate=source, shuffle=False)

    collect(module.val_dataloader().dataset, monkeypatch)
    with pytest.raises(TypeError, match="Arrow source schema changed"):
        collect(module.val_dataloader().dataset, monkeypatch)


def test_processed_schema_lock_survives_repeated_dataloader_creation(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    @rf.preprocess(produces=("id",))
    def change(batch: rf.Batch) -> rf.Batch:
        nonlocal calls
        calls += 1
        if calls == 1:
            return batch
        return batch.replace(pa.table({"id": pa.compute.cast(batch.data["id"], pa.string())}))

    module = rf.ArrowDataModule(
        model=model(),
        validate=pa.table({"id": [1]}),
        preprocessor=change,
        shuffle=False,
    )

    collect(module.val_dataloader().dataset, monkeypatch)
    with pytest.raises(TypeError, match="preprocessor schema changed"):
        collect(module.val_dataloader().dataset, monkeypatch)


def test_arrow_pipeline_batches_without_materializing_rows(monkeypatch: pytest.MonkeyPatch):
    source = pa.table({"id": list(range(5)), "nested": [[1, 2], [], [3], [4, 5], None]})
    module = rf.ArrowDataModule(model=model(), validate=source, shuffle=False)
    dataset = module.val_dataloader().dataset

    batches = collect(dataset, monkeypatch)

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert pa.concat_tables([batch.data for batch in batches]).equals(source)
    logical = pa.concat_arrays(
        [
            part
            for batch in batches
            for part in (batch.identity.chunks if isinstance(batch.identity, pa.ChunkedArray) else [batch.identity])
        ]
    ).field("logical")
    assert len(set(logical.to_pylist())) == 5


@pytest.mark.parametrize(
    ("drop_last", "expected_sizes", "expected_values"),
    [
        (False, [3, 2], set(range(20))),
        (True, [3], set(range(12))),
    ],
)
def test_distributed_batches_are_disjoint_and_have_equal_tails(drop_last, expected_sizes, expected_values):
    source = arrow.convert(pa.table({"id": list(range(23))}), namespace="distributed", offset=0)
    ranks = [
        list(
            arrow.distribute(
                (source,),
                size=3,
                global_rank=global_rank,
                world_size=4,
                drop_last=drop_last,
            )
        )
        for global_rank in range(4)
    ]

    assert [[len(batch) for batch in batches] for batches in ranks] == [expected_sizes] * 4
    values = [{value for batch in batches for value in batch.data["id"].to_pylist()} for batches in ranks]
    assert set().union(*values) == expected_values
    assert sum(map(len, values)) == len(set().union(*values))


def test_distributed_ownership_is_independent_of_arrow_batch_boundaries():
    table = pa.table({"id": list(range(24))})

    def assignments(sizes):
        offset = 0
        batches = []
        for size in sizes:
            batches.append(arrow.convert(table.slice(offset, size), namespace="distributed", offset=offset))
            offset += size
        return [
            [
                value
                for batch in arrow.distribute(
                    batches,
                    size=3,
                    global_rank=global_rank,
                    world_size=4,
                    drop_last=False,
                )
                for value in batch.data["id"].to_pylist()
            ]
            for global_rank in range(4)
        ]

    assert assignments([24]) == assignments([1, 7, 3, 13])


def test_arrow_dataset_uses_distributed_rank_and_world_size(monkeypatch: pytest.MonkeyPatch):
    configured = rf.Model(
        id=rf.Category(size=32),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
    )
    module = rf.ArrowDataModule(
        model=configured,
        validate=pa.table({"id": ["a", "b", "c", "d", "e"]}),
        shuffle=False,
    )
    monkeypatch.setattr(arrow, "rank", lambda: 1)
    monkeypatch.setattr(arrow, "world_size", lambda: 2)

    dataset = module.val_dataloader().dataset
    batches = collect(dataset, monkeypatch)

    assert [batch.data["id"].to_pylist() for batch in batches] == [["b", "d"]]
    context = dataset.encoding_context[next(iter(dataset.encoding_context))]
    assert context.global_rank == 1


def test_arrow_dataset_source_scans_through_the_shared_pipeline(monkeypatch: pytest.MonkeyPatch):
    source = ds.dataset(pa.table({"id": list(range(5))}))
    module = rf.ArrowDataModule(model=model(), validate=source, shuffle=False)

    batches = collect(module.val_dataloader().dataset, monkeypatch)

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert pa.concat_tables([batch.data for batch in batches])["id"].to_pylist() == list(range(5))


def test_dataset_scope_preprocessor_receives_one_logical_split(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    @rf.preprocess(scope="dataset")
    def inspect(batch: rf.Batch) -> rf.Batch:
        calls.append(len(batch))
        return batch

    source = ds.dataset(pa.table({"id": list(range(9))}))
    module = rf.ArrowDataModule(model=model(), validate=source, preprocessor=inspect, shuffle=False)

    collect(module.val_dataloader().dataset, monkeypatch)

    assert calls == [9]


def test_preprocessor_pipeline_runs_in_order_and_is_stored_as_a_tuple(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    @rf.preprocess(requires=("id",), produces=("amount",))
    def derive(batch: rf.Batch) -> rf.Batch:
        calls.append("derive")
        return batch.replace(batch.data.append_column("amount", pa.compute.add(batch.data["id"], 1)))

    @rf.preprocess(requires=("amount",), produces=("id",))
    def replace(batch: rf.Batch) -> rf.Batch:
        calls.append("replace")
        return batch.replace(pa.table({"id": pa.compute.multiply(batch.data["amount"], 2)}))

    configured = [derive, replace]
    module = rf.ArrowDataModule(
        model=model(),
        validate=pa.table({"id": [1, 2, 3]}),
        preprocessor=configured,
        shuffle=False,
    )
    configured.reverse()

    batches = collect(module.val_dataloader().dataset, monkeypatch)

    assert module.preprocessors[Strata.validate] == (derive, replace)
    assert calls == ["derive", "replace"]
    assert pa.concat_tables([batch.data for batch in batches])["id"].to_pylist() == [4, 6, 8]


def test_preprocessor_mapping_accepts_an_ordered_pipeline_per_split(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[Strata, str]] = []

    @rf.preprocess
    def first(batch: rf.Batch, *, strata: Strata) -> rf.Batch:
        calls.append((strata, "first"))
        return batch

    @rf.preprocess
    def second(batch: rf.Batch, *, strata: Strata) -> rf.Batch:
        calls.append((strata, "second"))
        return batch

    source = pa.table({"id": [1, 2]})
    module = rf.ArrowDataModule(
        model=model(),
        train=source,
        validate=source,
        preprocessor={
            "train": [first, second],
            "validate": (second,),
        },
        shuffle=False,
    )

    collect(module.train_dataloader().dataset, monkeypatch)
    collect(module.val_dataloader().dataset, monkeypatch)

    assert module.preprocessors[Strata.train] == (first, second)
    assert module.preprocessors[Strata.validate] == (second,)
    assert calls == [
        (Strata.train, "first"),
        (Strata.train, "second"),
        (Strata.validate, "second"),
    ]


def test_preprocessor_pipeline_flat_maps_each_stage():
    calls: list[int] = []

    @rf.preprocess
    def split(batch: rf.Batch):
        yield batch.slice(0, 1)
        yield batch.slice(1)

    @rf.preprocess
    def inspect(batch: rf.Batch) -> rf.Batch:
        calls.append(len(batch))
        return batch

    source = arrow.convert(pa.table({"id": [1, 2, 3]}), namespace="pipeline", offset=0)
    outputs = list(
        arrow.process(
            (source,),
            preprocessor=(split, inspect),
            strata=Strata.validate,
            schema=None,
            encoding_context={},
        )
    )

    assert calls == [1, 2]
    assert [batch.data["id"].to_pylist() for batch in outputs] == [[1], [2, 3]]


def test_dataset_scope_materializes_at_its_ordered_pipeline_stage():
    calls: list[tuple[str, int]] = []

    @rf.preprocess
    def before(batch: rf.Batch) -> rf.Batch:
        calls.append(("before", len(batch)))
        return batch.replace(batch.data.append_column("ready", pa.repeat(True, len(batch))))

    @rf.preprocess(scope="dataset", requires=("ready",))
    def global_order(batch: rf.Batch) -> rf.Batch:
        calls.append(("global", len(batch)))
        return batch

    @rf.preprocess
    def after(batch: rf.Batch) -> rf.Batch:
        calls.append(("after", len(batch)))
        return batch

    sources = (
        arrow.convert(pa.table({"id": [1, 2]}), namespace="pipeline", offset=0),
        arrow.convert(pa.table({"id": [3, 4, 5]}), namespace="pipeline", offset=2),
    )
    outputs = list(
        arrow.process(
            sources,
            preprocessor=(before, global_order, after),
            strata=Strata.validate,
            schema=None,
            encoding_context={},
        )
    )

    assert calls == [("before", 2), ("before", 3), ("global", 5), ("after", 5)]
    assert len(outputs) == 1
    assert outputs[0].data["id"].to_pylist() == [1, 2, 3, 4, 5]


def test_callable_source_rejects_dataset_scope_anywhere_in_pipeline():
    @rf.preprocess
    def local(batch: rf.Batch) -> rf.Batch:
        return batch

    @rf.preprocess(scope="dataset")
    def global_order(batch: rf.Batch) -> rf.Batch:
        return batch

    def source():
        yield pa.record_batch({"id": [1]})

    with pytest.raises(NotImplementedError, match="dataset-scoped preprocessing"):
        rf.ArrowDataModule(model=model(), validate=source, preprocessor=(local, global_order), shuffle=False)


def test_training_shuffle_is_reproducible_and_changes_by_epoch(monkeypatch: pytest.MonkeyPatch):
    source = pa.table({"id": list(range(32))})
    first = rf.ArrowDataModule(model=model(8), train=source, seed=7).train_dataloader().dataset
    second = rf.ArrowDataModule(model=model(8), train=source, seed=7).train_dataloader().dataset

    first_epoch = pa.concat_tables([batch.data for batch in collect(first, monkeypatch)])["id"].to_pylist()
    repeated_epoch = pa.concat_tables([batch.data for batch in collect(second, monkeypatch)])["id"].to_pylist()
    next_epoch = pa.concat_tables([batch.data for batch in collect(first, monkeypatch)])["id"].to_pylist()

    assert first_epoch == repeated_epoch
    assert next_epoch != first_epoch
    assert sorted(first_epoch) == list(range(32))
    assert sorted(next_epoch) == list(range(32))


def test_training_epoch_survives_dataloader_recreation(monkeypatch: pytest.MonkeyPatch):
    source = pa.table({"id": list(range(32))})
    module = rf.ArrowDataModule(model=model(8), train=source, seed=7)

    first = pa.concat_tables([batch.data for batch in collect(module.train_dataloader().dataset, monkeypatch)])[
        "id"
    ].to_pylist()
    second = pa.concat_tables([batch.data for batch in collect(module.train_dataloader().dataset, monkeypatch)])[
        "id"
    ].to_pylist()

    assert first != second
    assert sorted(first) == sorted(second) == list(range(32))


def test_evaluation_sampling_and_shuffle_are_stable_across_iterations(monkeypatch: pytest.MonkeyPatch):
    source = pa.table({"id": list(range(128))})
    dataset = (
        rf.ArrowDataModule(
            model=model(16),
            validate=source,
            seed=11,
            shuffle={"validate": True},
            sample={"validate": 0.5},
        )
        .val_dataloader()
        .dataset
    )

    first = pa.concat_tables([batch.data for batch in collect(dataset, monkeypatch)])["id"].to_pylist()
    second = pa.concat_tables([batch.data for batch in collect(dataset, monkeypatch)])["id"].to_pylist()

    assert first == second


def test_stream_epoch_limit_selects_before_shuffling(monkeypatch: pytest.MonkeyPatch):
    consumed: list[int] = []

    def source():
        for value in range(16):
            consumed.append(value)
            yield pa.record_batch({"id": [value]})

    dataset = (
        rf.ArrowDataModule(
            model=model(2),
            train=source,
            seed=19,
            shuffle_rows=4,
            epoch_size=2,
        )
        .train_dataloader()
        .dataset
    )

    selected = pa.concat_tables([batch.data for batch in collect(dataset, monkeypatch)])["id"].to_pylist()

    assert len(selected) == 2
    assert sorted(selected) == [0, 1]
    assert consumed == [0, 1]


def test_stream_randomization_is_independent_of_record_batch_boundaries(monkeypatch: pytest.MonkeyPatch):
    table = pa.table({"id": list(range(64))})

    def source(sizes):
        def factory():
            offset = 0
            for size in sizes:
                yield table.slice(offset, size).to_batches()[0]
                offset += size

        return factory

    options = dict(seed=23, shuffle_rows=7, sample=0.7)
    left = rf.ArrowDataModule(model=model(5), train=source([1] * 64), **options).train_dataloader().dataset
    right = rf.ArrowDataModule(model=model(5), train=source([17, 3, 29, 15]), **options).train_dataloader().dataset

    left_ids = pa.concat_tables([batch.data for batch in collect(left, monkeypatch)])["id"].to_pylist()
    right_ids = pa.concat_tables([batch.data for batch in collect(right, monkeypatch)])["id"].to_pylist()

    assert left_ids == right_ids


def test_one_row_stream_shuffle_window_preserves_order(monkeypatch: pytest.MonkeyPatch):
    def source():
        yield pa.record_batch({"id": list(range(8))})

    dataset = (
        rf.ArrowDataModule(
            model=model(3),
            train=source,
            shuffle=True,
            shuffle_rows=1,
        )
        .train_dataloader()
        .dataset
    )

    selected = pa.concat_tables([batch.data for batch in collect(dataset, monkeypatch)])["id"].to_pylist()

    assert selected == list(range(8))


def test_arrow_preprocessor_runs_before_model_batching(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    @rf.preprocess
    def positive(batch: rf.Batch) -> rf.Batch:
        calls.append(len(batch))
        return batch.filter(pa.compute.greater_equal(batch.data["id"], 0))

    source = pa.table({"id": [-2, -1, 0, 1, 2]})
    module = rf.ArrowDataModule(model=model(), validate=source, preprocessor=positive, shuffle=False)
    batches = collect(module.val_dataloader().dataset, monkeypatch)

    assert calls == [5]
    assert [batch.data["id"].to_pylist() for batch in batches] == [[0, 1], [2]]


def test_polars_is_a_one_time_conversion_adapter(monkeypatch: pytest.MonkeyPatch):
    calls = 0
    original = pl.DataFrame.to_arrow

    def track(frame):
        nonlocal calls
        calls += 1
        return original(frame)

    monkeypatch.setattr(pl.DataFrame, "to_arrow", track)
    frame = pl.DataFrame({"id": [1, 2]})
    module = rf.PolarsDataModule(model=model(), train=frame, validate=frame, shuffle=False)

    assert calls == 2
    assert all(isinstance(source, pa.Table) for source in module.sources.values())
    module.train_dataloader()
    module.val_dataloader()
    assert calls == 2


def test_polars_rejects_lazy_frames_and_removed_dataframe_alias():
    with pytest.raises(TypeError, match="collected"):
        rf.PolarsDataModule(model=model(), train=pl.DataFrame({"id": [1]}).lazy())
    with pytest.raises(TypeError, match="unexpected keyword argument 'dataframe'"):
        rf.PolarsDataModule(model=model(), dataframe=pl.DataFrame({"id": [1]}))


def test_custom_adapter_bounds_mapping_conversion_and_restarts():
    dataset = Records([{"id": index} for index in range(5)])
    module = rf.CustomDataModule(model=model(), train=dataset, ingress_rows=2, shuffle=False)
    source = module.sources[Strata.train]

    first = list(source())
    second = list(source())

    assert [batch.num_rows for batch in first] == [2, 2, 1]
    assert [batch.num_rows for batch in second] == [2, 2, 1]
    assert all(isinstance(batch, pa.RecordBatch) for batch in first)
    assert first[0].schema == second[0].schema


def test_custom_inference_uses_the_union_of_first_chunk_mapping_keys():
    source = adapt(
        lambda: iter([{"id": 1}, {"id": 2, "late_in_chunk": "kept"}]),
        schema=None,
        rows=2,
        name="train",
    )

    batch = list(source())[0]

    assert batch.schema.names == ["id", "late_in_chunk"]
    assert batch.to_pylist() == [
        {"id": 1, "late_in_chunk": None},
        {"id": 2, "late_in_chunk": "kept"},
    ]


def test_custom_schema_errors_explain_arrow_schema_remedy():
    empty = adapt(lambda: iter(()), schema=None, rows=2, name="train")
    nulls = adapt(lambda: iter([{"id": None}]), schema=None, rows=2, name="train")
    late = adapt(lambda: iter([{"id": 1}, {"id": 2, "late": True}]), schema=None, rows=1, name="train")

    with pytest.raises(ValueError, match="provide arrow_schema"):
        list(empty())
    with pytest.raises(TypeError, match="all-null.*provide arrow_schema"):
        list(nulls())
    with pytest.raises(TypeError, match="introduced field.*arrow_schema"):
        list(late())


def test_explicit_custom_schema_supports_empty_sources():
    schema = pa.schema([("id", pa.int64())])
    source = adapt(lambda: iter(()), schema=schema, rows=2, name="validate")

    batches = list(source())

    assert len(batches) == 1
    assert batches[0].num_rows == 0
    assert batches[0].schema == schema


def test_synthetic_generator_is_called_for_every_source_iteration():
    calls = 0

    def generate() -> Iterator[dict[str, int]]:
        nonlocal calls
        calls += 1
        yield {"id": calls}

    module = rf.SyntheticDataModule(model=model(), train=generate, shuffle=False)
    source = module.sources[Strata.train]

    assert list(source())[0]["id"].to_pylist() == [1]
    assert list(source())[0]["id"].to_pylist() == [2]


def test_deferred_parallel_and_replacement_modes_fail_explicitly():
    source = pa.table({"id": [1, 2]})
    with pytest.raises(NotImplementedError, match="workers are deferred"):
        rf.ArrowDataModule(model=model(), train=source, num_workers=1)
    with pytest.raises(NotImplementedError, match="replacement sampling is deferred"):
        rf.ArrowDataModule(model=model(), train=source, replacement=True)


def test_model_batch_size_remains_the_single_batch_authority(monkeypatch: pytest.MonkeyPatch):
    configured_model = model(2)
    module = rf.ArrowDataModule(
        model=configured_model,
        validate=pa.table({"id": list(range(5))}),
        shuffle=False,
    )
    configured_model.batch_size = 3

    batches = collect(module.val_dataloader().dataset, monkeypatch)

    assert module.batch_size == 3
    assert [len(batch) for batch in batches] == [3, 2]


def test_retain_mapping_matches_configured_splits_exactly():
    source = pa.table({"id": [1, 2], "request_id": ["a", "b"]})
    module = rf.ArrowDataModule(
        model=model(),
        train=source,
        validate=source,
        retain={"train": ("request_id",), "validate": "*"},
    )

    assert module.retain == {
        Strata.train: ("request_id",),
        Strata.validate: "*",
    }

    with pytest.raises(ValueError, match="match configured splits exactly"):
        rf.ArrowDataModule(model=model(), train=source, validate=source, retain={"train": ()})

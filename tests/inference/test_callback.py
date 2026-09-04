from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

import relflow as rf
from relflow.data.datasets.arrow import identity
from relflow.inference.callback import Writer


def result(data: pa.Table, *, namespace: str = "writer") -> rf.Batch:
    return rf.Batch(data=data, identity=identity(data.num_rows, namespace=namespace))


def write(writer: Writer, output: object, *, rank: int = 0, batch_idx: int = 0) -> None:
    writer.write_on_batch_end(
        trainer=SimpleNamespace(global_rank=rank),
        pl_module=SimpleNamespace(),
        output=output,
        batch_indices=None,
        batch=None,
        batch_idx=batch_idx,
        dataloader_idx=0,
    )


def test_writer_persists_the_written_arrow_batch_and_identity(tmp_path):
    output = result(
        pa.table(
            {
                "inputs": pa.array([{"request_id": "a"}, {"request_id": "b"}]),
                "predictions": pa.array([{"score": 0.25}, {"score": 0.75}]),
            }
        )
    )
    writer = Writer(tmp_path)

    write(writer, output, rank=7)
    writer.on_predict_end(SimpleNamespace(), SimpleNamespace())

    path = tmp_path / "rank-7.parquet"
    assert path.exists()
    assert not (tmp_path / "rank-0.parquet").exists()
    table = pq.read_table(path)
    assert table.column_names == ["identity", "inputs", "predictions"]
    assert table.select(["inputs", "predictions"]).equals(output.data)
    assert (
        pc.struct_field(table["identity"], "logical").to_pylist()
        == pc.struct_field(output.identity, "logical").to_pylist()
    )
    assert writer.writer is None


def test_writer_applies_arrow_postprocessors_in_order_before_persistence(tmp_path):
    calls = []

    @rf.postprocess(requires=("inputs", "predictions"), produces=("request_id", "score"))
    def compact(batch: rf.Batch) -> rf.Batch:
        calls.append("compact")
        predictions = batch.data["predictions"]
        return batch.replace(
            pa.table(
                {
                    "request_id": pc.struct_field(batch.data["inputs"], "request_id"),
                    "score": pc.struct_field(predictions, "score"),
                }
            )
        )

    @rf.postprocess(requires=("request_id", "score"), produces=("request_id", "review"))
    def decide(batch: rf.Batch) -> rf.Batch:
        calls.append("decide")
        return batch.replace(
            pa.table(
                {
                    "request_id": batch.data["request_id"],
                    "review": pc.greater_equal(batch.data["score"], 0.5),
                }
            )
        )

    output = result(
        pa.table(
            {
                "inputs": pa.array([{"request_id": "a"}]),
                "predictions": pa.array([{"score": 0.5}]),
            }
        )
    )
    writer = Writer(tmp_path, postprocessor=[compact, decide])

    write(writer, output)
    writer.close()

    assert calls == ["compact", "decide"]
    assert writer.postprocessors == (compact, decide)
    table = pq.read_table(tmp_path / "rank-0.parquet")
    assert table.column_names == ["identity", "request_id", "review"]
    assert table.select(["request_id", "review"]).to_pylist() == [{"request_id": "a", "review": True}]


def test_writer_stops_after_a_middle_postprocessor_breaks_identity(tmp_path):
    calls = []

    @rf.postprocess
    def first(batch: rf.Batch) -> rf.Batch:
        calls.append("first")
        return batch

    @rf.postprocess
    def reorder(batch: rf.Batch) -> rf.Batch:
        calls.append("reorder")
        return batch.take(pa.array([1, 0], type=pa.int64())).replace(batch.data)

    @rf.postprocess
    def last(batch: rf.Batch) -> rf.Batch:
        calls.append("last")
        return batch

    writer = Writer(tmp_path, postprocessor=(first, reorder, last))
    output = result(pa.table({"score": [0.25, 0.75]}))

    with pytest.raises(ValueError, match="postprocessor 'reorder'.*preserve Batch identity"):
        write(writer, output)

    assert calls == ["first", "reorder"]
    assert writer.writer is None
    assert not list(tmp_path.glob("*.parquet"))


def test_writer_locks_the_exact_first_batch_schema_and_closes_on_drift(tmp_path):
    writer = Writer(tmp_path)
    write(writer, result(pa.table({"score": pa.array([1], type=pa.int64())})), batch_idx=0)

    with pytest.raises(ValueError, match="schema differs from the first batch"):
        write(writer, result(pa.table({"score": pa.array([1.0], type=pa.float64())})), batch_idx=1)

    assert writer.writer is None
    table = pq.read_table(tmp_path / "rank-0.parquet")
    assert table["score"].type == pa.int64()
    assert table["score"].to_pylist() == [1]


def test_writer_rejects_the_reserved_identity_column(tmp_path):
    writer = Writer(tmp_path)

    with pytest.raises(ValueError, match="reserved column name 'identity'"):
        write(writer, result(pa.table({"identity": ["user-owned"]})))

    assert writer.writer is None
    assert not list(tmp_path.glob("*.parquet"))


def test_writer_requires_predict_step_to_return_batch(tmp_path):
    writer = Writer(tmp_path)

    with pytest.raises(TypeError, match="predict_step.*rf.Batch"):
        write(writer, pa.table({"score": [1.0]}))

    assert writer.writer is None

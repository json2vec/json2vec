from enum import StrEnum

import pyarrow as pa
import pyarrow.compute as pc
import pytest

import relflow as rf
from relflow.data import processors
from relflow.data.arrow import IDENTITY
from relflow.structs.enums import Strata


def make_batch(values: list[int]) -> rf.Batch:
    size = len(values)
    identity = pa.StructArray.from_arrays(
        [
            pa.array([index.to_bytes(32) for index in range(size)], type=pa.binary(32)),
            pa.array([(index + 10).to_bytes(32) for index in range(size)], type=pa.binary(32)),
            pa.array([index.to_bytes(8) for index in range(size)], type=pa.large_binary()),
        ],
        fields=list(IDENTITY),
    )
    return rf.Batch(pa.table({"value": values}), identity)


def test_preprocessor_providers_are_string_enums():
    assert issubclass(rf.PreprocessorProvider, StrEnum)
    assert rf.PreprocessorProvider.strata == "strata"
    assert rf.PreprocessorProvider.schema == "schema"
    assert rf.PreprocessorProvider.encoding_context == "encoding_context"


def test_preprocess_returns_callable_processor_object():
    @processors.preprocess
    def increment(batch: rf.Batch):
        return batch.replace(pa.table({"value": pc.add(batch.data["value"], 1)}))

    source = make_batch([1, 2])
    assert isinstance(increment, processors.Preprocessor)
    assert increment(source).data.to_pydict() == {"value": [2, 3]}
    assert list(increment.run(source, strata=Strata.train, schema=None, encoding_context={}))[0].identity.equals(
        source.identity
    )


def test_preprocess_declarations_are_validated():
    @processors.preprocess(
        scope="dataset",
        requires=("source",),
        produces=("value",),
    )
    def prepare(batch: rf.Batch):
        return batch

    assert prepare.scope == "dataset"
    assert prepare.requires == ("source",)
    assert prepare.produces == ("value",)

    with pytest.raises(ValueError, match="must be unique"):
        processors.preprocess(requires=("value", "value"))(lambda batch: batch)

    with pytest.raises(ValueError, match="scope must be"):
        processors.preprocess(scope="global")(lambda batch: batch)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="produces must be a tuple"):
        processors.preprocess(produces=None)(lambda batch: batch)  # type: ignore[arg-type]


def test_preprocessor_discards_none_and_expands_batches():
    @processors.preprocess
    def discard(batch: rf.Batch):
        return None

    @processors.preprocess
    def split(batch: rf.Batch):
        yield batch.slice(0, 1)
        yield batch.slice(1)

    source = make_batch([1, 2, 3])
    assert list(discard.run(source, strata=Strata.train, schema=None, encoding_context={})) == []
    assert [
        item.data["value"].to_pylist()
        for item in split.run(
            source,
            strata=Strata.train,
            schema=None,
            encoding_context={},
        )
    ] == [[1], [2, 3]]


@pytest.mark.parametrize("value", [pa.table({"value": [1]}), {"value": [1]}, [1]])
def test_preprocessor_rejects_non_batch_results(value):
    @processors.preprocess
    def invalid(batch: rf.Batch):
        return value

    with pytest.raises(TypeError, match="Batch"):
        list(invalid.run(make_batch([1]), strata=Strata.train, schema=None, encoding_context={}))


def test_preprocessor_receives_only_named_pipeline_providers():
    @processors.preprocess
    def inspect(batch: rf.Batch, *, strata, schema, encoding_context):
        assert strata is Strata.validate
        assert schema == "schema"
        assert encoding_context == {"marker": "seen"}
        return batch

    source = make_batch([1])
    assert list(
        inspect.run(
            source,
            strata=Strata.validate,
            schema="schema",
            encoding_context={"marker": "seen"},
        )
    ) == [source]


def test_required_user_parameters_are_bound_immutably():
    @processors.preprocess
    def offset(batch: rf.Batch, *, amount: int):
        data = pa.table({"value": pc.add(batch.data["value"], amount)})
        return batch.replace(data)

    with pytest.raises(ValueError, match="requires unbound parameter"):
        list(offset.run(make_batch([1]), strata=Strata.train, schema=None, encoding_context={}))

    configured = offset.partial(amount=4)
    assert configured.bound == {"amount": 4}
    assert offset.bound == {}
    assert list(configured.run(make_batch([1]), strata=Strata.train, schema=None, encoding_context={}))[0].data[
        "value"
    ].to_pylist() == [5]

    with pytest.raises(ValueError, match="already bound"):
        configured.partial(amount=5)


def test_pipeline_parameters_cannot_be_bound():
    @processors.preprocess
    def inspect(batch: rf.Batch, *, strata):
        return batch

    with pytest.raises(ValueError, match="provided by the pipeline"):
        inspect.partial(strata=Strata.train)


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (lambda value: value, "first parameter must be 'batch'"),
        (lambda batch, value: batch, "must be keyword-only"),
    ],
)
def test_processor_signatures_are_explicit(function, message):
    with pytest.raises(TypeError, match=message):
        processors.preprocess(function)


def test_preprocessor_normalize_rejects_raw_callable():
    with pytest.raises(TypeError, match="preprocessor must be a Preprocessor"):
        processors.Preprocessor.normalize(lambda batch: batch)


def test_postprocessor_preserves_rows_and_identity():
    @rf.postprocess
    def compact(batch: rf.Batch, *, threshold: int):
        data = pa.table({"large": pc.greater_equal(batch.data["value"], threshold)})
        return batch.replace(data)

    source = make_batch([1, 3])
    output = compact.partial(threshold=2).run(source)
    assert output.data.to_pydict() == {"large": [False, True]}
    assert output.identity.equals(source.identity)


def test_postprocessor_rejects_invalid_output_contracts():
    source = make_batch([1, 2])

    @rf.postprocess
    def table(batch: rf.Batch):
        return batch.data

    @rf.postprocess
    def rows(batch: rf.Batch):
        return batch.slice(0, 1)

    @rf.postprocess
    def identity(batch: rf.Batch):
        other = batch.take(pa.array([1, 0], type=pa.int64()))
        return rf.Batch(batch.data, other.identity)

    @rf.postprocess
    def empty(batch: rf.Batch):
        return batch.replace(pa.table({}))

    with pytest.raises(TypeError, match="must return Batch"):
        table.run(source)
    with pytest.raises(ValueError, match="returned 1 rows"):
        rows.run(source)
    with pytest.raises(ValueError, match="preserve Batch identity"):
        identity.run(source)
    with pytest.raises(ValueError, match="at least one Arrow column"):
        empty.run(source)


def test_postprocessor_normalize_rejects_raw_callable():
    with pytest.raises(TypeError, match="postprocessor must be a Postprocessor"):
        rf.Postprocessor.normalize(lambda batch: batch)

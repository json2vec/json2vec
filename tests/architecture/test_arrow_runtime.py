import pyarrow as pa
import pyarrow.compute as pc
import pytest

import relflow as rf
from relflow.architecture.runtime import ModelRuntime
from relflow.structs.enums import Component, Strata, TensorKey
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS


def model(*, nested: bool = False, embed: bool = False) -> rf.Model:
    fields = (
        {
            "items": rf.Branch(
                length=3,
                value=rf.Number,
                label=rf.Boolean(target=True),
            )
        }
        if nested
        else {"value": rf.Number, "label": rf.Boolean(target=True)}
    )
    return rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
        embed=embed,
        **fields,
    )


def test_predict_returns_canonical_arrow_table_with_retained_inputs():
    configured = model()
    source = pa.table({"request_id": ["a", "b"], "value": [1.0, 2.0]})

    result = configured.predict(source, retain=("request_id",))

    assert isinstance(result, pa.Table)
    assert result.column_names == ["inputs", "predictions"]
    assert len(result) == 2
    assert result["inputs"].combine_chunks().type == pa.struct([("request_id", pa.string())])
    prediction_type = result["predictions"].combine_chunks().type
    assert prediction_type.get_field_index("record/label") == 0
    assert result["inputs"].to_pylist() == [{"request_id": "a"}, {"request_id": "b"}]


def test_lightning_predict_step_uses_the_datamodule_retain_plan():
    configured = model()
    data = rf.ArrowDataModule(
        model=configured,
        predict=pa.table({"request_id": ["a"], "value": [1.0]}),
        retain=("request_id",),
        shuffle=False,
    )
    batch = next(iter(data.predict_dataloader()))

    result = configured.predict_step(batch, 0)

    assert isinstance(result, rf.Batch)
    assert result.data["inputs"].to_pylist() == [{"request_id": "a"}]


def test_predict_adapts_a_small_python_record_sequence_once():
    result = model().predict([{"value": 1.0}, {"value": 2.0}])

    assert isinstance(result, pa.Table)
    assert len(result) == 2


def test_python_prediction_ingress_preserves_keys_introduced_after_the_first_row():
    result = model().predict(
        [{"value": 1.0}, {"value": 2.0, "request_id": "second"}],
        retain="*",
    )

    assert result["inputs"].to_pylist() == [
        {"value": 1.0, "request_id": None},
        {"value": 2.0, "request_id": "second"},
    ]


def test_predict_requires_typed_arrow_for_an_empty_sequence():
    with pytest.raises(ValueError, match="empty Python prediction sequence has no Arrow schema"):
        model().predict([])


def test_typed_empty_prediction_compiles_the_exact_output_schema_without_forward():
    source = pa.table(
        {
            "request_id": pa.array([], type=pa.large_string()),
            "value": pa.array([], type=pa.float64()),
        }
    )

    result = model().predict(source, retain=("request_id",))

    assert len(result) == 0
    assert result["inputs"].type == pa.struct([pa.field("request_id", pa.large_string())])
    predictions = result["predictions"].type
    assert pa.types.is_struct(predictions)
    assert predictions.get_field_index("record/label") == 0


def test_preprocessor_may_filter_every_prediction_row_without_erasing_schema():
    @rf.preprocess(requires=("value",))
    def discard(batch: rf.Batch) -> rf.Batch:
        return batch.filter(pa.array([False] * len(batch)))

    result = model().predict(
        pa.table({"request_id": ["a"], "value": [1.0]}),
        preprocess=discard,
        retain=("request_id",),
    )

    assert len(result) == 0
    assert result["inputs"].type == pa.struct([pa.field("request_id", pa.string())])
    assert result["predictions"].type.get_field_index("record/label") == 0


def test_predict_preserves_nested_model_axes_as_fixed_lists():
    configured = model(nested=True)
    source = pa.Table.from_pylist(
        [
            {"items": [{"value": 1.0}, {"value": 2.0}]},
            {"items": [{"value": 3.0}]},
        ]
    )

    result = configured.predict(source)
    predictions = result["predictions"].combine_chunks()
    field = predictions.type.field("record/items/label")

    assert pa.types.is_fixed_size_list(field.type)
    assert field.type.list_size == 3
    assert all(len(row["record/items/label"]) == 3 for row in predictions.to_pylist())


def test_embedding_only_address_has_no_state_or_inferred_fields():
    configured = model(embed=True)

    result = configured.predict(pa.table({"value": [1.0]}))
    root = result["predictions"].combine_chunks().type.field("record").type

    assert [field.name for field in root] == ["embedding"]
    assert root.field("embedding").type == pa.list_(pa.float32(), 8)


def test_empty_retain_uses_typed_null_without_changing_row_count():
    result = model().predict(pa.table({"value": [1.0, 2.0, 3.0]}))

    assert result["inputs"].type == pa.null()
    assert result["inputs"].null_count == 3
    assert len(result) == 3


def test_target_without_public_plugin_output_keeps_a_typed_null_prediction_column():
    configured = rf.Model(
        value=rf.Number,
        identifier=rf.Hash(target=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )

    result = configured.predict(pa.table({"value": [1.0, 2.0]}))

    assert result["predictions"].type == pa.null()
    assert result["predictions"].null_count == 2


def test_predict_runs_arrow_preprocessor_and_postprocessor_once():
    calls = {"pre": 0, "post": 0}

    @rf.preprocess(requires=("value",), produces=("value",))
    def prepare(batch: rf.Batch) -> rf.Batch:
        calls["pre"] += 1
        data = batch.data.set_column(0, "value", pc.multiply(batch.data["value"], 2))
        return batch.replace(data)

    @rf.postprocess
    def compact(batch: rf.Batch) -> rf.Batch:
        calls["post"] += 1
        inputs = pc.struct_field(batch.data["inputs"], "value")
        return batch.replace(pa.table({"prepared": inputs}))

    result = model().predict(
        pa.table({"value": [1.0, 2.0]}),
        preprocess=prepare,
        postprocess=compact,
        retain=("value",),
    )

    assert calls == {"pre": 1, "post": 1}
    assert result.to_pydict() == {"prepared": [2.0, 4.0]}


def test_predict_restores_training_mode_after_inference():
    configured = model()
    configured.train()

    configured.predict(pa.table({"value": [1.0]}))

    assert configured.training


def test_predict_rejects_missing_or_duplicate_retain_columns():
    configured = model()
    source = pa.table({"value": [1.0]})

    with pytest.raises(KeyError, match="absent after preprocessing"):
        configured.predict(source, retain=("missing",))
    with pytest.raises(ValueError, match="must be unique"):
        configured.predict(source, retain=("value", "value"))


def test_output_plan_declares_once_and_passes_that_type_to_writer(monkeypatch: pytest.MonkeyPatch):
    plugin = TENSORFIELDS["boolean"]
    declare = plugin.output
    render = plugin.write
    calls = {"output": 0, "write": 0}
    declarations: dict[Address, pa.StructType] = {}

    def output(module, address):
        calls["output"] += 1
        datatype = declare(module=module, address=address)
        declarations[address] = datatype
        return datatype

    def write(module, prediction, datatype):
        calls["write"] += 1
        assert datatype is declarations[prediction.address]
        return render(module=module, prediction=prediction, datatype=datatype)

    monkeypatch.setitem(plugin.components, Component.output, output)
    monkeypatch.setitem(plugin.components, Component.write, write)

    model().predict(pa.table({"value": [1.0]}))

    assert calls == {"output": 1, "write": 1}


def test_write_rejects_an_ordinary_unplanned_forward_address():
    configured = model()
    source, inputs = ModelRuntime.prepare(
        configured,
        pa.table({"value": [1.0]}),
        preprocess=None,
        strata=Strata.predict,
        mask=True,
    )
    predictions = configured(inputs, strata=Strata.predict)
    target = predictions[0]
    extra = Prediction(
        address=Address("record/value"),
        payload=target.payload.clone(),
        batch_size=target.batch_size,
    )

    with pytest.raises(ValueError, match="unplanned prediction address.*record/value"):
        configured.write([*predictions, extra], source=source)


def test_output_plan_rejects_runtime_embedding_width_drift():
    configured = model(embed=True)
    source, inputs = ModelRuntime.prepare(
        configured,
        pa.table({"value": [1.0]}),
        preprocess=None,
        strata=Strata.predict,
        mask=True,
    )
    predictions = configured(inputs, strata=Strata.predict)
    root = next(prediction for prediction in predictions if prediction.address == Address("record"))
    root.payload[TensorKey.embedding] = root.payload[TensorKey.embedding][..., :-1]

    with pytest.raises(ValueError, match="must end with model width 8"):
        configured.write(predictions, source=source)

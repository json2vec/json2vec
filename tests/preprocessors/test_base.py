from enum import StrEnum

import pytest

import relflow as rf
from relflow.data import processors
from relflow.structs.enums import Strata


def test_processor_providers_are_string_enums():
    assert issubclass(rf.PreprocessorProvider, StrEnum)
    assert issubclass(rf.PostprocessorProvider, StrEnum)
    assert rf.PreprocessorProvider.strata == "strata"
    assert rf.PostprocessorProvider.metadata == "metadata"


def test_preprocess_returns_callable_processor_object():
    @processors.preprocess
    def add_marker(observation: dict):
        return processors.Observation({"id": observation["id"], "marked": True})

    assert isinstance(add_marker, processors.Preprocessor)
    assert add_marker({"id": 1}) == processors.Observation({"id": 1, "marked": True})
    assert list(add_marker.outputs({"id": 1}, strata=Strata.train, schema=None, encoding_context={})) == [
        [{"id": 1, "marked": True}]
    ]


def test_preprocess_rejects_decorator_configuration_kwargs():
    with pytest.raises(TypeError, match="unexpected preprocess keyword"):
        processors.preprocess(yields=True)


def test_preprocessor_outputs_discard_none():
    @processors.preprocess
    def drop_low_ids(observation: dict):
        if observation["id"] < 10:
            return None
        return processors.Observation(observation)

    assert list(drop_low_ids.outputs({"id": 1}, strata=Strata.train, schema=None, encoding_context={})) == []
    assert list(drop_low_ids.outputs({"id": 10}, strata=Strata.train, schema=None, encoding_context={})) == [
        [{"id": 10}]
    ]


def test_preprocessor_outputs_expand_iterables_and_skip_none_items():
    @processors.preprocess
    def fan_out(observation: dict):
        yield processors.Observation({"id": observation["id"]})
        yield None
        yield processors.Observation({"id": observation["id"] + 1})

    assert list(fan_out.outputs({"id": 1}, strata=Strata.train, schema=None, encoding_context={})) == [
        [{"id": 1}],
        [{"id": 2}],
    ]


def test_preprocessor_outputs_reject_plain_dict_return():
    @processors.preprocess
    def legacy_dict(observation: dict):
        return {"id": observation["id"]}

    with pytest.raises(TypeError, match="must return Observation"):
        list(legacy_dict.outputs({"id": 1}, strata=Strata.train, schema=None, encoding_context={}))


def test_preprocessor_outputs_reject_plain_dict_yield():
    @processors.preprocess
    def legacy_generator(observation: dict):
        yield {"id": observation["id"]}

    with pytest.raises(TypeError, match="expected Observation or None"):
        list(legacy_generator.outputs({"id": 1}, strata=Strata.train, schema=None, encoding_context={}))


def test_preprocessor_receives_named_pipeline_providers():
    @processors.preprocess
    def with_runtime(observation: dict, *, strata, encoding_context):
        return processors.Observation(
            {
                "id": observation["id"],
                "strata": strata,
                "marker": encoding_context["marker"],
            }
        )

    assert list(
        with_runtime.outputs(
            {"id": 1},
            strata=Strata.validate,
            schema=None,
            encoding_context={"marker": "seen"},
        )
    ) == [[{"id": 1, "strata": Strata.validate, "marker": "seen"}]]


def test_preprocessor_requires_user_params_to_be_bound():
    @processors.preprocess
    def with_user_param(observation: dict, *, marker: str):
        return processors.Observation({"id": observation["id"], "marker": marker})

    with pytest.raises(ValueError, match="requires unbound parameter"):
        list(with_user_param.outputs({"id": 1}, strata=Strata.train, schema=None, encoding_context={}))

    bound = with_user_param.partial(marker="ready")
    assert list(bound.outputs({"id": 1}, strata=Strata.train, schema=None, encoding_context={})) == [
        [{"id": 1, "marker": "ready"}]
    ]


def test_preprocessor_does_not_infer_provider_from_type_annotation():
    @processors.preprocess
    def typed_name_is_user_param(observation: dict, *, split: Strata):
        return processors.Observation({"id": observation["id"], "split": split})

    with pytest.raises(ValueError, match="requires unbound parameter"):
        list(typed_name_is_user_param.outputs({"id": 1}, strata=Strata.train, schema=None, encoding_context={}))


def test_preprocessor_rejects_binding_pipeline_provider():
    @processors.preprocess
    def with_strata(observation: dict, *, strata):
        return processors.Observation({"id": observation["id"], "strata": strata})

    with pytest.raises(ValueError, match="provided by the pipeline"):
        with_strata.partial(strata=Strata.train)


def test_preprocessor_normalize_rejects_raw_callable():
    def raw(observation: dict):
        return rf.Observation(observation)

    with pytest.raises(TypeError, match="preprocessor must be a Preprocessor object or None"):
        processors.Preprocessor.normalize(raw)


def test_postprocess_optional_unavailable_provider_receives_none():
    @rf.postprocess
    def add_batch_index(predictions: dict, *, batch_idx: int | None = None):
        return {"predictions": predictions, "batch_idx": batch_idx}

    assert add_batch_index.run({}, available={}) == {"predictions": {}, "batch_idx": None}


def test_postprocess_required_unavailable_provider_errors():
    @rf.postprocess
    def needs_batch_index(predictions: dict, *, batch_idx: int):
        return {"predictions": predictions, "batch_idx": batch_idx}

    with pytest.raises(ValueError, match="not available in this runtime"):
        needs_batch_index.run({}, available={})


def test_postprocessor_normalize_rejects_raw_callable():
    def raw(predictions: dict):
        return predictions

    with pytest.raises(TypeError, match="postprocessor must be a Postprocessor object or None"):
        rf.Postprocessor.normalize(raw)


def test_postprocess_rejects_context_dict_signature():
    with pytest.raises(TypeError, match="first parameter must be 'predictions'"):

        @rf.postprocess
        def legacy_context(context: dict, predictions: dict):
            return predictions


def test_processors_reject_var_keyword_parameters():
    with pytest.raises(TypeError, match="does not support \\*\\*kwargs"):

        @processors.preprocess
        def legacy_kwargs(observation: dict, **kwargs):
            return processors.Observation(observation)

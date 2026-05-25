import pytest

from json2vec.preprocessors import base


@pytest.fixture(autouse=True)
def restore_preprocessors():
    snapshot = dict(base.PREPROCESSORS)
    yield
    base.PREPROCESSORS.clear()
    base.PREPROCESSORS.update(snapshot)


def test_preprocess_assigns_preprocessor_mode():
    def transformation(observation: dict):
        return observation

    transformation.__name__ = "__test_transformation_preprocessor"

    def generator(observation: dict):
        yield observation

    generator.__name__ = "__test_generator_preprocessor"

    base.preprocess(yields=False)(transformation)
    base.preprocess(yields=True)(generator)

    assert base.PREPROCESSORS[transformation.__name__].mode == base.PreprocessorMode.transformation
    assert base.PREPROCESSORS[generator.__name__].mode == base.PreprocessorMode.generator


def test_preprocessor_mode_from_yields():
    assert base.PreprocessorMode.from_yields(False) is base.PreprocessorMode.transformation
    assert base.PreprocessorMode.from_yields(True) is base.PreprocessorMode.generator


def test_preprocess_overwrites_duplicate_preprocessor_names():
    def first(observation: dict):
        return {"first": observation}

    first.__name__ = "__duplicate_preprocessor"
    base.preprocess(yields=False)(first)

    def second(observation: dict):
        return observation

    second.__name__ = "__duplicate_preprocessor"

    base.preprocess(yields=True)(second)

    preprocessor = base.PREPROCESSORS["__duplicate_preprocessor"]
    assert preprocessor.func is second
    assert preprocessor.mode == base.PreprocessorMode.generator


def test_preprocess_accepts_yield_keyword_via_kwargs():
    def generator(observation: dict):
        yield observation

    generator.__name__ = "__yield_keyword_preprocessor"
    base.preprocess(**{"yield": True})(generator)

    assert base.PREPROCESSORS[generator.__name__].mode == base.PreprocessorMode.generator


def test_preprocess_rejects_non_boolean_mode():
    with pytest.raises(TypeError, match="yields must be a boolean"):
        base.preprocess(yields="yes")


def test_preprocessor_call_filters_unknown_kwargs():
    def returning(observation: dict, strata):
        return observation, strata

    preprocessor = base.Preprocessor(name="filtered", func=returning, mode=base.PreprocessorMode.transformation)

    output = preprocessor({"id": 1}, strata="train", interprocess_encoding_context={"unused": True})
    assert output == ({"id": 1}, "train")


def test_transformation_outputs_wrap_dict_result():
    def transformation(observation: dict):
        return {"id": observation["id"]}

    preprocessor = base.Preprocessor(
        name="wrapped-transformation",
        func=transformation,
        mode=base.PreprocessorMode.transformation,
    )

    assert list(preprocessor.outputs({"id": 1})) == [[{"id": 1}]]


def test_generator_outputs_wrap_each_object_from_list():
    def generator(observation: dict):
        return [{"id": observation["id"]}, {"id": observation["id"] + 1}]

    preprocessor = base.Preprocessor(name="list-generator", func=generator, mode=base.PreprocessorMode.generator)

    assert list(preprocessor.outputs({"id": 1})) == [[{"id": 1}], [{"id": 2}]]


def test_generator_outputs_wrap_each_yielded_object():
    def generator(observation: dict):
        yield {"id": observation["id"]}
        yield {"id": observation["id"] + 1}

    preprocessor = base.Preprocessor(name="yield-generator", func=generator, mode=base.PreprocessorMode.generator)

    assert list(preprocessor.outputs({"id": 1})) == [[{"id": 1}], [{"id": 2}]]


def test_transformation_outputs_reject_non_dict():
    def transformation(observation: dict):
        return observation["id"]

    preprocessor = base.Preprocessor(
        name="invalid-transformation",
        func=transformation,
        mode=base.PreprocessorMode.transformation,
    )

    with pytest.raises(TypeError, match="must produce dict objects"):
        list(preprocessor.outputs({"id": 1}))


def test_generator_outputs_reject_non_list_return():
    def generator(observation: dict):
        return observation

    preprocessor = base.Preprocessor(name="invalid-generator", func=generator, mode=base.PreprocessorMode.generator)

    with pytest.raises(TypeError, match="must yield dict objects or return a list of dict objects"):
        list(preprocessor.outputs({"id": 1}))

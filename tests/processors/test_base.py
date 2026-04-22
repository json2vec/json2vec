import pytest

from json2vec.processors import base


@pytest.fixture(autouse=True)
def restore_processors():
    snapshot = dict(base.PROCESSORS)
    yield
    base.PROCESSORS.clear()
    base.PROCESSORS.update(snapshot)


def test_register_assigns_processor_mode():
    def transformation(observation: dict):
        return observation

    transformation.__name__ = "__test_transformation_processor"

    def generator(observation: dict):
        yield observation

    generator.__name__ = "__test_generator_processor"

    base.register.transformation(transformation)
    base.register.generator(generator)

    assert base.PROCESSORS[transformation.__name__].mode == base.ProcessorMode.transformation
    assert base.PROCESSORS[generator.__name__].mode == base.ProcessorMode.generator


def test_register_rejects_duplicate_processor_names():
    def first(observation: dict):
        return observation

    first.__name__ = "__duplicate_processor"
    base.register.transformation(first)

    def second(observation: dict):
        return observation

    second.__name__ = "__duplicate_processor"

    with pytest.raises(ValueError, match="already registered"):
        base.register.transformation(second)


def test_processor_call_filters_unknown_kwargs():
    def returning(observation: dict, strata):
        return observation, strata

    processor = base.Processor(name="filtered", func=returning, mode=base.ProcessorMode.transformation)

    output = processor({"id": 1}, strata="train", state={"unused": True})
    assert output == ({"id": 1}, "train")


def test_transformation_outputs_wrap_dict_result():
    def transformation(observation: dict):
        return {"id": observation["id"]}

    processor = base.Processor(
        name="wrapped-transformation",
        func=transformation,
        mode=base.ProcessorMode.transformation,
    )

    assert list(processor.outputs({"id": 1})) == [[{"id": 1}]]


def test_generator_outputs_wrap_each_object_from_list():
    def generator(observation: dict):
        return [{"id": observation["id"]}, {"id": observation["id"] + 1}]

    processor = base.Processor(name="list-generator", func=generator, mode=base.ProcessorMode.generator)

    assert list(processor.outputs({"id": 1})) == [[{"id": 1}], [{"id": 2}]]


def test_generator_outputs_wrap_each_yielded_object():
    def generator(observation: dict):
        yield {"id": observation["id"]}
        yield {"id": observation["id"] + 1}

    processor = base.Processor(name="yield-generator", func=generator, mode=base.ProcessorMode.generator)

    assert list(processor.outputs({"id": 1})) == [[{"id": 1}], [{"id": 2}]]


def test_transformation_outputs_reject_non_dict():
    def transformation(observation: dict):
        return observation["id"]

    processor = base.Processor(
        name="invalid-transformation",
        func=transformation,
        mode=base.ProcessorMode.transformation,
    )

    with pytest.raises(TypeError, match="must produce dict objects"):
        list(processor.outputs({"id": 1}))


def test_generator_outputs_reject_non_list_return():
    def generator(observation: dict):
        return observation

    processor = base.Processor(name="invalid-generator", func=generator, mode=base.ProcessorMode.generator)

    with pytest.raises(TypeError, match="must yield dict objects or return a list of dict objects"):
        list(processor.outputs({"id": 1}))

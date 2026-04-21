import pytest

from json2vec.processors import base


@pytest.fixture(autouse=True)
def restore_processors():
    snapshot = dict(base.PROCESSORS)
    yield
    base.PROCESSORS.clear()
    base.PROCESSORS.update(snapshot)


def test_is_yielding_processor():
    def returns(observation: dict):
        return [observation]

    def yields(observation: dict):
        yield [observation]

    assert not base.is_yielding_processor(returns)
    assert base.is_yielding_processor(yields)


def test_register_assigns_processor_mode():
    def returning(observation: dict):
        return observation

    returning.__name__ = "__test_returning_processor"

    def yielding(observation: dict):
        yield observation

    yielding.__name__ = "__test_yielding_processor"

    base.register(returning)
    base.register(yielding)

    assert base.PROCESSORS[returning.__name__].mode == base.ProcessorMode.returning
    assert base.PROCESSORS[yielding.__name__].mode == base.ProcessorMode.yielding


def test_register_rejects_duplicate_processor_names():
    def first(observation: dict):
        return observation

    first.__name__ = "__duplicate_processor"
    base.register(first)

    def second(observation: dict):
        return observation

    second.__name__ = "__duplicate_processor"

    with pytest.raises(ValueError, match="already registered"):
        base.register(second)


def test_processor_call_filters_unknown_kwargs():
    def returning(observation: dict, strata):
        return observation, strata

    processor = base.Processor(name="filtered", func=returning, mode=base.ProcessorMode.returning)

    output = processor({"id": 1}, strata="train", state={"unused": True})
    assert output == ({"id": 1}, "train")

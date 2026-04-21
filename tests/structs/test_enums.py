import pytest

from json2vec.structs.enums import ShardingStrategy, Stage, Strata


def test_strata_from_stage_fit():
    assert Strata.from_stage(Stage.fit) == [Strata.train, Strata.validate]


def test_strata_from_stage_predict():
    assert Strata.from_stage(Stage.predict) == [Strata.predict]


def test_strata_from_stage_accepts_strings():
    assert Strata.from_stage("test") == [Strata.test]


def test_strata_from_stage_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown stage"):
        Strata.from_stage("unknown")


def test_sharding_strategy_values_stable():
    assert {member.value for member in ShardingStrategy} == {"file", "chunk", "record"}

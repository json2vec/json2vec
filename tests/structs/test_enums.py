from json2vec.structs.enums import ShardingStrategy


def test_sharding_strategy_values_stable():
    assert {member.value for member in ShardingStrategy} == {"file", "chunk", "record"}

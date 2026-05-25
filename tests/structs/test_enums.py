from json2vec.structs.enums import AttentionMode, ShardingStrategy, Strata


def test_strata_normalizes_strings():
    assert Strata.normalize(" TRAIN ") is Strata.train
    assert Strata.normalize(Strata.predict) is Strata.predict


def test_strata_expands_scalar_and_mapping_values():
    assert Strata.expand("all", default="default")[Strata.test] == "all"

    expanded = Strata.expand({"TRAIN": 1}, default=0)
    assert expanded[Strata.train] == 1
    assert expanded[Strata.validate] == 0


def test_sharding_strategy_values_stable():
    assert {member.value for member in ShardingStrategy} == {"file", "chunk", "record"}


def test_sharding_strategy_normalizes_strings():
    assert ShardingStrategy.normalize(" RECORD ") is ShardingStrategy.record
    assert ShardingStrategy.normalize(ShardingStrategy.chunk) is ShardingStrategy.chunk


def test_sharding_strategy_expands_by_strata():
    expanded = ShardingStrategy.expand({"train": "record"}, default=ShardingStrategy.chunk)
    assert expanded[Strata.train] is ShardingStrategy.record
    assert expanded[Strata.validate] is ShardingStrategy.chunk


def test_attention_mode_kv_heads():
    assert AttentionMode.mha.kv_heads(8) == 8
    assert AttentionMode.gqa.kv_heads(8) == 4
    assert AttentionMode.mqa.kv_heads(8) == 1

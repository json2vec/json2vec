from relflow.structs.enums import AttentionMode, Strata


def test_strata_normalizes_strings():
    assert Strata.normalize(" TRAIN ") is Strata.train
    assert Strata.normalize(Strata.predict) is Strata.predict


def test_strata_expands_scalar_and_mapping_values():
    assert Strata.expand("all", default="default")[Strata.test] == "all"

    expanded = Strata.expand({"TRAIN": 1}, default=0)
    assert expanded[Strata.train] == 1
    assert expanded[Strata.validate] == 0


def test_attention_mode_kv_heads():
    assert AttentionMode.mha.kv_heads(8) == 8
    assert AttentionMode.gqa.kv_heads(8) == 4
    assert AttentionMode.mqa.kv_heads(8) == 1

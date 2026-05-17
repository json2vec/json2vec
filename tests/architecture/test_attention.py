import pytest
import torch

from json2vec.architecture.attention import RotaryMultiheadAttention


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_rotary_attention_supports_grouped_kv_heads(n_kv_heads: int):
    attention = RotaryMultiheadAttention(d_model=16, nhead=4, n_kv_heads=n_kv_heads, dropout=0.0)
    inputs = torch.randn(3, 5, 16)

    output = attention(query=inputs, key=inputs, value=inputs)

    assert output.shape == inputs.shape
    assert attention.k_proj.out_features == n_kv_heads * attention.head_dim
    assert attention.v_proj.out_features == n_kv_heads * attention.head_dim


def test_rotary_attention_rejects_kv_heads_that_do_not_divide_query_heads():
    with pytest.raises(ValueError, match="nhead must be divisible by n_kv_heads"):
        RotaryMultiheadAttention(d_model=16, nhead=4, n_kv_heads=3, dropout=0.0)


def test_rotary_attention_passes_grouped_heads_to_sdpa(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def sdpa(query, key, value, attn_mask, dropout_p, enable_gqa):
        seen["query_heads"] = query.shape[1]
        seen["key_heads"] = key.shape[1]
        seen["value_heads"] = value.shape[1]
        seen["attn_mask"] = attn_mask
        seen["dropout_p"] = dropout_p
        seen["enable_gqa"] = enable_gqa
        return torch.zeros_like(query)

    monkeypatch.setattr("json2vec.architecture.attention.F.scaled_dot_product_attention", sdpa)
    attention = RotaryMultiheadAttention(d_model=16, nhead=4, n_kv_heads=1, dropout=0.2)
    inputs = torch.randn(3, 5, 16)

    attention(query=inputs, key=inputs, value=inputs)

    assert seen == {
        "query_heads": 4,
        "key_heads": 1,
        "value_heads": 1,
        "attn_mask": None,
        "dropout_p": 0.2,
        "enable_gqa": True,
    }


def test_rotary_attention_padding_mask_uses_sdpa_keep_mask(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def sdpa(query, key, value, attn_mask, dropout_p, enable_gqa):
        seen["attn_mask"] = attn_mask
        return torch.zeros_like(query)

    monkeypatch.setattr("json2vec.architecture.attention.F.scaled_dot_product_attention", sdpa)
    attention = RotaryMultiheadAttention(d_model=16, nhead=4, dropout=0.0)
    inputs = torch.randn(2, 3, 16)

    attention(
        query=inputs,
        key=inputs,
        value=inputs,
        key_padding_mask=torch.tensor(
            [
                [False, True, False],
                [True, True, True],
            ]
        ),
    )

    assert torch.equal(
        seen["attn_mask"],
        torch.tensor(
            [
                [[[True, False, True]]],
                [[[True, False, False]]],
            ]
        ),
    )

import pytest
import torch

import json2vec as j2v
from json2vec.data.iterables import encode, mask
from json2vec.structs.enums import Strata, TensorKey, Tokens


def test_array_mask_count_targets_recent_real_slots_and_excludes_padding():
    model = j2v.Model.from_schema(
        j2v.Array(
            j2v.Number("amount"),
            name="items",
            max_length=5,
            mask=j2v.Mask(count=1, window=2),
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    inputs = encode(
        batch=[[{"items": [{"amount": 1.0}, {"amount": 2.0}, {"amount": 3.0}]}]],
        hyperparameters=model.hyperparameters,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    field = next(mask([inputs], model.hyperparameters, Strata.train))["record/items/amount"]

    assert int(field.trainable.sum()) == 1
    assert field.trainable[0, 0, :3].any()
    assert not field.trainable[0, 0, 3:].any()
    assert field.state[0, 0, 3:].tolist() == [Tokens.padded.value, Tokens.padded.value]
    assert torch.equal(field.targets[TensorKey.state], torch.tensor([[[0, 0, 0, 2, 2]]]))


def test_model_encode_mask_flag_applies_and_skips_array_masks():
    model = j2v.Model.from_schema(
        j2v.Array(
            j2v.Number("amount"),
            name="items",
            max_length=5,
            mask=j2v.Mask(count=1, window=2),
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    batch = [{"items": [{"amount": 1.0}, {"amount": 2.0}, {"amount": 3.0}]}]

    unmasked = model.encode(batch, strata=Strata.train, mask=False)["record/items/amount"]
    masked = model.encode(batch, strata=Strata.train)["record/items/amount"]

    assert not unmasked.trainable.any()
    assert list(unmasked.targets.keys()) == []
    assert unmasked.state[0, 0].tolist() == [
        Tokens.valued.value,
        Tokens.valued.value,
        Tokens.valued.value,
        Tokens.padded.value,
        Tokens.padded.value,
    ]
    assert int(masked.trainable.sum()) == 1
    assert masked.state[0, 0, :3].eq(Tokens.masked.value).any()
    assert masked.state[0, 0, 3:].tolist() == [Tokens.padded.value, Tokens.padded.value]
    assert set(masked.targets.keys()) == {TensorKey.content, TensorKey.state}


def test_array_mask_exclude_predicate_skips_matching_leaf():
    model = j2v.Model.from_schema(
        j2v.Array(
            j2v.Number("amount"),
            j2v.Category("code", max_vocab_size=8),
            name="items",
            max_length=2,
            mask=j2v.Mask(count=1, exclude=j2v.where("type") == "category"),
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    inputs = encode(
        batch=[[{"items": [{"amount": 1.0, "code": "A"}, {"amount": 2.0, "code": "B"}]}]],
        hyperparameters=model.hyperparameters,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    masked = next(mask([inputs], model.hyperparameters, Strata.train))

    assert int(masked["record/items/amount"].trainable.sum()) == 1
    assert not masked["record/items/code"].trainable.any()


def test_mask_literal_is_predict_only_and_does_not_enter_vocabulary():
    model = j2v.Model.from_schema(
        j2v.Category("code", max_vocab_size=8),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    model.encode([{"code": "A"}], strata=Strata.train)

    with pytest.raises(ValueError, match="only valid during predict"):
        model.encode([{"code": "<MASK>"}], strata=Strata.train)

    inputs = model.encode([{"code": "<MASK>"}], strata=Strata.predict)
    field = inputs["record/code"]

    assert field.state.tolist() == [[Tokens.masked.value]]
    assert not field.trainable.any()
    assert list(field.targets.keys()) == []
    assert model.nodes["record/code"].embedder.vocab.snapshot() == ["A"]
    prediction = model.predict([{"code": "<MASK>"}])["record/code"]
    assert prediction[TensorKey.inferred.name] == [True]


def test_inferred_marks_only_masked_predict_slots():
    model = j2v.Model.from_schema(
        j2v.Array(
            j2v.Category("letter", max_vocab_size=8, p_unavailable=0.0),
            name="letters",
            max_length=5,
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    model.encode(
        [{"letters": [{"letter": "A"}, {"letter": "B"}, {"letter": "C"}]}],
        strata=Strata.train,
        mask=False,
    )

    prediction = model.predict(
        [
            {
                "letters": [
                    {"letter": "A"},
                    {"letter": "<MASK>"},
                    {"letter": "C"},
                ]
            }
        ]
    )["record/letters/letter"]

    assert prediction[TensorKey.inferred.name] == [[False, True, False, False, False]]


def test_mask_literal_can_mask_whole_structured_leaf_but_not_leaf_items():
    model = j2v.Model.from_schema(
        j2v.Vector("embedding", n_dim=2),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    inputs = model.encode([{"embedding": "<MASK>"}], strata=Strata.predict)
    assert inputs["record/embedding"].state.tolist() == [[Tokens.masked.value]]

    with pytest.raises(ValueError, match="whole field value"):
        model.encode([{"embedding": [1.0, "<MASK>"]}], strata=Strata.predict)

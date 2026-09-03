import torch

import relflow as rf
from relflow.data.iterables import encode, mask
from relflow.structs.enums import Strata, TensorKey, Tokens
from tests.arrow import batch as arrow_batch
from tests.arrow import table


def test_branch_mask_count_targets_recent_real_slots_and_excludes_padding():
    model = rf.Model(
        rf.Branch(
            rf.Number("amount"),
            name="items",
            length=5,
            mask=rf.Mask(count=1, window=2),
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    inputs = encode(
        batch=arrow_batch([{"items": [{"amount": 1.0}, {"amount": 2.0}, {"amount": 3.0}]}]),
        schema=model.schema,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    field = next(mask([inputs], model.schema, Strata.train))["record/items/amount"]

    assert int(field.trainable.sum()) == 1
    assert field.trainable[0, 0, :3].any()
    assert not field.trainable[0, 0, 3:].any()
    assert field.state[0, 0, 3:].tolist() == [Tokens.padded.value, Tokens.padded.value]
    assert torch.equal(field.targets[TensorKey.state], torch.tensor([[[0, 0, 0, 2, 2]]]))


def test_model_encode_mask_flag_applies_and_skips_branch_masks():
    model = rf.Model(
        rf.Branch(
            rf.Number("amount"),
            name="items",
            length=5,
            mask=rf.Mask(count=1, window=2),
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    batch = table([{"items": [{"amount": 1.0}, {"amount": 2.0}, {"amount": 3.0}]}])

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


def test_branch_mask_exclude_address_skips_matching_leaf():
    model = rf.Model(
        rf.Branch(
            rf.Number("amount"),
            rf.Category("code", size=8),
            name="items",
            length=2,
            mask=rf.Mask(count=1, exclude="record/items/code"),
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    inputs = encode(
        batch=arrow_batch([{"items": [{"amount": 1.0, "code": "A"}, {"amount": 2.0, "code": "B"}]}]),
        schema=model.schema,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    masked = next(mask([inputs], model.schema, Strata.train))

    assert int(masked["record/items/amount"].trainable.sum()) == 1
    assert not masked["record/items/code"].trainable.any()


def test_branch_mask_exclude_relative_address_skips_matching_leaf():
    model = rf.Model(
        rf.Branch(
            rf.Number("amount"),
            rf.Category("code", size=8),
            name="items",
            length=2,
            mask=rf.Mask(count=1, exclude="code"),
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    inputs = encode(
        batch=arrow_batch([{"items": [{"amount": 1.0, "code": "A"}, {"amount": 2.0, "code": "B"}]}]),
        schema=model.schema,
        strata=Strata.train,
        interprocess_encoding_context=model.interprocess_encoding_context,
    )

    masked = next(mask([inputs], model.schema, Strata.train))

    assert int(masked["record/items/amount"].trainable.sum()) == 1
    assert not masked["record/items/code"].trainable.any()

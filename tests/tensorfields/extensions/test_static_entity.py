import pytest
import torch

import relflow as rf
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.tensorfields.extensions.static_entity import (
    Decoder,
    Embedder,
    EpochSaltCallback,
    TensorField,
    _bucketize,
    _get_epoch_salt,
    _hash_value,
    _set_epoch_salt,
    _static_hash_content,
)

ADDRESS = "root/items/identifier"


def _structure_payload(
    *,
    length: int = 2,
    n_hashes: int = 4,
    n_bands: int = 4,
    offset: int = 2,
    n_buckets: int = 4,
) -> dict:
    field: dict = {
        "name": "identifier",
        "type": "static_entity",
        "query": "[*].items[*].id",
        "n_hashes": n_hashes,
        "n_bands": n_bands,
        "offset": offset,
        "n_buckets": n_buckets,
    }
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "dropout": 0.1,
            "fields": [
                {
                    "name": "items",
                    "type": "branch",
                    "length": length,
                    "fields": [field],
                }
            ],
        },
    }


# --- request / schema validation --------------------------------------------------


def test_static_entity_request_defaults_load_without_extra_config():
    payload = _structure_payload()
    del payload["fields"]["fields"][0]["fields"][0]["n_hashes"]
    del payload["fields"]["fields"][0]["fields"][0]["n_bands"]
    del payload["fields"]["fields"][0]["fields"][0]["offset"]
    del payload["fields"]["fields"][0]["fields"][0]["n_buckets"]

    schema = Schema.model_validate(payload)
    request = schema.requests[ADDRESS]

    assert request.type == "static_entity"
    assert request.n_hashes == 1
    assert request.n_bands == 8
    assert request.offset == 4
    assert request.n_buckets == 4


def test_static_entity_request_rejects_non_positive_config():
    with pytest.raises(ValueError):
        Schema.model_validate(_structure_payload(n_hashes=0))
    with pytest.raises(ValueError):
        Schema.model_validate(_structure_payload(n_bands=0))
    with pytest.raises(ValueError):
        Schema.model_validate(_structure_payload(offset=0))
    with pytest.raises(ValueError):
        Schema.model_validate(_structure_payload(n_buckets=1))


def test_static_entity_allows_single_slot_length():
    # Unlike Entity, StaticEntity has no reidentification objective, so a single
    # slot is a legal configuration.
    schema = Schema.model_validate(_structure_payload(length=1))
    assert schema.shapes[ADDRESS] == (1, 1)


# --- hash primitive verification --------------------------------------------------


def test_hash_value_is_deterministic_and_bounded():
    value = "alice"
    outputs = [_hash_value(value, seed=seed) for seed in range(8)]

    for h in outputs:
        assert -1.0 <= h < 1.0
    # calling again yields the exact same floats.
    assert outputs == [_hash_value(value, seed=seed) for seed in range(8)]


def test_hash_value_uses_seed_independence():
    value = "alice"
    a = _hash_value(value, seed=0)
    b = _hash_value(value, seed=1)
    assert a != b


def test_hash_value_distinguishes_common_python_types():
    # repr() canonicalization makes 1 (int) and "1" (str) distinct inputs.
    assert _hash_value(1, seed=0) != _hash_value("1", seed=0)
    assert _hash_value(True, seed=0) != _hash_value(1, seed=0)
    assert _hash_value(1.0, seed=0) != _hash_value(1, seed=0)


# --- tensorfield content behaviour ------------------------------------------------


def test_static_entity_tensorfield_content_is_deterministic_across_observations():
    schema = Schema.model_validate(_structure_payload(n_hashes=4))
    values = [
        [["alice", "bob"]],
        [["alice", "carol"]],
    ]

    field = TensorField.new(
        values=values,
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )

    assert field.content.shape == (2, 1, 2, 4)
    assert torch.equal(field.content[0, 0, 0], field.content[1, 0, 0])
    assert not torch.equal(field.content[0, 0, 0], field.content[0, 0, 1])


def test_static_entity_tensorfield_zero_pads_nulls_and_padding():
    schema = Schema.model_validate(_structure_payload(n_hashes=4))
    values = [
        [["alice", None]],
        [["alice"]],
    ]

    field = TensorField.new(
        values=values,
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )

    assert field.state[0, 0, 0].item() == Tokens.valued.value
    assert field.state[0, 0, 1].item() == Tokens.null.value
    assert field.state[1, 0, 1].item() == Tokens.padded.value

    assert torch.all(field.content[0, 0, 1] == 0.0)
    assert torch.all(field.content[1, 0, 1] == 0.0)


def test_static_entity_tensorfield_rejects_unhashable_values():
    schema = Schema.model_validate(_structure_payload())
    values = [
        [[[1, 2], "ok"]],
        [["x", "y"]],
    ]

    with pytest.raises(ValueError, match="only accepts hashable scalar values"):
        TensorField.new(
            values=values,
            address=ADDRESS,
            schema=schema,
            strata=Strata.train,
        )


def test_static_entity_mask_caches_targets_before_zeroing():
    schema = Schema.model_validate(_structure_payload(n_hashes=4))
    values = [
        [["a", "b"]],
        [["c", "d"]],
    ]

    field = TensorField.new(
        values=values,
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )
    original_state = field.state.clone()
    original_content = field.content.clone()

    field.mask(1.0)

    assert torch.equal(field.targets[TensorKey.state], original_state)
    assert torch.equal(field.targets[TensorKey.content], original_content)
    assert torch.all(field.state == Tokens.masked.value)
    assert torch.all(field.content == 0.0)


# --- injectivity probes -----------------------------------------------------------


def _hash_matrix(values, n_hashes: int) -> torch.Tensor:
    """Return a `(len(values), n_hashes)` tensor of raw hash floats."""
    import numpy as np

    data = np.asarray(values, dtype=object).reshape(-1)
    states = np.full_like(data, fill_value=Tokens.valued.value, dtype=object)
    content = _static_hash_content(data=data, states=states, n_hashes=n_hashes)
    return torch.tensor(content, dtype=torch.float)


def test_static_entity_hashes_never_collide_over_10000_distinct_strings():
    """Empirical injectivity probe.

    Blake2b is a cryptographic hash (not a perfect hash function). With an
    8-byte truncated digest per seed and `n_hashes` seeds we have 64 * n_hashes
    bits of effective output space per input, so raw-float collisions on
    10k distinct short strings should be astronomically improbable.
    """
    n_hashes = 4
    values = [f"user-{index}" for index in range(10_000)]

    matrix = _hash_matrix(values, n_hashes=n_hashes)

    # Convert each row to a fingerprint tuple; count uniques.
    fingerprints = {tuple(row.tolist()) for row in matrix}
    assert len(fingerprints) == len(values)


def test_static_entity_hashes_are_stable_across_calls():
    n_hashes = 3
    values = ["alpha", "bravo", "charlie", "delta"]

    matrix_a = _hash_matrix(values, n_hashes=n_hashes)
    matrix_b = _hash_matrix(values, n_hashes=n_hashes)

    assert torch.equal(matrix_a, matrix_b)


def test_static_entity_sign_fingerprint_has_bounded_capacity():
    """Bucket-collapsed fingerprints cap at `n_buckets ** n_hashes` distinct identities.

    Deliberate probe of the reconstruction capacity ceiling the loss enforces.
    """
    n_hashes = 4
    n_buckets = 2
    values = [f"user-{index}" for index in range(500)]

    matrix = _hash_matrix(values, n_hashes=n_hashes)
    bucket_ids = _bucketize(matrix, n_buckets=n_buckets)
    fingerprints = {tuple(row.tolist()) for row in bucket_ids}

    assert len(fingerprints) <= n_buckets**n_hashes
    # With 500 inputs into 2**4 = 16 buckets we necessarily observe collisions.
    assert len(fingerprints) < len(values)


def test_static_entity_more_hashes_increase_sign_fingerprint_capacity():
    values = [f"user-{index}" for index in range(2_000)]
    n_buckets = 2

    small = _bucketize(_hash_matrix(values, n_hashes=4), n_buckets=n_buckets)
    large = _bucketize(_hash_matrix(values, n_hashes=16), n_buckets=n_buckets)

    small_uniques = len({tuple(row.tolist()) for row in small})
    large_uniques = len({tuple(row.tolist()) for row in large})

    assert small_uniques <= n_buckets**4
    assert large_uniques > small_uniques


def test_static_entity_more_buckets_increase_fingerprint_capacity():
    values = [f"user-{index}" for index in range(4_000)]
    n_hashes = 4

    coarse = _bucketize(_hash_matrix(values, n_hashes=n_hashes), n_buckets=2)
    fine = _bucketize(_hash_matrix(values, n_hashes=n_hashes), n_buckets=16)

    coarse_uniques = len({tuple(row.tolist()) for row in coarse})
    fine_uniques = len({tuple(row.tolist()) for row in fine})

    assert coarse_uniques <= 2**n_hashes
    assert fine_uniques > coarse_uniques
    assert fine_uniques <= 16**n_hashes


def test_static_entity_bucketize_range_is_valid():
    matrix = _hash_matrix([f"v-{i}" for i in range(2_000)], n_hashes=4)
    for k in (2, 4, 8, 16):
        buckets = _bucketize(matrix, n_buckets=k)
        assert int(buckets.min().item()) >= 0
        assert int(buckets.max().item()) < k


# --- embedder / decoder / loss end-to-end -----------------------------------------


def test_static_entity_embedder_has_no_learned_embedding_tables():
    model = rf.Model(
        rf.Branch(rf.StaticEntity("id", n_hashes=4, n_bands=4, offset=2), name="items", length=2),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
    )
    embedder = model.nodes["record/items/id"].embedder

    named_modules = dict(embedder.named_children())

    # only a projection Linear; no nn.Embedding tables anywhere.
    assert any(isinstance(m, torch.nn.Linear) for m in named_modules.values())
    assert not any(isinstance(m, torch.nn.Embedding) for m in named_modules.values())


def test_static_entity_embedder_forward_produces_finite_projections():
    model = rf.Model(
        rf.Branch(rf.StaticEntity("id", n_hashes=4), name="items", length=3),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
    )
    inputs = model.encode(
        [
            {"items": [{"id": "a"}, {"id": "b"}, {"id": "c"}]},
            {"items": [{"id": "d"}, {"id": "e"}, {"id": None}]},
        ],
        strata=Strata.train,
        mask=False,
    )

    predictions = model(inputs, strata=Strata.train)
    for prediction in predictions:
        for tensor in prediction.payload.values():
            assert torch.isfinite(tensor).all()


def test_static_entity_training_loss_covers_state_and_content_heads():
    torch.manual_seed(0)
    n_hashes = 4
    n_buckets = 4
    model = rf.Model(
        rf.Branch(
            rf.StaticEntity("id", n_hashes=n_hashes, n_buckets=n_buckets),
            name="items",
            length=3,
        ),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
    )
    address = "record/items/id"
    inputs = model.encode(
        [
            {"items": [{"id": "a"}, {"id": "b"}, {"id": "c"}]},
            {"items": [{"id": "d"}, {"id": "e"}, {"id": "f"}]},
        ],
        strata=Strata.train,
        mask=False,
    )
    field = inputs[address]
    field.hide(torch.ones_like(field.state, dtype=torch.bool))

    predictions = model(inputs, strata=Strata.train)
    prediction = next(p for p in predictions if p.address == address)

    assert prediction.payload[TensorKey.state].shape[-1] == len(Tokens)
    assert prediction.payload[TensorKey.content].shape[-1] == n_hashes * n_buckets

    output = model.training_step(inputs, batch_idx=0)
    assert torch.isfinite(output["loss"])


def test_static_entity_content_target_matches_deterministic_recomputation():
    schema = Schema.model_validate(_structure_payload(n_hashes=5, length=2))
    values = [[["alice", "bob"]], [["carol", "dave"]]]

    field = TensorField.new(
        values=values,
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )

    # Cache targets then re-derive from a fresh TensorField and compare.
    field.mask(1.0)

    fresh = TensorField.new(
        values=values,
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )

    assert torch.equal(field.targets[TensorKey.content], fresh.content)


def test_static_entity_decoder_content_head_width_matches_n_hashes():
    schema = Schema.model_validate(_structure_payload(n_hashes=7, n_buckets=5))
    decoder = Decoder(schema=schema, address=ADDRESS)

    assert decoder.n_hashes == 7
    assert decoder.n_buckets == 5
    assert decoder.content_linear.out_features == 7 * 5
    assert decoder.state_linear.out_features == len(Tokens)


def test_static_entity_embedder_projection_input_width_matches_features():
    schema = Schema.model_validate(_structure_payload(n_hashes=3, n_bands=4, offset=2))
    embedder = Embedder(schema=schema, address=ADDRESS)

    n_frequencies = 4 + 2 + 1  # logspace(-n_bands, offset, n_bands + offset + 1)
    expected = 3 * 2 * n_frequencies + len(Tokens)
    assert embedder.linear.in_features == expected
    assert embedder.linear.out_features == schema.d_model


# --- group parameter sharing ------------------------------------------------------


def _grouped_model(group: str | None = "user", **kwargs) -> "rf.Model":
    return rf.Model(
        rf.Branch(
            rf.StaticEntity("id", n_hashes=4, n_buckets=4, group=group, **kwargs),
            name="items",
            length=2,
        ),
        rf.StaticEntity("owner", n_hashes=4, n_buckets=4, group=group, **kwargs),
        d_model=16,
        n_layers=1,
        n_heads=2,
        batch_size=2,
    )


def test_static_entity_group_shares_encoder_and_decoder_linears_by_identity():
    model = _grouped_model()
    items = model.nodes["record/items/id"]
    owner = model.nodes["record/owner"]

    assert items.embedder.linear is owner.embedder.linear
    assert items.decoder.content_linear is owner.decoder.content_linear
    # state_linear is intentionally per-address (state one-hot is not identity).
    assert items.decoder.state_linear is not owner.decoder.state_linear


def test_static_entity_group_defaults_to_none_and_keeps_linears_distinct():
    model = _grouped_model(group=None)
    items = model.nodes["record/items/id"]
    owner = model.nodes["record/owner"]

    assert items.embedder.linear is not owner.embedder.linear
    assert items.decoder.content_linear is not owner.decoder.content_linear


def test_static_entity_group_gives_same_input_value_identical_projections():
    model = _grouped_model()
    inputs = model.encode(
        [
            {"items": [{"id": "alice"}, {"id": "bob"}], "owner": "alice"},
            {"items": [{"id": "carol"}, {"id": "dave"}], "owner": "carol"},
        ],
        strata=Strata.train,
        mask=False,
    )

    items_projection = model.nodes["record/items/id"].embedder(inputs["record/items/id"]).payload
    owner_projection = model.nodes["record/owner"].embedder(inputs["record/owner"]).payload

    # observation 0: "alice" appears as items[0][0] and as owner.
    assert torch.allclose(items_projection[0, 0, 0], owner_projection[0, 0])
    # observation 1: "carol" appears as items[0][0] and as owner.
    assert torch.allclose(items_projection[1, 0, 0], owner_projection[1, 0])


def test_static_entity_group_rejects_inconsistent_config():
    with pytest.raises(ValueError, match="inconsistent"):
        rf.Model(
            rf.Branch(
                rf.StaticEntity("id", n_hashes=4, n_buckets=4, group="user"),
                name="items",
                length=2,
            ),
            rf.StaticEntity("owner", n_hashes=8, n_buckets=4, group="user"),
            d_model=16,
            n_layers=1,
            n_heads=2,
            batch_size=2,
        )

    with pytest.raises(ValueError, match="inconsistent"):
        rf.Model(
            rf.Branch(
                rf.StaticEntity("id", n_hashes=4, n_buckets=4, group="user"),
                name="items",
                length=2,
            ),
            rf.StaticEntity("owner", n_hashes=4, n_buckets=8, group="user"),
            d_model=16,
            n_layers=1,
            n_heads=2,
            batch_size=2,
        )


def test_static_entity_group_deduplicates_shared_parameters_in_model_iterator():
    model = _grouped_model()

    unique_ids: set[int] = set()
    duplicates: set[int] = set()
    for _, tensor in model.named_parameters(remove_duplicate=False):
        identifier = id(tensor)
        if identifier in unique_ids:
            duplicates.add(identifier)
        unique_ids.add(identifier)

    # With remove_duplicate=False we should see both aliases for the shared params.
    shared_embedder = id(model.nodes["record/items/id"].embedder.linear.weight)
    shared_decoder = id(model.nodes["record/items/id"].decoder.content_linear.weight)
    assert shared_embedder in duplicates
    assert shared_decoder in duplicates

    # And remove_duplicate=True should hide the aliases.
    deduped_ids = {id(tensor) for _, tensor in model.named_parameters(remove_duplicate=True)}
    assert shared_embedder in deduped_ids
    assert shared_decoder in deduped_ids
    duplicate_count = sum(1 for _, t in model.named_parameters(remove_duplicate=False) if id(t) == shared_embedder)
    unique_count = sum(1 for _, t in model.named_parameters(remove_duplicate=True) if id(t) == shared_embedder)
    assert duplicate_count == 2
    assert unique_count == 1


def test_static_entity_group_training_step_backpropagates_through_shared_linears():
    torch.manual_seed(0)
    model = _grouped_model()
    address_items = "record/items/id"
    address_owner = "record/owner"

    inputs = model.encode(
        [
            {"items": [{"id": "alice"}, {"id": "bob"}], "owner": "alice"},
            {"items": [{"id": "carol"}, {"id": "dave"}], "owner": "carol"},
        ],
        strata=Strata.train,
        mask=False,
    )
    inputs[address_items].hide(torch.ones_like(inputs[address_items].state, dtype=torch.bool))
    inputs[address_owner].hide(torch.ones_like(inputs[address_owner].state, dtype=torch.bool))

    output = model.training_step(inputs, batch_idx=0)
    assert torch.isfinite(output["loss"])

    output["loss"].backward()
    shared_encoder = model.nodes[address_items].embedder.linear.weight
    shared_decoder = model.nodes[address_items].decoder.content_linear.weight
    assert shared_encoder.grad is not None and torch.isfinite(shared_encoder.grad).all()
    assert shared_decoder.grad is not None and torch.isfinite(shared_decoder.grad).all()


# --- epoch salt rotation ----------------------------------------------------------


def test_hash_value_rotates_with_salt():
    value = "alice"
    baseline = _hash_value(value, seed=0)
    rotated = _hash_value(value, seed=0, salt=1)
    assert baseline != rotated
    assert baseline == _hash_value(value, seed=0, salt=0)


def test_static_hash_content_rotates_with_salt():
    import numpy as np

    data = np.array([["alice", "bob"]], dtype=object)
    states = np.full(data.shape, Tokens.valued.value, dtype=np.int64)

    epoch0 = _static_hash_content(data=data, states=states, n_hashes=4, salt=0)
    epoch1 = _static_hash_content(data=data, states=states, n_hashes=4, salt=1)
    assert not np.array_equal(epoch0, epoch1)
    assert np.array_equal(epoch0, _static_hash_content(data=data, states=states, n_hashes=4, salt=0))


def test_tensorfield_content_reflects_current_epoch_salt():
    schema = Schema.model_validate(_structure_payload(length=2, n_hashes=4))
    values = [[["alice", "bob"]]]

    try:
        _set_epoch_salt(schema, 0)
        field_a = TensorField.new(values=values, address=ADDRESS, schema=schema, strata=Strata.train)
        _set_epoch_salt(schema, 1)
        field_b = TensorField.new(values=values, address=ADDRESS, schema=schema, strata=Strata.train)
        _set_epoch_salt(schema, 0)
        field_c = TensorField.new(values=values, address=ADDRESS, schema=schema, strata=Strata.train)
    finally:
        _set_epoch_salt(schema, 0)

    assert not torch.equal(field_a.content, field_b.content)
    assert torch.equal(field_a.content, field_c.content)


def test_epoch_salt_defaults_to_zero_per_schema():
    schema = Schema.model_validate(_structure_payload())
    assert _get_epoch_salt(schema) == 0


def test_epoch_salt_isolated_between_schemas():
    schema_a = Schema.model_validate(_structure_payload())
    schema_b = Schema.model_validate(_structure_payload())

    _set_epoch_salt(schema_a, 7)
    assert _get_epoch_salt(schema_a) == 7
    assert _get_epoch_salt(schema_b) == 0


def test_epoch_salt_registry_is_cleaned_up_after_schema_gc():
    import gc

    from relflow.tensorfields.extensions import static_entity as module

    schema = Schema.model_validate(_structure_payload())
    _set_epoch_salt(schema, 3)
    key = id(schema)
    assert key in module._EPOCH_SALT

    del schema
    gc.collect()
    assert key not in module._EPOCH_SALT


def test_grouped_values_stay_matched_under_rotating_salt():
    model = _grouped_model()
    records = [{"items": [{"id": "alice"}, {"id": "bob"}], "owner": "alice"}]

    projections_by_epoch: list[tuple[torch.Tensor, torch.Tensor]] = []
    try:
        for salt in (0, 1, 2):
            _set_epoch_salt(model.schema, salt)
            inputs = model.encode(records, strata=Strata.train, mask=False)
            items_projection = model.nodes["record/items/id"].embedder(inputs["record/items/id"]).payload
            owner_projection = model.nodes["record/owner"].embedder(inputs["record/owner"]).payload
            projections_by_epoch.append((items_projection[0, 0, 0].detach(), owner_projection[0, 0].detach()))
    finally:
        _set_epoch_salt(model.schema, 0)

    # Same value at both addresses matches within each epoch (group sharing preserved under rotation).
    for items_alice, owner_alice in projections_by_epoch:
        assert torch.allclose(items_alice, owner_alice)

    # But the alice representation itself rotates across epochs.
    for later in projections_by_epoch[1:]:
        assert not torch.allclose(projections_by_epoch[0][0], later[0])


def test_epoch_salt_callback_registered_via_plugin():
    from relflow.tensorfields.extensions.static_entity import static_entity

    factories = static_entity.callback_factories
    assert EpochSaltCallback in factories


def test_epoch_salt_callback_hooks_track_lightning_epoch():
    schema = Schema.model_validate(_structure_payload())

    class _Stub:
        def __init__(self, schema: Schema, current_epoch: int) -> None:
            self.schema = schema
            self.current_epoch = current_epoch

    callback = EpochSaltCallback()

    callback.on_train_epoch_start(trainer=None, pl_module=_Stub(schema, current_epoch=0))
    assert _get_epoch_salt(schema) == 1

    callback.on_train_epoch_start(trainer=None, pl_module=_Stub(schema, current_epoch=4))
    assert _get_epoch_salt(schema) == 5

    callback.on_validation_epoch_start(trainer=None, pl_module=_Stub(schema, current_epoch=4))
    assert _get_epoch_salt(schema) == 5

    callback.on_test_epoch_start(trainer=None, pl_module=_Stub(schema, current_epoch=4))
    assert _get_epoch_salt(schema) == 0

    callback.on_predict_epoch_start(trainer=None, pl_module=_Stub(schema, current_epoch=4))
    assert _get_epoch_salt(schema) == 0


def test_configure_callbacks_includes_epoch_salt_callback_when_static_entity_active():
    model = rf.Model(
        rf.StaticEntity("owner", n_hashes=2, n_buckets=4),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=1,
    )

    callback_types = {type(cb) for cb in model.configure_callbacks()}
    assert EpochSaltCallback in callback_types

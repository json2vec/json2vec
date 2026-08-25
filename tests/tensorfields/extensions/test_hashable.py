from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import relflow as rf
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.tensorfields.extensions.hashable import (
    Decoder,
    Embedder,
    TensorField,
    salt,
)

ADDRESS = "root/items/identifier"


def _structure_payload(
    *,
    length: int = 2,
    n_hashes: int = 4,
    n_bands: int = 4,
    offset: int = 2,
    n_buckets: int = 4,
    deterministic: bool = False,
) -> dict:
    field: dict = {
        "name": "identifier",
        "type": "hash",
        "query": "[*].items[*].id",
        "n_hashes": n_hashes,
        "n_bands": n_bands,
        "offset": offset,
        "n_buckets": n_buckets,
        "deterministic": deterministic,
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


def test_hashable_request_defaults_load_without_extra_config():
    payload = _structure_payload()
    del payload["fields"]["fields"][0]["fields"][0]["n_hashes"]
    del payload["fields"]["fields"][0]["fields"][0]["n_bands"]
    del payload["fields"]["fields"][0]["fields"][0]["offset"]
    del payload["fields"]["fields"][0]["fields"][0]["n_buckets"]
    del payload["fields"]["fields"][0]["fields"][0]["deterministic"]

    schema = Schema.model_validate(payload)
    request = schema.requests[ADDRESS]

    assert request.type == "hash"
    assert request.n_hashes == 1
    assert request.n_bands == 8
    assert request.offset == 4
    assert request.n_buckets == 4
    assert request.deterministic is False


def test_hashable_request_accepts_deterministic_mode():
    schema = Schema.model_validate(_structure_payload(deterministic=True))
    restored = Schema.model_validate(schema.model_dump())

    assert schema.requests[ADDRESS].deterministic is True
    assert restored.requests[ADDRESS].deterministic is True


def test_hashable_request_rejects_non_positive_config():
    with pytest.raises(ValueError):
        Schema.model_validate(_structure_payload(n_hashes=0))
    with pytest.raises(ValueError):
        Schema.model_validate(_structure_payload(n_bands=0))
    with pytest.raises(ValueError):
        Schema.model_validate(_structure_payload(offset=0))
    with pytest.raises(ValueError):
        Schema.model_validate(_structure_payload(n_buckets=1))


def test_hashable_allows_single_slot_length():
    # Hash has no local-vocabulary reidentification objective, so a single
    # slot is a legal configuration.
    schema = Schema.model_validate(_structure_payload(length=1))
    assert schema.shapes[ADDRESS] == (1, 1)


# --- hash primitive verification --------------------------------------------------


def test_hash_vector_is_deterministic_int64():
    outputs = _hash_matrix(["alice"], n_hashes=8)[0]

    assert outputs.dtype == torch.int64
    assert torch.equal(outputs, _hash_matrix(["alice"], n_hashes=8)[0])


def test_hash_vector_channels_are_independent():
    outputs = _hash_matrix(["alice"], n_hashes=8)[0]
    assert outputs.unique().numel() == outputs.numel()


def test_hash_value_distinguishes_common_python_types():
    integer, string, boolean, floating = _hash_matrix([1, "1", True, 1.0], n_hashes=4)
    assert not torch.equal(integer, string)
    assert not torch.equal(boolean, integer)
    assert not torch.equal(floating, integer)


# --- tensorfield content behaviour ------------------------------------------------


def test_hashable_tensorfield_content_is_deterministic_across_observations():
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
    assert field.content.dtype == torch.int64
    assert torch.equal(field.content[0, 0, 0], field.content[1, 0, 0])
    assert not torch.equal(field.content[0, 0, 0], field.content[0, 0, 1])


def test_hashable_tensorfield_zero_pads_nulls_and_padding():
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


@pytest.mark.parametrize("unsupported", [[1, 2], object()])
def test_hashable_tensorfield_rejects_unsupported_values(unsupported):
    schema = Schema.model_validate(_structure_payload())
    values = [
        [[unsupported, "ok"]],
        [["x", "y"]],
    ]

    with pytest.raises(ValueError, match="only accepts MessagePack-compatible hashable scalar values"):
        TensorField.new(
            values=values,
            address=ADDRESS,
            schema=schema,
            strata=Strata.train,
        )


def test_hashable_mask_caches_targets_before_zeroing():
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
    """Tensorize values into a `(len(values), n_hashes)` integer matrix."""
    schema = Schema.model_validate(_structure_payload(length=len(values), n_hashes=n_hashes))
    field = TensorField.new(
        values=[[values]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.test,
    )
    return field.content.reshape(len(values), n_hashes)


def _bucket_ids(content: torch.Tensor, n_buckets: int) -> torch.Tensor:
    return (
        content.to(dtype=torch.get_default_dtype())
        .div(1 << 63)
        .add(1.0)
        .mul(0.5 * n_buckets)
        .floor()
        .long()
        .clamp(min=0, max=n_buckets - 1)
    )


def test_hashable_hashes_never_collide_over_10000_distinct_strings():
    """Empirical injectivity probe.

    BLAKE3 is a cryptographic hash (not a perfect hash function). Its extended
    output provides 64 * n_hashes bits per input, so fingerprint collisions on
    10k short strings should be improbable.
    """
    n_hashes = 4
    values = [f"user-{index}" for index in range(10_000)]

    matrix = _hash_matrix(values, n_hashes=n_hashes)

    # Convert each row to a fingerprint tuple; count uniques.
    fingerprints = {tuple(row.tolist()) for row in matrix}
    assert len(fingerprints) == len(values)


def test_hashable_hashes_are_stable_across_calls():
    n_hashes = 3
    values = ["alpha", "bravo", "charlie", "delta"]

    matrix_a = _hash_matrix(values, n_hashes=n_hashes)
    matrix_b = _hash_matrix(values, n_hashes=n_hashes)

    assert torch.equal(matrix_a, matrix_b)


def test_hashable_sign_fingerprint_has_bounded_capacity():
    """Bucket-collapsed fingerprints cap at `n_buckets ** n_hashes` distinct identities.

    Deliberate probe of the reconstruction capacity ceiling the loss enforces.
    """
    n_hashes = 4
    n_buckets = 2
    values = [f"user-{index}" for index in range(500)]

    matrix = _hash_matrix(values, n_hashes=n_hashes)
    bucket_ids = _bucket_ids(matrix, n_buckets=n_buckets)
    fingerprints = {tuple(row.tolist()) for row in bucket_ids}

    assert len(fingerprints) <= n_buckets**n_hashes
    # With 500 inputs into 2**4 = 16 buckets we necessarily observe collisions.
    assert len(fingerprints) < len(values)


def test_hashable_more_hashes_increase_sign_fingerprint_capacity():
    values = [f"user-{index}" for index in range(2_000)]
    n_buckets = 2

    small = _bucket_ids(_hash_matrix(values, n_hashes=4), n_buckets=n_buckets)
    large = _bucket_ids(_hash_matrix(values, n_hashes=16), n_buckets=n_buckets)

    small_uniques = len({tuple(row.tolist()) for row in small})
    large_uniques = len({tuple(row.tolist()) for row in large})

    assert small_uniques <= n_buckets**4
    assert large_uniques > small_uniques


def test_hashable_more_buckets_increase_fingerprint_capacity():
    values = [f"user-{index}" for index in range(4_000)]
    n_hashes = 4

    coarse = _bucket_ids(_hash_matrix(values, n_hashes=n_hashes), n_buckets=2)
    fine = _bucket_ids(_hash_matrix(values, n_hashes=n_hashes), n_buckets=16)

    coarse_uniques = len({tuple(row.tolist()) for row in coarse})
    fine_uniques = len({tuple(row.tolist()) for row in fine})

    assert coarse_uniques <= 2**n_hashes
    assert fine_uniques > coarse_uniques
    assert fine_uniques <= 16**n_hashes


def test_hashable_bucketize_range_is_valid():
    matrix = _hash_matrix([f"v-{i}" for i in range(2_000)], n_hashes=4)
    for k in (2, 4, 8, 16):
        buckets = _bucket_ids(matrix, n_buckets=k)
        assert int(buckets.min().item()) >= 0
        assert int(buckets.max().item()) < k


# --- embedder / decoder / loss end-to-end -----------------------------------------


def test_hashable_embedder_only_learns_state_embeddings():
    model = rf.Model(
        rf.Branch(rf.Hash("id", n_hashes=4, n_bands=4, offset=2), name="items", length=2),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
    )
    embedder = model.nodes["record/items/id"].embedder

    named_modules = dict(embedder.named_children())

    assert named_modules == {"state_embeddings": embedder.state_embeddings}
    assert isinstance(embedder.state_embeddings, torch.nn.Embedding)


def test_hashable_embedder_forward_produces_finite_projections():
    model = rf.Model(
        rf.Branch(rf.Hash("id", n_hashes=4), name="items", length=3),
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


def test_hashable_training_loss_covers_state_and_content_heads():
    torch.manual_seed(0)
    n_hashes = 4
    n_buckets = 4
    model = rf.Model(
        rf.Branch(
            rf.Hash("id", n_hashes=n_hashes, n_buckets=n_buckets),
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


def test_hashable_content_target_matches_deterministic_recomputation():
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


def test_hashable_decoder_content_head_width_matches_n_hashes():
    schema = Schema.model_validate(_structure_payload(n_hashes=7, n_buckets=5))
    decoder = Decoder(schema=schema, address=ADDRESS)

    assert decoder.n_hashes == 7
    assert decoder.n_buckets == 5
    assert decoder.content_linear.out_features == 7 * 5
    assert decoder.state_linear.out_features == len(Tokens)


def test_hashable_embedder_uses_raw_sinusoidal_content_and_state_embeddings():
    schema = Schema.model_validate(_structure_payload(n_hashes=3, n_bands=4, offset=2))
    embedder = Embedder(schema=schema, address=ADDRESS)
    field = TensorField.new(
        values=[[["alice", "bob"]]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.test,
    )

    output = embedder(field).payload
    normalized = field.content.reshape(-1, 3).to(embedder.weights.dtype).div(1 << 63)
    weighted = normalized.unsqueeze(-1).mul(embedder.weights)
    expected = torch.stack([torch.sin(weighted), torch.cos(weighted)], dim=-1)
    expected = expected.flatten(start_dim=-2)[..., : schema.d_model].sum(dim=1)

    assert torch.allclose(output.reshape(-1, schema.d_model), expected)
    assert embedder.state_embeddings.embedding_dim == schema.d_model
    assert not hasattr(embedder, "linear")


# --- identity across fields -------------------------------------------------------


def _hashable_model(**kwargs) -> "rf.Model":
    return rf.Model(
        rf.Branch(
            rf.Hash("id", n_hashes=4, n_buckets=4, **kwargs),
            name="items",
            length=2,
        ),
        rf.Hash("owner", n_hashes=4, n_buckets=4, **kwargs),
        d_model=16,
        n_layers=1,
        n_heads=2,
        batch_size=2,
    )


def test_hashable_fields_keep_state_embeddings_and_decoders_local():
    model = _hashable_model()
    items = model.nodes["record/items/id"]
    owner = model.nodes["record/owner"]

    assert items.embedder.state_embeddings is not owner.embedder.state_embeddings
    assert items.decoder.content_linear is not owner.decoder.content_linear
    assert items.decoder.state_linear is not owner.decoder.state_linear


def test_hashable_gives_same_input_value_identical_raw_embeddings_across_fields():
    model = _hashable_model()
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

    assert torch.allclose(items_projection[0, 0, 0], owner_projection[0, 0])
    assert torch.allclose(items_projection[1, 0, 0], owner_projection[1, 0])


# --- device-side optimizer-step salt ---------------------------------------------


def test_cpu_hashes_are_static_across_calls_and_strata(monkeypatch: pytest.MonkeyPatch):
    from relflow.data import iterables

    model = _hashable_model()
    deterministic_model = _hashable_model(deterministic=True)
    records = [{"items": [{"id": "alice"}, {"id": "bob"}], "owner": "alice"}]

    def unexpected_random_salt(bits: int) -> int:
        raise AssertionError("CPU encoding must not generate a random hash salt")

    monkeypatch.setattr(iterables.random, "getrandbits", unexpected_random_salt)

    baseline = model.encode(records, strata=Strata.train, mask=False)
    deterministic = deterministic_model.encode(records, strata=Strata.train, mask=False)
    for strata in Strata:
        first = model.encode(records, strata=strata, mask=False)
        second = model.encode(records, strata=strata, mask=False)

        for address in ("record/items/id", "record/owner"):
            assert torch.equal(deterministic[address].content, baseline[address].content)
            assert torch.equal(first[address].content, baseline[address].content)
            assert torch.equal(second[address].content, baseline[address].content)


def test_global_step_hash_is_deterministic_per_lane_and_does_not_mutate_content():
    content = torch.zeros(3, 4, dtype=torch.int64)
    original = content.clone()
    module = SimpleNamespace(global_step=0)

    first_salted = salt(
        module,
        deterministic=False,
        strata=Strata.train,
        n_hashes=4,
        device=content.device,
    )
    repeated_salted = salt(
        module,
        deterministic=False,
        strata=Strata.train,
        n_hashes=4,
        device=content.device,
    )
    module.global_step = 1
    next_salted = salt(
        module,
        deterministic=False,
        strata=Strata.train,
        n_hashes=4,
        device=content.device,
    )

    assert first_salted is not None
    assert repeated_salted is not None
    assert next_salted is not None
    first = torch.bitwise_xor(content, first_salted)

    assert first.shape == content.shape
    assert first.dtype == content.dtype
    assert first.device == content.device
    assert first[0].unique().numel() == content.shape[-1]
    assert torch.equal(first_salted, repeated_salted)
    assert not torch.equal(first_salted, next_salted)
    assert torch.equal(content, original)


def test_reconstruction_buckets_use_the_same_global_step_salt_as_embeddings():
    content = _hash_matrix([f"value-{index}" for index in range(64)], n_hashes=4)
    module = SimpleNamespace(global_step=0)
    first_salted = salt(
        module,
        deterministic=False,
        strata=Strata.train,
        n_hashes=4,
        device=content.device,
    )
    module.global_step = 1
    next_salted = salt(
        module,
        deterministic=False,
        strata=Strata.train,
        n_hashes=4,
        device=content.device,
    )

    assert first_salted is not None
    assert next_salted is not None
    first = _bucket_ids(torch.bitwise_xor(content, first_salted), n_buckets=4)
    following = _bucket_ids(torch.bitwise_xor(content, next_salted), n_buckets=4)

    assert not torch.equal(first, following)


def test_hashable_embedder_rotates_valued_content_by_global_step():
    schema = Schema.model_validate(_structure_payload(length=2, n_hashes=4))
    module = SimpleNamespace(global_step=0)
    embedder = Embedder(schema=schema, address=ADDRESS, module=module)
    field = TensorField.new(
        values=[[["alice", None]]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )
    original = field.content.clone()

    unsalted = embedder(field).payload
    module.global_step = 99
    test = embedder(field, strata=Strata.test).payload
    predict = embedder(field, strata=Strata.predict).payload

    valued_index = (0, 0, 0)
    null_index = (0, 0, 1)
    for strata in (Strata.train, Strata.validate):
        module.global_step = 0
        first = embedder(field, strata=strata).payload
        repeated = embedder(field, strata=strata).payload
        module.global_step = 1
        following = embedder(field, strata=strata).payload
        module.global_step = 0
        repeated_first_step = embedder(field, strata=strata).payload

        assert not torch.allclose(first[valued_index], unsalted[valued_index])
        assert torch.allclose(first[valued_index], repeated[valued_index])
        assert not torch.allclose(first[valued_index], following[valued_index])
        assert torch.allclose(first, repeated_first_step)
        assert torch.allclose(first[null_index], following[null_index])

    assert torch.allclose(test, unsalted)
    assert torch.allclose(predict, unsalted)
    assert torch.equal(field.content, original)


def test_deterministic_hashable_embedder_disables_training_and_validation_salt():
    schema = Schema.model_validate(_structure_payload(length=2, n_hashes=4, deterministic=True))
    module = SimpleNamespace(global_step=0)
    embedder = Embedder(schema=schema, address=ADDRESS, module=module)
    field = TensorField.new(
        values=[[["alice", "bob"]]],
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
    )

    expected = embedder(field).payload
    for strata in Strata:
        module.global_step = 0
        assert torch.allclose(embedder(field, strata=strata).payload, expected)
        module.global_step = 99
        assert torch.allclose(embedder(field, strata=strata).payload, expected)

    assert not hasattr(embedder, "_salt")


@pytest.mark.parametrize("deterministic", [False, True])
def test_equal_values_stay_matched_across_fields_at_each_global_step(deterministic: bool):
    model = _hashable_model(deterministic=deterministic)
    trainer = SimpleNamespace(global_step=0)
    model.trainer = trainer
    inputs = model.encode(
        [{"items": [{"id": "alice"}, {"id": "bob"}], "owner": "alice"}],
        strata=Strata.train,
        mask=False,
    )

    projections_by_step: list[torch.Tensor] = []
    for global_step in range(3):
        trainer.global_step = global_step
        items_embedder = model.nodes["record/items/id"].embedder
        owner_embedder = model.nodes["record/owner"].embedder
        items_projection = items_embedder(inputs["record/items/id"], strata=Strata.train).payload
        owner_projection = owner_embedder(inputs["record/owner"], strata=Strata.train).payload

        assert torch.allclose(items_projection[0, 0, 0], owner_projection[0, 0])
        projections_by_step.append(items_projection[0, 0, 0].detach())

    if deterministic:
        assert torch.allclose(projections_by_step[0], projections_by_step[1])
        assert torch.allclose(projections_by_step[1], projections_by_step[2])
    else:
        assert not torch.allclose(projections_by_step[0], projections_by_step[1])
        assert not torch.allclose(projections_by_step[1], projections_by_step[2])


def test_model_runtime_uses_global_step_salt_and_disables_it_for_inference():
    model = _hashable_model()
    trainer = SimpleNamespace(global_step=0)
    model.trainer = trainer
    inputs = model.encode(
        [{"items": [{"id": "alice"}, {"id": "bob"}], "owner": "alice"}],
        strata=Strata.train,
        mask=False,
    )
    captured: list[torch.Tensor] = []
    handle = model.nodes["record/items/id"].embedder.register_forward_hook(
        lambda _module, _args, output: captured.append(output.payload.detach().clone())
    )
    try:
        model(inputs, strata=Strata.train)
        trainer.global_step = 1
        model(inputs, strata=Strata.train)
        trainer.global_step = 100
        model(inputs, strata=Strata.predict)
        trainer.global_step = 200
        model(inputs, strata=Strata.predict)
    finally:
        handle.remove()

    assert not torch.allclose(captured[0], captured[1])
    assert torch.allclose(captured[2], captured[3])


def test_hashable_embedder_root_reference_is_not_a_registered_submodule():
    model = _hashable_model()
    embedder = model.nodes["record/items/id"].embedder

    assert model.global_step == 0
    assert embedder._module is model
    assert "_module" not in embedder._modules
    assert all(not key.startswith("nodes.record/items/id.embedder._module") for key in model.state_dict())

    copied = deepcopy(model)
    assert copied.nodes["record/items/id"].embedder._module is copied

    model._rebuild()
    assert model.nodes["record/items/id"].embedder._module is model


@pytest.mark.parametrize(("deterministic", "expects_salt"), [(False, True), (True, False)])
def test_hashable_loss_obeys_deterministic_mode(
    monkeypatch: pytest.MonkeyPatch,
    deterministic: bool,
    expects_salt: bool,
):
    model = rf.Model(
        rf.Hash("context", n_hashes=4, deterministic=deterministic),
        rf.Hash("label", n_hashes=4, target=True, deterministic=deterministic),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
    )
    trainer = SimpleNamespace(global_step=3)
    model.trainer = trainer
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    inputs = model.encode(
        [{"context": "alice", "label": "alice"}, {"context": "bob", "label": "bob"}],
        strata=Strata.train,
    )

    expected_salted = salt(
        model,
        deterministic=deterministic,
        strata=Strata.train,
        n_hashes=4,
        device=inputs["record/label"].content.device,
    )
    raw_targets = inputs["record/label"].targets[TensorKey.content]
    salted_targets = raw_targets if expected_salted is None else torch.bitwise_xor(raw_targets, expected_salted)
    expected_buckets = _bucket_ids(salted_targets.reshape(-1, 4), n_buckets=4).reshape(-1)

    captured_targets: list[torch.Tensor] = []
    original_cross_entropy = torch.nn.functional.cross_entropy

    def capture_targets(
        input: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        captured_targets.append(target.detach().clone())
        return original_cross_entropy(input=input, target=target, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", capture_targets)
    output = model.training_step(inputs, batch_idx=99)

    assert torch.isfinite(output["loss"])
    assert torch.equal(captured_targets[-1], expected_buckets)
    if expects_salt:
        assert expected_salted is not None
    else:
        assert expected_salted is None


def test_hashable_does_not_register_a_salt_callback():
    from relflow.tensorfields.extensions.hashable import hashable

    assert hashable.callback_factories == []

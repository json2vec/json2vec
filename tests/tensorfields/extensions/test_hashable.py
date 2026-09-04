import pytest
import torch

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.tree import Mask
from relflow.tensorfields.base import TENSORFIELDS, Context
from relflow.tensorfields.extensions.hashable import (
    Decoder,
    Embedder,
    TensorField,
)
from tests.arrow import batch as arrow_batch
from tests.arrow import table
from tests.tensorfields.helpers import tensorize

ADDRESS = "root/items/identifier"


def _structure_payload(
    *,
    length: int = 2,
    n_hashes: int = 4,
    n_bands: int = 4,
    offset: int = 2,
    n_buckets: int = 4,
    mask: bool | Mask = False,
) -> dict:
    field: dict = {
        "name": "identifier",
        "type": "hash",
        "n_hashes": n_hashes,
        "n_bands": n_bands,
        "offset": offset,
        "n_buckets": n_buckets,
        "mask": mask,
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


def _new_tensorfield(
    *,
    values: list,
    schema: Schema,
    strata: Strata,
    salt: int = 0,
) -> TensorField:
    batch = arrow_batch([{"items": [{"identifier": value} for value in root]} for (root,) in values])
    projection = coalesce(batch, schema=schema, strata=strata)[ADDRESS]
    return tensorize(
        TensorField,
        projection,
        TENSORFIELDS["hash"],
        address=ADDRESS,
        schema=schema,
        strata=strata,
        context=Context(salt=salt),
    )


# --- request / schema validation --------------------------------------------------


def test_hashable_request_defaults_load_without_extra_config():
    payload = _structure_payload()
    del payload["fields"]["fields"][0]["fields"][0]["n_hashes"]
    del payload["fields"]["fields"][0]["fields"][0]["n_bands"]
    del payload["fields"]["fields"][0]["fields"][0]["offset"]
    del payload["fields"]["fields"][0]["fields"][0]["n_buckets"]

    schema = Schema.model_validate(payload)
    request = schema.requests[ADDRESS]

    assert request.type == "hash"
    assert request.n_hashes == 1
    assert request.n_bands == 8
    assert request.offset == 4
    assert request.n_buckets == 4


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


def test_hash_value_preserves_python_scalar_identity_across_batches():
    integer = _hash_matrix([1], n_hashes=4)[0]
    string = _hash_matrix(["1"], n_hashes=4)[0]
    boolean = _hash_matrix([True], n_hashes=4)[0]
    floating = _hash_matrix([1.0], n_hashes=4)[0]

    assert not torch.equal(integer, string)
    assert not torch.equal(boolean, integer)
    assert not torch.equal(floating, integer)


def test_hash_value_uses_one_canonical_arrow_type_within_a_batch():
    outputs = _hash_matrix([1, 1.0], n_hashes=4)

    assert torch.equal(outputs[0], outputs[1])


# --- tensorfield content behaviour ------------------------------------------------


def test_hashable_tensorfield_content_is_deterministic_across_observations():
    schema = Schema.model_validate(_structure_payload(n_hashes=4))
    values = [
        [["alice", "bob"]],
        [["alice", "carol"]],
    ]

    field = _new_tensorfield(
        values=values,
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

    field = _new_tensorfield(
        values=values,
        schema=schema,
        strata=Strata.train,
    )

    assert field.state[0, 0, 0].item() == Tokens.valued.value
    assert field.state[0, 0, 1].item() == Tokens.null.value
    assert field.state[1, 0, 1].item() == Tokens.padded.value

    assert torch.all(field.content[0, 0, 1] == 0.0)
    assert torch.all(field.content[1, 0, 1] == 0.0)


def test_hashable_tensorfield_rejects_non_scalar_values():
    schema = Schema.model_validate(_structure_payload())
    values = [
        [[[1, 2], [3, 4]]],
        [[[5, 6], [7, 8]]],
    ]

    with pytest.raises(ValueError, match="expects scalar Arrow values"):
        _new_tensorfield(
            values=values,
            schema=schema,
            strata=Strata.train,
        )


def test_hashable_rejects_arrow_struct_values():
    schema = Schema.model_validate(_structure_payload())
    values = [[[{"key": 1}, {"key": 2}]], [[{"key": 3}, {"key": 4}]]]

    with pytest.raises(TypeError, match="root/items/identifier.*does not accept Arrow type"):
        _new_tensorfield(values=values, schema=schema, strata=Strata.train)


def test_hashable_reconstruction_projects_targets_before_zeroing_input():
    schema = Schema.model_validate(_structure_payload(n_hashes=4, mask=Mask(reconstruct=True)))
    values = [
        [["a", "b"]],
        [["c", "d"]],
    ]

    field = _new_tensorfield(
        values=values,
        schema=schema,
        strata=Strata.train,
    )
    visible = _new_tensorfield(
        values=values,
        schema=Schema.model_validate(_structure_payload(n_hashes=4)),
        strata=Strata.train,
    )

    assert torch.equal(field.targets[TensorKey.state], visible.state)
    assert torch.equal(field.targets[TensorKey.content], visible.content)
    assert torch.all(field.state == Tokens.masked.value)
    assert torch.all(field.content == 0.0)


# --- injectivity probes -----------------------------------------------------------


def _hash_matrix(values, n_hashes: int, *, salt: int = 0) -> torch.Tensor:
    """Tensorize values into a `(len(values), n_hashes)` integer matrix."""
    schema = Schema.model_validate(_structure_payload(length=len(values), n_hashes=n_hashes))
    field = _new_tensorfield(
        values=[[values]],
        schema=schema,
        strata=Strata.test,
        salt=salt,
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
        table(
            [
                {"items": [{"id": "a"}, {"id": "b"}, {"id": "c"}]},
                {"items": [{"id": "d"}, {"id": "e"}, {"id": None}]},
            ]
        ),
        strata=Strata.train,
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
            rf.Hash(
                "id",
                n_hashes=n_hashes,
                n_buckets=n_buckets,
                mask=rf.Mask(reconstruct=True),
            ),
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
        table(
            [
                {"items": [{"id": "a"}, {"id": "b"}, {"id": "c"}]},
                {"items": [{"id": "d"}, {"id": "e"}, {"id": "f"}]},
            ]
        ),
        strata=Strata.train,
    )

    predictions = model(inputs, strata=Strata.train)
    prediction = next(p for p in predictions if p.address == address)

    assert prediction.payload[TensorKey.state].shape[-1] == len(Tokens)
    assert prediction.payload[TensorKey.content].shape[-1] == n_hashes * n_buckets

    output = model.training_step(inputs, batch_idx=0)
    assert torch.isfinite(output["loss"])


def test_hashable_content_target_matches_deterministic_recomputation():
    schema = Schema.model_validate(_structure_payload(n_hashes=5, length=2, mask=Mask(reconstruct=True)))
    values = [[["alice", "bob"]], [["carol", "dave"]]]

    field = _new_tensorfield(
        values=values,
        schema=schema,
        strata=Strata.train,
    )

    fresh = _new_tensorfield(
        values=values,
        schema=Schema.model_validate(_structure_payload(n_hashes=5, length=2)),
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
    field = _new_tensorfield(
        values=[[["alice", "bob"]]],
        schema=schema,
        strata=Strata.test,
    )

    output = embedder.embed(field).payload
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
    model = _hashable_model(mask=True)
    items = model.nodes["record/items/id"]
    owner = model.nodes["record/owner"]

    assert items.embedder.state_embeddings is not owner.embedder.state_embeddings
    assert items.decoder.content_linear is not owner.decoder.content_linear
    assert items.decoder.state_linear is not owner.decoder.state_linear


def test_hashable_gives_same_input_value_identical_raw_embeddings_across_fields():
    model = _hashable_model()
    inputs = model.encode(
        table(
            [
                {"items": [{"id": "alice"}, {"id": "bob"}], "owner": "alice"},
                {"items": [{"id": "carol"}, {"id": "dave"}], "owner": "carol"},
            ]
        ),
        strata=Strata.train,
    )

    items_projection = model.nodes["record/items/id"].embedder.embed(inputs["record/items/id"]).payload
    owner_projection = model.nodes["record/owner"].embedder.embed(inputs["record/owner"]).payload

    assert torch.allclose(items_projection[0, 0, 0], owner_projection[0, 0])
    assert torch.allclose(items_projection[1, 0, 0], owner_projection[1, 0])


# --- batch salt rotation ----------------------------------------------------------


def test_hash_matrix_rotates_with_salt():
    baseline = _hash_matrix(["alice", "bob"], n_hashes=4)
    rotated = _hash_matrix(["alice", "bob"], n_hashes=4, salt=1)
    assert not torch.equal(baseline, rotated)
    assert torch.equal(baseline, _hash_matrix(["alice", "bob"], n_hashes=4, salt=0))


def test_tensorfield_content_reflects_explicit_salt():
    schema = Schema.model_validate(_structure_payload(length=2, n_hashes=4))
    values = [[["alice", "bob"]]]

    field_a = _new_tensorfield(values=values, schema=schema, strata=Strata.train, salt=0)
    field_b = _new_tensorfield(values=values, schema=schema, strata=Strata.train, salt=1)
    field_c = _new_tensorfield(values=values, schema=schema, strata=Strata.train, salt=0)

    assert not torch.equal(field_a.content, field_b.content)
    assert torch.equal(field_a.content, field_c.content)


def test_equal_values_stay_matched_within_each_rotating_batch(monkeypatch: pytest.MonkeyPatch):
    from relflow.data import iterables

    model = _hashable_model()
    records = table([{"items": [{"id": "alice"}, {"id": "bob"}], "owner": "alice"}])
    salts = iter((1, 2, 3))
    monkeypatch.setattr(iterables.random, "getrandbits", lambda bits: next(salts))

    projections_by_batch: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(3):
        inputs = model.encode(records, strata=Strata.train)
        items_projection = model.nodes["record/items/id"].embedder.embed(inputs["record/items/id"]).payload
        owner_projection = model.nodes["record/owner"].embedder.embed(inputs["record/owner"]).payload
        projections_by_batch.append((items_projection[0, 0, 0].detach(), owner_projection[0, 0].detach()))

    for items_alice, owner_alice in projections_by_batch:
        assert torch.allclose(items_alice, owner_alice)

    for later in projections_by_batch[1:]:
        assert not torch.allclose(projections_by_batch[0][0], later[0])


@pytest.mark.parametrize("strata", [Strata.test, Strata.predict])
def test_inference_uses_stable_unsalted_hashes(strata: Strata, monkeypatch: pytest.MonkeyPatch):
    from relflow.data import iterables

    model = _hashable_model()
    records = table([{"items": [{"id": "alice"}, {"id": "bob"}], "owner": "alice"}])

    def unexpected_random_salt(bits: int) -> int:
        raise AssertionError("inference must not generate a random salt")

    monkeypatch.setattr(iterables.random, "getrandbits", unexpected_random_salt)

    first = model.encode(records, strata=strata)
    second = model.encode(records, strata=strata)

    assert torch.equal(first["record/items/id"].content, second["record/items/id"].content)
    assert torch.equal(first["record/owner"].content, second["record/owner"].content)


def test_hashable_does_not_register_a_salt_callback():
    from relflow.tensorfields.extensions.hashable import hashable

    assert hashable.callback_factories == []

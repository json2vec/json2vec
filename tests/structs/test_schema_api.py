import pytest

import relflow as rf
from relflow.structs.enums import TensorKey


def test_model_constructor_supports_direct_binding_and_opt_in_queries():
    model = rf.Model(
        rf.Category(
            "job_code",
            query='source["job code"]',
            description="job code",
            size=128,
            source="openml",
        ),
        rf.Number("amount"),
        rf.Category("label", mask=True, embed=False, metric="roc_auc", topk=[2, 3]),
        d_model=32,
        n_layers=2,
        n_heads=4,
        batch_size=8,
    )
    params = model.schema

    assert model.batch_size == 8
    assert params.d_model == 32
    assert params.fields.name == "record"
    assert params.fields.length == 1
    assert params.fields.n_layers == 2
    assert params.fields.n_heads == 4

    job = params.requests["record/job_code"]
    assert job.name == "job_code"
    assert job.description == "job code"
    assert job.query == 'source["job code"]'
    assert job.size == 128
    assert job.source == "openml"

    amount = params.requests["record/amount"]
    assert amount.query is None
    assert amount.active is True
    assert amount.embed is False

    label = params.requests["record/label"]
    assert label.mask == (rf.Mask(skip=True, dropout=False, reconstruct=True),)
    assert label.embed is False
    assert label.metric == "roc_auc"
    assert label.topk == [2, 3]
    assert params.reconstruct == ["record/label"]


def test_model_from_tree_remains_a_compatibility_wrapper():
    direct = rf.Model(rf.Number("amount"), d_model=16, n_layers=1, n_heads=4)
    compatible = rf.Model.from_tree(rf.Number("amount"), d_model=16, n_layers=1, n_heads=4)

    assert compatible.schema.model_dump(mode="python") == direct.schema.model_dump(mode="python")


def test_model_constructor_rejects_duplicate_sources():
    with pytest.raises(ValueError, match="duplicate schema source field"):
        rf.Model(
            rf.Number("amount"),
            rf.Number("amount"),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )


def test_model_constructor_accepts_branch_nodes_with_optional_leaf_queries():
    model = rf.Model(
        rf.Branch(
            rf.Number("amount"),
            rf.Category(
                "merchant_code",
                query='source["merchant code"]',
                description="merchant code",
                size=32,
            ),
            name="transactions",
            length=4,
        ),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.schema

    assert "record/transactions" in params.branches

    amount = params.requests["record/transactions/amount"]
    assert amount.query is None
    assert params.shapes["record/transactions/amount"] == (1, 4)

    merchant = params.requests["record/transactions/merchant_code"]
    assert merchant.name == "merchant_code"
    assert merchant.description == "merchant code"
    assert merchant.query == 'source["merchant code"]'
    assert merchant.size == 32


def test_branch_mask_shorthand_normalizes_and_exports_public_api():
    policy = rf.Mask(query="recent", rate=0.5)
    branch = rf.Branch(
        rf.Number("amount"),
        name="transactions",
        length=4,
        mask=policy,
    )
    model = rf.Model(branch, d_model=16, n_layers=1, n_heads=4)

    bound = model.schema.branches["record/transactions"]
    assert bound.mask == (policy,)


def test_branch_mask_validation_rejects_invalid_bound_configs():
    with pytest.raises(ValueError, match="removed node field"):
        rf.Schema.model_validate(
            {
                "d_model": 16,
                "fields": {
                    "name": "record",
                    "type": "branch",
                    "length": 1,
                    "masks": [{"count": 1}],
                    "fields": [{"name": "amount", "type": "number"}],
                },
            }
        )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        rf.Model(
            rf.Branch(
                rf.Number("amount"),
                name="transactions",
                length=2,
                mask=rf.Mask.model_validate({"offset": 2}),
            ),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )

    with pytest.raises(TypeError, match="entries must be Mask"):
        rf.Model(
            rf.Branch(
                rf.Number("amount"),
                name="transactions",
                length=2,
                mask=[0.5],
            ),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )


def test_model_constructor_accepts_root_branch_options():
    model = rf.Model(
        rf.Number("amount"),
        d_model=16,
        n_layers=2,
        n_heads=4,
        name="events",
        description="event records",
        embed=True,
        attention="none",
        n_linear=2,
        dropout=0.2,
    )
    params = model.schema

    assert params.fields.name == "events"
    assert params.fields.description == "event records"
    assert params.fields.embed is True
    assert params.fields.attention == "none"
    assert params.fields.length == 1
    assert params.fields.n_linear == 2
    assert params.fields.dropout == 0.2
    assert not hasattr(params.fields, "p_mask")
    assert params.embed == ["events"]
    assert params.shapes["events/amount"] == (1,)


def test_model_constructor_rejects_root_length_argument():
    with pytest.raises(TypeError, match="tree field 'length'"):
        rf.Model(
            rf.Number("amount"),
            d_model=16,
            n_layers=2,
            n_heads=4,
            length=3,
        )


def test_model_constructor_accepts_root_mask():
    model = rf.Model(
        rf.Number("amount"),
        d_model=16,
        n_layers=2,
        n_heads=4,
        mask=True,
    )

    assert model.schema.fields.mask == (rf.Mask(skip=True, dropout=False, reconstruct=True),)
    assert model.schema.reconstruct == ["record/amount"]


def test_model_select_returns_nodes_and_update_refreshes_cached_role_views():
    model = rf.Model(
        rf.Number("amount"),
        rf.Category("label", mask=True, embed=False),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.schema

    numeric = rf.where("type") == "number"
    assert model.select(numeric) == model.select(rf.where("type") == "number")

    model.update(numeric, weight=2.0)
    assert params.requests["record/amount"].weight == 2.0

    model.update(rf.where("name") == "amount", benchmark="schema_api", allow_extra=True)
    assert model.select(rf.where("benchmark") == "schema_api") == [params.requests["record/amount"]]

    reconstruct = rf.where("reconstruct")
    assert model.select(reconstruct, include_root=False) == [params.requests["record/label"]]

    model.update(rf.where("name") == "amount", mask=True)
    assert params.requests["record/amount"].mask == (rf.Mask(skip=True, dropout=False, reconstruct=True),)
    assert model.select(reconstruct, include_root=False) == [
        params.requests["record/amount"],
        params.requests["record/label"],
    ]

    model.update(rf.where("name") == "amount", mask=False)
    assert params.requests["record/amount"].mask == ()
    assert model.select(reconstruct, include_root=False) == [params.requests["record/label"]]


def test_schema_helper_classmethods_back_public_dsl():
    predicate = rf.NodePredicate.from_callable("amount-name", lambda node: node.name == "amount")
    attribute = rf.NodeAttribute.named("name")

    assert rf.predicate("amount-name", lambda node: node.name == "amount").key == predicate.key
    assert rf.where("name") == attribute


def test_schema_select_returns_nodes_and_accepts_boolean_predicates():
    model = rf.Model(
        rf.Number("amount"),
        rf.Number("memo", active=False),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.schema

    active = params.select(rf.where("active"), include_root=False)
    inactive = params.select(~rf.where("active"), include_root=False)

    assert isinstance(active, list)
    assert active == [params.requests["record/amount"]]
    assert inactive == [params.requests["record/memo"]]

    model.update(rf.where("name") == "memo", mask=True)
    assert params.requests["record/memo"].mask == (rf.Mask(skip=True, dropout=False, reconstruct=True),)
    assert params.select(rf.where("reconstruct"), include_root=False) == []

    with pytest.raises(TypeError, match="Python 'not where"):
        not rf.where("active")


def test_model_update_can_deactivate_and_reactivate_leaf_nodes():
    model = rf.Model(
        rf.Number("amount"),
        rf.Number("memo", active=False, mask=0.5, embed=True),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.schema

    assert "record/memo" in params.requests
    assert "record/memo" not in params.active_requests
    assert "record/memo" in model.nodes
    assert params.embed == []
    inactive = model.select(lambda node: getattr(node, "active", True) is False)
    assert inactive[0].address == "record/memo"

    model.update(rf.where("name") == "memo", active=True)

    assert "record/memo" in params.requests
    assert "record/memo" in params.active_requests
    assert "record/memo" in model.nodes
    assert params.embed == ["record/memo"]


def test_model_update_applies_validated_values_before_rebuilding_modules():
    model = rf.Model(
        rf.Category("label", size=8, topk=[2]),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    address = "record/label"
    before = model.nodes[address]

    model.update(rf.where("name") == "label", size=16, topk=[3, 2])

    request = model.schema.requests[address]
    assert request.size == 16
    assert request.topk == [2, 3]
    assert model.nodes[address] is not before
    assert model.nodes[address].embedder.size == 16
    assert model.nodes[address].embedder.embeddings[TensorKey.content.name].num_embeddings == 16


def test_model_update_uses_current_schema_when_selection_cache_is_stale():
    model = rf.Model(
        rf.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    predicate = rf.where("name") == "amount"

    assert model.select(predicate) == [model.schema.requests["record/amount"]]

    request = model.schema.requests["record/amount"]
    request.name = "renamed"

    model.update(predicate, weight=2.0)

    assert request.weight == 1.0
    assert "record/amount" not in model.schema.requests
    assert "record/renamed" in model.schema.requests
    assert "record/amount" not in model.nodes
    assert "record/renamed" in model.nodes


def test_model_extend_appends_fields_under_one_selected_array_and_rebuilds_modules():
    model = rf.Model(
        rf.Branch(
            rf.Number("amount"),
            name="transactions",
            length=4,
        ),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.schema

    model.extend(rf.where("address") == "record/transactions", rf.Number("risk_score"))

    assert "record/transactions/risk_score" in params.requests
    assert "record/transactions/risk_score" in model.nodes


def test_model_extend_appends_category_field_and_preserves_existing_vocabulary():
    model = rf.Model(
        rf.Category("label", size=10),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )

    label_vocab = model.nodes["record/label"].embedder.vocab
    label_vocab.extend(["alpha", "beta"])

    model.extend(rf.where("name") == "record", rf.Category("caretaker", size=10))

    assert "record/caretaker" in model.schema.requests
    assert "record/caretaker" in model.nodes
    assert model.nodes["record/label"].embedder.vocab.snapshot() == ["alpha", "beta"]
    assert model.nodes["record/caretaker"].embedder.vocab.snapshot() == []


def test_model_extend_defaults_to_root_when_only_one_array_matches():
    model = rf.Model(
        rf.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )

    model.extend(rf.Number("risk_score"))

    assert "record/risk_score" in model.schema.requests
    assert "record/risk_score" in model.nodes


def test_model_delete_removes_nodes_permanently_and_rebuilds_modules():
    model = rf.Model(
        rf.Number("amount"),
        rf.Number("risk_score"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.schema

    model.delete(rf.where("name") == "risk_score")

    assert "record/risk_score" not in params.requests
    assert "record/risk_score" not in model.nodes


def test_model_delete_rejects_removing_the_final_request():
    model = rf.Model(
        rf.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )

    with pytest.raises(ValueError, match="every request"):
        model.delete(rf.where("name") == "amount")


def test_model_reset_reinitializes_runtime_node_without_changing_schema():
    model = rf.Model(
        rf.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    before = model.nodes["record/amount"]

    model.reset(rf.where("name") == "amount")

    assert model.nodes["record/amount"] is not before
    assert "record/amount" in model.schema.requests


def test_model_override_temporarily_updates_schema_and_rebuilds_modules():
    model = rf.Model(
        rf.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    before = model.nodes["record/amount"]

    with model.override(rf.where("name") == "amount", active=False):
        assert "record/amount" not in model.schema.active_requests
        assert model.nodes["record/amount"] is not before

    assert "record/amount" in model.schema.active_requests


def test_model_override_mask_restores_original_policy():
    model = rf.Model(
        rf.Number(name="amount", mask=0.25),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    request = model.schema.requests["record/amount"]

    with model.override(rf.where("name") == "amount", mask=True):
        assert request.mask == (rf.Mask(skip=True, dropout=False, reconstruct=True),)

    assert request.mask == (rf.Mask(rate=0.25),)


def test_model_mutations_are_blocked_inside_training_loop_lock():
    model = rf.Model(
        rf.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    lock = rf.MutationLockCallback()

    lock.on_train_start(trainer=None, pl_module=model)
    try:
        with pytest.raises(RuntimeError, match="active loop: train"):
            model.update(rf.where("name") == "amount", weight=2.0)
    finally:
        lock.on_train_end(trainer=None, pl_module=model)

    model.update(rf.where("name") == "amount", weight=2.0)
    assert model.schema.requests["record/amount"].weight == 2.0

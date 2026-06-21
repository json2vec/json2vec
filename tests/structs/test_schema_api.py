import pytest

import json2vec as jv
from json2vec.structs.enums import TensorKey


def test_model_from_tree_builds_record_branch_and_infers_queries():
    model = jv.Model.from_tree(
        jv.Category(
            "job_code",
            query='[*]."job code"',
            description="job code",
            size=128,
            source="openml",
        ),
        jv.Number("amount"),
        jv.Category("label", target=True, embed=False, metric="roc_auc", topk=[2, 3]),
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
    assert job.query == '[*]."job code"'
    assert job.size == 128
    assert job.source == "openml"

    amount = params.requests["record/amount"]
    # Inferred queries are request-level expressions. The encoder prepends the
    # outer batch selector at search time, so this intentionally is not
    # `[*][*].amount`.
    assert amount.query == "[*].amount"
    assert amount.active is True
    assert amount.embed is False

    label = params.requests["record/label"]
    assert label.p_prune == 1.0
    assert label.embed is False
    assert label.metric == "roc_auc"
    assert label.topk == [2, 3]
    assert params.target == ["record/label"]


def test_model_from_tree_rejects_duplicate_sources():
    with pytest.raises(ValueError, match="duplicate schema source field"):
        jv.Model.from_tree(
            jv.Number("amount"),
            jv.Number("amount"),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )


def test_model_from_tree_accepts_branch_nodes_and_infers_nested_queries():
    model = jv.Model.from_tree(
        jv.Branch(
            jv.Number("amount"),
            jv.Category(
                "merchant_code",
                query='[*].transactions[*]."merchant code"',
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
    # Nested defaults follow the same convention: one leading selector here,
    # with the outer batch selector added by the encoder.
    assert amount.query == "[*].transactions[*].amount"
    assert params.shapes["record/transactions/amount"] == (1, 4)

    merchant = params.requests["record/transactions/merchant_code"]
    assert merchant.name == "merchant_code"
    assert merchant.description == "merchant code"
    assert merchant.query == '[*].transactions[*]."merchant code"'
    assert merchant.size == 32


def test_branch_mask_shorthand_normalizes_and_exports_public_api():
    policy = jv.Mask(name="recent", count=1, window=2)
    branch = jv.Branch(
        jv.Number("amount"),
        name="transactions",
        length=4,
        mask=policy,
    )
    model = jv.Model.from_tree(branch, d_model=16, n_layers=1, n_heads=4)

    bound = model.schema.branches["record/transactions"]
    assert bound.masks == [policy]
    assert jv.MASK_LITERAL == "<MASK>"
    assert jv.MaskLiteral is not None


def test_branch_mask_validation_rejects_invalid_bound_configs():
    with pytest.raises(ValueError, match="root branch"):
        jv.Schema.model_validate(
            {
                "d_model": 16,
                "fields": {
                    "name": "record",
                    "type": "branch",
                    "length": 1,
                    "masks": [{"count": 1}],
                    "fields": [{"name": "amount", "type": "number", "query": "[*].amount"}],
                },
            }
        )

    with pytest.raises(ValueError, match="offset must be less than length=2"):
        jv.Model.from_tree(
            jv.Branch(
                jv.Number("amount"),
                name="transactions",
                length=2,
                masks=[jv.Mask(count=1, offset=2)],
            ),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )

    with pytest.raises(ValueError, match="duplicate mask name"):
        jv.Model.from_tree(
            jv.Branch(
                jv.Number("amount"),
                name="transactions",
                length=2,
                masks=[jv.Mask(name="same", count=1), jv.Mask(name="same", rate=0.5)],
            ),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )

    with pytest.raises(ValueError, match="excludes every active descendant leaf"):
        jv.Model.from_tree(
            jv.Branch(
                jv.Number("amount"),
                name="transactions",
                length=2,
                masks=[jv.Mask(count=1, exclude=jv.where("type") == "number")],
            ),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )


def test_model_from_tree_accepts_root_branch_options():
    model = jv.Model.from_tree(
        jv.Number("amount"),
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


def test_model_from_tree_rejects_root_length_argument():
    with pytest.raises(TypeError, match="tree field 'length'"):
        jv.Model.from_tree(
            jv.Number("amount"),
            d_model=16,
            n_layers=2,
            n_heads=4,
            length=3,
        )


def test_model_from_tree_rejects_root_branch_mask_options():
    with pytest.raises(TypeError, match="tree field 'p_mask'"):
        jv.Model.from_tree(
            jv.Number("amount"),
            d_model=16,
            n_layers=2,
            n_heads=4,
            p_mask=0.1,
        )


def test_model_select_returns_nodes_and_update_refreshes_cached_role_views():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        jv.Category("label", target=True, embed=False),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.schema

    numeric = jv.where("type") == "number"
    assert model.select(numeric) == model.select(jv.where("type") == "number")

    model.update(numeric, weight=2.0)
    assert params.requests["record/amount"].weight == 2.0

    model.update(jv.where("name") == "amount", benchmark="schema_api", allow_extra=True)
    assert model.select(jv.where("benchmark") == "schema_api") == [params.requests["record/amount"]]

    target = jv.where("target")
    assert model.select(target, include_root=False) == [params.requests["record/label"]]

    model.update(jv.where("name") == "amount", target=True)
    assert params.requests["record/amount"].p_prune == 1.0
    assert model.select(target, include_root=False) == [
        params.requests["record/amount"],
        params.requests["record/label"],
    ]

    model.update(jv.where("name") == "amount", target=False)
    assert params.requests["record/amount"].p_prune == 0.0
    assert model.select(target, include_root=False) == [params.requests["record/label"]]

    model.update(jv.where("name") == "amount", p_prune=0.25)
    assert params.requests["record/amount"].p_prune == 0.25
    assert model.select(target, include_root=False) == [params.requests["record/label"]]


def test_schema_helper_classmethods_back_public_dsl():
    predicate = jv.NodePredicate.from_callable("amount-name", lambda node: node.name == "amount")
    attribute = jv.NodeAttribute.named("name")

    assert jv.predicate("amount-name", lambda node: node.name == "amount").key == predicate.key
    assert jv.where("name") == attribute
    assert jv.Schema.update_values({"target": True}) == {"target": True}


def test_schema_select_returns_nodes_and_accepts_boolean_predicates():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        jv.Number("memo", active=False),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.schema

    active = params.select(jv.where("active"), include_root=False)
    inactive = params.select(~jv.where("active"), include_root=False)

    assert isinstance(active, list)
    assert active == [params.requests["record/amount"]]
    assert inactive == [params.requests["record/memo"]]

    model.update(jv.where("name") == "memo", target=True)
    assert params.requests["record/memo"].p_prune == 1.0
    assert params.select(jv.where("target"), include_root=False) == []

    with pytest.raises(TypeError, match="Python 'not where"):
        not jv.where("active")


def test_model_update_can_deactivate_and_reactivate_leaf_nodes():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        jv.Number("memo", active=False, p_mask=0.5, embed=True),
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

    model.update(jv.where("name") == "memo", active=True)

    assert "record/memo" in params.requests
    assert "record/memo" in params.active_requests
    assert "record/memo" in model.nodes
    assert params.embed == ["record/memo"]


def test_model_update_applies_validated_values_before_rebuilding_modules():
    model = jv.Model.from_tree(
        jv.Category("label", size=8, topk=[2]),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    address = "record/label"
    before = model.nodes[address]

    model.update(jv.where("name") == "label", size=16, topk=[3, 2])

    request = model.schema.requests[address]
    assert request.size == 16
    assert request.topk == [2, 3]
    assert model.nodes[address] is not before
    assert model.nodes[address].embedder.size == 16
    assert model.nodes[address].embedder.embeddings[TensorKey.content.name].num_embeddings == 16


def test_model_update_uses_current_schema_when_selection_cache_is_stale():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    predicate = jv.where("name") == "amount"

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
    model = jv.Model.from_tree(
        jv.Branch(
            jv.Number("amount"),
            name="transactions",
            length=4,
        ),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.schema

    model.extend(jv.where("address") == "record/transactions", jv.Number("risk_score"))

    assert "record/transactions/risk_score" in params.requests
    assert "record/transactions/risk_score" in model.nodes
    assert params.requests["record/transactions/risk_score"].query == "[*].transactions[*].risk_score"


def test_model_extend_appends_category_field_and_preserves_existing_vocabulary():
    model = jv.Model.from_tree(
        jv.Category("label", size=10),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )

    label_vocab = model.nodes["record/label"].embedder.vocab
    label_vocab.extend(["alpha", "beta"])

    model.extend(jv.where("name") == "record", jv.Category("caretaker", size=10))

    assert "record/caretaker" in model.schema.requests
    assert "record/caretaker" in model.nodes
    assert model.schema.requests["record/caretaker"].query == "[*].caretaker"
    assert model.nodes["record/label"].embedder.vocab.snapshot() == ["alpha", "beta"]
    assert model.nodes["record/caretaker"].embedder.vocab.snapshot() == []


def test_model_extend_defaults_to_root_when_only_one_array_matches():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )

    model.extend(jv.Number("risk_score"))

    assert "record/risk_score" in model.schema.requests
    assert "record/risk_score" in model.nodes


def test_model_delete_removes_nodes_permanently_and_rebuilds_modules():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        jv.Number("risk_score"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.schema

    model.delete(jv.where("name") == "risk_score")

    assert "record/risk_score" not in params.requests
    assert "record/risk_score" not in model.nodes


def test_model_delete_rejects_removing_the_final_request():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )

    with pytest.raises(ValueError, match="every request"):
        model.delete(jv.where("name") == "amount")


def test_model_reset_reinitializes_runtime_node_without_changing_schema():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    before = model.nodes["record/amount"]

    model.reset(jv.where("name") == "amount")

    assert model.nodes["record/amount"] is not before
    assert "record/amount" in model.schema.requests


def test_model_override_temporarily_updates_schema_and_rebuilds_modules():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    before = model.nodes["record/amount"]

    with model.override(jv.where("name") == "amount", active=False):
        assert "record/amount" not in model.schema.active_requests
        assert model.nodes["record/amount"] is not before

    assert "record/amount" in model.schema.active_requests


def test_model_override_target_restores_original_prune_rate():
    model = jv.Model.from_tree(
        jv.Number(name="amount", p_prune=0.25),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    request = model.schema.requests["record/amount"]

    with model.override(jv.where("name") == "amount", target=True):
        assert request.p_prune == 1.0
        assert request.target is True

    assert request.p_prune == 0.25
    assert request.target is False


def test_model_mutations_are_blocked_inside_training_loop_lock():
    model = jv.Model.from_tree(
        jv.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    lock = jv.MutationLockCallback()

    lock.on_train_start(trainer=None, pl_module=model)
    try:
        with pytest.raises(RuntimeError, match="active loop: train"):
            model.update(jv.where("name") == "amount", weight=2.0)
    finally:
        lock.on_train_end(trainer=None, pl_module=model)

    model.update(jv.where("name") == "amount", weight=2.0)
    assert model.schema.requests["record/amount"].weight == 2.0

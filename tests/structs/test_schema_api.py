import pytest

import json2vec as j2v
from json2vec.structs.enums import TensorKey


def test_model_from_schema_builds_record_array_and_infers_queries():
    model = j2v.Model.from_schema(
        j2v.Category(
            "job_code",
            query='[*]."job code"',
            description="job code",
            max_vocab_size=128,
            source="openml",
        ),
        j2v.Number("amount"),
        j2v.Category("label", target=True, embed=False, metric="roc_auc", topk=[2, 3]),
        d_model=32,
        n_layers=2,
        n_heads=4,
        batch_size=8,
    )
    params = model.hyperparameters

    assert model.batch_size == 8
    assert params.d_model == 32
    assert params.fields.name == "record"
    assert params.fields.max_length == 1
    assert params.fields.n_layers == 2
    assert params.fields.n_heads == 4

    job = params.requests["record/job_code"]
    assert job.name == "job_code"
    assert job.description == "job code"
    assert job.query == '[*]."job code"'
    assert job.max_vocab_size == 128
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


def test_model_from_schema_rejects_duplicate_sources():
    with pytest.raises(ValueError, match="duplicate schema source field"):
        j2v.Model.from_schema(
            j2v.Number("amount"),
            j2v.Number("amount"),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )


def test_model_from_schema_accepts_array_nodes_and_infers_nested_queries():
    model = j2v.Model.from_schema(
        j2v.Array(
            j2v.Number("amount"),
            j2v.Category(
                "merchant_code",
                query='[*].transactions[*]."merchant code"',
                description="merchant code",
                max_vocab_size=32,
            ),
            name="transactions",
            max_length=4,
        ),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.hyperparameters

    assert "record/transactions" in params.arrays

    amount = params.requests["record/transactions/amount"]
    # Nested defaults follow the same convention: one leading selector here,
    # with the outer batch selector added by the encoder.
    assert amount.query == "[*].transactions[*].amount"
    assert params.shapes["record/transactions/amount"] == (1, 4)

    merchant = params.requests["record/transactions/merchant_code"]
    assert merchant.name == "merchant_code"
    assert merchant.description == "merchant code"
    assert merchant.query == '[*].transactions[*]."merchant code"'
    assert merchant.max_vocab_size == 32


def test_model_from_schema_accepts_root_array_options():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        d_model=16,
        n_layers=2,
        n_heads=4,
        root="events",
        description="event records",
        embed=True,
        attention="none",
        max_length=3,
        n_outputs=2,
        n_linear=2,
        dropout=0.2,
        p_mask=0.1,
    )
    params = model.hyperparameters

    assert params.fields.name == "events"
    assert params.fields.description == "event records"
    assert params.fields.embed is True
    assert params.fields.attention == "none"
    assert params.fields.max_length == 3
    assert params.fields.n_outputs == 2
    assert params.fields.n_linear == 2
    assert params.fields.dropout == 0.2
    assert params.fields.p_mask == 0.1
    assert params.embed == ["events"]
    assert params.shapes["events/amount"] == (3,)


def test_model_select_returns_nodes_and_update_refreshes_cached_role_views():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        j2v.Category("label", target=True, embed=False),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.hyperparameters

    numeric = j2v.where("type") == "number"
    assert model.select(numeric) == model.select(j2v.where("type") == "number")

    model.update(numeric, weight=2.0)
    assert params.requests["record/amount"].weight == 2.0

    model.update(j2v.where("name") == "amount", benchmark="schema_api", allow_extra=True)
    assert model.select(j2v.where("benchmark") == "schema_api") == [params.requests["record/amount"]]

    target = j2v.where("target")
    assert model.select(target, include_root=False) == [params.requests["record/label"]]

    model.update(j2v.where("name") == "amount", target=True)
    assert params.requests["record/amount"].p_prune == 1.0
    assert model.select(target, include_root=False) == [
        params.requests["record/amount"],
        params.requests["record/label"],
    ]

    model.update(j2v.where("name") == "amount", target=False)
    assert params.requests["record/amount"].p_prune is None
    assert model.select(target, include_root=False) == [params.requests["record/label"]]

    model.update(j2v.where("name") == "amount", p_prune=0.25)
    assert params.requests["record/amount"].p_prune == 0.25
    assert model.select(target, include_root=False) == [params.requests["record/label"]]


def test_schema_helper_classmethods_back_public_dsl():
    predicate = j2v.NodePredicate.from_callable("amount-name", lambda node: node.name == "amount")
    attribute = j2v.NodeAttribute.named("name")

    assert j2v.predicate("amount-name", lambda node: node.name == "amount").key == predicate.key
    assert j2v.where("name") == attribute
    assert j2v.Hyperparameters.update_values({"target": True}) == {"p_prune": 1.0}


def test_hyperparameters_select_returns_nodes_and_accepts_boolean_predicates():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        j2v.Number("memo", active=False),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.hyperparameters

    active = params.select(j2v.where("active"), include_root=False)
    inactive = params.select(~j2v.where("active"), include_root=False)

    assert isinstance(active, list)
    assert active == [params.requests["record/amount"]]
    assert inactive == [params.requests["record/memo"]]

    model.update(j2v.where("name") == "memo", target=True)
    assert params.requests["record/memo"].p_prune == 1.0
    assert params.select(j2v.where("target"), include_root=False) == []

    with pytest.raises(TypeError, match="Python 'not where"):
        not j2v.where("active")


def test_model_update_can_deactivate_and_reactivate_leaf_nodes():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        j2v.Number("memo", active=False, p_mask=0.5, embed=True),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.hyperparameters

    assert "record/memo" in params.requests
    assert "record/memo" not in params.active_requests
    assert "record/memo" in model.nodes
    assert params.embed == []
    inactive = model.select(lambda node: getattr(node, "active", True) is False)
    assert inactive[0].address == "record/memo"

    model.update(j2v.where("name") == "memo", active=True)

    assert "record/memo" in params.requests
    assert "record/memo" in params.active_requests
    assert "record/memo" in model.nodes
    assert params.embed == ["record/memo"]


def test_model_update_applies_validated_values_before_rebuilding_modules():
    model = j2v.Model.from_schema(
        j2v.Category("label", max_vocab_size=8, topk=[2]),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    address = "record/label"
    before = model.nodes[address]

    model.update(j2v.where("name") == "label", max_vocab_size=16, topk=[3, 2])

    request = model.hyperparameters.requests[address]
    assert request.max_vocab_size == 16
    assert request.topk == [2, 3]
    assert model.nodes[address] is not before
    assert model.nodes[address].embedder.max_vocab_size == 16
    assert model.nodes[address].embedder.embeddings[TensorKey.content.name].num_embeddings == 17


def test_model_update_uses_current_schema_when_selection_cache_is_stale():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    predicate = j2v.where("name") == "amount"

    assert model.select(predicate) == [model.hyperparameters.requests["record/amount"]]

    request = model.hyperparameters.requests["record/amount"]
    request.name = "renamed"

    model.update(predicate, weight=2.0)

    assert request.weight == 1.0
    assert "record/amount" not in model.hyperparameters.requests
    assert "record/renamed" in model.hyperparameters.requests
    assert "record/amount" not in model.nodes
    assert "record/renamed" in model.nodes


def test_model_extend_appends_fields_under_one_selected_array_and_rebuilds_modules():
    model = j2v.Model.from_schema(
        j2v.Array(
            j2v.Number("amount"),
            name="transactions",
            max_length=4,
        ),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.hyperparameters

    model.extend(j2v.where("address") == "record/transactions", j2v.Number("risk_score"))

    assert "record/transactions/risk_score" in params.requests
    assert "record/transactions/risk_score" in model.nodes
    assert params.requests["record/transactions/risk_score"].query == "[*].transactions[*].risk_score"


def test_model_extend_appends_category_field_and_preserves_existing_vocabulary():
    model = j2v.Model.from_schema(
        j2v.Category("label", max_vocab_size=10),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )

    label_vocab = model.nodes["record/label"].embedder.vocab
    label_vocab.extend(["alpha", "beta"])

    model.extend(j2v.where("name") == "record", j2v.Category("caretaker", max_vocab_size=10))

    assert "record/caretaker" in model.hyperparameters.requests
    assert "record/caretaker" in model.nodes
    assert model.hyperparameters.requests["record/caretaker"].query == "[*].caretaker"
    assert model.nodes["record/label"].embedder.vocab.snapshot() == ["alpha", "beta"]
    assert model.nodes["record/caretaker"].embedder.vocab.snapshot() == []


def test_model_extend_defaults_to_root_when_only_one_array_matches():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )

    model.extend(j2v.Number("risk_score"))

    assert "record/risk_score" in model.hyperparameters.requests
    assert "record/risk_score" in model.nodes


def test_model_delete_removes_nodes_permanently_and_rebuilds_modules():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        j2v.Number("risk_score"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.hyperparameters

    model.delete(j2v.where("name") == "risk_score")

    assert "record/risk_score" not in params.requests
    assert "record/risk_score" not in model.nodes


def test_model_delete_rejects_removing_the_final_request():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )

    with pytest.raises(ValueError, match="every request"):
        model.delete(j2v.where("name") == "amount")


def test_model_reset_reinitializes_runtime_node_without_changing_schema():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    before = model.nodes["record/amount"]

    model.reset(j2v.where("name") == "amount")

    assert model.nodes["record/amount"] is not before
    assert "record/amount" in model.hyperparameters.requests


def test_model_override_temporarily_updates_schema_and_rebuilds_modules():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    before = model.nodes["record/amount"]

    with model.override(j2v.where("name") == "amount", active=False):
        assert "record/amount" not in model.hyperparameters.active_requests
        assert model.nodes["record/amount"] is not before

    assert "record/amount" in model.hyperparameters.active_requests


def test_model_mutations_are_blocked_inside_training_loop_lock():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    lock = j2v.MutationLockCallback()

    lock.on_train_start(trainer=None, pl_module=model)
    try:
        with pytest.raises(RuntimeError, match="active loop: train"):
            model.update(j2v.where("name") == "amount", weight=2.0)
    finally:
        lock.on_train_end(trainer=None, pl_module=model)

    model.update(j2v.where("name") == "amount", weight=2.0)
    assert model.hyperparameters.requests["record/amount"].weight == 2.0

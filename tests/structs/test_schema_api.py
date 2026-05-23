import pytest

import json2vec as j2v


def test_model_from_schema_builds_record_array_and_infers_queries():
    model = j2v.Model.from_schema(
        j2v.Column("job code", j2v.Category, kwargs={"max_vocab_size": 128}, source="openml"),
        j2v.Number("amount"),
        j2v.Column("label", "category", target=True, embed=False, metric="roc_auc", topk=[2, 3]),
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
    assert amount.query == "[*].amount"
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
            j2v.Column("amount", "number"),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )


def test_model_from_schema_accepts_array_nodes_and_infers_nested_queries():
    model = j2v.Model.from_schema(
        j2v.Array(
            j2v.Number("amount"),
            j2v.Column("merchant code", j2v.Category, max_vocab_size=32),
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
    assert amount.query == "[*].transactions[*].amount"
    assert params.shapes["record/transactions/amount"] == (1, 4)

    merchant = params.requests["record/transactions/merchant_code"]
    assert merchant.name == "merchant_code"
    assert merchant.description == "merchant code"
    assert merchant.query == '[*].transactions[*]."merchant code"'
    assert merchant.max_vocab_size == 32


def test_model_selector_set_and_cached_role_views():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        j2v.Column("label", j2v.Category, target=True, embed=False),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.hyperparameters

    numeric = j2v.where("type") == "number"
    assert model.select(numeric).to_list() == model.select(j2v.where("type") == "number").to_list()

    model.set(numeric, weight=2.0)
    assert params.requests["record/amount"].weight == 2.0
    assert params.last_mutation is not None
    assert params.last_mutation.updated == 1

    model.set(j2v.where("name") == "amount", benchmark="schema_api", allow_extra=True)
    assert model.select(j2v.where("benchmark") == "schema_api").to_list() == [params.requests["record/amount"]]

    model.set(j2v.where("name") == "amount", target=True)
    assert params.requests["record/amount"].p_prune == 1.0

    model.select(j2v.where("name") == "amount").set(p_prune=0.25)
    assert params.requests["record/amount"].p_prune == 0.25

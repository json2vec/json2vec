import pytest

import json2vec as j2v


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


def test_from_spec_aliases_schema_constructor():
    params = j2v.Hyperparameters.from_spec(
        j2v.Number("amount"),
        d_model=16,
        n_layers=1,
        n_heads=4,
        embed=True,
    )
    model = j2v.Model.from_spec(
        fields=[j2v.Number("amount")],
        d_model=16,
        n_layers=1,
        n_heads=4,
        embed=True,
    )

    assert params.embed == ["record"]
    assert model.hyperparameters.embed == ["record"]


def test_model_selector_update_and_cached_role_views():
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        j2v.Category("label", target=True, embed=False),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.hyperparameters

    numeric = j2v.where("type") == "number"
    assert model.select(numeric).to_list() == model.select(j2v.where("type") == "number").to_list()

    model.update(numeric, weight=2.0)
    assert params.requests["record/amount"].weight == 2.0
    assert params.last_mutation is not None
    assert params.last_mutation.updated == 1

    model.update(j2v.where("name") == "amount", benchmark="schema_api", allow_extra=True)
    assert model.select(j2v.where("benchmark") == "schema_api").to_list() == [params.requests["record/amount"]]

    model.update(j2v.where("name") == "amount", target=True)
    assert params.requests["record/amount"].p_prune == 1.0

    model.update(j2v.where("name") == "amount", target=False)
    assert params.requests["record/amount"].p_prune is None

    model.select(j2v.where("name") == "amount").update(p_prune=0.25)
    assert params.requests["record/amount"].p_prune == 0.25

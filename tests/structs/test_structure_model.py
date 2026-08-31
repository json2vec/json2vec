import pytest

from relflow.structs.enums import Overflow
from relflow.structs.experiment import Schema
from relflow.structs.structure import Branch
from relflow.tensorfields.extensions.category import Request as Category


def _payload() -> dict:
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "description": "root branch docs",
            "dropout": 0.1,
            "length": 2,
            "fields": [
                {
                    "name": "branch",
                    "type": "branch",
                    "description": "branch docs",
                    "length": 4,
                    "fields": [
                        {
                            "name": "category_leaf",
                            "type": "category",
                            "description": "category docs",
                        }
                    ],
                }
            ],
        },
    }


def test_branch_accepts_positional_children():
    branch = Branch(
        Category(name="category_leaf"),
        name="branch",
    )

    assert branch.fields[0].name == "category_leaf"


def test_branch_rejects_positional_and_keyword_children():
    with pytest.raises(TypeError, match="both positionally and by keyword"):
        Branch(
            Category(name="category_leaf"),
            name="branch",
            fields=[],
        )


def test_branch_rejects_leaf_mask_and_target_options():
    with pytest.raises(TypeError, match="tree field 'p_mask'"):
        Branch(
            Category(name="category_leaf"),
            name="branch",
            p_mask=0.1,
        )


def test_schema_derives_branches_requests_and_shapes():
    structure = Schema.model_validate(_payload())

    assert "root" in structure.branches
    assert "root/branch" in structure.branches
    assert "root/branch/category_leaf" in structure.requests
    assert structure.branches["root"].length == 1
    assert structure.branches["root"].overflow == Overflow.error
    assert structure.shapes["root/branch/category_leaf"] == (1, 4)
    assert structure.overflows("root/branch/category_leaf") == (Overflow.error, Overflow.error, Overflow.head)


def test_branch_accepts_overflow_policy():
    branch = Branch(
        Category(name="category_leaf"),
        name="branch",
        overflow="tail",
    )

    assert branch.overflow == Overflow.tail


def test_branch_rejects_invalid_overflow_policy():
    with pytest.raises(ValueError):
        Branch(
            Category(name="category_leaf"),
            name="branch",
            overflow="middle",
        )


def test_schema_converts_leaf_instances_nested_in_branches():
    structure = Schema(
        d_model=16,
        fields={
            "name": "root",
            "type": "branch",
            "fields": [
                {
                    "name": "branch",
                    "type": "branch",
                    "fields": [
                        Category(name="category_leaf"),
                    ],
                }
            ],
        },
    )

    request = structure.requests["root/branch/category_leaf"]
    assert request.size == 1024


def test_schema_depthwise_contains_branch_levels():
    structure = Schema.model_validate(_payload())
    assert structure.depthwise == [["root"], ["root/branch"]]


def test_schema_string_representation_contains_tree_nodes():
    structure = Schema.model_validate(_payload())
    rendered = str(structure)
    assert "schema [schema]" in rendered
    root_line = next(line for line in rendered.splitlines() if "root [root]" in line)
    assert "length=" not in root_line
    assert "overflow=" not in root_line
    assert "category_leaf [category]" in rendered


def test_schema_preserves_field_and_branch_descriptions():
    structure = Schema.model_validate(_payload())
    assert structure.branches["root"].description == "root branch docs"
    assert structure.branches["root/branch"].description == "branch docs"
    assert structure.requests["root/branch/category_leaf"].description == "category docs"


def test_schema_uses_direct_branch_dropout():
    structure = Schema.model_validate(_payload())

    assert structure.branches["root"].dropout == 0.1
    assert structure.branches["root/branch"].dropout is None
    assert structure.requests["root/branch/category_leaf"].dropout is None


def test_schema_allows_missing_dropout():
    payload = _payload()
    payload["fields"].pop("dropout")
    structure = Schema.model_validate(payload)

    assert structure.branches["root"].dropout is None
    assert structure.branches["root/branch"].dropout is None


def test_schema_preserves_direct_field_dropout():
    payload = _payload()
    payload["fields"]["fields"][0]["fields"][0]["dropout"] = 0.4

    structure = Schema.model_validate(payload)

    assert structure.requests["root/branch/category_leaf"].dropout == 0.4


def test_schema_rejects_branch_mask_and_target_rates():
    payload = _payload()
    payload["fields"]["p_mask"] = 0.2
    payload["fields"]["p_prune"] = 0.1

    with pytest.raises(TypeError, match="tree field 'p_mask'"):
        Schema.model_validate(payload)


def test_schema_preserves_direct_leaf_mask_and_target_rates():
    payload = _payload()
    payload["fields"]["fields"][0]["p_mask"] = 0.3
    payload["fields"]["fields"][0]["p_prune"] = 0.4
    payload["fields"]["fields"][0]["fields"][0]["p_mask"] = 0.5
    payload["fields"]["fields"][0]["fields"][0]["p_prune"] = 0.6

    with pytest.raises(TypeError, match="tree field 'p_mask'"):
        Schema.model_validate(payload)

    payload["fields"]["fields"][0].pop("p_mask")
    payload["fields"]["fields"][0].pop("p_prune")

    structure = Schema.model_validate(payload)

    assert not hasattr(structure.branches["root"], "p_mask")
    assert not hasattr(structure.branches["root/branch"], "p_mask")
    assert structure.requests["root/branch/category_leaf"].p_mask == 0.5
    assert structure.requests["root/branch/category_leaf"].p_prune == 0.6


def test_schema_allows_missing_mask_and_target_rates():
    structure = Schema.model_validate(_payload())

    assert structure.requests["root/branch/category_leaf"].p_mask == 0.0
    assert structure.requests["root/branch/category_leaf"].p_prune == 0.0


def test_inactive_leaf_nodes_are_kept_in_tree_but_removed_from_runtime_maps():
    payload = _payload()
    payload["fields"]["fields"][0]["fields"][0]["active"] = False
    payload["fields"]["fields"][0]["fields"][0]["p_prune"] = 1.0
    payload["fields"]["fields"][0]["fields"][0]["embed"] = True

    structure = Schema.model_validate(payload)
    inactive = structure.select(lambda node: getattr(node, "name", None) == "category_leaf")[0]

    assert inactive.active is False
    assert inactive.address == "root/branch/category_leaf"
    assert "root/branch/category_leaf" in structure.requests
    assert "root/branch/category_leaf" not in structure.active_requests
    assert structure.shapes["root/branch/category_leaf"] == (1, 4)
    assert structure.target == []
    assert structure.embed == []

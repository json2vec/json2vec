from json2vec.structs.structure import Structure


def _payload() -> dict:
    return {
        "name": "demo",
        "type": "structure",
        "batch_size": 3,
        "dropout": 0.1,
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "context",
            "description": "root context docs",
            "context_size": 2,
            "n_outputs": 1,
            "fields": [
                {
                    "name": "branch",
                    "type": "context",
                    "description": "branch docs",
                    "context_size": 4,
                    "n_outputs": 1,
                    "fields": [
                        {
                            "name": "category_leaf",
                            "type": "category",
                            "description": "category docs",
                            "query": "[*].code",
                        }
                    ],
                }
            ],
        },
    }

def test_structure_derives_contexts_requests_and_shapes():
    structure = Structure.model_validate(_payload())

    assert "root" in structure.contexts
    assert "root/branch" in structure.contexts
    assert "root/branch/category_leaf" in structure.requests
    assert structure.shapes["root/branch/category_leaf"] == (2, 4)


def test_structure_depthwise_contains_context_levels():
    structure = Structure.model_validate(_payload())
    assert structure.depthwise == [["root"], ["root/branch"]]


def test_structure_string_representation_contains_tree_nodes():
    structure = Structure.model_validate(_payload())
    rendered = str(structure)
    assert "demo (structure)" in rendered
    assert "root (context)" in rendered
    assert "category_leaf (category)" in rendered


def test_structure_preserves_field_and_context_descriptions():
    structure = Structure.model_validate(_payload())
    assert structure.contexts["root"].description == "root context docs"
    assert structure.contexts["root/branch"].description == "branch docs"
    assert structure.requests["root/branch/category_leaf"].description == "category docs"

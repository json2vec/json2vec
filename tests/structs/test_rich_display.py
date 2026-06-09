from pprint import pformat

import torch
from rich.console import Console
from rich.pretty import Pretty

import json2vec as j2v
from json2vec.structs.tree import Renderable


def render_text(node: object) -> str:
    console = Console(record=True, width=120)
    console.print(node)
    return console.export_text(clear=False).rstrip("\n")


def test_renderable_owns_shared_rich_styles() -> None:
    assert Renderable.RICH_NAME_STYLE == "bold white on #1f2937"
    assert Renderable.RICH_TYPE_STYLE == "bold yellow on #3f3f46"


def test_leaf_rich_display_uses_schema_summary() -> None:
    rendered = render_text(j2v.Number("amount"))

    assert "amount [number] active" in rendered
    assert "query=" not in rendered
    assert "pooling=query weight=1 p_mask=0 p_prune=0 n_heads=4 n_linear=1" in rendered
    assert "jitter=0 n_bands=8 offset=4 objective=mae" in rendered
    assert "model_config" not in rendered
    assert "model_fields_set" not in rendered

    lines = rendered.splitlines()
    assert lines[1].startswith(" pooling=")
    assert lines[2].startswith(" jitter=")


def test_leaf_display_flags() -> None:
    target = render_text(j2v.Category("returned", target=True, max_vocab_size=2)).splitlines()[0].split()
    embedded = render_text(j2v.Number("amount", embed=True)).splitlines()[0].split()
    inactive = render_text(j2v.Category("customer_id", active=False)).splitlines()[0].split()

    assert "target" in target
    assert "embed" in embedded
    assert "inactive" in inactive
    assert "active" not in inactive


def test_names_and_type_labels_have_background_styles() -> None:
    number_html = j2v.Number("amount")._repr_html_()
    category_html = j2v.Category("sku")._repr_html_()
    array_html = j2v.Array(j2v.Number("amount"), name="items")._repr_html_()
    hyperparameters_html = j2v.Hyperparameters.from_schema(
        j2v.Number("amount"),
        j2v.Number("label", target=True),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )._repr_html_()
    model_html = j2v.Model.from_schema(
        j2v.Number("amount"),
        j2v.Number("label", target=True),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )._repr_html_()

    for html in (number_html, category_html, array_html, hyperparameters_html, model_html):
        assert "color: #808000" in html
        assert "background-color: #3f3f46" in html
        assert "background-color: #1f2937" in html


def test_leaf_display_separates_common_and_specific_attributes() -> None:
    number_lines = render_text(j2v.Number("amount", p_mask=0.15, objective="huber")).splitlines()
    category_lines = render_text(j2v.Category("sku", max_vocab_size=2048)).splitlines()

    assert "p_mask=0.15" in number_lines[1]
    assert number_lines[1].startswith(" ")
    assert "objective=huber" not in number_lines[1]
    assert "objective=huber" in number_lines[2]
    assert number_lines[2].startswith(" ")

    assert "pooling=query" in category_lines[1]
    assert category_lines[1].startswith(" ")
    assert "max_vocab_size=2048" not in category_lines[1]
    assert "max_vocab_size=2048" in category_lines[2]
    assert category_lines[2].startswith(" ")


def test_array_rich_display_renders_child_subtree() -> None:
    rendered = render_text(
        j2v.Array(
            j2v.Category("sku", max_vocab_size=2048),
            j2v.Number("quantity"),
            name="line_items",
            max_length=32,
        )
    )

    assert "line_items [array] max_length=32 overflow=head attention=mha n_layers=1 n_heads=4 n_linear=1" in rendered
    assert "embed=False" not in rendered
    assert "|-- sku [category] active" in rendered
    assert "`-- quantity [number] active" in rendered
    assert "query=" not in rendered


def test_tree_prefixes_have_bold_html_style() -> None:
    html = j2v.Array(
        j2v.Number("amount"),
        j2v.Number("quantity"),
        name="line_items",
    )._repr_html_()

    assert 'font-weight: bold">|-- </span>' in html
    assert 'font-weight: bold">`-- </span>' in html


def test_array_embed_renders_as_flag() -> None:
    rendered = render_text(
        j2v.Array(
            j2v.Number("amount"),
            name="line_items",
            max_length=32,
            embed=True,
        )
    )

    assert "line_items [array] embed max_length=32 overflow=head" in rendered
    assert "embed=True" not in rendered


def test_root_array_embed_renders_as_flag() -> None:
    rendered = render_text(
        j2v.Hyperparameters.from_schema(
            j2v.Number("amount"),
            name="record",
            d_model=8,
            n_layers=1,
            n_heads=4,
            embed=True,
        )
    )

    assert "`-- record [root] embed attention=mha" in rendered
    assert "embed=True" not in rendered


def test_nested_array_rich_display_renders_nested_tree_prefixes() -> None:
    rendered = render_text(
        j2v.Array(
            j2v.Array(
                j2v.Number("amount"),
                j2v.Category("merchant", max_vocab_size=4096),
                name="transactions",
                max_length=360,
                overflow="tail",
            ),
            j2v.Category("churned", target=True, max_vocab_size=2),
            name="customer",
        )
    )

    assert "|-- transactions [array] max_length=360 overflow=tail" in rendered
    assert "|   |-- amount [number] active" in rendered
    assert "|   `-- merchant [category] active" in rendered
    assert "`-- churned [category] active target" in rendered
    assert "query=" not in rendered


def test_common_display_surfaces_are_backed_by_rich() -> None:
    node = j2v.Number("amount")

    assert str(node) == render_text(node)

    bundle = node._repr_mimebundle_()
    assert bundle["text/plain"] == str(node)
    assert "<!DOCTYPE html>" not in bundle["text/html"]
    assert bundle["text/html"].startswith("<pre")
    assert "background: transparent" in bundle["text/html"]
    assert "amount [number]" in bundle["text/plain"]
    assert "font-weight: bold" in bundle["text/html"]
    assert "color:" in bundle["text/html"]
    assert "background-color: #1f2937" in bundle["text/html"]
    assert "background-color: #3f3f46" in bundle["text/html"]

    mime, data = node._mime_()
    assert mime == "text/html"
    assert data == node._repr_html_()


def test_hyperparameters_rich_display_uses_root_schema_tree() -> None:
    hyperparameters = j2v.Hyperparameters.from_schema(
        j2v.Number("amount"),
        j2v.Category("label", target=True, max_vocab_size=2),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    assert isinstance(hyperparameters, Renderable)
    rendered = render_text(hyperparameters)

    assert rendered == str(hyperparameters)
    assert "hyperparameters [hyperparameters] d_model=8 arrays=1 fields=2 targets=1 embeds=0" in rendered
    root_line = next(line for line in rendered.splitlines() if "`-- record [root]" in line)
    assert "max_length=" not in root_line
    assert "overflow=" not in root_line
    assert "embed=False" not in root_line
    assert "    |-- amount [number] active query=[*].amount" in rendered
    assert "    `-- label [category] active target query=[*].label" in rendered


def test_model_rich_display_uses_runtime_summary_and_schema_tree() -> None:
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        j2v.Category("label", target=True, max_vocab_size=2),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=3,
    )

    assert isinstance(model, Renderable)
    rendered = render_text(model)

    assert rendered == str(model)
    assert "Model [model] batch_size=3 d_model=8 parameters=" in rendered
    assert "arrays=1 fields=2 targets=1 embeds=0" in rendered
    root_line = next(line for line in rendered.splitlines() if "`-- record [root]" in line)
    assert "max_length=" not in root_line
    assert "overflow=" not in root_line
    assert "embed=False" not in root_line
    assert "    |-- amount [number] active query=[*].amount" in rendered
    assert "    `-- label [category] active target query=[*].label" in rendered


def test_model_select_pprint_uses_rich_node_display() -> None:
    model = j2v.Model.from_schema(
        j2v.Number("amount"),
        j2v.Category("species", target=True, max_vocab_size=4),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    selection = model.select(j2v.where("address") == "record/species")
    rendered = pformat(selection)
    console = Console(record=True, width=120)
    console.print(Pretty(selection))
    rich_rendered = console.export_text(clear=False)

    assert isinstance(selection, list)
    for output in (rendered, rich_rendered):
        assert "species [category] active target query=[*].species" in output
        assert " pooling=query weight=1 p_mask=0 p_prune=1 n_heads=4 n_linear=1" in output
        assert " max_vocab_size=4 p_unavailable=0.01 topk=[]" in output
        assert "Request(name=" not in output
        assert "Selection(" not in output


def test_tensorfield_rich_display_previews_state_tokens() -> None:
    model = j2v.Model.from_schema(
        j2v.Array(
            j2v.Category("letter", max_vocab_size=4, p_unavailable=0.0),
            name="letters",
            max_length=4,
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode(
        [{"letters": [{"letter": "A"}, {"letter": "B"}]}],
        strata=j2v.Strata.train,
    )["record/letters/letter"]

    field.hide(torch.tensor([[[False, True, False, False]]]))
    rendered = render_text(field)

    assert isinstance(field, Renderable)
    assert "TensorField [tensorfield] state=(1, 1, 4) device=cpu trainable=1" in rendered
    assert "counts V=1 N=0 P=2 M=1 O=0" in rendered
    assert "state V M P P" in rendered
    assert "targets=content, state" in rendered


def test_tensorfield_rich_display_separates_nested_array_state_tokens() -> None:
    model = j2v.Model.from_schema(
        j2v.Array(
            j2v.Array(
                j2v.Category("letter", max_vocab_size=8, p_unavailable=0.0),
                name="letters",
                max_length=3,
            ),
            name="words",
            max_length=2,
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode(
        [
            {
                "words": [
                    {"letters": [{"letter": "A"}]},
                    {"letters": [{"letter": "B"}, {"letter": "C"}]},
                ]
            }
        ],
        strata=j2v.Strata.train,
        mask=False,
    )["record/words/letters/letter"]

    rendered = render_text(field)

    assert "TensorField [tensorfield] state=(1, 1, 2, 3) device=cpu trainable=0" in rendered
    assert "state V P P\n       V V P" in rendered


def test_rich_display_does_not_replace_repr_or_mutate_serialization() -> None:
    node = j2v.Number("amount", p_mask=0.15)
    dumped = node.model_dump(mode="python")

    assert "query=<inferred>" not in repr(node)
    assert "query=" not in str(node)
    assert "name='amount'" in repr(node)

    str(node)

    assert node.model_dump(mode="python") == dumped

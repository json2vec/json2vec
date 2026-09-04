from pprint import pformat

import pyarrow as pa
import torch
from rich.console import Console
from rich.pretty import Pretty

import relflow as rf
from relflow.structs.tree import Renderable


def render_text(node: object) -> str:
    console = Console(record=True, width=120)
    console.print(node)
    return console.export_text(clear=False).rstrip("\n")


def test_renderable_owns_shared_rich_styles() -> None:
    assert Renderable.RICH_NAME_STYLE == "bold white on #1f2937"
    assert Renderable.RICH_TYPE_STYLE == "bold yellow on #3f3f46"


def test_leaf_rich_display_uses_schema_summary() -> None:
    rendered = render_text(rf.Number("amount"))

    assert "amount [number] active" in rendered
    assert "query=" not in rendered
    assert "pooling=query weight=1 n_heads=4 n_linear=1" in rendered
    assert "jitter=0 n_bands=8 offset=4 objective=mae" in rendered
    assert "model_config" not in rendered
    assert "model_fields_set" not in rendered

    lines = rendered.splitlines()
    assert lines[1].startswith(" pooling=")
    assert lines[2].startswith(" jitter=")


def test_leaf_rich_display_shows_only_explicit_query() -> None:
    direct = render_text(rf.Number("amount"))
    queried = render_text(rf.Number("amount", query="payload.amount"))

    assert "query=" not in direct
    assert "amount [number] active query=payload.amount" in queried


def test_leaf_display_flags() -> None:
    reconstruct = render_text(rf.Category("returned", mask=True, size=2)).splitlines()[0].split()
    embedded = render_text(rf.Number("amount", embed=True)).splitlines()[0].split()
    inactive = render_text(rf.Category("customer_id", active=False)).splitlines()[0].split()

    assert "reconstruct" in reconstruct
    assert "embed" in embedded
    assert "inactive" in inactive
    assert "active" not in inactive


def test_names_and_type_labels_have_background_styles() -> None:
    number_html = rf.Number("amount")._repr_html_()
    category_html = rf.Category("sku")._repr_html_()
    branch_html = rf.Branch(rf.Number("amount"), name="items")._repr_html_()
    schema_html = rf.Schema.from_tree(
        rf.Number("amount"),
        rf.Number("label", mask=True),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )._repr_html_()
    model_html = rf.Model(
        rf.Number("amount"),
        rf.Number("label", mask=True),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )._repr_html_()

    for html in (number_html, category_html, branch_html, schema_html, model_html):
        assert "color: #808000" in html
        assert "background-color: #3f3f46" in html
        assert "background-color: #1f2937" in html


def test_leaf_display_separates_common_and_specific_attributes() -> None:
    number_lines = render_text(rf.Number("amount", mask=0.15, objective="huber")).splitlines()
    category_lines = render_text(rf.Category("sku", size=2048)).splitlines()

    assert number_lines[1].startswith(" ")
    assert "objective=huber" not in number_lines[1]
    assert "objective=huber" in number_lines[2]
    assert number_lines[2].startswith(" ")

    assert "pooling=query" in category_lines[1]
    assert category_lines[1].startswith(" ")
    assert "size=2048" not in category_lines[1]
    assert "size=2048" in category_lines[2]
    assert category_lines[2].startswith(" ")


def test_branch_rich_display_renders_child_subtree() -> None:
    rendered = render_text(
        rf.Branch(
            rf.Category("sku", size=2048),
            rf.Number("quantity"),
            name="line_items",
            length=32,
        )
    )

    assert "line_items [branch] length=32 overflow=head attention=mha n_layers=1 n_heads=4 n_linear=1" in rendered
    assert "embed=False" not in rendered
    assert "|-- sku [category] active" in rendered
    assert "`-- quantity [number] active" in rendered
    assert "query=" not in rendered


def test_tree_prefixes_have_bold_html_style() -> None:
    html = rf.Branch(
        rf.Number("amount"),
        rf.Number("quantity"),
        name="line_items",
    )._repr_html_()

    assert 'font-weight: bold">|-- </span>' in html
    assert 'font-weight: bold">`-- </span>' in html


def test_branch_embed_renders_as_flag() -> None:
    rendered = render_text(
        rf.Branch(
            rf.Number("amount"),
            name="line_items",
            length=32,
            embed=True,
        )
    )

    assert "line_items [branch] embed length=32 overflow=head" in rendered
    assert "embed=True" not in rendered


def test_root_branch_embed_renders_as_flag() -> None:
    rendered = render_text(
        rf.Schema.from_tree(
            rf.Number("amount"),
            name="record",
            d_model=8,
            n_layers=1,
            n_heads=4,
            embed=True,
        )
    )

    assert "`-- record [root] embed attention=mha" in rendered
    assert "embed=True" not in rendered


def test_nested_branch_rich_display_renders_nested_tree_prefixes() -> None:
    rendered = render_text(
        rf.Branch(
            rf.Branch(
                rf.Number("amount"),
                rf.Category("merchant", size=4096),
                name="transactions",
                length=360,
                overflow="tail",
            ),
            rf.Category("churned", mask=True, size=2),
            name="customer",
        )
    )

    assert "|-- transactions [branch] length=360 overflow=tail" in rendered
    assert "|   |-- amount [number] active" in rendered
    assert "|   `-- merchant [category] active" in rendered
    assert "`-- churned [category] active reconstruct" in rendered
    assert "query=" not in rendered


def test_common_display_surfaces_are_backed_by_rich() -> None:
    node = rf.Number("amount")

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


def test_schema_rich_display_uses_root_schema_tree() -> None:
    schema = rf.Schema.from_tree(
        rf.Number("amount"),
        rf.Category("label", mask=True, size=2),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    assert isinstance(schema, Renderable)
    rendered = render_text(schema)

    assert rendered == str(schema)
    assert "schema [schema] d_model=8 branches=1 fields=2 reconstruct=1 embeds=0" in rendered
    root_line = next(line for line in rendered.splitlines() if "`-- record [root]" in line)
    assert "length=" not in root_line
    assert "overflow=" not in root_line
    assert "embed=False" not in root_line
    assert "    |-- amount [number] active" in rendered
    assert "    `-- label [category] active reconstruct" in rendered
    assert "query=" not in rendered


def test_model_rich_display_uses_runtime_summary_and_schema_tree() -> None:
    model = rf.Model(
        rf.Number("amount"),
        rf.Category("label", mask=True, size=2),
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
    assert "branches=1 fields=2 reconstruct=1 embeds=0" in rendered
    root_line = next(line for line in rendered.splitlines() if "`-- record [root]" in line)
    assert "length=" not in root_line
    assert "overflow=" not in root_line
    assert "embed=False" not in root_line
    assert "    |-- amount [number] active" in rendered
    assert "    `-- label [category] active reconstruct" in rendered
    assert "query=" not in rendered


def test_model_select_pprint_uses_rich_node_display() -> None:
    model = rf.Model(
        rf.Number("amount"),
        rf.Category("species", mask=True, size=4),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    selection = model.select(rf.where("address") == "record/species")
    rendered = pformat(selection)
    console = Console(record=True, width=120)
    console.print(Pretty(selection))
    rich_rendered = console.export_text(clear=False)

    assert isinstance(selection, list)
    for output in (rendered, rich_rendered):
        assert "species [category] active reconstruct" in output
        assert "query=" not in output
        assert " pooling=query weight=1 n_heads=4 n_linear=1" in output
        assert " size=4 p_unavailable=0.01 topk=[]" in output
        assert "Request(name=" not in output
        assert "Selection(" not in output


def test_tensorfield_rich_display_previews_state_tokens() -> None:
    model = rf.Model(
        rf.Branch(
            rf.Category("letter", size=4, p_unavailable=0.0),
            name="letters",
            length=4,
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode(
        pa.Table.from_pylist([{"letters": [{"letter": "A"}, {"letter": "B"}]}]),
        strata=rf.Strata.train,
    )["record/letters/letter"]

    selected = torch.tensor([[[False, True, False, False]]])
    field.state[selected] = rf.Tokens.masked
    field.trainable[selected] = True
    rendered = render_text(field)

    assert isinstance(field, Renderable)
    assert "TensorField [tensorfield] state=(1, 1, 4) device=cpu trainable=1" in rendered
    assert "counts V=1 N=0 P=2 M=1 O=0" in rendered
    assert "state V M P P" in rendered
    assert "targets=content, state" in rendered


def test_tensorfield_rich_display_separates_nested_array_state_tokens() -> None:
    model = rf.Model(
        rf.Branch(
            rf.Branch(
                rf.Category("letter", size=8, p_unavailable=0.0),
                name="letters",
                length=3,
            ),
            name="words",
            length=2,
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode(
        pa.Table.from_pylist(
            [
                {
                    "words": [
                        {"letters": [{"letter": "A"}]},
                        {"letters": [{"letter": "B"}, {"letter": "C"}]},
                    ]
                }
            ]
        ),
        strata=rf.Strata.train,
    )["record/words/letters/letter"]

    rendered = render_text(field)

    assert "TensorField [tensorfield] state=(1, 1, 2, 3) device=cpu trainable=0" in rendered
    assert "state V P P\n       V V P" in rendered


def test_rich_display_does_not_replace_repr_or_mutate_serialization() -> None:
    node = rf.Number("amount", mask=0.15)
    dumped = node.model_dump(mode="python")

    assert "query=<inferred>" not in repr(node)
    assert "query=" not in str(node)
    assert "name='amount'" in repr(node)

    str(node)

    assert node.model_dump(mode="python") == dumped

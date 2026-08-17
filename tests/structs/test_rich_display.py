from io import StringIO
from pprint import pformat

import pytest
import torch
from rich.console import Console
from rich.pretty import Pretty

import relflow as rf
from relflow.rich import MimeBundleDisplay, theme
from relflow.structs.tree import Renderable


def render_text(node: object, *, width: int = 120) -> str:
    console = Console(
        file=StringIO(),
        record=True,
        width=width,
        force_terminal=False,
        theme=theme,
    )
    console.print(node)
    return console.export_text(clear=False).rstrip("\n")


def test_renderable_owns_shared_rich_styles() -> None:
    assert Renderable.RICH_NAME_STYLE == "relflow.name"
    assert Renderable.RICH_TYPE_STYLE == "relflow.type"
    assert Renderable.RICH_TREE_STYLE == "relflow.dim"


def test_leaf_rich_display_uses_schema_summary() -> None:
    rendered = render_text(rf.Number("amount"))

    assert "amount [number] active" in rendered
    assert "query=" not in rendered
    assert "pooling=query weight=1 p_mask=0 p_prune=0 n_heads=4 n_linear=1" in rendered
    assert "jitter=0 n_bands=8 offset=4 objective=mae" in rendered
    assert "model_config" not in rendered
    assert "model_fields_set" not in rendered

    lines = rendered.splitlines()
    assert lines[1].startswith("  pooling=")
    assert lines[2].startswith("  jitter=")


def test_leaf_display_flags() -> None:
    target = render_text(rf.Category("returned", target=True, size=2)).splitlines()[0].split()
    embedded = render_text(rf.Number("amount", embed=True)).splitlines()[0].split()
    inactive = render_text(rf.Category("customer_id", active=False)).splitlines()[0].split()

    assert "target" in target
    assert "embed" in embedded
    assert "inactive" in inactive
    assert "active" not in inactive


def test_unbound_nodes_render_with_placeholder_names() -> None:
    leaf = rf.Number()
    branch = rf.Branch(rf.Number("amount"))

    assert render_text(leaf).startswith("<unnamed> [number] active")
    assert render_text(branch).startswith("<unnamed> [branch]")


def test_leaf_display_includes_meaningful_optional_metadata() -> None:
    rendered = render_text(
        rf.Number(
            "amount",
            description="Normalized transaction amount",
            nullable=False,
        )
    )

    assert "Normalized transaction amount" in rendered
    assert "nullable=False" in rendered


def test_names_and_type_labels_use_shared_semantic_styles() -> None:
    number_html = rf.Number("amount")._repr_html_()
    category_html = rf.Category("sku")._repr_html_()
    branch_html = rf.Branch(rf.Number("amount"), name="items")._repr_html_()
    schema_html = rf.Schema.from_tree(
        rf.Number("amount"),
        rf.Number("label", target=True),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )._repr_html_()
    model_html = rf.Model(
        rf.Number("amount"),
        rf.Number("label", target=True),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )._repr_html_()

    for html in (number_html, category_html, branch_html, schema_html, model_html):
        assert "font-weight: bold" in html
        assert "color: #" not in html


def test_leaf_display_separates_common_and_specific_attributes() -> None:
    number_lines = render_text(rf.Number("amount", p_mask=0.15, objective="huber")).splitlines()
    category_lines = render_text(rf.Category("sku", size=2048)).splitlines()

    assert "p_mask=0.15" in number_lines[1]
    assert number_lines[1].startswith("  ")
    assert "objective=huber" not in number_lines[1]
    assert "objective=huber" in number_lines[2]
    assert number_lines[2].startswith("  ")

    assert "pooling=query" in category_lines[1]
    assert category_lines[1].startswith("  ")
    assert "size=2048" not in category_lines[1]
    assert "size=2048" in category_lines[2]
    assert category_lines[2].startswith("  ")


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
    assert "├── sku [category] active" in rendered
    assert "└── quantity [number] active" in rendered
    assert "query=" not in rendered


def test_tree_uses_native_rich_guides() -> None:
    rendered = render_text(
        rf.Branch(
            rf.Number("amount"),
            rf.Number("quantity"),
            name="line_items",
        )
    )

    assert "├── amount [number]" in rendered
    assert "└── quantity [number]" in rendered


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

    assert "└── record [root] embed attention=mha" in rendered
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
            rf.Category("churned", target=True, size=2),
            name="customer",
        )
    )

    assert "├── transactions [branch] length=360 overflow=tail" in rendered
    assert "│   ├── amount [number] active" in rendered
    assert "│   └── merchant [category] active" in rendered
    assert "└── churned [category] active target" in rendered
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
    assert "color: #" not in bundle["text/html"]

    mime, data = node._mime_()
    assert mime == "text/html"
    assert data == node._repr_html_()


def test_marimo_display_protocol_preserves_html_and_plain_text() -> None:
    model = rf.Model(
        amount=rf.Number,
        label=rf.Number(target=True),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    for value in (model, model.select()):
        display = value._display_()
        mime, bundle = display._mime_()

        assert isinstance(display, MimeBundleDisplay)
        assert mime == "application/vnd.marimo+mimebundle"
        assert set(bundle) == {"text/html", "text/plain"}
        assert bundle["text/html"].startswith("<pre")
        assert "\x1b" not in bundle["text/html"]
        assert "\x1b" not in bundle["text/plain"]

    assert "Model [model]" in model._display_()._mime_()[1]["text/plain"]
    assert "record/amount [number]" in model.select()._display_()._mime_()[1]["text/plain"]


def test_mime_bundle_filters_formats_and_escapes_html() -> None:
    node = rf.Number("amount", description="<unsafe>& text")

    assert set(node._repr_mimebundle_(include={"text/plain"})) == {"text/plain"}
    assert set(node._repr_mimebundle_(include={"text/html"})) == {"text/html"}
    assert node._repr_mimebundle_(exclude={"text/plain", "text/html"}) == {}

    html = node._repr_html_()
    assert "&lt;unsafe&gt;&amp; text" in html
    assert "<unsafe>" not in html
    assert "\x1b[" not in html


def test_schema_rich_display_uses_root_schema_tree() -> None:
    schema = rf.Schema.from_tree(
        rf.Number("amount"),
        rf.Category("label", target=True, size=2),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    assert isinstance(schema, Renderable)
    rendered = render_text(schema)

    assert rendered == str(schema)
    assert "schema [schema] d_model=8 branches=1 fields=2 targets=1 embeds=0" in rendered
    root_line = next(line for line in rendered.splitlines() if "└── record [root]" in line)
    assert "length=" not in root_line
    assert "overflow=" not in root_line
    assert "embed=False" not in root_line
    assert "    ├── amount [number] active query=[*].amount" in rendered
    assert "    └── label [category] active target query=[*].label" in rendered


def test_model_rich_display_uses_runtime_summary_and_schema_tree() -> None:
    model = rf.Model(
        rf.Number("amount"),
        rf.Category("label", target=True, size=2),
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
    assert "branches=1 fields=2 targets=1 embeds=0" in rendered
    root_line = next(line for line in rendered.splitlines() if "└── record [root]" in line)
    assert "length=" not in root_line
    assert "overflow=" not in root_line
    assert "embed=False" not in root_line
    assert "    ├── amount [number] active query=[*].amount" in rendered
    assert "    └── label [category] active target query=[*].label" in rendered


def test_model_select_pprint_uses_rich_node_display() -> None:
    model = rf.Model(
        rf.Number("amount"),
        rf.Category("species", target=True, size=4),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    selection = model.select(rf.where("address") == "record/species")
    rendered = pformat(selection)
    console = Console(file=StringIO(), record=True, width=120, theme=theme)
    console.print(Pretty(selection))
    rich_rendered = console.export_text(clear=False)

    assert isinstance(selection, list)
    for output in (rendered, rich_rendered):
        assert "record/species [category] active target query=[*].species" in output
        assert "pooling=" not in output
        assert "Request(name=" not in output
        assert "Selection(" not in output


def test_nested_rich_repr_is_compact_and_not_a_quoted_direct_view() -> None:
    node = rf.Number("amount", p_mask=0.15)
    model = rf.Model(
        amount=rf.Number,
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    rendered = render_text(Pretty({"node": node, "model": model}, expand_all=True))

    assert "Request(" in rendered
    assert "name='amount'" in rendered
    assert "Model(" in rendered
    assert "parameters=" in rendered
    assert "'amount [number] active\\n" not in rendered


def test_selection_renders_each_overlapping_match_once() -> None:
    model = rf.Model(
        items=rf.Branch(
            amount=rf.Number,
            quantity=rf.Number,
            length=4,
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    rendered = render_text(model.select())

    assert rendered.count("record [root]") == 1
    assert rendered.count("record/items [branch]") == 1
    assert rendered.count("record/items/amount [number]") == 1
    assert rendered.count("record/items/quantity [number]") == 1
    assert "pooling=" not in rendered


def test_rendering_does_not_populate_schema_or_address_caches() -> None:
    schema = rf.Schema.from_tree(
        rf.Number("amount"),
        rf.Number("label", target=True),
        name="record",
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    selection = schema.select()
    schema._clear_tree_caches()
    selection_cache = dict(schema._selection_cache)

    render_text(schema)
    render_text(selection)

    assert schema._selection_cache == selection_cache
    assert not {"branches", "requests", "active_requests", "shapes", "depthwise"} & schema.__dict__.keys()
    assert all("address" not in node.__dict__ for node in selection)


def test_model_tree_wraps_with_guides_intact_at_narrow_width() -> None:
    model = rf.Model(
        long_repeated_context=rf.Branch(
            long_numeric_field_name=rf.Number,
            length=32,
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )

    rendered = render_text(model, width=40)

    assert "└── record [root]" in rendered
    assert "    └── long_repeated_context" in rendered
    assert all(len(line) <= 40 for line in rendered.splitlines())


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
        [{"letters": [{"letter": "A"}, {"letter": "B"}]}],
        strata=rf.Strata.train,
    )["record/letters/letter"]

    field.hide(torch.tensor([[[False, True, False, False]]]))
    rendered = render_text(field)

    assert isinstance(field, Renderable)
    assert "TensorField [category] state=(1, 1, 4) dtype=int64 device=cpu trainable=(1, 1, 4)" in rendered
    assert "preview counts V=1 N=0 P=2 M=1 O=0" in rendered
    assert "content=(1, 1, 4)" in rendered
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
        [
            {
                "words": [
                    {"letters": [{"letter": "A"}]},
                    {"letters": [{"letter": "B"}, {"letter": "C"}]},
                ]
            }
        ],
        strata=rf.Strata.train,
        mask=False,
    )["record/words/letters/letter"]

    rendered = render_text(field)

    assert "TensorField [category] state=(1, 1, 2, 3) dtype=int64 device=cpu" in rendered
    assert "state V P P\n        V V P" in rendered


def test_tensorfield_default_render_avoids_reductions_and_device_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    model = rf.Model(
        amount=rf.Number,
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode([{"amount": 1.0}], strata=rf.Strata.predict)["record/amount"]

    def forbidden(*args, **kwargs):
        raise AssertionError("default rendering must not move or reduce tensors")

    monkeypatch.setattr(torch.Tensor, "to", forbidden)
    monkeypatch.setattr(torch.Tensor, "sum", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)

    rendered = render_text(field)

    assert "TensorField [number]" in rendered
    assert "state V" in rendered


def test_tensorfield_accelerator_like_state_omits_value_preview() -> None:
    model = rf.Model(
        amount=rf.Number,
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode([{"amount": 1.0}], strata=rf.Strata.predict)["record/amount"]
    field.state = torch.empty(field.state.shape, dtype=field.state.dtype, device="meta")
    field.trainable = torch.empty(field.trainable.shape, dtype=field.trainable.dtype, device="meta")

    rendered = render_text(field)

    assert "device=meta" in rendered
    assert "preview omitted for meta" in rendered
    assert "preview counts" not in rendered


def test_tensorfield_empty_cpu_state_renders_without_indexing_values() -> None:
    model = rf.Model(
        amount=rf.Number,
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode([], strata=rf.Strata.predict)["record/amount"]

    rendered = render_text(field)

    assert "state=(0, 1)" in rendered
    assert "preview counts V=0 N=0 P=0 M=0 O=0" in rendered
    assert "state <empty>" in rendered


def test_rich_display_does_not_replace_repr_or_mutate_serialization() -> None:
    node = rf.Number("amount", p_mask=0.15)
    dumped = node.model_dump(mode="python")

    assert "query=<inferred>" not in repr(node)
    assert "query=" not in str(node)
    assert "name='amount'" in repr(node)

    str(node)

    assert node.model_dump(mode="python") == dumped

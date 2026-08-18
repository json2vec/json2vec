import inspect
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


def render_direct_console(node: object, *, width: int = 120) -> str:
    output = StringIO()
    console = Console(
        file=output,
        width=width,
        force_terminal=False,
        force_jupyter=False,
        theme=theme,
    )
    console.print(node)
    return output.getvalue().rstrip("\n")


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


def test_selector_rich_repr_uses_stable_expressions() -> None:
    predicate = (rf.where("type") == "number") & ~rf.where("active")
    raw_callable = rf.NodePredicate.from_selector(lambda node: node.name == "amount")

    assert repr(rf.where("type")) == "where('type')"
    assert repr(predicate) == "(where('type') == 'number') & (~where('active'))"

    for value in (predicate, raw_callable):
        rendered = render_text(Pretty(value))
        assert "lambda" not in rendered
        assert "0x" not in rendered
        assert "func=" not in rendered

    assert "where('type') == 'number'" in render_text(Pretty(predicate))
    assert "predicate('callable')" in render_text(Pretty(raw_callable))


def test_selector_fallback_type_labels_are_sanitized_and_bounded() -> None:
    hostile_type = type(f"Value_{'x' * 1000}\x1b]52;c;payload\x07", (), {})
    predicate = rf.where("metadata") == hostile_type()

    rendered = repr(predicate)

    assert "\x1b" not in rendered
    assert "x" * 80 not in rendered
    assert len(rendered) <= 320


def test_processor_and_observation_rich_repr_hide_values_and_stay_bounded() -> None:
    secret = "SENSITIVE_BOUND_VALUE"

    @rf.preprocess
    def prepare(observation: dict, *, api_key: str, schema=None):
        return rf.Observation(observation)

    unbound = render_text(Pretty(prepare), width=160)
    bound = prepare.partial(api_key=secret)
    bound_outputs = (repr(bound), render_text(Pretty(bound), width=160))

    assert "name='prepare'" in unbound
    assert "signature='(observation, *, api_key, schema)'" in unbound
    assert "ready=False" in unbound
    assert "missing=('api_key',)" in unbound
    for output in bound_outputs:
        assert "ready=True" in output
        assert "bound=('api_key',)" in output
        assert secret not in output
        assert "0x" not in output

    observation = rf.Observation(
        {
            "credential": secret,
            **{f"field_{index}": index for index in range(10)},
        }
    )
    observation_outputs = (repr(observation), render_text(Pretty(observation), width=160))
    for output in observation_outputs:
        assert "count=11" in output
        assert "field_0" in output
        assert "int" in output
        assert "credential" in output
        assert "str" in output
        assert "omitted=3" in output
        assert secret not in output


def test_processor_and_observation_labels_and_parameter_lists_are_bounded() -> None:
    def wide_processor(observation, **kwargs):
        return rf.Observation(observation)

    parameters = [inspect.Parameter("observation", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    parameters.extend(inspect.Parameter(f"parameter_{index}", inspect.Parameter.KEYWORD_ONLY) for index in range(40))
    setattr(wide_processor, "__signature__", inspect.Signature(parameters))
    wide_processor.__name__ = f"processor_{'x' * 200}\x1b[31m"
    processor = rf.preprocess(wide_processor).partial(
        **{f"parameter_{index}": f"SECRET_{index}" for index in range(40)}
    )

    hostile_type = type(f"Value_{'y' * 1000}\x1b[31m", (), {})
    observation = rf.Observation({f"key_{'z' * 1000}\x1b]52;c;payload\x07": hostile_type()})

    outputs = (
        repr(processor),
        render_text(Pretty(processor), width=160),
        repr(observation),
        render_text(Pretty(observation), width=160),
    )
    for output in outputs:
        assert "\x1b" not in output
        assert "SECRET_" not in output
        assert len(output) < 1000

    assert "+33 parameters" in outputs[0]
    assert "… +32" in outputs[0]
    assert "z" * 80 not in outputs[2]
    assert "y" * 80 not in outputs[2]


def test_branch_rich_display_includes_bounded_mask_policy_details() -> None:
    masks = [
        rf.Mask(
            name="recent",
            count=3,
            window=10,
            branch=True,
            offset=2,
            exclude="amount",
        ),
        rf.Mask(name="early", rate=0.25, window=4, start=True),
        rf.Mask(name="third", count=1),
        rf.Mask(name="fourth", count=1),
        rf.Mask(name="hidden", count=1),
    ]
    branch = rf.Branch(amount=rf.Number, name="events", length=32, masks=masks)

    rendered = render_text(branch, width=180)

    assert "masks=5" in rendered
    assert "mask recent count=3 window=10 extent=capacity edge=end offset=2 exclude=('amount',)" in rendered
    assert "mask early rate=0.25 window=4 extent=occupied edge=start" in rendered
    assert "… +1 masks" in rendered
    assert "mask hidden" not in rendered


def test_leaf_rich_values_normalize_and_bound_nested_configuration() -> None:
    dateparts = rf.DateParts(
        "created_at",
        dateparts=["month_of_year", "day_of_week", "hour_of_day"],
    )
    node = rf.Number(
        "amount",
        tags=list(range(12)),
        note="x" * 200,
        nested={"roles": ["input", "audit"]},
    )

    dateparts_rendered = render_text(dateparts)
    node_rendered = render_text(node, width=200)

    assert "dateparts=[month_of_year, day_of_week, hour_of_day]" in dateparts_rendered
    assert "<DatePart." not in dateparts_rendered
    assert "metadata tags=[0, 1, 2, 3, 4, 5, 6, 7, … +4]" in node_rendered
    assert "nested={'roles': ['input', 'audit']}" in node_rendered
    assert "x" * 100 not in node_rendered
    assert "…" in node_rendered


def test_direct_rich_rendering_sanitizes_and_bounds_user_strings() -> None:
    terminal_controls = "\x1b]8;;https://evil.example\x07LINK\x1b]8;;\x07\x1b[31mRED\x1b[0m\x00"
    visible_prefix = "[bold]literal[/bold]<tag>"

    def hostile_callable() -> None:
        return None

    hostile_callable.__qualname__ = f"callable:{terminal_controls}{'C' * 1_000}"
    HostileType = type(f"type:{terminal_controls.replace(chr(0), '')}{'T' * 1_000}", (), {})
    leaf = rf.Number(
        "amount",
        description=f"description:{visible_prefix}{terminal_controls}{'D' * 1_000}",
        callback=hostile_callable,
        instance=HostileType(),
        **{f"metadata{terminal_controls}": f"value:{visible_prefix}{terminal_controls}{'V' * 1_000}"},
    )
    leaf.query = f"query:{visible_prefix}{terminal_controls}{'Q' * 1_000}"
    branch = rf.Branch(
        leaf,
        name="events",
        length=4,
        description=f"branch:{visible_prefix}{terminal_controls}{'B' * 1_000}",
        mask=rf.Mask(name=f"mask:{visible_prefix}{terminal_controls}{'M' * 1_000}", count=1),
    )

    direct = render_direct_console(branch, width=500)
    bundle = leaf._repr_mimebundle_()

    for output in (direct, bundle["text/plain"], bundle["text/html"]):
        assert "\x1b" not in output
        assert "\x00" not in output
        assert "\x07" not in output
        assert "evil.example" not in output
        assert "Q" * 81 not in output
        assert "V" * 81 not in output
        assert "C" * 81 not in output
        assert "T" * 81 not in output
        assert "…" in output

    assert "[bold]literal[/bold]" in direct
    assert "<tag>" in direct
    assert "&lt;tag&gt;" in bundle["text/html"]

    query_value = next(line for line in direct.splitlines() if "query=" in line).split("query=", maxsplit=1)[1]
    description_value = next(line for line in direct.splitlines() if line.startswith("      description:"))[6:]
    mask_name = next(line for line in direct.splitlines() if "mask mask:" in line).split("mask ", maxsplit=1)[1]
    mask_name = mask_name.split(" count=", maxsplit=1)[0]
    metadata_value = next(line for line in direct.splitlines() if "metadataLINKRED=" in line)
    metadata_value = metadata_value.split("metadataLINKRED=", maxsplit=1)[1]

    assert len(query_value) <= leaf.RICH_STRING_LIMIT
    assert len(description_value) <= leaf.RICH_STRING_LIMIT
    assert len(mask_name) <= leaf.RICH_STRING_LIMIT
    assert len(metadata_value) <= leaf.RICH_STRING_LIMIT


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

    field.hide(torch.tensor([[[True, False, False, False]]]), trainable=False)
    field.hide(torch.tensor([[[False, True, False, False]]]))
    rendered = render_text(field)

    assert isinstance(field, Renderable)
    assert "TensorField [category] state=(1, 1, 4) dtype=int64 device=cpu trainable=(1, 1, 4)" in rendered
    assert "preview counts V=0 N=0 P=2 M=2 O=0 *=1 trainable" in rendered
    assert "content=(1, 1, 4)" in rendered
    assert "axes=/record/letters=4 (singleton batch, /record hidden)" in rendered
    assert "state [letters] M M* P P" in rendered
    assert "legend V valued  N null  P padded  M masked  O other  * trainable" in rendered
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
    assert "axes=/record/words=2 × /record/words/letters=3" in rendered
    assert "state [words × letters]" in rendered
    assert "0 │ V P P" in rendered
    assert "1 │ V V P" in rendered


def test_tensorfield_rich_display_renders_the_exact_sliced_state() -> None:
    model = rf.Model(
        rf.Branch(
            rf.Branch(
                rf.Category("letter", size=8, p_unavailable=0.0),
                name="letters",
                length=4,
            ),
            name="words",
            length=3,
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode(
        [
            {
                "words": [
                    {"letters": [{"letter": "A"}, {"letter": "B"}]},
                    {"letters": [{"letter": "C"}]},
                    {"letters": [{"letter": "D"}, {"letter": "E"}, {"letter": "F"}]},
                ]
            }
        ],
        strata=rf.Strata.train,
        mask=False,
    )["record/words/letters/letter"]

    observation = field[0]
    window = field[0, 0, 1:, :3]
    row = field[0, 0, 2]
    scalar = field[0, 0, 2, 1]

    observation_rendered = render_text(observation)
    window_rendered = render_text(window)
    row_rendered = render_text(row)
    scalar_rendered = render_text(scalar)

    assert tuple(field.state.shape) == (1, 1, 3, 4)
    assert "state=(1, 3, 4)" in observation_rendered
    assert "axes=/record/words=3 × /record/words/letters=4" in observation_rendered
    assert "0 │ V V P P" in observation_rendered
    assert "1 │ V P P P" in observation_rendered
    assert "2 │ V V V P" in observation_rendered

    assert "state=(2, 3)" in window_rendered
    assert "axes=/record/words=2 × /record/words/letters=3" in window_rendered
    assert "0 │ V P P" in window_rendered
    assert "1 │ V V V" in window_rendered

    assert "state=(4,)" in row_rendered
    assert "axes=/record/words/letters=4" in row_rendered
    assert "state [letters] V V V P" in row_rendered

    assert "state=()" in scalar_rendered
    assert "axes=<scalar>" in scalar_rendered
    assert "state V" in scalar_rendered


def test_tensorfield_rich_display_requires_a_slice_for_more_than_two_axes() -> None:
    model = rf.Model(
        outer=rf.Branch(
            length=2,
            middle=rf.Branch(
                length=3,
                inner=rf.Branch(
                    length=4,
                    amount=rf.Number,
                ),
            ),
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode([{"outer": []}], mask=False)["record/outer/middle/inner/amount"]

    rendered = render_text(field)

    assert "state=(1, 1, 2, 3, 4)" in rendered
    assert "axes=/record/outer=2 × /record/outer/middle=3 × /record/outer/middle/inner=4" in rendered
    assert "state preview omitted: 3 non-singleton axes remain; slice to at most 2 to preview state" in rendered
    assert "preview counts" not in rendered
    assert "state [" not in rendered


def test_tensorfield_axis_labels_preserve_distinguishing_path_tails() -> None:
    shared = "/root_segment_is_extremely_long"

    first = rf.TensorFieldBase._state_axis_name(f"{shared}/first_segment_is_extremely_long", 0)
    second = rf.TensorFieldBase._state_axis_name(f"{shared}/second_segment_is_extremely_long", 1)
    third = rf.TensorFieldBase._state_axis_name(f"{shared}/third_segment_is_extremely_long", 2)

    for label in (first, second, third):
        assert len(label) <= rf.TensorFieldBase.STATE_AXIS_NAME_LIMIT
        assert label.endswith("long")

    assert "first_" in first
    assert "second_" in second
    assert "third_" in third
    assert len({first, second, third}) == 3


def test_tensorfield_high_rank_axes_put_omission_between_head_and_tail() -> None:
    node: rf.SchemaField = rf.Number("amount")
    for index in reversed(range(10)):
        node = rf.Branch(node, name=f"level_{index}", length=2)

    model = rf.Model(
        node,
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode([{"level_0": []}], mask=False)[
        "record/level_0/level_1/level_2/level_3/level_4/level_5/level_6/level_7/level_8/level_9/amount"
    ]

    axes_line = next(line for line in render_text(field, width=1_000).splitlines() if line.startswith("  axes="))

    assert "… +2 axes" in axes_line
    assert axes_line.index("/record/level_0=2") < axes_line.index("… +2 axes")
    assert axes_line.index("… +2 axes") < axes_line.index("/level_9=2")


def test_tensorfield_rich_display_bounds_a_large_two_dimensional_slice() -> None:
    model = rf.Model(
        rows=rf.Branch(
            length=20,
            columns=rf.Branch(
                length=40,
                amount=rf.Number,
            ),
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode([{"rows": []}], mask=False)["record/rows/columns/amount"]

    rendered = render_text(field[0, 0])

    assert "state=(20, 40)" in rendered
    assert "axes=/record/rows=20 × /record/rows/columns=40" in rendered
    assert "(sample)" in rendered
    assert len([line for line in rendered.splitlines() if "│" in line]) == field.STATE_PREVIEW_ROW_LIMIT
    assert len(rendered) < 2_500


def test_tensorfield_preview_columns_fit_the_current_console_width() -> None:
    model = rf.Model(
        items=rf.Branch(
            length=32,
            amount=rf.Number(target=True),
        ),
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode(
        [{"items": [{"amount": float(index)} for index in range(32)]}],
        strata=rf.Strata.train,
    )["record/items/amount"]

    rendered = render_text(field[0, 0], width=88)
    state_line = next(line for line in rendered.splitlines() if line.startswith("  state [items]"))

    assert len(state_line) <= 88
    assert "M*" in state_line
    assert "(sample)" in rendered
    assert not any(line.startswith(("M*", "V*", "N*", "P*", "O*")) for line in rendered.splitlines())


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


def test_tensorfield_accelerator_like_state_omits_value_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    model = rf.Model(
        amount=rf.Number,
        d_model=8,
        n_layers=1,
        n_heads=4,
    )
    field = model.encode([{"amount": 1.0}], strata=rf.Strata.predict)["record/amount"]
    field.state = torch.empty(field.state.shape, dtype=field.state.dtype, device="meta")
    field.trainable = torch.empty(field.trainable.shape, dtype=field.trainable.dtype, device="meta")

    def forbidden(*args, **kwargs):
        raise AssertionError("accelerator rendering must not transfer or extract tensor values")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    monkeypatch.setattr(torch.Tensor, "item", forbidden)

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
    assert "axes=batch=0 (singleton /record hidden)" in rendered
    assert "state [batch] <empty>" in rendered


def test_rich_display_does_not_replace_repr_or_mutate_serialization() -> None:
    node = rf.Number("amount", p_mask=0.15)
    dumped = node.model_dump(mode="python")

    assert "query=<inferred>" not in repr(node)
    assert "query=" not in str(node)
    assert "name='amount'" in repr(node)

    str(node)

    assert node.model_dump(mode="python") == dumped

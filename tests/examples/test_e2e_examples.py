from __future__ import annotations

import re
from pathlib import Path

import jmespath
import lightning.pytorch as lit
import polars as pl
import pytest
import torch
from rich.pretty import Pretty

import relflow as rf


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _quarto_anchors(source: Path) -> set[str]:
    text = source.read_text()
    anchors = set(re.findall(r"\{#([^}]+)\}", text))

    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", text, flags=re.MULTILINE):
        heading = re.sub(r"\s*\{[^}]*\}\s*$", "", heading)
        heading = re.sub(r"`([^`]*)`", r"\1", heading)
        slug = re.sub(r"[^a-z0-9\s-]", "", heading.lower())
        slug = re.sub(r"\s+", "-", slug).strip("-")
        if slug:
            anchors.add(slug)

    return anchors


def _marimo_cells(source: str) -> list[str]:
    return re.findall(r"```\{python \.marimo[^}]*\}\n(.*?)\n```", source, flags=re.DOTALL)


def _contrast_ratio(first: str, second: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_quarto_docs_snapshot_contains_expected_pages() -> None:
    root = _repo_root()

    expected = {
        "docs/index.qmd",
        "docs/getting-started.qmd",
        "docs/ai-quickstart.qmd",
        "docs/core-concepts/binding-data.qmd",
        "docs/core-concepts/data-flow.qmd",
        "docs/core-concepts/model-tree.qmd",
        "docs/core-concepts/querypaths.qmd",
        "docs/core-concepts/data-types.qmd",
        "docs/core-concepts/embeddings.qmd",
        "docs/core-concepts/dynamic-masking.qmd",
        "docs/data-types/boolean.qmd",
        "docs/data-types/branch.qmd",
        "docs/data-types/category.qmd",
        "docs/data-types/dateparts.qmd",
        "docs/data-types/hash.qmd",
        "docs/data-types/number.qmd",
        "docs/data-types/set.qmd",
        "docs/data-types/text.qmd",
        "docs/data-types/vector.qmd",
        "docs/guides/batch-inference.qmd",
        "docs/guides/custom-tensorfields.qmd",
        "docs/guides/data-modules.qmd",
        "docs/guides/evaluation.qmd",
        "docs/guides/field-importance.qmd",
        "docs/guides/field-stacking.qmd",
        "docs/guides/lightning.qmd",
        "docs/guides/model-configuration.qmd",
        "docs/guides/model-lifecycle.qmd",
        "docs/guides/performance.qmd",
        "docs/guides/postprocessors.qmd",
        "docs/guides/prediction-output.qmd",
        "docs/guides/schema-mutation.qmd",
        "docs/guides/serving.qmd",
        "docs/guides/temporal-validation.qmd",
        "docs/guides/troubleshooting.qmd",
        "docs/reference/public-api.qmd",
        "docs/case-studies/device-tenure.qmd",
        "docs/case-studies/iris-reproducible.qmd",
    }

    missing = {path for path in expected if not (root / path).is_file()}
    assert missing == set()


def test_quarto_docs_use_current_public_api_style() -> None:
    root = _repo_root()
    sources = "\n".join(path.read_text() for path in sorted((root / "docs").rglob("*.qmd")))

    assert "rf.Model(" in sources
    assert "rf.Model.from_tree(" not in sources
    assert "rf.Branch(" in sources
    assert "Struct(" not in sources
    assert not re.search(r"Model\.from_tree\([^)]*\broot\s*=", sources, re.DOTALL)


def test_quarto_rich_examples_use_live_mime_display_and_semantic_styles() -> None:
    docs = _repo_root() / "docs"
    model_tree = (docs / "core-concepts/model-tree.qmd").read_text()
    mutation = (docs / "guides/schema-mutation.qmd").read_text()
    public_api = (docs / "reference/public-api.qmd").read_text()
    troubleshooting = (docs / "guides/troubleshooting.qmd").read_text()
    stylesheet = (docs / "assets/stylesheets/quarto-marimo.css").read_text()

    model_cells = _marimo_cells(model_tree)
    mutation_cells = _marimo_cells(mutation)
    public_cells = _marimo_cells(public_api)

    assert any(cell.rstrip().endswith("model") for cell in model_cells)
    assert "from rich.padding import Padding" not in model_tree
    assert "print(model)" not in model_tree

    assert any(cell.rstrip().endswith("numbers") for cell in mutation_cells)
    assert "print([str(node.address)" not in mutation
    assert '"relflow @ git+' in mutation

    assert any(cell.rstrip().endswith("inspection_model") for cell in public_cells)
    assert "from rich.padding import Padding" not in public_api
    assert any(re.search(r"Pretty\(.*expand_all=True,\s*\)\s*$", cell, flags=re.DOTALL) for cell in public_cells)
    assert "print(inspection_model)" not in public_api
    assert '"relflow @ git+' in public_api

    troubleshooting_prose = " ".join(troubleshooting.split())
    assert "incident-only" in troubleshooting
    assert "optimizer steps, and epochs do not emit diagnostic lines" in troubleshooting_prose
    assert "show_locals=False" in troubleshooting
    assert "from rich.traceback import install" in troubleshooting

    published = "\n".join(path.read_text() for path in docs.rglob("*.qmd"))
    assert "rf.console" not in published
    assert "rf.configure_console" not in published
    assert "rf.install_tracebacks" not in published

    nested = Pretty({"amount": rf.Number("amount")}, expand_all=True)
    bundle = nested._repr_mimebundle_(include=None, exclude=None)
    assert set(bundle) == {"text/html", "text/plain"}
    assert bundle["text/html"].startswith("<pre")
    assert "<span" in bundle["text/html"]
    assert "Request" in bundle["text/html"]

    assert stylesheet.count("--qmo-output-fg:") == 2
    assert stylesheet.count("--qmo-output-accent-fg:") == 2
    assert "--qmo-rich-" not in stylesheet
    assert "#008000" not in stylesheet
    assert "#800080" not in stylesheet
    assert "#808000" not in stylesheet
    assert 'span[style*="color:"]' in stylesheet

    def values(name: str) -> list[str]:
        return re.findall(rf"{re.escape(name)}:\s*(#[0-9a-fA-F]{{6}});", stylesheet)

    cell_backgrounds = values("--qmo-cell-bg")
    accents = values("--qmo-output-accent-fg")
    emphasis_backgrounds = values("--qmo-output-emphasis-bg")
    emphasis_foregrounds = values("--qmo-output-emphasis-fg")
    assert len(cell_backgrounds) == len(accents) == 2
    assert len(emphasis_backgrounds) == len(emphasis_foregrounds) == 2
    assert all(
        _contrast_ratio(foreground, background) >= 4.5
        for foreground, background in zip(accents, cell_backgrounds, strict=True)
    )
    assert all(
        _contrast_ratio(foreground, background) >= 4.5
        for foreground, background in zip(emphasis_foregrounds, emphasis_backgrounds, strict=True)
    )


def test_workflow_guides_label_example_status() -> None:
    guides = _repo_root() / "docs/guides"
    unlabeled = [path.name for path in guides.glob("*.qmd") if "**Example status" not in path.read_text()]
    assert unlabeled == []


def test_quarto_navigation_and_relative_links_resolve() -> None:
    root = _repo_root()
    docs = root / "docs"

    navigation = (docs / "_quarto.yml").read_text()
    hrefs = re.findall(r"^\s*- href:\s+([^\s]+)", navigation, flags=re.MULTILINE)
    missing_navigation = [href for href in hrefs if not (docs / href).is_file()]
    assert missing_navigation == []

    missing_links: list[tuple[str, str]] = []
    missing_anchors: list[tuple[str, str]] = []
    for source in docs.rglob("*.qmd"):
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", source.read_text()):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            pathname, _, anchor = target.partition("#")
            destination = (source.parent / pathname).resolve() if pathname else source.resolve()
            if not destination.is_file():
                missing_links.append((source.relative_to(root).as_posix(), target))
            elif anchor and anchor not in _quarto_anchors(destination):
                missing_anchors.append((source.relative_to(root).as_posix(), target))

    assert missing_links == []
    assert missing_anchors == []


def test_documented_contracts_match_public_enums_and_requests() -> None:
    root = _repo_root()
    docs = root / "docs"
    data_types = (docs / "core-concepts/data-types.qmd").read_text()
    branch = (docs / "data-types/branch.qmd").read_text()
    dateparts = (docs / "data-types/dateparts.qmd").read_text()
    published = "\n".join(
        path.read_text() for path in docs.rglob("*.qmd") if path.name != "documentation-content-spec.md"
    )

    assert {token.name for token in rf.Tokens} == {"valued", "null", "padded", "masked", "other"}
    assert all(f"`{token.name}`" in data_types for token in rf.Tokens)
    assert all(mode.value in branch for mode in rf.AttentionMode)
    assert "`second_of_minute`" in dateparts
    assert "One extra internal bucket is reserved" not in published
    assert "reserved unavailable bucket" not in published
    assert "Pruning is an operation" in data_types

    common_fields = set(rf.RequestBase.model_fields)
    for name in ("Number", "Boolean", "Category", "Set", "Hash", "DateParts", "Vector", "Text"):
        request = getattr(rf, name)
        reference = (docs / f"data-types/{name.lower()}.qmd").read_text()
        for field_name, field in request.model_fields.items():
            if field_name in common_fields:
                continue
            public_name = field.serialization_alias or field.alias or field_name
            assert f"`{public_name}`" in reference, f"{name}.{public_name} is missing from its reference"


def test_documented_streaming_and_attention_constraints_match_runtime() -> None:
    root = _repo_root()
    readme = (root / "README.md").read_text()
    data_modules = (root / "docs/guides/data-modules.qmd").read_text()
    model_configuration = (root / "docs/guides/model-configuration.qmd").read_text()
    public_api = (root / "docs/reference/public-api.qmd").read_text()
    readme_prose = " ".join(readme.split())
    data_module_prose = " ".join(data_modules.split())

    assert "`ndjson` is local-only" in readme
    assert "Avro is not supported" in readme_prose
    assert "The `ndjson` reader uses local file access" in data_module_prose
    assert "Avro is not supported" in data_module_prose

    for page in (model_configuration, public_api):
        assert "`d_model // n_heads >= 2`" in page


def test_datatype_option_tables_cover_public_type_specific_fields() -> None:
    docs = _repo_root() / "docs/data-types"
    common = {
        "name",
        "type",
        "description",
        "embed",
        "n_heads",
        "dropout",
        "active",
        "query",
        "nullable",
        "pooling",
        "weight",
        "p_mask",
        "p_prune",
        "n_linear",
    }
    pages = {
        rf.Boolean: docs / "boolean.qmd",
        rf.Category: docs / "category.qmd",
        rf.DateParts: docs / "dateparts.qmd",
        rf.Hash: docs / "hash.qmd",
        rf.Number: docs / "number.qmd",
        rf.Set: docs / "set.qmd",
        rf.Text: docs / "text.qmd",
        rf.Vector: docs / "vector.qmd",
    }

    missing: list[tuple[str, str]] = []
    for request, page in pages.items():
        text = page.read_text()
        for name, field in request.model_fields.items():
            if name in common:
                continue
            option = field.serialization_alias or field.alias or name
            if f"`{option}`" not in text:
                missing.append((request.__name__, option))

    assert missing == []


def test_documented_sparse_stacking_and_map_queries_preserve_coordinates() -> None:
    observation = [
        {
            "events": [
                {"ip_country": "US", "amount": None},
                {"ip_country": None, "amount": 950.0},
                {"ip_country": "CA", "amount": None},
            ]
        }
    ]

    assert jmespath.search("[*].events[*].ip_country", observation) == [["US", "CA"]]
    assert jmespath.search("[*].events[*].amount", observation) == [[950.0]]
    assert jmespath.search("[*].map(&ip_country, events)", observation) == [["US", None, "CA"]]
    assert jmespath.search("[*].map(&amount, events)", observation) == [[None, 950.0, None]]

    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            length=3,
            ip_country=rf.Category(size=4),
            amount=rf.Number,
        ),
    )
    encoded = model.encode(observation)
    assert encoded[rf.Address("record", "events", "ip_country")].state.tolist() == [[[0, 1, 0]]]
    assert encoded[rf.Address("record", "events", "amount")].state.tolist() == [[[1, 0, 1]]]

    filtered = [
        {
            "events": [
                {"event_type": "login", "device_id": "a", "risk_score": None},
                {"event_type": "other", "device_id": "b", "risk_score": 1.0},
                {"event_type": "login", "device_id": "c", "risk_score": 2.0},
            ]
        }
    ]
    selection = "events[?event_type == 'login']"
    assert jmespath.search(f"[*].map(&device_id, {selection})", filtered) == [["a", "c"]]
    assert jmespath.search(f"[*].map(&risk_score, {selection})", filtered) == [[None, 2.0]]


def test_documented_prediction_envelope_is_executable() -> None:
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=1,
        embed=True,
        amount=rf.Number,
        label=rf.Category(target=True, size=2, p_unavailable=0.0),
    )
    model.encode(
        [
            {"amount": 1.0, "label": "no"},
            {"amount": 2.0, "label": "yes"},
        ],
        strata="train",
    )

    output = model.predict([{"amount": 1.5}])
    target = output[rf.Address("record", "label")]
    root = output[rf.Address("record")]

    assert set(target) == {"state", "content", "inferred"}
    assert set(target["state"]) == {token.name for token in rf.Tokens}
    assert set(target["content"]) == {"value", "probability", "topk"}
    assert target["inferred"] == [True]
    assert torch.linalg.vector_norm(torch.tensor(root["embedding"][0])).item() == pytest.approx(1.0)


def test_getting_started_lifecycle_is_executable(tmp_path: Path) -> None:
    torch.manual_seed(7)
    train_records = pl.DataFrame(
        [
            {"sepal_length": 5.1, "petal_length": 1.4, "species": "setosa"},
            {"sepal_length": 4.9, "petal_length": 1.4, "species": "setosa"},
            {"sepal_length": 6.4, "petal_length": 4.5, "species": "versicolor"},
            {"sepal_length": 6.0, "petal_length": 4.5, "species": "versicolor"},
            {"sepal_length": 6.3, "petal_length": 6.0, "species": "virginica"},
            {"sepal_length": 5.8, "petal_length": 5.1, "species": "virginica"},
        ]
    )
    validate_records = pl.DataFrame(
        [
            {"sepal_length": 5.0, "petal_length": 1.5, "species": "setosa"},
            {"sepal_length": 5.9, "petal_length": 4.2, "species": "versicolor"},
            {"sepal_length": 6.5, "petal_length": 5.2, "species": "virginica"},
        ]
    )
    model = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        batch_size=3,
        embed=True,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=1e-2),
        sepal_length=rf.Number,
        petal_length=rf.Number,
        species=rf.Category(target=True, size=3, topk=[2]),
    )
    datamodule = rf.PolarsDataModule(
        model=model,
        train=train_records,
        validate=validate_records,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        observation_buffer_size=8,
        sample_rate=1.0,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        limit_train_batches=1,
        limit_val_batches=1,
    )

    trainer.fit(model=model, datamodule=datamodule)
    metrics = trainer.validate(model=model, datamodule=datamodule, verbose=False)[0]
    assert "loss/validate" in metrics
    assert "record.species/validate.accuracy.content" in metrics

    artifact = tmp_path / "getting-started.rf"
    model.save(artifact)
    restored = rf.Model.load(artifact)
    predictions = restored.predict([{"sepal_length": 5.0, "petal_length": 1.5}])
    assert predictions[rf.Address("record", "species")]["inferred"] == [True]
    assert len(predictions[rf.Address("record")]["embedding"][0]) == 16


def test_iris_case_study_reproduces_documented_result(tmp_path: Path) -> None:
    lit.seed_everything(7, workers=True)
    records = pl.read_ndjson(_repo_root() / "docs/data/iris.jsonl").with_row_index()
    train_records = records.filter((pl.col("index") % 5) < 3).drop("index")
    validate_records = records.filter((pl.col("index") % 5) == 3).drop("index")
    test_records = records.filter((pl.col("index") % 5) == 4).drop("index")

    model = rf.Model(
        name="flower",
        d_model=32,
        n_layers=2,
        n_heads=4,
        batch_size=15,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        sepal_length=rf.Number,
        sepal_width=rf.Number,
        petal_length=rf.Number,
        petal_width=rf.Number,
        species=rf.Category(target=True, size=3, p_unavailable=0.0),
    )
    datamodule = rf.PolarsDataModule(
        model=model,
        train=train_records,
        validate=validate_records,
        test=test_records,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    checkpoint = rf.RollbackCheckpoint(
        dirpath=tmp_path / "iris-checkpoints",
        filename="best",
        monitor="loss/validate",
        mode="min",
        save_top_k=1,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=100,
        callbacks=[checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
    )

    trainer.fit(model=model, datamodule=datamodule)
    metrics = trainer.test(model=model, datamodule=datamodule, verbose=False)[0]
    assert metrics["flower.species/test.accuracy.content"] == pytest.approx(0.9)

    artifact = tmp_path / "iris-model.rf"
    model.save(artifact)
    restored = rf.Model.load(artifact)
    requests = test_records.drop("species").head(3).to_dicts()
    predictions = restored.predict(requests)
    assert predictions[rf.Address("flower", "species")]["inferred"] == [True, True, True]


def test_only_allowed_standalone_examples_are_present() -> None:
    root = _repo_root()
    examples = root / "examples"
    if not examples.exists():
        return

    discovered = {
        path.relative_to(root).as_posix()
        for path in examples.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    assert discovered == {
        "examples/dynamic-masking/run.py",
        "examples/inference-masking/run.py",
        "examples/realtime-serving/run.py",
        "examples/text-encoder/run.py",
    }

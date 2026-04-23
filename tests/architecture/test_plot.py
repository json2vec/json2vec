from __future__ import annotations

import json
from pathlib import Path

import torch

from json2vec.architecture.root import JSON2Vec
from json2vec.structs.enums import Stage, Strata, Suffix
from json2vec.structs.experiment import Dataset, Session
from json2vec.structs.structure import Structure


def _structure() -> Structure:
    return Structure.model_validate(
        {
            "name": "plot-demo",
            "type": "structure",
            "batch_size": 2,
            "dropout": 0.1,
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "context",
                "context_size": 1,
                "n_outputs": 1,
                "fields": [
                    {
                        "name": "amount",
                        "type": "number",
                        "query": "amount",
                    },
                    {
                        "name": "items",
                        "type": "context",
                        "context_size": 2,
                        "n_outputs": 1,
                        "fields": [
                            {
                                "name": "label",
                                "type": "category",
                                "query": "[*].label",
                                "max_vocab_size": 4,
                            },
                            {
                                "name": "identifier",
                                "type": "entity",
                                "query": "[*].id",
                                "topk": [2],
                            },
                        ],
                    },
                ],
            },
        }
    )


def _session(dataset_root: Path) -> Session:
    dataset = Dataset.model_validate(
        {
            "root": str(dataset_root),
            "sample_rate": 1.0,
            "file_buffer_size": 4,
            "observation_buffer_size": 4,
            "processor": "default",
            "kwargs": {},
            "suffix": Suffix.ndjson,
            "patterns": {strata: r".*\.ndjson$" for strata in Strata},
        }
    )

    return Session.model_validate(
        {
            "name": "plot-session",
            "dataset": dataset,
            "structure": _structure(),
            "task": Stage.predict,
            "output": ["root/items/label"],
        }
    )


def _model(tmp_path: Path) -> JSON2Vec:
    dataset_path = tmp_path / "plot_records.ndjson"
    dataset_path.write_text(json.dumps({"label": "alpha"}), encoding="utf-8")
    model = JSON2Vec.get_or_create(session=_session(dataset_root=dataset_path))

    label = model.nodes["root/items/label"]
    label.embedder.vocab.master.append("alpha")
    label.embedder.vocab.master.append("beta")
    label.decoder.counters["state"].counts.copy_(torch.tensor([3, 2, 1, 1, 1, 1], dtype=torch.int64))
    label.decoder.counters["content"].counts.copy_(torch.tensor([4, 2, 1, 1, 1], dtype=torch.int64))

    amount = model.nodes["root/amount"]
    amount.embedder.normalizer.mean.fill_(12.0)
    amount.embedder.normalizer.var.fill_(9.0)
    amount.embedder.normalizer.count.fill_(5.0)
    amount.decoder.counter.counts.copy_(torch.tensor([6, 2, 1, 1, 1, 1], dtype=torch.int64))

    return model


def test_plot_renders_full_model_and_writes_output(tmp_path: Path) -> None:
    model = _model(tmp_path)
    output_path = tmp_path / "model-tree.html"

    rendered = model.plot(detail=True, out=output_path)

    assert rendered.startswith("<!DOCTYPE html>")
    assert "<style>" in rendered
    assert "background-color: #ffffff;" in rendered
    assert "background-color: #000000" in rendered
    assert "background-color: #800000" in rendered
    assert "background-color: #008000" in rendered
    assert '<span class="r' in rendered
    assert "JSON2Vec (plot-demo)" in rendered
    assert "root (context)" in rendered
    assert "amount (number)" in rendered
    assert "address: root/amount" in rendered
    assert "items (context)" in rendered
    assert "address: root/items" in rendered
    assert "label (category)" in rendered
    assert "address: root/items/label" in rendered
    assert "identifier (entity)" in rendered
    assert "address: root/items/identifier" in rendered
    assert "vocabulary:" in rendered
    assert "alpha" in rendered
    assert "beta" in rendered
    assert "std_dev: 3.0000016689300537" in rendered
    assert "counts: [4, 2, 1, 1, 1]" in rendered
    assert "children" in rendered
    assert output_path.read_text(encoding="utf-8") == rendered


def test_plot_address_limits_output_to_selected_branch(tmp_path: Path) -> None:
    model = _model(tmp_path)

    rendered = model.plot(address="root/items", detail=False)

    assert rendered.startswith("<!DOCTYPE html>")
    assert '<span class="r' in rendered
    assert "background-color: #ffffff;" in rendered
    assert "background-color: #000000" in rendered
    assert "background-color: #800000" in rendered
    assert "items (context)" in rendered
    assert "address: root/items" in rendered
    assert "label (category)" in rendered
    assert "address: root/items/label" in rendered
    assert "identifier (entity)" in rendered
    assert "address: root/items/identifier" in rendered
    assert "address: root/amount" not in rendered
    assert "JSON2Vec (plot-demo)" not in rendered


def test_plot_leaf_uses_default_extension_renderer(tmp_path: Path) -> None:
    model = _model(tmp_path)

    rendered = model.plot(address="root/items/identifier", detail=True)

    assert rendered.startswith("<!DOCTYPE html>")
    assert '<span class="r' in rendered
    assert "background-color: #ffffff;" in rendered
    assert "background-color: #000000" in rendered
    assert "identifier (entity)" in rendered
    assert "address: root/items/identifier" in rendered
    assert "topk: [2]" in rendered
    assert "query: [*].id" in rendered
    assert "counters" not in rendered

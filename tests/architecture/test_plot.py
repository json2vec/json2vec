from __future__ import annotations

from pathlib import Path

import torch

from json2vec.architecture.plot import format_value
from json2vec.architecture.root import Model
from json2vec.structs.experiment import Hyperparameters


def _hyperparameters() -> Hyperparameters:
    return Hyperparameters.model_validate(
        {
            "d_model": 8,
            "fields": {
                "name": "root",
                "type": "array",
                "dropout": 0.1,
                "max_length": 1,
                "n_outputs": 1,
                "fields": [
                    {
                        "name": "amount",
                        "type": "number",
                        "query": "amount",
                    },
                    {
                        "name": "items",
                        "type": "array",
                        "max_length": 2,
                        "n_outputs": 1,
                        "fields": [
                            {
                                "name": "label",
                                "type": "category",
                                "query": "[*].label",
                                "embed": True,
                                "max_vocab_size": 4,
                            },
                            {
                                "name": "identifier",
                                "type": "entity",
                                "query": "[*].id",
                                "embed": False,
                                "p_prune": 1.0,
                                "topk": [2],
                            },
                        ],
                    },
                ],
            },
        }
    )


def _model(tmp_path: Path) -> Model:
    model = Model(hyperparameters=_hyperparameters(), batch_size=2)

    label = model.nodes["root/items/label"]
    label.embedder.vocab.master.append("alpha")
    label.embedder.vocab.master.append("beta")
    label.embedder.counters["state"].counts.copy_(torch.tensor([3, 2, 1, 1, 1], dtype=torch.int64))
    label.embedder.counters["content"].counts.copy_(torch.tensor([4, 2, 1, 1, 1], dtype=torch.int64))

    amount = model.nodes["root/amount"]
    amount.embedder.normalizer.mean.fill_(12.0)
    amount.embedder.normalizer.var.fill_(9.0)
    amount.embedder.normalizer.count.fill_(5.0)
    amount.embedder.counter.counts.copy_(torch.tensor([6, 2, 1, 1, 1], dtype=torch.int64))

    return model


def test_plot_renders_full_model_and_writes_output(tmp_path: Path, capsys) -> None:
    model = _model(tmp_path)
    output_path = tmp_path / "model-tree.html"
    capsys.readouterr()

    rendered = model.plot(detail=True, out=output_path)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert rendered.startswith("<!DOCTYPE html>")
    assert "<style>" in rendered
    assert "background-color: #ffffff;" in rendered
    assert "background-color: #800000" not in rendered
    assert "background-color: #008000" not in rendered
    assert '<span class="r' not in rendered
    assert "Model" in rendered
    assert f"parameters: {sum(parameter.numel() for parameter in model.parameters())}" in rendered
    assert "root (array)" in rendered
    assert "amount (number)" in rendered
    assert "address: root/amount" in rendered
    assert "items (array)" in rendered
    assert "address: root/items" in rendered
    assert "label (category)" in rendered
    assert "address: root/items/label" in rendered
    assert "identifier (entity)" in rendered
    assert "address: root/items/identifier" in rendered
    assert "┏━ label (category)" in rendered
    assert "┏━ identifier (entity)" in rendered
    assert "vocabulary:" in rendered
    assert "alpha" in rendered
    assert "beta" in rendered
    assert "std_dev: 3.0000016689300537" in rendered
    assert "counts: [4, 2, 1, 1, 1]" in rendered
    assert "children" not in rendered
    assert output_path.read_text(encoding="utf-8") == rendered


def test_plot_address_limits_output_to_selected_branch(tmp_path: Path) -> None:
    model = _model(tmp_path)

    rendered = model.plot(address="root/items", detail=False)

    assert rendered.startswith("<!DOCTYPE html>")
    assert "background-color: #ffffff;" in rendered
    assert "background-color: #800000" not in rendered
    assert '<span class="r' not in rendered
    assert "items (array)" in rendered
    assert "address: root/items" in rendered
    assert "label (category)" in rendered
    assert "address: root/items/label" in rendered
    assert "identifier (entity)" in rendered
    assert "address: root/items/identifier" in rendered
    assert "┏━ label (category)" in rendered
    assert "┏━ identifier (entity)" in rendered
    assert "address: root/amount" not in rendered
    assert "Model" not in rendered


def test_plot_leaf_uses_default_extension_renderer(tmp_path: Path) -> None:
    model = _model(tmp_path)

    rendered = model.plot(address="root/items/identifier", detail=True)

    assert rendered.startswith("<!DOCTYPE html>")
    assert "background-color: #ffffff;" in rendered
    assert '<span class="r' not in rendered
    assert "identifier (entity)" in rendered
    assert "address: root/items/identifier" in rendered
    assert "┏━ identifier (entity)" in rendered
    assert "topk: [2]" in rendered
    assert "query: [*].id" in rendered
    assert "counters" not in rendered


def test_plot_formats_scalar_lists_inline() -> None:
    vocabulary = [f"label_{index}" for index in range(12)]

    rendered = format_value(vocabulary)

    assert "\n" not in rendered
    assert rendered == "[" + ", ".join(vocabulary) + "]"

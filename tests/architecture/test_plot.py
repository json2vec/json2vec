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
    output_path = tmp_path / "model-tree.txt"
    capsys.readouterr()

    rendered = model.plot(detail=True, out=output_path)

    captured = capsys.readouterr()
    written = output_path.read_text(encoding="utf-8")
    assert rendered is None
    assert captured.out
    assert written
    assert "<!DOCTYPE html>" not in written
    assert ".json2vec" not in written
    assert "Schema" in captured.out
    assert "Model" not in captured.out
    assert "d_model=8" in captured.out
    assert "batch_size=2" in captured.out
    assert f"{sum(parameter.numel() for parameter in model.parameters()):,}" in captured.out
    assert "root [array]" in captured.out
    assert "amount [number]" in captured.out
    assert "root/amount" in captured.out
    assert "items [array]" in captured.out
    assert "root/items" in captured.out
    assert "label [category]" in captured.out
    assert "root/items/label" in captured.out
    assert "identifier [entity]" in captured.out
    assert "root/items/identifier" in captured.out
    assert "vocabulary:" in written
    assert "alpha" in written
    assert "beta" in written
    assert "std_dev: 3.0000016689300537" in written
    assert "counts: [4, 2, 1, 1, 1]" in written
    assert "children" not in written


def test_plot_uses_rich_environment_detection_for_display(tmp_path: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class ConsoleStub:
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def print(self, renderable) -> None:
            self.renderable = renderable

        def export_text(self, clear: bool = False) -> str:
            return "recorded"

    monkeypatch.setattr("json2vec.architecture.plot.Console", ConsoleStub)

    model = _model(tmp_path)
    model.plot(out=tmp_path / "model-tree.txt")

    assert "force_jupyter" not in calls[0]
    assert calls[1]["force_jupyter"] is False


def test_plot_address_limits_output_to_selected_branch(tmp_path: Path, capsys) -> None:
    model = _model(tmp_path)
    capsys.readouterr()

    rendered = model.plot(address="root/items", detail=False)
    captured = capsys.readouterr()

    assert rendered is None
    assert "items [array]" in captured.out
    assert "root/items" in captured.out
    assert "label [category]" in captured.out
    assert "root/items/label" in captured.out
    assert "identifier [entity]" in captured.out
    assert "root/items/identifier" in captured.out
    assert "root/amount" not in captured.out
    assert "Model" not in captured.out


def test_plot_leaf_uses_default_extension_renderer(tmp_path: Path, capsys) -> None:
    model = _model(tmp_path)
    capsys.readouterr()

    rendered = model.plot(address="root/items/identifier", detail=True)
    captured = capsys.readouterr()

    assert rendered is None
    assert "identifier [entity]" in captured.out
    assert "root/items/identifier" in captured.out
    assert "topk=[2]" in captured.out
    assert "query=[*].id" in captured.out
    assert "counters" not in captured.out


def test_plot_formats_scalar_lists_inline() -> None:
    vocabulary = [f"label_{index}" for index in range(12)]

    rendered = format_value(vocabulary)

    assert "\n" not in rendered
    assert rendered == "[" + ", ".join(vocabulary) + "]"

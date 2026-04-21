# JSON2Vec

`json2vec` is a framework for learning embeddings directly from nested, semi-structured records without flattening them into static feature tables.

The full project overview, architecture notes, and usage examples live in [docs/README.md](docs/README.md).

## Install

```bash
pip install json2vec
```

## Quickstart

```bash
uv sync
uv run python -m json2vec --experiment taxml --name local-dev --notes "baseline run"
```

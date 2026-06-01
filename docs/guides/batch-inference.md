# Batch Inference

`model.predict([...])` is the interactive path for a list of raw dictionaries.
For large prediction jobs, use a data module with Lightning
`trainer.predict(...)`. Attach `j2v.Writer(...)` to write each prediction batch
to Parquet as it finishes.

## Minimal Batch Prediction

```python
import lightning.pytorch as lit
import polars as pl

import json2vec as j2v

predict_frame = pl.DataFrame(
    {
        "amount": [12.5, 8.0],
        "merchant": ["books", "coffee"],
    }
)

datamodule = j2v.PolarsDataModule(
    model=model,
    predict=predict_frame,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)

writer = j2v.Writer("predictions")

trainer = lit.Trainer(
    accelerator="cpu",
    callbacks=[writer],
    logger=False,
)

trainer.predict(model=model, datamodule=datamodule)
```

`Writer` is a Lightning prediction callback. It belongs in
`Trainer(callbacks=[...])` and runs during `trainer.predict(...)`.

## Output Files

`Writer` creates one Parquet file per local rank:

```text
predictions/
  rank-0.parquet
  rank-1.parquet
```

Single-process jobs usually write only `rank-0.parquet`. Distributed jobs write
one file per local rank, and downstream jobs should read all rank files.

Each row contains:

- `inputs`: row metadata captured during encoding.
- `predictions`: a nested struct keyed by schema address.

Target predictions are included for configured targets. Embeddings are included
for nodes configured with `embed=True`.

## Writer Options

| Argument | Meaning |
| --- | --- |
| `path` | Local or mounted output directory for rank-partitioned Parquet files. |
| `flush_every_n_batches` | Optional Parquet writer flush interval. |
| `postprocessor` | Optional callable that rewrites prediction dictionaries before write. |

```python
writer = j2v.Writer(
    "predictions",
    flush_every_n_batches=100,
)
```

## Postprocess Before Writing

Use a postprocessor when downstream consumers need flat columns, renamed
addresses, filtered payloads, or fewer fields.

```python
import json2vec as j2v


def compact_rows(context, predictions):
    label = predictions.get(j2v.Address("record", "label"), {}).get("content", {})
    metadata = context[j2v.TensorKey.metadata]

    return {
        j2v.Address("record", "warehouse_row"): {
            "request_id": [item.get("request_id") for item in metadata],
            "label": label.get("value"),
            "label_probability": label.get("probability"),
        }
    }


writer = j2v.Writer(
    "warehouse-predictions",
    postprocessor=compact_rows,
)
```

Keep returned lists row-aligned with the prediction batch. That lets the writer
build a valid Parquet table.

Writer postprocessors receive context keys such as:

| Key | Meaning |
| --- | --- |
| `"input"` | Encoded Lightning batch. |
| `"batch"` | Alias for the encoded batch. |
| `j2v.TensorKey.metadata` | Raw input metadata for each row. |
| `"batch_indices"` | Lightning batch indices when provided. |
| `"batch_idx"` | Prediction batch index. |
| `"dataloader_idx"` | Prediction dataloader index. |

For more output-shaping patterns, see [Postprocessors](postprocessors.md).

## Streaming Prediction

For file-backed local or S3 inputs, use `StreamingDataModule` with a `predict`
split:

```python
import re

import lightning.pytorch as lit

import json2vec as j2v

datamodule = j2v.StreamingDataModule(
    model=model,
    root="s3://my-bucket/events",
    suffix="parquet",
    predict=re.compile(r"/predict/.*\.parquet$"),
    sharding="chunk",
)

trainer = lit.Trainer(
    accelerator="gpu",
    devices=1,
    callbacks=[j2v.Writer("predictions")],
)

trainer.predict(model=model, datamodule=datamodule)
```

`StreamingDataModule` can read from S3. `Writer` writes to a local or mounted
filesystem path, so copy or sync those rank files downstream when the final
destination is object storage.

## Choosing An Inference Path

| Need | Use |
| --- | --- |
| Debug one request shape | `model.predict([...])` |
| Notebook demo | `model.predict([...])` |
| Batch DataFrame prediction | `PolarsDataModule(..., predict=...)` plus `Writer` |
| Batch file or S3 prediction | `StreamingDataModule(..., predict=...)` plus `Writer` |
| Online service | `j2v.Deployment` |

## Common Mistakes

- Forgetting to configure a `predict` split on the data module.
- Calling `model.predict(...)` in a Python loop for a large batch job.
- Returning non-row-aligned lists from a writer postprocessor.
- Reading only one `rank-*.parquet` file from a distributed prediction job.
- Using a different preprocessor at prediction time than the one used during
  training.

## Where Next

- Use [Data Modules](data-modules.md) to choose Polars or streaming inputs.
- Use [Postprocessors](postprocessors.md) to reshape outputs.
- Use [Learning Modes & Embeddings](../core-concepts/embeddings.md) to configure
  embedding outputs.
- Use the [API Reference](../reference/api.md) for `Writer`.

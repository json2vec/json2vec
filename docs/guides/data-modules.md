# Data Modules

`json2vec` data modules are Lightning `LightningDataModule` implementations.
They load raw records, apply optional preprocessing, batch observations,
tensorize values from the model schema, apply training-time masking and target
pruning, and return Lightning batches.

The data module does not define the model schema. It reads schema state from the
`model` passed to the constructor.

The batch path is:

1. Raw records are read from a DataFrame or files.
2. An optional preprocessor emits processed observations.
3. Observations are sampled and shuffled.
4. Observations are grouped into model batches.
5. Query paths tensorize values from the model schema.
6. `p_mask` hides selected leaf values for reconstruction.
7. `p_prune` and `target=True` hide selected leaf instances for decoding.
8. The encoded batch is handed to the Lightning loop.

## Shared Options

Both data modules use the same core ideas:

| Option | Meaning |
| --- | --- |
| `model` | Supplies schema hyperparameters, `batch_size`, and tensorfield encoding state. |
| `preprocessor` | Callable, registered preprocessor name, or `Preprocessor` object. |
| `**kwargs` | Passed to the preprocessor. |
| `num_workers` | PyTorch dataloader worker count. |
| `persistent_workers` | Keeps worker processes alive between epochs when workers are enabled. |
| `pin_memory` | Enables dataloader pinning when useful for accelerator transfer. |
| `sample_rate` | Samples observations before batching. |
| `observation_buffer_size` | Local shuffle buffer for processed observations. |
| `chunk_batch_size` | Read chunk size. This is separate from `model.batch_size`. |

Most execution options accept either one value or a mapping keyed by
`"train"`, `"validate"`, `"test"`, or `"predict"`.

```python
datamodule = j2v.PolarsDataModule(
    model=model,
    train=train_frame,
    validate=valid_frame,
    num_workers={"train": 8, "validate": 2},
    sample_rate={"train": 0.25, "validate": 1.0},
)
```

## Choosing A Module

| Use case | Recommended module |
| --- | --- |
| Tutorial or notebook | `PolarsDataModule` |
| Unit test or tiny local sample | `PolarsDataModule` |
| Data already in memory as a Polars DataFrame | `PolarsDataModule` |
| Many local files | `StreamingDataModule` |
| S3-backed data | `StreamingDataModule` |
| Distributed training or prediction over large inputs | `StreamingDataModule` |

## PolarsDataModule

Use `PolarsDataModule` for in-memory Polars DataFrames. It is the right default
for examples, notebooks, tests, and small-to-medium local workflows.

```python
import polars as pl

import json2vec as j2v

train = pl.read_ndjson("docs/data/iris.jsonl")
valid = train.head(16)

datamodule = j2v.PolarsDataModule(
    model=model,
    train=train,
    validate=valid,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)
```

You may pass named splits:

```python
datamodule = j2v.PolarsDataModule(
    model=model,
    train=train_frame,
    validate=valid_frame,
    test=test_frame,
    predict=predict_frame,
)
```

Or pass one split mapping:

```python
datamodule = j2v.PolarsDataModule(
    model=model,
    dataframe={
        "train": train_frame,
        "validate": valid_frame,
        "predict": predict_frame,
    },
)
```

Do not pass `dataframe=...` and named split arguments in the same constructor.
At least one split is required.

### Polars Prediction

Configure a `predict` split before using the Lightning prediction loop:

```python
datamodule = j2v.PolarsDataModule(
    model=model,
    predict=predict_frame,
)

trainer.predict(model=model, datamodule=datamodule)
```

For writing outputs to disk, add `j2v.Writer`; see
[Batch Inference](batch-inference.md).

## StreamingDataModule

Use `StreamingDataModule` when data lives in files and should not be loaded into
one in-memory DataFrame. It supports local paths and `s3://...` roots.

Supported suffixes:

- `ndjson`
- `parquet`
- `feather`
- `avro`
- `csv`
- `orc`
- `json`

Split arguments are compiled regular expressions matched against discovered
file paths.

```python
import re

import json2vec as j2v

datamodule = j2v.StreamingDataModule(
    model=model,
    root="data/events",
    suffix="ndjson",
    train=re.compile(r"/train/.*\.jsonl$"),
    validate=re.compile(r"/validate/.*\.jsonl$"),
    predict=re.compile(r"/predict/.*\.jsonl$"),
    sharding="file",
)
```

For S3:

```python
datamodule = j2v.StreamingDataModule(
    model=model,
    root="s3://my-bucket/events",
    suffix="parquet",
    train=re.compile(r"/train/.*\.parquet$"),
    validate=re.compile(r"/validate/.*\.parquet$"),
)
```

### Sharding

`StreamingDataModule` assigns work across dataloader workers and distributed
ranks.

| Sharding | Behavior |
| --- | --- |
| `"file"` | Assigns whole files to workers. |
| `"chunk"` | Assigns read chunks to workers. |
| `"record"` | Assigns individual records to workers. |

Default sharding for streaming data is `"file"`. Use `"chunk"` when individual
files are large and you need more parallelism. Use `"record"` when distribution
needs to be fine-grained and record-order locality is not important.

### Streaming Buffers

| Option | Meaning |
| --- | --- |
| `file_buffer_size` | Shuffles file order before reading. |
| `chunk_batch_size` | Read chunk size and chunk ownership unit. |
| `observation_buffer_size` | Shuffles processed observations before batching. |

When `replacement=None`, training uses replacement sampling and non-training
splits do not. Set `replacement` explicitly when you need different behavior.

!!! Note
    `StreamingDataModule` expects compiled regex patterns, not glob strings.
    Use `re.compile(...)` for split patterns.

## Preprocessors

Both data modules accept the same preprocessor forms:

```python
datamodule = j2v.PolarsDataModule(
    model=model,
    train=records,
    preprocessor=my_preprocessor,
    request_time="2026-05-31",
)
```

The keyword arguments after data module options are passed to the preprocessor.
Use this for stable input shaping such as type normalization, sorting, windowing,
or deriving fields before query paths run.

## Schema Mutation

Data modules keep a reference to the model when possible. If you mutate the
model schema between Lightning runs, the data module reads the current
hyperparameters from the model. If you detach or replace the model, rebuild the
data module so it uses the intended schema and tensorfield encoding context.

## Where Next

- Use [Training With Lightning](lightning.md) for the execution model.
- Use [Batch Inference](batch-inference.md) for `trainer.predict(...)` and
  `j2v.Writer`.
- Use [Preprocessors](preprocessors.ipynb) for input-side transformations.
- Use [Query Paths](../core-concepts/querypaths.md) to map processed records to
  leaf fields.

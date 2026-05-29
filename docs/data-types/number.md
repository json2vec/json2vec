# Number

Use `Number` for continuous scalar values: prices, counts, measurements,
scores, datetime durations, and other numeric features.

```json
{
  "amount": 42.75,
  "days_since_transaction": 3
}
```

```python
import json2vec as j2v

amount = j2v.Number(
    "amount",
    jitter=0.02,
    objective="huber",
    p_mask=0.10,
)
```

Use `Number` only when numeric distance is meaningful. Zip codes, merchant IDs,
product codes, account IDs, and class labels should usually be `Category` or
`Entity`, even when they look numeric.

## Input Values

`Number` expects scalar numeric values. `None` is encoded as a null state, and
missing array positions are encoded as padded state. For repeated values, place
the field inside an `Array`.

Numeric content is normalized online by the embedder before Fourier features are
computed, so models do not require pre-normalized inputs.

Date-derived numbers are appropriate when elapsed time is the signal. For
example, compute `days_since_transaction` in preprocessing when risk depends on
the difference between inference time and transaction time.

Avoid raw timestamps unless absolute time is truly the intended signal. Raw
timestamps can let the model memorize collection windows, rollout dates, policy
changes, label leakage, or train/test split boundaries, and they may drift when
future inference dates fall outside the training range.

## Examples

Common number fields include:

- Monetary values such as amount, price, balance, or tax.
- Counts and quantities such as item count, attempts, clicks, or stock on hand.
- Measurements such as length, weight, intensity, duration, or distance.
- Scores and rates where numeric distance is meaningful.

## Configuration

| Option | Default | Notes |
| --- | --- | --- |
| `jitter` | `0.0` | Adds training-time uniform noise after normalization to prevent overfitting. Must be `>= 0.0`. |
| `n_bands` | `8` | Advanced: negative exponent bound for log-spaced Fourier frequencies. Must be positive. |
| `offset` | `4` | Advanced: positive exponent bound for log-spaced Fourier frequencies. Must be positive. |
| `alpha` | `None` | Optional exponential update rate for the online normalizer. `None` uses cumulative statistics. |
| `objective` | `"mae"` | Content reconstruction objective: `"mae"`, `"mse"`, or `"huber"`. |

The total number of Fourier frequencies is `n_bands + offset + 1`.

## Shared Preprocessing State

`Number` keeps normalization state on the field embedder: mean, variance, and
count. During training, valued inputs update those statistics before the Fourier
features are computed. During validation, test, and prediction, the learned
normalization values are reused without updating them.

With `alpha=None`, updates are cumulative over observed training values. With an
`alpha` value, the normalizer uses exponential moving updates. The normalization
buffers are part of the model state and are used again after loading a
checkpoint.

## Target Behavior

When a `Number` is masked or used as a target, the decoder predicts:

- `state`: probabilities for `valued`, `null`, `padded`, and `masked`.
- `content`: a scalar numeric value.

The content loss is computed in normalized units, while tracked `mae` and
`rmse` metrics are reported in the original value scale.

## Prediction Output

`Model.predict(...)` returns a state probability map plus numeric content:

```python
{
    "record/amount": {
        "state": {"valued": ..., "null": ..., "padded": ..., "masked": ...},
        "content": ...,
    }
}
```

## Notes

Use `Category` for numeric-looking identifiers or class labels. `Number` assumes
the magnitude and distance between values are meaningful.

There are some cases in which users may wish to use binned numerical data or CDFs of input values.
Both can be implemented as a custom data type, or using custom preprocessors.
# Entity

Use `Entity` for local identity matching inside repeated records. It is useful
when the model should notice that two items in the same encoded batch share the
same identifier, without maintaining a global vocabulary for that identifier.

```json
{
  "transactions": [
    {"account_id": "acct_1", "amount": 20.0},
    {"account_id": "acct_1", "amount": 35.0},
    {"account_id": "acct_2", "amount": 11.0}
  ]
}
```

```python
import json2vec as j2v

transactions = j2v.Array(
    j2v.Entity("account_id", topk=[5]),
    name="transactions",
    max_length=16,
)
```

Use `Category` when identifiers should be stable labels learned across training
and prediction. Use `Entity` only when equality inside the current repeated
context is the useful signal.

## Input Values

`Entity` expects hashable scalar values such as strings or integers. `None` is
encoded as a null state, and missing array positions are encoded as padded
state.

Entity IDs are local to the tensorized batch. The same raw value receives the
same local token within that batch, but token IDs are not stable across batches
or checkpoints.

## Examples

Common entity fields include:

- Repeated IP addresses, device IDs, hostnames, or user identifiers in one session.
- Tokenized card fingerprints, account IDs, or payment instruments within one record.
- Transaction counterparties that may repeat across line items.
- Supplier, customer, warehouse, or route identifiers that should be matched locally.

## Configuration

| Option | Default | Notes |
| --- | --- | --- |
| `topk` | `[]` | Optional top-k identity accuracy metrics. Values must be positive and not `1`. |

An `Entity` field must have at least two slots per observation. In practice that
means placing it under an `Array(max_length=...)` greater than `1`, or otherwise
configuring the model root shape to provide repeated values.

## Target Behavior

When an `Entity` is masked or used as a target, the decoder predicts:

- `state`: probabilities for `valued`, `null`, `padded`, and `masked`.
- `content`: a feature vector scored against the batch-local entity codebook.

Top-k metrics compare the predicted vector against the local codebook. Large
`topk` values are skipped when the local codebook is smaller than `k`.

## Prediction Output

`Entity` currently trains and reports losses and accuracies, but it does not
emit user-facing `Model.predict(...)` payloads. It is primarily an internal
representation for learning identity relationships.

## Notes

Use `j2v.Category` when identifiers should map to a persistent global vocabulary and
be emitted as labels. Use `j2v.Entity` when only equality relationships within the
current repeated context matter, or there are simply too many unique global values to track.

Users may also consider defining a "superbloom" style custom extension to maintain a larger set of unique categorical values without the linear memory costs associated with `j2v.Category`.

# Field Stacking

Field stacking is a schema and query pattern for repeated semantic roles. It
maps multiple source fields into one repeated schema field so they share
datatype parameters, vocabulary, decoder behavior, and embedding behavior.

Use it when source fields represent the same kind of thing in different roles:
origin and destination, buyer and seller, payer and payee, source and target, or
before and after values.

## Basic Pattern

Without stacking, two locations become two separate categorical fields:

```python
import json2vec as j2v

model = j2v.Model.from_schema(
    j2v.Category("origin", max_vocab_size=4096),
    j2v.Category("destination", max_vocab_size=4096),
    name="itinerary",
    d_model=64,
    n_layers=2,
    n_heads=4,
)
```

That is simple, but the model learns separate parameters for
`itinerary/origin` and `itinerary/destination`.

With stacking, the source fields become repeated values under one schema field:

```python
model = j2v.Model.from_schema(
    j2v.Array(
        j2v.Category(
            "location",
            query="[*].[origin, destination]",
            max_vocab_size=4096,
        ),
        name="locations",
        max_length=2,
    ),
    name="itinerary",
    d_model=64,
    n_layers=2,
    n_heads=4,
)
```

For a raw record like:

```json
{
  "origin": "IAD",
  "destination": "SFO"
}
```

the query `[*].[origin, destination]` produces two location values for the
`locations` array. The modeled field is `itinerary/locations/location`, so both
positions share one categorical vocabulary.

## Buyer And Seller Accounts

Stacking also works for repeated identity roles.

```python
model = j2v.Model.from_schema(
    j2v.Array(
        j2v.Entity(
            "account_id",
            query="[*].[buyer_account_id, seller_account_id]",
        ),
        name="parties",
        max_length=2,
    ),
    j2v.Number("amount"),
    name="payment",
    d_model=64,
    n_layers=2,
    n_heads=4,
)
```

Use `Entity` when the useful signal is local equality inside the observation,
for example whether the same account appears in multiple roles or events. Use
`Category` when the identifier should be learned as a persistent global label.

## Preserve Roles When They Matter

Stacking removes role-specific field names. If the role carries important
meaning, add it back explicitly. A short preprocessor is often clearer than a
dense query.

```python
import json2vec as j2v


def stack_parties(record: dict) -> dict:
    return {
        **record,
        "parties": [
            {"party_id": record["buyer_account_id"], "role": "buyer"},
            {"party_id": record["seller_account_id"], "role": "seller"},
        ],
    }


model = j2v.Model.from_schema(
    j2v.Array(
        j2v.Entity("party_id"),
        j2v.Category("role", max_vocab_size=8),
        name="parties",
        max_length=2,
    ),
    j2v.Number("amount"),
    name="payment",
    d_model=64,
    n_layers=2,
    n_heads=4,
)
```

This keeps the shared identity representation while letting the model distinguish
buyer from seller.

## Before And After Values

Stacking can represent state changes:

```python
model = j2v.Model.from_schema(
    j2v.Array(
        j2v.Category(
            "status",
            query="[*].[previous_status, new_status]",
            max_vocab_size=64,
        ),
        name="status_values",
        max_length=2,
    ),
    name="profile_change",
    d_model=32,
    n_layers=1,
    n_heads=4,
)
```

Add a role marker such as `before` and `after` if direction matters.

## Check The Shape

Before training, validate that the query produces the repeated shape expected by
the schema. For top-level stacking, the request query usually looks like:

```python
query = "[*].[origin, destination]"
```

For values inside an array, include the parent path:

```python
query = "[*].segments[*].[origin, destination]"
```

The resulting values should line up with the `Array(max_length=...)` that owns
the stacked field.

## When To Use It

Use field stacking when:

- The values have the same datatype and should share parameters.
- You want one vocabulary for a concept that appears in multiple roles.
- Role-specific meaning is optional or can be modeled with an explicit role
  field.
- You want embeddings at the shared concept level.

## When Not To Use It

Do not stack fields just because their storage types match. Airport locations,
merchant categories, and fraud labels may all be strings, but they are not the
same semantic field. Do not stack values if sharing a vocabulary would hide
meaning that the model needs. Do not use a clever query when a preprocessor
would be easier to test and maintain.

## Where Next

- Use [Query Paths](../core-concepts/querypaths.md) for request query conventions.
- Use [Array](../data-types/array.md) for repeated context behavior.
- Use [Preprocessors](preprocessors.ipynb) when role-aware stacking needs Python
  logic.
- See the [Device Tenure case study](../case-studies/device-tenure.md) for a
  repeated-identity pattern in event histories.

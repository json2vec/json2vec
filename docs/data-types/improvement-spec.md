# Data Types Documentation Improvement Spec

## Purpose

Improve the built-in data type docs for first-time JSON2Vec users. The current
pages are technically dense and mostly accurate, but they assume users already
know how to choose between tensorfields, how query inference works, and which
types produce user-facing predictions. The revised docs should answer three
questions quickly:

1. Which data type should I use for this source value?
2. What input shape and schema declaration does the type expect?
3. What happens if I use the field as a feature, reconstruction target, or
   prediction target?

## Non-Goals

- Do not change tensorfield behavior or public APIs as part of this doc pass.
- Do not turn the data type docs into a full schema tutorial. Link to
  `guides/model-schemas.md` for the complete JMESPath explanation.
- Do not document every internal tensor shape unless it directly prevents user
  confusion.

## Audience Assumptions

New users are likely to arrive with ordinary JSON records and ask practical
questions:

- Is an integer ID a `Number`, `Category`, or `Entity`?
- Is a list of labels a `Set`, an `Array(Category)`, or a `Text` field?
- If a timestamp is available, should it be `DateParts` or `Number`?
- If I set `target=True`, will `Model.predict(...)` return a decoded value?
- Do I need to write `query=` for fields inside `Array`?
- Why are `null`, missing values, padded array slots, and masked values treated
  differently?

The revised docs should make those answers visible before advanced options.

## Cross-Cutting Problems

### 1. Missing Decision Model

The overview lists constructors but does not teach selection. Users need a
decision table organized by source semantics, not by implementation class.

Add a "Choose A Data Type" section to `index.md`:

| Source value | Recommended type | Use a different type when |
| --- | --- | --- |
| Continuous scalar | `Number` | Numeric value is an ID, code, or class label. |
| One label | `Category` | The label only needs equality matching within each batch; use `Entity`. |
| Zero or more labels | `Set` | Labels have nested attributes or order; use `Array`. |
| Repeated objects | `Array` | Repetition is only an upstream storage artifact; preprocess or flatten. |
| Timestamp/calendar value | `DateParts` | Elapsed time or monotonic distance matters; use `Number`. |
| Local repeated identity | `Entity` | The ID must be stable across training and prediction; use `Category`. |
| Precomputed dense vector | `Vector` | JSON2Vec should compute embeddings from text; use `Text`. |
| Raw string for semantic encoding | `Text` | The string is a small bounded label; use `Category` or `Set`. |

### 2. Page Template Is Inconsistent

Each data type page should follow the same user-facing template:

1. "Use When" and "Avoid When" bullets.
2. Minimal `json`-like input example.
3. Minimal schema example.
4. Inferred query or explicit query note.
5. Input state behavior: valued, null, padded, masked.
6. Feature behavior.
7. Target and `Model.predict(...)` behavior.
8. Common mistakes.
9. Configuration table, split into "common" and "advanced" where useful.

This order keeps internals available without forcing new users through them
before they understand the type.

Every page must include at least one fenced `json` example that looks like a
source record or a list of source records. These examples should be valid JSON
when possible, but they may be "JSON-like" when the concept is easier to show
with comments or placeholder values. The matching Python schema should appear
immediately after the input example so users can map values to fields.

### 3. Prediction Support Is Not Obvious

Some tensorfields train decoders but do not emit public prediction payloads.
The overview should include a prediction matrix:

| Type | Public `Model.predict(...)` content | State probabilities | Notes |
| --- | --- | --- | --- |
| `Number` | Yes, scalar value | Yes | Metrics are reported in original value scale. |
| `Category` | Yes, best label plus optional top-k candidates | Yes | Unknown bucket is internal only. |
| `Set` | Yes, per-label probabilities, optionally thresholded | Yes | Thresholding should reduce API response size when configured. |
| `Vector` | Yes, reconstructed vector | Yes | Should follow the same optional-state contract as scalar fields. |
| `DateParts` | No | No public payload | Trains losses and accuracies only. |
| `Entity` | No | No public payload | Batch-local identity representation. |
| `Text` | No | No public payload | Reconstructs frozen encoder embeddings, not text. |
| `Array` | No direct payload | No | Child fields may emit predictions. |

### 4. Query Inference Needs Repetition

The array and entity examples mix inferred and explicit child queries. New users
may conclude that `Array` children usually require manual `query=...`.

Requirements:

- In every page with nested examples, show the no-`query` version first when it
  is valid.
- Add a short callout: "Field `name` controls the public schema name. `query`
  controls where values are read. When omitted, JSON2Vec infers the request
  query from the field and parent array names."
- For explicit `query=...`, state why the example needs it.

### 5. Shared Options Are Overwhelming

The shared options table mixes basic schema metadata, training corruption,
decoder configuration, and embedding extraction. Split it into sections:

- Schema identity: `name`, `query`, `description`, `active`.
- Training roles: `target`, `p_mask`, `p_prune`, `weight`.
- Outputs: `embed`.
- Advanced decoder options: `pooling`, `n_heads`, `dropout`, `n_linear`.

Add plain-language definitions:

- `target=True`: hide this field from input and train the model to reconstruct
  it.
- `p_mask`: randomly hide individual values during training.
- `p_prune`: randomly hide whole field instances during training.
- `embed=True`: return this node from `Model.embed(...)`; it does not make a
  field a supervised target.

### 6. Null, Missing, Padded, And Empty Values Need Examples

The state explanation is useful but abstract. Add a small concrete example in
`index.md`:

```python
[
    {"tags": ["vip"]},
    {"tags": []},
    {"tags": None},
    {},
]
```

Explain how this differs for `Set`, `Category`, and fields inside an `Array`.
This is especially important because an empty `Set` is valued content, while
`None` is null and a missing array slot is padded.

Also explain why these states are encoded separately from content values. For a
numeric field, a single scalar cannot safely represent both the observed value
and why it may be unavailable:

- A sentinel such as `0`, `-1`, or `999999` can collide with real values and can
  distort normalization.
- `NaN` may preserve "not a number" but does not distinguish null source data,
  padded array slots, and training-time masks.
- A masked value usually has a real target value hidden from the model, while a
  null value has no content target and a padded slot is structural absence.

The docs should frame `state` as a separate modeling task: the model predicts
whether the value exists, and only then should users interpret the content
prediction.

## Page-by-Page Requirements

### `index.md`

Add:

- A task-oriented decision table before the constructor table.
- A "Same Value, Different Semantics" section:
  - `"12345"` as `Number` only if magnitude matters.
  - `"12345"` as `Category` if it is a stable global ID or code.
  - `"12345"` as `Entity` if only repeated equality within a batch matters.
  - `["red", "sale"]` as `Set` if they are unordered labels.
  - `[{"name": "red"}, {"name": "sale"}]` as `Array` if each item has fields.
- The prediction support matrix.
- Reorganized shared options.

Clarify that serialized schema `type` values are lower-case, while Python users
usually use constructor names from the package root.

### `array.md`

Add:

- A JSON input example beside the schema example.
- A visual explanation of `max_length`: retained slots, truncated overflow, and
  padded missing slots.
- A warning that `Array` is for repeated child objects, not scalar vectors or
  unordered labels.
- A "Queries In Arrays" section with inferred child queries and a separate
  explicit-query example.
- A "When To Flatten Instead" section for melted tabular data and upstream
  storage artifacts.

Clarify:

- `Array` has no direct `target=True` workflow and does not emit prediction
  payloads.
- `p_mask` and `p_prune` are stored on the array node, but runtime corruption is
  applied to active child tensorfields.
- Nested arrays add repeated dimensions and increase padding/truncation
  complexity.

### `number.md`

Add:

- "Use `Number` only when numeric distance is meaningful."
- Examples of numeric-looking values that should not be numbers: zip codes,
  merchant IDs, product codes, account IDs, diagnosis classes.
- A derived-date example, such as `days_since_transaction`, where the source
  date is converted into a numeric duration relative to the inference or
  scoring time.
- A small before/after prediction example that shows scalar content and state
  probabilities.

Clarify:

- Normalization is learned from valued training inputs and reused for
  validation, test, and prediction.
- `jitter` is training-only noise after normalization.
- `objective` changes reconstruction loss, not the public prediction shape.
- Date-derived numbers should represent elapsed time, age, or recency when that
  duration is the signal. For example, use `Number("days_since_transaction")`
  when fraud risk depends on how many days have passed between the transaction
  date and the time of inference.
- Avoid raw timestamps as `Number` unless absolute time is truly intended.
  Absolute timestamps can encourage the model to memorize training-window
  artifacts, calendar cutoffs, rollout dates, label leakage, or dataset split
  boundaries. They can also drift badly when future inference dates fall outside
  the range seen during training.
- `n_bands` and `offset` control Fourier feature frequencies. The current names
  are easy to misread because `n_bands` is also used as the negative exponent
  bound and the total frequency count is `n_bands + offset + 1`; document this
  as an advanced option.

### `category.md`

Add:

- A "Category vs Set vs Entity" comparison near the top.
- A JSON input example with a string label and a numeric-looking label.
- A small output example for `topk=[3]`.

Clarify:

- `Category` represents exactly one label when the field is valued.
- Labels should be stable strings after preprocessing.
- The vocabulary grows only during training and stops at `max_vocab_size`.
- Unknown labels after training remain `valued` but route to an internal
  unavailable bucket.
- `p_unavailable` simulates unknown labels during training; it is not dropout
  over field presence.
- `n_bands` should be removed from the `Category` request model without a
  backwards-compatibility shim. It is unused by the built-in embedding-based
  category encoder and should not appear in docs, schemas, examples, tests, or
  generated API references.

### `set.md`

Add:

- A "Set vs Array" comparison:
  - `Set("tags")` for unordered labels with no per-label attributes.
  - `Array(Category("tag"), Number("score"))` when each tag has extra data.
- A state example that distinguishes empty iterable, `None`, missing value, and
  repeated labels.
- A prediction example that explains per-label probabilities.

Clarify:

- A string input becomes one label, not a sequence of characters.
- Duplicate labels collapse into one active vocabulary entry.
- Unknown labels use the internal unavailable slot but are not emitted as
  predicted labels.
- Add an optional prediction decision threshold to `Set`. When configured,
  `Model.predict(...)` should only emit labels whose probability is at or above
  the threshold. This can reduce response size for API requests with large
  vocabularies.
- Use `threshold: float | None = None` as the schema option name.
  Values must be between `0.0` and `1.0`, inclusive.
- If no threshold is configured, preserve the current per-label probability map
  so applications can choose their own operating point.
- Document threshold semantics clearly: it is an inference-output filter, not a
  training loss change.

### `dateparts.md`

Add:

- A "DateParts vs Number" section:
  - `DateParts` for recurrence and seasonality.
  - `Number` for elapsed time, age, recency, durations, or ordering.
- Several input examples: Python datetime, ISO-like string with `pattern`, and
  date-only values if supported by NumPy coercion.
- A table documenting bucket conventions, including whether values are
  zero-based or one-based and whether `week_of_year` is ISO-compatible.

Clarify:

- Use `DateParts` when the signal is tied to calendar buckets, such as hour of
  day, day of week, month, seasonality, business cycles, or operational
  schedules.
- Use a derived `Number` when the signal is the difference between two dates,
  such as the number of days between the inference time and the observed date
  of a transaction. This should usually be computed in preprocessing so the
  model receives a stable scalar like `days_since_transaction`.
- Do not pass raw timestamps by default. A raw timestamp can let the model learn
  dataset-specific time boundaries instead of general patterns, especially when
  labels are correlated with collection period, product launch date, policy
  changes, or train/test split time.
- Internal precision is minutes; seconds may parse but no second-level part is
  exposed.
- Time zone handling should be explicit. If JSON2Vec relies on NumPy
  `datetime64[m]` without a timezone policy, say so and recommend preprocessing
  to a consistent timezone.
- Accept friendly date-part names in schemas. Normalize inputs such as
  `"day_of_week"`, `"day of week"`, `"Day-Of-Week"`, `"DAY_OF_WEEK"`, and
  `"DayOfWeek"` to the same canonical enum value.
- Prefer deterministic normalization over fuzzy automatic matching:
  casefolding, CamelCase splitting, and replacing non-alphanumeric runs with
  underscores are enough for the known date-part names. Use fuzzy matching only
  for error suggestions when an input cannot be normalized unambiguously.
- Recommended implementation: keep this dependency-free with `re`,
  `str.casefold()`, and a small alias map. If a library is desired for better
  suggestions, use standard-library `difflib.get_close_matches`; avoid adding a
  runtime dependency just to humanize casing.
- Display friendly labels in docs and error messages, such as "day of week",
  while keeping serialized schema values canonical and stable.
- `DateParts` does not emit user-facing prediction payloads today.

### `entity.md`

Add:

- A stronger warning that `Entity` is batch-local identity matching, not a
  persistent ID vocabulary.
- A no-explicit-query array example first:

```python
transactions = j2v.Array(
    j2v.Entity("account_id", topk=[5]),
    name="transactions",
    max_length=16,
)
```

- A comparison table:
  - `Entity`: "same value appears again in this encoded batch/context."
  - `Category`: "this value is a stable label learned across training."
  - `Number`: "distance between IDs is meaningful."

Clarify:

- The same raw ID can receive different local token IDs across batches and
  checkpoints.
- `Entity` requires at least two slots per observation because singletons have
  no useful identity-matching task.
- It trains losses and top-k identity metrics but does not emit public
  prediction payloads.

### `vector.md`

Add:

- A JSON input example with exactly `n_dim` values.
- A "Vector vs Array vs Text" comparison:
  - `Vector` for fixed-width dense numeric inputs.
  - `Array(Number(...))` for repeated scalar measurements with item structure.
  - `Text` when JSON2Vec should compute embeddings from strings.
- A prediction example that includes both `state` probabilities and `content`.

Clarify:

- Values must be 1D and exactly `n_dim`.
- `objective` changes vector reconstruction loss only.
- `Vector` does not normalize values, build vocabularies, or infer dimensions.
- `Vector` should be optional like other leaf fields: null source values, padded
  array slots, and masked targets should be represented with state values.
- The built-in vector decoder should emit state logits/probabilities as well as
  content. For non-valued states, content should default to a zero vector so API
  consumers have a stable shape without confusing zeros for observed content.

### `text.md`

Add:

- A first-screen warning: `Text` is semantic feature encoding, not text
  generation.
- A "Text vs Category" section:
  - `Text` for free-form language where semantic similarity matters.
  - `Category` for bounded labels such as merchant names when exact identity is
    the desired signal.
- A small resource note: optional dependency, Hugging Face model loading,
  memory cost, and offline mode with `local_files_only`.

Clarify:

- The encoder is frozen and cached; JSON2Vec trains against hidden embeddings.
- `max_length` truncates text through the tokenizer.
- `encoder_pooling="pooler"` requires `pooler_output`; `"cls"` and `"mean"`
  require `last_hidden_state`.
- `Text` does not emit user-facing prediction payloads or generated strings.

## Acceptance Criteria

- Each page answers "when should I use this?" before listing advanced options.
- Every data type page includes a concrete source JSON example and matching
  Python schema snippet.
- The overview includes both a decision table and a prediction support matrix.
- Pages consistently distinguish feature use, reconstruction target use, and
  public prediction output.
- Array examples default to inferred child queries unless explicit queries are
  the point of the example.
- `Category.n_bands` is removed from the request model and all docs.
- `Vector` prediction output includes state probabilities and keeps zero-vector
  content placeholders for non-valued states.
- `Set` supports an optional prediction threshold that filters emitted labels.
- `DateParts` accepts common human-readable spellings and casings for date-part
  names while preserving canonical serialized values.
- The docs build without broken links after the revised pages are added to the
  MkDocs navigation.

## Suggested Implementation Order

1. Update `index.md` with the decision model, reorganized shared options, state
   example, and prediction matrix.
2. Normalize `array.md`, because it explains the shape and query behavior that
   affects every nested tensorfield.
3. Update `number.md`, `category.md`, and `set.md`, because these are the most
   likely first tensorfields for new users.
4. Update `dateparts.md`, `entity.md`, `vector.md`, and `text.md` with stronger
   comparison sections and prediction-output caveats.
5. Implement the API-aligned changes: remove `Category.n_bands`, add vector
   state prediction, add `Set` threshold filtering, and normalize `DateParts`
   date-part names.
6. Run `uv run mkdocs build --strict` and fix any link or navigation warnings.

## Resolved API Decisions

- Remove `Category.n_bands` outright. The option is unused and should not be
  preserved for backwards compatibility.
- Make `Vector` prediction state-aware. It should emit state probabilities, and
  non-valued vector content should default to a zero vector.
- Add an optional `Set` decision threshold. When provided, the prediction writer
  should omit labels below the threshold to reduce large API responses.
- Add friendly `DateParts` input normalization. Support common casing and
  separator variants deterministically, and reserve fuzzy matching for helpful
  validation errors rather than silent guesses.

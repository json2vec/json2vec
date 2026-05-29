# Text

Use `Text` for string fields encoded by a frozen Hugging Face model. The text
encoder produces a dense representation that JSON2Vec projects into the model
dimension.

```json
{
  "body": "Customer reported a delayed international transfer."
}
```

```python
import json2vec as j2v

body = j2v.Text(
    "body",
    model_name="bert-base-uncased",
    max_length=128,
    encoder_pooling="mean",
)
```

`Text` is semantic feature encoding, not text generation. Use `Category` for
bounded labels such as merchant names or product codes when exact identity is
the desired signal.

## Dependency

The text tensorfield requires the optional `transformers` dependency.

```bash
uv sync --extra text
```

Without that extra, using `type: text` raises an import error when JSON2Vec
tries to load the tokenizer or encoder.

## Input Values

`Text` expects string values. `None` is encoded as a null state, and missing
array positions are encoded as padded state.

Text is tokenized with the configured Hugging Face tokenizer using max-length
padding and truncation. Token IDs and attention masks are stored as tensorfield
content.

## Examples

Common text fields include:

- Titles, descriptions, reviews, tickets, notes, or messages.
- Merchant names, product names, search queries, or support subjects.
- Short explanations or free-form metadata attached to structured events.
- Text that should be encoded as a semantic feature rather than generated as output.

## Configuration

| Option | Default | Notes |
| --- | --- | --- |
| `model_name` | required | Hugging Face model or local path. Must be a non-empty string. |
| `max_length` | `128` | Tokenizer max length. Must be positive. |
| `encoder_batch_size` | `32` | Number of flattened text values encoded per Hugging Face forward pass. |
| `encoder_pooling` | `"cls"` | Pooling mode: `"cls"`, `"mean"`, or `"pooler"`. |
| `objective` | `"l2"` | Hidden-embedding reconstruction objective: `"l1"` or `"l2"`. |
| `revision` | `None` | Optional Hugging Face revision. Blank strings normalize to `None`. |
| `local_files_only` | `False` | Passes through to Hugging Face loading for offline or cached-only use. |

`encoder_pooling="pooler"` requires the selected Hugging Face model to expose
`pooler_output`. The `"cls"` and `"mean"` modes require `last_hidden_state`.

## Target Behavior

When `Text` is masked or used as a target, the decoder predicts:

- `state`: probabilities for `valued`, `null`, `padded`, and `masked`.
- `content`: the frozen encoder hidden embedding for the original text.

The decoder does not generate text tokens. It reconstructs the frozen text
embedding used as the target representation.

## Prediction Output

`Text` currently trains and reports losses and metrics, but it does not emit
user-facing `Model.predict(...)` payloads. Configure it as an input feature or
embedding-reconstruction target, not as a generated-text output.

## Notes

The Hugging Face encoder is cached, run in evaluation mode, and not fine-tuned
by JSON2Vec. Use `encoder_batch_size` to manage memory when many text values are
flattened from nested arrays.

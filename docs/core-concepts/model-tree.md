# Model Tree

The base structure of a model is a tree of `anytree`-backed `pydantic` models.

`Model.from_schema(...)` creates a "root" `Array` node, and any
nested `Array(...)` declarations become "branch" nodes, and tensorfields such as
`Number`, `Category`, and `Text` become "leaves".

Each node has a stable slash-delimited address (`j2v.Address`):

```python
import json2vec as j2v

model = j2v.Model.from_schema(
    j2v.Category("customer_id", active=False, max_vocab_size=100_000),
    j2v.Array(
        j2v.Category("sku", max_vocab_size=2048),
        j2v.Number("quantity"),
        j2v.Number("price", p_mask=0.15),
        name="line_items",
        max_length=32,
        embed=True,
    ),
    j2v.Category("returned", target=True, max_vocab_size=2),
    name="order",
    d_model=64,
    n_layers=2,
    n_heads=4,
    embed=True,
)
```

```text
order                         array, root context encoder, embed
|-- customer_id               category leaf, inactive
|-- line_items                array, branch context encoder, embed
|   |-- sku                   category leaf
|   |-- quantity              number leaf
|   `-- price                 number leaf, masked reconstruction
`-- returned                  category leaf, supervised target
```

`order` is the root context. `j2v.Address("order", "line_items")` is a branching context. The
leaves are the typed requests that read values from input records.

Thus, an order is defined by a customer ID, a list a items purchased, and whether or not the customer returned any of the items.

## Node Kinds

| Node kind | Created by | Runtime role |
| --- | --- | --- |
| Root array | `Model.from_schema(..., name=...)` | Encodes the whole processed observation. |
| Branch array | `Array(..., name=..., max_length=...)` | Encodes a repeated nested context and passes one pooled representation to its parent. |
| Leaf tensorfield | `Number`, `Category`, `Set`, `Vector`, and other tensorfields | Reads source values, stores typed tensors, embeds visible input, and decodes predictions when trained or requested. |

!!! Note
    Tensorfields represent an extensible typing system. There are many builtin data types, with more in development. Users are empowered to create and register their own [custom data types](../data-types/tensorfields.ipynb).

Array nodes are context encoders with cross-attention pooling.
Leaf nodes are the only nodes that bind directly to values with a request-level `query`. See
[Query Paths](querypaths.md) for how leaf queries are inferred.

!!! Note
    Under the hood, the "Root" array is just another branch (`Array`), but has a `max_length=1`.

## Nodes As N-Dimensional Arrays

Every schema node corresponds to an N-dimensional tensor shape. Array nodes add
dimensions. Leaf tensorfields store typed arrays at the shape defined by their
array ancestors.

For the `order/line_items/price` leaf above, the inherited shape is:

```text
order.max_length       -> 1
line_items.max_length  -> 32
leaf shape             -> (1, 32)
```

At runtime, a scalar tensorfield such as `price` has arrays like:

```text
state      (batch, 1, 32)
content    (batch, 1, 32)
trainable  (batch, 1, 32)
```

Some field types add content dimensions. For example, `Vector` content includes
the configured vector width, and `Set` content includes vocabulary dimensions.
The shared rule is that `state` carries the node's value-state array and
`content` carries the type-specific value array.

Embedders convert leaf arrays into `d_model` vectors:

```text
leaf embedding payload  (batch, ..., d_model)
array embedding payload (batch, ..., d_model)
```

The repeated dimensions before `d_model` come from the schema path. Array
encoders pool a child context before handing a representation to the parent.

This extends to all model trees. Users may define multiple branches, and each branch can have child branches of their own:

```text
customer (1)
|
|-- login_sessions (1, 128)
|   |-- login_session_device (1, 128)
|   `-- session_events (1, 128, 36)
|       |-- timestamp (1, 128, 36)
|       `-- event_type (1, 128, 36)
|
|-- transactions (1, 360)
|   |-- transaction_type (1, 360)
|   |-- transaction_amount (1, 360)
|   `-- is_transaction_fraud (1, 360)
|
`-- customer_tenure (1)
```

## Forward Pass

A model forward pass follows the tree from leaves upward, then decodes selected
leaves from their available context.

1. Raw records are preprocessed and encoded into active leaf tensorfields.
2. Training-time masking and pruning mark hidden positions as `masked`, save the
   original values in `targets`, and set `trainable=True` where reconstruction
   loss should be computed.
3. Active, non-target leaves run their tensorfield embedders and send `Parcel`
   objects to their parent arrays.
4. Array encoders run from deepest branches back to the root. Each array
   concatenates child parcels, applies its configured attention layers, pools the
   child context, and sends one parcel to its parent.
5. Arrays configured with `embed=True` emit embedding predictions from their
   pooled payloads.
6. Leaves that are trainable, targets, or configured with `embed=True` run their
   decoders.
7. Losses are computed for decoded leaf predictions that have trainable targets.
   Pure embedding outputs are outputted only at inference time.

## Pooling And Heritage

Pooling happens in two places.

Array pooling compresses a branch context before passing it upward to its parent context. After an
array concatenates its child payloads and runs attention layers, learned-query
cross-attention produces the vector payload for that array node.

Decoder pooling builds a prediction context for a leaf. Each leaf has a
`heritage`: the addresses along its path from the root through the leaf itself.
For `order/line_items/price`, the heritage is:

```text
order
order/line_items
order/line_items/price
```

The decoder gathers the available outgoing parcels from that heritage,
concatenates them, then pools them into the target leaf shape. The default
`pooling="query"` uses learned-query cross-attention. `pooling="mean"` repeats
the mean of the heritage context for each target slot.

This is why a decoded leaf can use information from its own branch, ancestor
contexts, and visible sibling fields. If the leaf itself was masked, its own
parcel still carries masked-state information, while the original value is kept
only in `targets` for loss computation.

## Roles And Mutations

Schema roles change how nodes participate in training and prediction:

| Setting | Effect |
| --- | --- |
| `p_mask=0.15` | Randomly hides individual leaf values for reconstruction. |
| `p_prune=0.15` | Randomly hides whole leaf instances for reconstruction. |
| `target=True` | Always prunes/masks a leaf and decodes it as a supervised target. |
| `embed=True` | Emits an embedding for root, branch, or leaf nodes during prediction. |
| `active=False` | Keeps a leaf in the schema but removes it from encoding, forward passes, losses, and prediction until it is reactivated. |

Model mutations edit the same tree:

| Method | Use |
| --- | --- |
| `model.update(predicate, **values)` | Change selected node attributes such as `weight`, `target`, `p_mask`, `embed`, or `active`. |
| `model.extend(predicate, new_field, ...)` | Add new children under one selected array node. |
| `model.delete(predicate)` | Permanently remove selected nodes and their descendants. |
| `model.reset(predicate)` | Reinitialize selected runtime modules while keeping schema values. |
| `model.override(predicate, **values)` | Temporarily change nodes inside a context manager, then restore them. |

Mutations rebuild the runtime graph and reload compatible state where shapes
still match. They are blocked while Lightning owns an active training,
validation, test, or prediction loop.

Use predicates such as `j2v.where("name") == "price"` or
`j2v.where("type") == "number"` to select nodes. See
[Mutations](mutations.ipynb) for predicate examples and mutation workflows.

## Where Next

- Use [Built-In Data Types](data-types.md) to choose leaf tensorfields.
- Use [Array](../data-types/array.md) for repeated branch contexts.
- Use [Embeddings & Self-Supervised Learning](embeddings.md) for `p_mask`,
  `p_prune`, and `embed=True` patterns.

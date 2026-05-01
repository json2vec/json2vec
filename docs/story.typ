#set heading(numbering: "1.")

== Background

Nearly four years ago, when I had worked at Capital One, I enjoyed a lunch with my organization's VP. He asked me a simple question that defined the following several years of my career and curiosity:

> Is there a better way to model complex business data?

The organization had just spent eighteen months building the data engineering capabilities required to produce a _single_ high-value feature for a tabular (GBM) model. That very same year the team had to scale back a different tabular feature for another model that would have required approximately \$1 million each year in computation alone.

This is the ugly face of predictive modeling that people don't like to talk about. Some problems, at scale, are limited by the modeling frameworks available.

ML practitioners are forced to simplify, reduce, and redact massive amounts of invaluable information. This process of _tabular feature engineering_ is extraordinarily expensive, time consuming, prone to error, and requires continuous monitoring and validation when deployed for real-time predictive modeling at scale.

What my VP was really asking was this:

> Is there a way to model complex business data without tabular feature engineering?

Several organizations have pursued this problem over the last decade. When I was at Capital One, I had implemented several of them while modeling various fraud use cases.

They work. They outperform traditional machine learning solutions. However, up to this point, there is no modern, open-source, extensible framework to make them accessible.

Additionally, they all suffer from a common gap: they are not able to handle multiple hierarchical, nested contexts. As a non-limited example: separately encoding a customer's monthly statements, and their transactions, and their login sessions, and each login sessions' clickstream events. This topic is discussed in more detail @sec:nested-contexts[here].

I am releasing `json2vec`, a modeling framework I have been developing for several years, to help fill this gap.

== Requirements

=== Dynamic Model Architecture Constructed from Schema

The basis of `json2vec` is the schema.

All model architectures can be defined by a schema that defines the necessary "contexts".

For the sake of simplicity, I will use `yaml` to illustrate schemas.

==== Hello World Example

With `json2vec`, you can easily define a basic tabular model with a schema like so:

```yaml
name: record
n_layers: 4
n_heads: 4
context_size: 1
fields:
  - name: x2
    type: category
    max_vocab_size: 1000

  - name: x1
    type: category
    max_vocab_size: 1000
```

This model has a single context, which has expects two tabular inputs (`x1` and `x2`), both of which are of categorical data and may learn up to 1000 unique categorical inputs.
These two categorical inputs are processed as followed:
1. Tokenized
2. Embedded to vectors of width `d_model` (defined elsewhere)
3. Passed to a transformer encoder block (`record`) with 4 layers and 4 heads
4. Pooled together with a cross-attention block

During pretraining, the model will randomly mask values according to a masking rate hyperparameter.
This will randomly mask `x1` and `x2` according to the mask rate, and attempt to impute the masked values.

==== Basic BERT-like Model

Similarly, you can rebuild a model like BERT by nesting a context:

```yaml

name: observation
n_layers: 4
n_heads: 4
context_size: 1
fields:

  - name: context
    n_layers: 8
    n_heads: 4
    context_size: 768
    fields:

      - name: tokens
        description: each unique wordpiece token
        type: category
        max_vocab_size: 20000
```

This "BERT" example is not _exactly_ BERT, because it uses cross-attention to extract information from the nested context instead of a `[CLS]` token. however, it is functionally similar.

Encoding text like this is more of a thought exercise in nested contexts because in practice developers can use a dedicated `text` data type, which uses pretrained `BERT` models downloaded from HuggingFace on the fly.

During pretraining, the model will randomly mask each token, and impute the masked wordpiece token with the surrounding context.

You can utilize other fields in the schema as well:

```yaml

name: observation
n_layers: 4
n_heads: 4
context_size: 1
fields:

  - name: context
    n_layers: 8
    n_heads: 4
    context_size: 768
    fields:

      - name: tokens
        description: each unique wordpiece token
        type: category
        max_vocab_size: 20000

      - name: part_of_speech
        description: part of speech for each word (verb, noun, adjective, etc.)
        type: category
        max_vocab_size: 100

  - name: sentiment
    description: sentiment of message (positive, neutral, negative)
    type: category
    max_vocab_size: 3
```

You can pretrain the model with this schema, and then finetune it by setting the model to never mask out `tokens` but to always mask out `sentiment` and `parts_of_speech` with the following training parameters:

```yaml
name: "my-finetune-job"
task: "fit"
dataset: ...
structure: ...
trainer: ...
pruned:
  - "observation/context/parts_of_speech"
  - "observation/sentiment"
```

This illustrates a unique component of `json2vec`. The inputs and the outputs are defined in the same schema. The same underlying code is handling pretraining and finetuning. This is discussed in more detail @sec:training[here]. 

This integration between pretraining and finetuning enables multi-task models, with each task having different dimensionality. 

Keep in mind, the model supports modification of the schema. You can add and remove fields where required.

==== Basic Chess Encoding

You can also encode chess positions:

```yaml
name: observation
n_layers: 4
n_heads: 4
context_size: 1
fields:

  - name: board
    n_layers: 8
    n_heads: 4
    context_size: 64
    fields:

      - name: piece_type
        description: each unique piece type (pawn, bishop, knight, rook, queen, king) ... empty squares are marked by `None`
        type: category
        max_vocab_size: 6

      - name: piece_color
        description: player colors (black & white) ... empty squared are marked by `None`
        type: category
        max_vocab_size: 2

        # you should probably also add some information around castling rights !

  - name: player_to_move
    type: category
    max_vocab_size: 2

  - name: centipawn_score
    description: centipawn score of position
    type: number
```

With enough training observations, you may pretrain a model to encode chess positions, and then finetune it to learn centipawn scores, which may be used to evaluate chess positions.

=== Hierarchical Context Encoding <sec:nested-contexts>

Many architectures already enable tabular inputs (1D), or a single context of two-dimensional inputs (2D).

`json2vec` uniquely enables multiple contexts, each of which may have their own contexts. I refer to this as hierarchical context encoding.

Hierarchical context encoding is not just a party trick. It is surprisingly useful. For example:
- Every clickstream event among every login session
- Every purchased item among every purchase order

Moreover, it can enable the sharing of vocabulary embeddings that would otherwise not be possible.

Consider the following example of a travel itinerary:

```yaml
name: itinerary
n_layers: 4
n_heads: 4
context_size: 1
fields:

  - name: origin
    type: category
    max_vocab_size: ...

  - name: destination
    type: category
    max_vocab_size: ...
```

This is simple and easy to read, but it is actually harder for the model to understand because it needs to learn distinct embeddings for both `itinerary/origin` and `itinerary/destination`.
However, you can simplify this by stacking the origin and destination into a new context, which will share the embeddings:

```yaml
name: itinerary
n_layers: 4
n_heads: 4
context_size: 1
fields:

  - name: locations
    n_layers: 1
    context_size: 2

      - name: location
        type: category
        max_vocab_size: ...
        query: "[*].[origin, destination]"
```

Now, `itinerary/locations/location` will share parameters. This requires no change to the structure of the data because of the provided `jmespath` query.

`jmespath` querying enables succinct extraction of data from complex `json`-like data structures without needed to modify the data on the fly. Every field requires a `query`, which I have removed from the schemas referenced in the document.

Broadly speaking support for `jmespath` is meant to enable modeling from any data structure, whether a dictionary of lists, a list of dictionaries, a dictionary of dictionaries of lists of lists, or whatever else. It is flexible and fast.

==== Fraud Detection

All the examples so far have been fairly trivial. However, hierarchical context encoding scales as the data becomes more complex. Users may trivially define extremely complex schemas to model complex data structures

The following example uses multiple contexts, one of which is nested inside of another.

```yaml
name: customer
n_layers: 4
context_size: 1
fields:

  - name: transaction
    n_layers: 6
    description: up to 512 most recent trailing transactions
    context_size: 512
    fields:

      - name: type
        description: transaction type (card swipe, ACH, wire, etc.)
        type: category
        max_vocab_size: 20

      - name: amount
        description: transaction amount
        type: number

      - name: timestamp
        type: dateparts
        dateparts:
        - day_of_week
        - day_of_month

  - name: statement
    n_layers: 4
    description: up to five years of trailing monthly statements
    context_size: 60
    fields:

      - name: balance
        type: number

      - name: fees_accrued
        type: number

      - name: total_spent
        type: number

  - name: login_sessions
    n_layers: 1
    description: recent login sessions
    context_size: 24
    fields:

      - name: device
        description: hash of device used for login session - helpful for fraud use case
        type: entity

      - name: region
        type: category
        description: region / state of device used for logic session
        max_vocab_size: 20

      - name: clickstream_events
        n_layers: 2
        description: set of events happening within each login session
        context_size: 128
        fields:

          - name: type
            description: clickstream event type ()
            type: category
            max_vocab_size: 20

          - name: timestamp
            type: dateparts
            dateparts:
            - hour_of_day
            - minute_of_hour
```

This schema may be pretrained on slices of a customer's lifestream (a time-windowed snapshot of observed behavior).
This can be effectively done at scale by streaming customer data, sampling a time window, and then filtering the data down to the time window.
One may yield multiple observations from the same customer, but it is typically prudent to prevent leakage by stratifying training / validation / testing data by a unique customer identifier.

Upon pretraining the model on customer behavior, users may finetune multiple fraud models with different fraud tagging strategies at any level. For example, one may create the field `customer/transaction/is_account_takeover_fraud` at a transaction level and then `prune` it such that the model is focused exclusively on imputing whether or not each individual transaction is indicative of account takeover fraud. Alternatively, one may create the field `customer/is_first_party_fraud` to predict first party fraud at a customer level. 

=== Unified Self-Supervised and Supervised Learning Tasks <sec:training> 

Unification of self-supervised learning with supervised learning simplifies the control flow, loss functions, and logging to all reuse the same resources. Because another requirement of this project is the ability to manage an extensible library of datatypes (categories, numbers, text, dateparts, embeddings, entities, etc.), reusing components is absolutely critical.

The idea is simple: the loss functions are used between self-supervised learning and supervised learning.

During pretraining, all masked values are imputed regardless of their dimensionality. During supervised learning, all "pruned" values are predicted regardless of their dimensionality.
The only difference is masking happens at a value-by-value basis depending on the masking rate. Pruning, however, is enforced for all values of the same field.

This means that the control flow is exactly the same for pretraining and finetuning. The difference between pretraining and finetuning is just a hyperparameter. 

```yaml
- name: "my-pretrain-job"
  task: "fit"
  dataset: ...
  structure: ...
  p_mask: 0.15 # mask 15% of all values
  p_prune: 0.05 # mask 5% of all observations' entire fields
  ...
```

```yaml
- name: "my-finetune-job"
  task: "fit"
  dataset: ...
  structure: ...
  pruned: ["customer/transactions/is_fraud"] # don't mask or prune anything except for this, but always prune this
  ...
```

=== Transfer Learning with Schema Evolution

The schema is the basis of modeling with `json2vec`. The schema is meant to be flexible and adaptable to accommodate changes to upstream data.

If you load an old model checkpoint with an altered schema, the model will initialize any new parameters, and freeze stale parameters.

Fields may be added and removed at a whim because each parent context has a flexible context width.

Enterprise organizations may build foundation models for their organization with only the most generalizable fields, and then share the checkpoints to individual data science teams. These individual teams may then add new use-case specific fields before pretraining some more, and then eventually fine tune to their target(s).

Once the data changes (new schema, or just new customer behavior), the organization can refit their organization foundation model before sharing it to the individual teams, at which they refit as well.

Without schema evolution the organization would need to resort to one or both of the following unpleasant alternatives:
1. Integrate only a subset of fields to maximize generalizability
2. Manage and periodically train multiple foundation models independently (wasting compute)

By utilizing transfer learning with schema evolution, teams can individually adapt foundation models with new fields for their individual use cases.


=== Extensible Datatype Plugin System

==== Online Tokenizer Vocabulary

==== Unified Enumerable State Management

==== Built-In Entity Encoding


=== Explanability is built-in

==== Via Pruning

==== Via Embedding Trees


=== Integrated Querying, Wrangling, and Logging


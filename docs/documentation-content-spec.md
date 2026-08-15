# Documentation Content Revision Spec

- Status: Implemented, with the hypersphere-specific Category work deferred
- Date: 2026-08-01
- Scope owner: Documentation maintainers and public API owners

## Summary

The documentation is visually polished and already explains many individual
features well. The next revision should focus on the content contract:

1. make every behavioral claim agree with the implementation;
2. give a new user a complete path from installation through operation;
3. introduce concepts in the order a reader needs them;
4. give each recurring concept one authoritative page;
5. separate public behavior from implementation detail, and verified capability
   from illustrative or aspirational claims.

This is not a request for a new theme or a wholesale prose rewrite. Existing
examples, diagrams, and page structure should be retained when they serve the
revised story.

Implementation note (2026-08-01): the site, lifecycle guides, factual
corrections, navigation, references, and contract tests in this spec are
implemented against `main` at the audit baseline. At the owner's request, the
`dev/categorical-unit-circle` source integration and its hypersphere/CosFace
documentation remain a later atomic change. Current Category documentation
therefore describes the checked-out linear classifier while preserving the
already implemented rule that unavailable valued content contributes zero
content and retains state `valued`.

## Audit Baseline

This spec was built from:

- the published Quarto sources on `main` at `224ff43`;
- the current public package surface and tensorfield implementations;
- the categorical hypersphere work on `dev/categorical-unit-circle` at
  `a7f81b1`, where it changes a behavior that documentation must describe when
  that branch lands;
- the README and whitepaper where they duplicate or contradict the website.

Implementation behavior must be rechecked at the commit where each docs change
lands. The commit identifiers above identify the audit, not a permanent source
of truth.

## Scope

Included:

- all published pages under `docs/`;
- navigation and learning order in `docs/_quarto.yml`;
- the README where it owns installation or repeats the product introduction;
- `docs/whitepaper.typ` where it states current product behavior;
- examples, output payloads, cross-links, terminology, and capability claims;
- missing guides needed to complete the user journey.

Not included:

- visual redesign, theme work, or cosmetic reformatting;
- changing package behavior merely to make an existing explanation true;
- documenting every private class or implementation helper;
- adding large examples that cannot be run or maintained;
- marketing copy that cannot be connected to a supported workflow or evidence.

## Audiences And Jobs

The docs should explicitly support these readers:

| Reader | Job the docs must help complete |
| --- | --- |
| Evaluator | Understand what problem RelFlow solves, where it differs from a flat model, its maturity, and its constraints. |
| First-time builder | Install the package, build a small model, train it, inspect metrics, save it, load it, and predict. |
| Model designer | Map nested records to branches and datatypes, choose context sizes, avoid leakage, and configure learning roles. |
| Training practitioner | Choose data modules and model/trainer settings, interpret metrics, checkpoint safely, and diagnose failures. |
| Inference engineer | Reproduce preprocessing, understand the output contract, and choose interactive, batch, or online inference. |
| Extension author | Determine whether the public plugin surface is supported and, if so, implement and test a custom tensorfield. |

The primary narrative should optimize for the first-time builder without hiding
the routes for the other readers.

## Desired Reader Story

A reader should be able to move through this sequence without an unexplained
jump:

1. **Why:** hierarchical business records lose useful structure when flattened.
2. **What:** a RelFlow schema is both a data contract and a model blueprint.
3. **How data moves:** raw record to preprocessing, querying, state/content
   tensors, model tree, decoder/loss, and prediction writer.
4. **First result:** build, train, evaluate, save, load, and predict a tiny model.
5. **Design:** choose branches, datatypes, learning roles, embeddings, and model
   capacity for a real problem.
6. **Operate:** preserve the same schema and preprocessing contract in batch or
   online inference, then monitor quality and throughput.
7. **Go deeper:** use masking, field stacking, field importance, extensions, and
   architecture internals when the use case requires them.

The current docs cover parts of all seven steps, but the material is ordered
mostly by internal concepts and stops short of a complete artifact lifecycle.

## Findings And Required Corrections

### P0: Factual Integrity

These items are incorrect or internally contradictory. They should be fixed
before adding new conceptual material.

| Area | Current problem | Required correction |
| --- | --- | --- |
| Home example | The prose says the shown model names its root `order` and refers to `order/returned`, but the constructor omits `name="order"`. | Add the argument or describe the default `record` address. Keep schema and output examples executable together. |
| Model Tree example | The prose says the example order includes a customer ID, but the shown schema does not. | Add the field or remove the claim. |
| Category unavailable values | The main docs describe an extra unavailable embedding/output class. Tensorization uses sentinel index `size`; the embedder converts it to zero content. The implementation has exactly `size` content rows/classes, excludes unavailable values from content accuracy, and applies a uniform content target for unavailable targets. | Describe unavailable as `state=valued` with unavailable/zero embedded content, never as a learned label, state, or emitted class. Keep implementation-specific training details in the Category page. |
| Set unavailable values | The Set page describes a reserved unavailable slot. Unknown labels are omitted from the `size`-wide multi-hot vector, and `p_unavailable` drops known positive bits during training. | Document Set-specific omission/dropout semantics. Explain that valued empty, all-OOV, and fully dropped sets can all have zero content. State that unavailable behavior is datatype-specific. |
| Hash semantics | The type uses batch-salted deterministic hashes rather than a learned vocabulary. | Describe equality across fields in one encoded batch, per-batch salt rotation during training, deterministic inference, quantized hash reconstruction, and the correspondence lost when sibling branches pool independently. |
| DateParts precision | The page claims minute precision and omits seconds. The implementation stores second precision and supports `second_of_minute`. | Add the implemented part, remove the minute-only claim, and document per-part angular error in radians rather than generic content “accuracies.” |
| Branch attention | The Branch page documents only `mha` and `none`; the public enum also includes `gqa` and `mqa`. | Document all supported modes and their intended tradeoffs, or classify unsupported values as non-public. |
| State vocabulary | Core pages omit `other`, most output examples omit it, and the whitepaper invents a separate `pruned` state. There is no `pruned` token: pruning hides input with `masked` while retaining targets/trainability. | Make the Data Types overview authoritative for `valued`, `null`, `padded`, `masked`, and reserved `other`. Explain pruning as an operation, not a value state. |
| Prediction payloads | Datatype examples are incomplete without saying so. They usually omit `state.other` and the public `inferred` mask. | Show one complete canonical payload, then label all smaller payloads as excerpts and link to it. Explain that `inferred` is true at positions masked in prediction input, while a visible leaf decoded for `embed=True` can carry `inferred=false`. |
| Category capacity examples | Several tutorials allocate one more slot than the labels shown (`size=4` for three Iris labels and `size=3` for binary examples), reinforcing the false extra-bucket model. | Use exact capacity in introductory examples or explicitly label extra capacity as future-vocabulary headroom and explain its cost. |
| Invalid top-k example | AI / Expert Quickstart mutates a `Category(size=2)` to `topk=[2]`, but `topk` must be less than `size`. | Use a legal value on a larger category or remove the mutation. Execute this example as a docs contract test. |
| Dynamic masking | The prose says `p_mask` samples observed values, while leaf masking can select null and padded positions; branch masking excludes padding but may select null positions. | Make an explicit product decision: change the implementation to valued-only selection or document and test the exact position/state behavior. Use one term consistently. |
| Device Tenure case study | The schema puts separate `Hash` fields under sibling branches, while the narrative claims the model can match that identity across those branches. The page later warns that this does not happen automatically. | Restructure the example around a shared repeated context/stacked field, or remove the cross-branch identity claim. Keep hypotheses distinct from demonstrated capability. |
| Whitepaper category/state model | The whitepaper repeats the old unavailable bucket and separate `pruned` state. | Align its high-level architecture story with the canonical Data Types and Category pages; do not duplicate option-level details. |

Every correction should add or update a test when a small executable example can
protect the documented contract.

### P0: Trust And Evidence

The home page claims training over billions of observations and throughput over
100,000 observations per second without workload, hardware, precision, batch
size, or benchmark links. Either:

- publish a reproducible benchmark with those dimensions and name it next to
  the claim; or
- soften the language to a qualitative scalability statement.

Apply the same rule throughout the docs: identify statements as verified
behavior, measured results, illustrative hypotheses, or future direction. A
case-study sketch must not read like an evaluated result.

## Missing Content

### P0: Complete The Basic Lifecycle

The following material is required for the main reader journey:

| Deliverable | Content contract |
| --- | --- |
| Installation and compatibility | Supported Python versions, package installation channel, optional extras, CPU/GPU expectations, a one-command import/version check, and a clear distinction between user installation and contributor checkout. If the package is intentionally checkout-only, say so. |
| End-to-end first model | One small runnable path through build, train, validation metric, save, load, and prediction. Explain what successful output looks like and keep nested modeling as the next step rather than a second competing quickstart. |
| Data-flow mental model | One canonical record-to-output walkthrough: raw record → preprocessor → query → state/content tensorfield → model tree → decoder/loss → writer/postprocessor. Define user-facing terms before `parcel`, `heritage`, or other internals. |
| Evaluation and metrics | Metric naming by schema address and stage, weighted total-loss composition, state versus content metrics, class imbalance, thresholds/top-k, selecting checkpoint monitors, validation leakage, and how to tell whether the model learned anything useful. |
| Model lifecycle | `save`, `load`, checkpoint restoration, resuming, vocabulary/normalizer preservation, schema compatibility, `RollbackCheckpoint`, and train/serve artifact parity. Clarify lightweight RelFlow persistence versus Lightning checkpoints with optimizer/training state. |
| Online serving | A dedicated supported path for `Deployment`, request/response contracts, batching, accelerator/backend choices, preprocessors/postprocessors, concurrency, failure behavior, and production caveats. If serving is experimental, mark it prominently. |

### P1: Make Design And Operation Deliberate

| Deliverable | Content contract |
| --- | --- |
| Model configuration | Explain `d_model`, `n_layers`, `n_heads`, attention mode, dropout, branch length, optimizer/scheduler, and batch size as decisions with costs—not merely copied constructor values. Include starting points, constraints, and signals to change them. |
| Performance and distributed operation | Padding and branch-length cost, vocabulary/cardinality cost, attention modes, data workers/buffers, precision, sharding, DDP vocabulary synchronization, profiling, and throughput measurement. Connect tuning advice to emitted metrics. |
| Temporal validation and leakage | Time cutoffs, identity/entity splits, `as_of` windows, target leakage through nested history, and consistent preprocessing across splits. The case study may demonstrate these rules but should not be their only home. |
| Schema mutation | Turn the mutation summary into a runnable workflow for `select`, `update`, `extend`, `delete`, `reset`, and temporary `override`. Cover predicates, parent/descendant rules, state reuse or reinitialization after shape changes, active-loop restrictions, and saving the result. |
| Troubleshooting | Query/shape mismatches, missing targets, overflow, empty/no-trainable schemas, full vocabularies, checkpoint incompatibility, distributed data duplication, and unexpected output addresses. Each symptom should point to a diagnostic and likely fix. |
| Extension authoring | If the plugin API is supported, document the request/tensorfield/embedder/decoder contract, registration, loss/writer hooks, tests, serialization, and compatibility expectations. Otherwise mark those exports as experimental rather than advertising extensibility without a path. |

### P2: Improve Discoverability And Proof

- Add an API/configuration reference for intended public exports and constructor
  options. Generate tables from the implementation where practical.
- Publish at least one fully runnable case study with a baseline, split design,
  metric definition, result, and limitations.
- State the documentation/version compatibility policy and support level of
  serving, plugins, and distributed workflows.
- Add a lightweight troubleshooting index that can grow from recurring issues.
- Correct smaller contract language during page revisions: Number jitter is the
  difference of two uniforms (triangular noise), not uniform noise; ordinary
  dictionary transformation helpers should not be called RelFlow
  preprocessors unless they implement the public processor contract; remove
  copy errors such as “Supplies schema schema.”

## Narrative And Information Architecture

### Current Flow Problems

- Home, Motivation, and README repeat the same positioning instead of handing
  the reader from a short promise to a detailed rationale.
- Getting Started assumes a repository checkout and bundled data, so it is not
  an external installation path. It also ends before evaluation and persistence.
- AI / Expert Quickstart combines an expert reference with instructions for AI
  agents and duplicates much of Getting Started.
- Model Tree introduces architecture internals before the basic data-flow and
  value-state mental models are established.
- Query Paths is comprehensive but long for the first learning path; common
  inference should come before advanced JMESPath patterns.
- Dynamic Masking appears too early in the main sequence relative to evaluation,
  checkpointing, and inference.
- The navbar and sidebar expose different conceptual groups; Motivation, AI /
  Expert Quickstart, and Field Stacking are not equally discoverable.
- Datatype pages use similar but inconsistent templates, depth, payload examples,
  terminology, and next-step links.
- Important concepts such as `target=True`, `p_mask`, `p_prune`, and `embed=True`
  are fully re-explained on several pages, which makes drift likely.

### Proposed Learning Order

The exact navigation labels may change, but the content should be ordered as
follows:

1. **Start Here**
   - Home: concise promise, limits, and routes by reader intent.
   - Getting Started: installation and one complete first-model lifecycle.
   - Motivation: the detailed case for preserving hierarchy.
   - Mental Model / Data Flow: the canonical record-to-output walkthrough and
     progressive glossary.
2. **Model Structured Data**
   - Model Tree and addresses.
   - Data Types overview and universal state/content semantics.
   - Individual datatype references.
   - Binding Data: same-name defaults, nested defaults, query versus
     preprocessor.
   - Advanced Query Paths / JMESPath recipes.
   - Learning roles and exported embeddings.
   - Branch/window design and dynamic masking.
3. **Train And Evaluate**
   - Data Modules.
   - Training with Lightning.
   - Evaluation and metrics.
   - Model configuration and temporal validation.
   - Checkpoints and model lifecycle.
4. **Predict And Operate**
   - Interactive prediction and the canonical output contract.
   - Batch inference.
   - Online serving.
   - Postprocessors.
   - Performance and distributed operation.
5. **Advanced Workflows**
   - Field stacking.
   - Field importance.
   - Schema mutation.
   - Extension authoring.
   - Troubleshooting and API reference.
6. **Case Studies**
   - Runnable, evidence-backed workflows; sketches clearly marked as sketches.

The AI / Expert Quickstart should become a terse map to these authoritative
pages. Agent-specific repository rules belong in an agent-oriented artifact,
not interleaved with the human learning path.

## Canonical Content Ownership

Each recurring concept gets one authoritative explanation. Other pages may give
a one-paragraph summary and link to it.

| Concept | Canonical owner | Other pages should do |
| --- | --- | --- |
| Product promise | Home | Link to Motivation for the longer argument. |
| Why hierarchy matters | Motivation | Use only the context needed by an example. |
| Installation and first success | Getting Started | Link rather than repeat environment setup. |
| End-to-end data flow and glossary | New Mental Model / Data Flow page | Define only page-specific internals. |
| Tree, branch, leaf, context, address | Model Tree | Query/type pages assume and link to these terms. |
| Choosing default binding, a query, or preprocessing | New short Binding Data page | Tutorials show one common case and link. |
| Query syntax and inference | Advanced Query Paths reference | Datatype pages show only a relevant example. |
| Universal state/content vocabulary | Data Types overview | Type pages explain only type-specific content behavior. |
| `target`, masking, pruning, and `embed` roles | Learning Modes & Embeddings | Tutorials summarize in a compact table and link. |
| Public prediction envelope and `inferred` | Prediction/output-contract section | Type pages show their `content` member and link to the envelope. |
| Type-specific encoding, loss, metrics, options, and output | Individual datatype page | Core pages compare types without duplicating contracts. |
| Category vocabulary and unavailable semantics | Category | Set and whitepaper must not generalize Category behavior. |
| Exported versus internal embeddings | Learning Modes & Embeddings | Type pages name internal vectors precisely and link. |
| Training-loop boundary | Training with Lightning | Getting Started demonstrates one loop only. |
| Ingestion, splits, sharding, preprocessing | Data Modules | Inference pages link to train/serve parity requirements. |
| Metric interpretation and model selection | New Evaluation guide | Type pages list only metrics they emit. |
| Persistence and schema compatibility | New Model Lifecycle guide | Training and serving pages link to it. |
| Online runtime contract | New Serving guide | AI Quickstart contains at most a linked teaser. |
| Architecture rationale | Whitepaper | Link to live references for current API behavior. |

## Standard Page Contracts

### Datatype Reference

Every leaf datatype page should answer these questions in this order:

1. What kind of real value is this for, and when should it not be used?
2. What is the smallest valid schema and source record?
3. What source values are accepted, and how are missing, malformed, unknown, or
   overflow values handled?
4. How are state and content represented? Clearly label public semantics versus
   internal implementation.
5. What vocabulary, normalizer, or other preprocessing state is learned, and
   how is it preserved?
6. What loss and metrics are used for state and content?
7. What does every public option mean, what is its default, and what constraints
   or cost does it introduce?
8. What is the complete prediction payload, including the shared envelope?
9. What are the cardinality, memory, compute, calibration, and distributed
   limitations?
10. Which alternative datatype should the reader choose for adjacent cases?
11. Where should the reader go next?

Boolean should receive the same configuration and evaluation sections as the
other datatype pages, including how scalar or list `threshold` values configure
threshold-qualified metrics without changing prediction output.

### Workflow Guide

Every guide should contain:

1. intended reader and prerequisite concepts;
2. the outcome and a minimal end-to-end example;
3. the decisions the user must make and their tradeoffs;
4. expected output or a verification step;
5. common failures and diagnostic signals;
6. train/evaluate/serve consistency concerns where applicable;
7. operational limits or maturity caveats;
8. links to the next workflow and the authoritative references it uses.

Examples must be labeled as one of:

- **runnable:** exercised against the package in CI or a docs example suite;
- **excerpt:** intentionally incomplete and linked to a runnable example;
- **sketch:** illustrative and explicitly not an evaluated or directly runnable
  system.

## Category As The Pilot Revision

Category is the best first page for applying the content contract because it
touches vocabulary state, unavailable content, metrics, embeddings, outputs,
and capacity.

The page should flow as follows:

1. bounded categorical value and decision boundary against Set, Hash, and
   Text;
2. minimal schema and record using a genuinely bounded example such as
   `merchant_category`, not a massive `merchant_id`;
3. option summary;
4. online vocabulary lifecycle, capacity, and what happens when capacity is
   full;
5. available versus unavailable content, while both retain state `valued`;
6. internal state-plus-content encoder input;
7. public prediction semantics, including the populated-vocabulary denominator,
   top-k behavior, and why low confidence is not a calibrated OOD detector;
8. training objective, metrics, and tuning guidance;
9. memory/scoring cost and monitoring `vocabulary:size`;
10. alternatives and next links.

For `dev/categorical-unit-circle`, the docs and code must land atomically. The
branch-specific explanation must say:

- all non-valued states have zero content;
- unavailable categorical content also has a zero content vector while keeping
  state `valued`;
- only available valued content receives a hypersphere direction;
- the state embedding remains distinct from the content vector;
- the tied category direction/CosFace training vector is internal and is not the
  same object as the public normalized embedding emitted by `embed=True`;
- unavailable is neither an output class nor a separate availability
  probability.

The advanced training section must include the correct-class CosFace equation,
qualify learned directions as task-discriminative geometry rather than a
guaranteed human taxonomy, distinguish capacity-wide training scores from
populated-vocabulary prediction scores, and explain diagnostic telemetry only
when it leads to a user decision.

The whitepaper should retain only the architectural motivation and link to this
page for live behavior. Set must document its own unknown-label behavior rather
than borrowing Category terminology.

## Delivery Plan

### Phase 0: Establish Truth

- Build a small source-of-truth matrix from public request models, enums,
  writers, persistence APIs, and deployment APIs.
- Fix every P0 factual issue above on the website and in the whitepaper.
- Mark incomplete payloads and case studies honestly.
- Add focused contract tests for examples most likely to drift.

Exit gate: no known contradiction between published behavior and source/tests.

### Phase 1: Repair The Front Door

- Resolve the supported installation story.
- Rewrite Home as a concise router and remove repeated Motivation prose.
- Make Getting Started one complete artifact lifecycle. Predict from records
  with the target omitted, interpret the returned output, and label reuse of
  training data for validation as tutorial-only.
- Add the Data Flow / Mental Model page and progressive glossary.
- Reorder the navigation so advanced masking follows core modeling.
- Keep Model Tree focused on the public mental model; move `anytree`, `Parcel`,
  `heritage`, and similar implementation vocabulary into an advanced section.

Exit gate: an external reader can reach a verified prediction from a clean
environment without repository-specific assumptions.

### Phase 2: Complete The Practitioner Lifecycle

- Add Evaluation and Metrics.
- Add Model Lifecycle.
- Add Online Serving or clearly mark the feature experimental.
- Align batch, interactive, and online output/preprocessing contracts.

Exit gate: a reader can build, validate, select, persist, reload, and operate the
same model artifact using documented public APIs.

### Phase 3: Add Design And Scale Guidance

- Add model configuration, temporal validation, performance/distributed, and
  troubleshooting guidance.
- Apply the datatype page contract to all built-ins, beginning with Category,
  Hash, Set, Boolean, DateParts, and Branch.
- Document or explicitly de-scope the extension API.

Exit gate: important choices have stated tradeoffs, diagnostics, and limits.

### Phase 4: Consolidate And Prove

- Remove duplicate explanations after canonical pages exist.
- Add consistent next-step links and route pages by reader intent.
- Add public API/config reference coverage.
- Promote one case study from sketch to runnable, measured example.

Exit gate: site search and navigation expose every supported public workflow,
and no concept has competing authoritative explanations.

## Acceptance Criteria

### Reader Outcomes

- A new user can install the supported distribution and verify the import.
- A new user can train, inspect a validation result, save, load, and predict from
  one coherent example.
- A reader can explain the difference between state, content, trainability,
  pruning, and exported embeddings.
- A reader can choose among Category, Set, Hash, Text, Number, Boolean,
  DateParts, and Vector from explicit decision boundaries.
- A practitioner can choose interactive, batch, or online prediction and knows
  whether all three are supported at their version.
- A practitioner can find the option, metric, output shape, limit, and failure
  behavior for each supported public workflow.

### Correctness And Maintainability

- Every option table matches the public request/model fields at the tested
  version.
- Complete output examples match an actual `Model.predict` or writer result;
  excerpts are labeled.
- Runnable snippets are executed in CI or by a documented example-validation
  command.
- `make render` succeeds and internal links resolve.
- Claims about scale include benchmark context or are explicitly qualitative.
- Sketches, experiments, and stable APIs are visually and verbally distinct.
- README, website, and whitepaper do not define conflicting runtime semantics.
- Each new public option or output field has a named canonical docs owner in the
  same change that introduces it.

## Open Product Decisions

These decisions affect the docs and should be resolved by the responsible API
owner, but they do not block the Phase 0 correctness fixes:

1. Is the supported user installation PyPI, a Git URL, or repository checkout?
2. Which serving and deployment APIs are stable enough for a production guide?
3. What benchmark evidence supports the current scale and throughput claims?
4. Is the tensorfield plugin surface a supported extension API or an
   experimental internal interface?
5. Which audience has priority when expert brevity conflicts with first-time
   explanation?
6. Does the whitepaper describe a versioned implementation or only architectural
   intent?
7. Will documentation be versioned with releases, and how will older model
   artifacts map to those versions?

## Definition Of Done

The revision is complete when the docs tell one technically accurate story from
problem to production, a reader can complete the supported lifecycle without
guessing at hidden contracts, advanced concepts appear only after their
prerequisites, and implementation changes cannot silently leave duplicated
behavioral explanations behind.

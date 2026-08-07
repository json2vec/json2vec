"""Shared tensorfield helper implementations.

This package contains runtime helpers used by tensorfield extensions, such as
online vocabularies, counters, and their Lightning callbacks. These helpers are
extension implementation details. They may be reused by multiple tensorfield
extensions, but they should not become a second public architecture layer.

Rules for adding helpers here:

1. Keep ownership inside tensorfields.
   Helpers in `tensorfields.shared` should support tensorfield extensions. They
   should not require `architecture`, `data`, `inference`, or orchestration code
   to understand a concrete helper such as `VocabularyState` or `Counter`.

2. Do not leak extension-specific helpers downstream.
   Downstream modules may call generic lifecycle methods on encoding context
   resources, but they must not import helper-specific operations. Prefer a
   generic data-layer hook like
   `share_interprocess_encoding_context(...)`, which duck-types an optional
   `share()` method on each context resource.

3. Use narrow optional protocols for cross-layer behavior.
   Encoding context resources may expose small, generic lifecycle methods:
   `configure_distributed(...)` for rank/worker identity and `share()` for
   opt-in multiprocessing-safe storage. Those names should mean the same thing
   for any future helper that implements them. Extension-specific operations
   belong in the helper module or its callback, not in data/root code.

4. Keep multiprocessing lazy and loop-scoped.
   Helper construction must be cheap and local. Creating a `Model` with a
   tensorfield must not start `multiprocessing.Manager`, shared memory, worker
   processes, network connections, or other process-global resources. Activate
   multiprocessing only when a train dataloader actually needs worker-shared
   mutable state, and return to local state when the fit loop ends.

5. Preserve inference and notebook ergonomics.
   Prediction, validation, docs, notebooks, and realtime serving should work
   without multiprocessing. Loaded or freshly constructed models should use
   local, frozen/read-optimized helper state unless a training loop explicitly
   opts into shared mutable state.

6. Put extension-specific synchronization in callbacks.
   If a helper needs distributed synchronization, implement it in a callback
   owned by the helper module. The callback may inspect concrete helper types
   because it is part of that helper's implementation. Data modules and the root
   model should only attach callbacks through the tensorfield plugin registry.

7. Avoid generic registries for one-off behavior.
   Add a shared abstraction only when multiple tensorfields can use it or when
   a lifecycle hook is truly generic. If the behavior is specific to one
   tensorfield, keep it in that extension module.

8. Make serialization explicit.
   Helper state that affects inference must be included in the owning module's
   checkpoint state. Nonpersistent process resources, worker proposals, locks,
   and cached snapshots should be rebuilt or discarded on load.

9. Test the boundary, not only the behavior.
   Tests for helpers should assert that model construction does not activate
   multiprocessing, that train worker paths opt in when needed, that predict
   paths remain local, and that downstream packages do not import concrete
   helper-specific functions.
"""

from __future__ import annotations

from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback
from relflow.tensorfields.shared.vocabulary import (
    OnlineVocabularyModel,
    VocabularyState,
    VocabularySyncCallback,
)

__all__ = [
    "Counter",
    "CounterUpdateCallback",
    "OnlineVocabularyModel",
    "VocabularyState",
    "VocabularySyncCallback",
]

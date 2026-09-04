# ty: ignore[invalid-method-override,unknown-argument]
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast
from weakref import ReferenceType, ref

import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from lightning.pytorch import Callback, Trainer
from loguru import logger
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.distributed import broadcast_object
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    Context,
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
    TensorInput,
)
from relflow.tensorfields.output import array, labels, struct
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularyState, VocabularySyncCallback

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.datasets.base import InterprocessEncodingContext
    from relflow.structs.experiment import Schema

cluster: Plugin = Plugin(name="cluster", types=(bool, int, float, str, bytes))

cluster.callback(VocabularySyncCallback, CounterUpdateCallback)

# Algorithmic constants: fixed values from the SwAV/DINO literature or symmetry-breaking
# magnitudes that have no reason to be user-tuned.
_BALANCE_EPSILON: float = 0.05
_BALANCE_ITERS: int = 3
_REVIVE_NOISE: float = 0.02
_REVIVE_WARMUP: int = 1
_OVERRIDE_LOCK: str = "cluster assignment overrides"

_Assignment = int | torch.Tensor | list[float] | tuple[float, ...]


@cluster.register
class Request(RequestBase):
    """Clustering scalar tensorfield request."""

    model_config = pydantic.ConfigDict(extra="allow", populate_by_name=True, serialize_by_alias=True)

    type: Literal["cluster"] = "cluster"
    capacity: Annotated[
        int,
        pydantic.Field(gt=0, default=1024),
    ] = 1024
    p_unavailable: Annotated[float, pydantic.Field(ge=0.0, le=1.0, default=0.01)] = 0.01

    ema_decay: Annotated[float, pydantic.Field(ge=0.0, le=1.0, default=0.99)] = 0.99
    revive_temperature: Annotated[float, pydantic.Field(ge=0.0, default=10.0)] = 10.0

    n_clusters: Annotated[
        tuple[int, int],
        pydantic.Field(
            alias="bounds",
            serialization_alias="bounds",
            description="(lower_bound, upper_bound) for the number of clusters. A single int is broadcast to (K, K) to specify an exact number of clusters.",
        ),
    ]

    @pydantic.field_validator("n_clusters", mode="before")
    @classmethod
    def _broadcast_n_clusters(cls, value: Any) -> Any:
        if isinstance(value, int) and not isinstance(value, bool):
            return (value, value)
        return value

    @property
    def size(self) -> int:
        return self.n_clusters[-1]

    @pydantic.model_validator(mode="before")
    @classmethod
    def reject_removed_options(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "max_vocab_size" in data:
            raise ValueError("max_vocab_size was removed; use size")

        return data

    @pydantic.model_validator(mode="after")
    def check_n_clusters(self):
        if len(self.n_clusters) != 2:
            raise ValueError("n_clusters tuple must have exactly 2 elements (lower_bound, upper_bound)")
        lower, upper = self.n_clusters
        if lower <= 0:
            raise ValueError("n_clusters lower bound must be greater than 0")
        if lower > upper:
            raise ValueError("n_clusters lower bound must be less than or equal to upper bound")
        return self

    @classmethod
    def assign(
        cls,
        model: "Model",
        address: Address | str,
        token: Any,
        assignment: int | torch.Tensor | list[float] | tuple[float, ...],
        /,
    ) -> int:
        """Persistently assign one token to a cluster or cluster distribution."""
        return _ClusterRuntime(model=model, address=address).assign(token, assignment)

    @classmethod
    def vocabulary(
        cls,
        source: "Model | InterprocessEncodingContext",
        address: Address | str,
        /,
    ) -> tuple[Any, ...]:
        """Return an immutable snapshot of one Cluster vocabulary."""
        from relflow.architecture.root import Model

        normalized = Address(str(address))
        if isinstance(source, Model):
            embedder = _resolve_cluster_embedder(source, normalized)
            with embedder.vocab.lock:
                return tuple(embedder.vocab.master)
        elif isinstance(source, Mapping):
            if normalized not in source:
                raise KeyError(f"no encoding context at address {str(normalized)!r}")

            state = source[normalized]
            if not isinstance(state, VocabularyState):
                raise TypeError(
                    f"encoding context at {str(normalized)!r} is not a VocabularyState (got {type(state).__name__})"
                )
        else:
            raise TypeError(
                f"Cluster.vocabulary source must be a Model or InterprocessEncodingContext, got {type(source).__name__}"
            )

        with state.lock:
            return tuple(state.master)

    @classmethod
    def assignments(
        cls,
        model: "Model",
        address: Address | str,
        /,
    ) -> dict[Any, dict[str, int | tuple[float, ...]]]:
        """Return evaluation-time assignments for every populated token."""
        embedder = _resolve_cluster_embedder(model, address)
        with embedder.vocab.lock:
            vocabulary = tuple(embedder.vocab.master)
            logits = embedder.embeddings[TensorKey.cluster.name].weight[: len(vocabulary)].detach().clone()

        if not vocabulary:
            return {}

        probabilities = torch.softmax(logits, dim=-1)
        clusters: list[int] = logits.argmax(dim=-1).cpu().tolist()
        rows: list[list[float]] = probabilities.cpu().tolist()
        return {
            token: {
                "cluster": int(cluster_id),
                "probabilities": tuple(float(probability) for probability in row),
            }
            for token, cluster_id, row in zip(vocabulary, clusters, rows, strict=True)
        }

    @classmethod
    def status(
        cls,
        model: "Model",
        address: Address | str,
        /,
    ) -> dict[str, tuple[int, ...] | tuple[float, ...]]:
        """Return committed cluster IDs and their usage EMA snapshot."""
        embedder = _resolve_cluster_embedder(model, address)
        committed = embedder.committed.detach().cpu().clone()
        usage = embedder.usage_ema.detach().cpu().clone()

        return {
            "committed": tuple(index for index, value in enumerate(committed.tolist()) if value),
            "usage": tuple(float(value) for value in usage.tolist()),
        }

    @classmethod
    def override(
        cls,
        model: "Model",
        address: Address | str,
        assignments: Mapping[Any, int | torch.Tensor | list[float] | tuple[float, ...]],
        /,
    ) -> AbstractContextManager[None]:
        """Temporarily override token assignments within a context."""
        return _ClusterRuntime(model=model, address=address).override(assignments)


@cluster.register
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Reserve and count the complete pristine clustering exposure."""

    if not learn:
        return None
    if not isinstance(state, VocabularyState):
        raise RuntimeError(f"cluster field at '{address}' requires a vocabulary encoding context")

    indices = state.indices(field.values, learn=True)
    request: Request = schema.requests[address]
    return TensorDict(
        {
            TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens)),
            TensorKey.content: tally(torch.from_numpy(indices.copy()), request.capacity + 1),
        },
        batch_size=[],
    )


@cluster.register
@tensorclass
class TensorField(TensorFieldBase):
    state: torch.Tensor
    content: torch.Tensor
    present: torch.Tensor
    trainable: torch.Tensor
    inferred: torch.Tensor
    targets: TensorDict[TensorKey, torch.Tensor]

    @classmethod
    def new(
        cls,
        input: RaggedField,
        target: RaggedField,
        present: torch.Tensor,
        trainable: torch.Tensor,
        inferred: torch.Tensor,
        address: Address,
        schema: Schema,
        strata: Strata,
        context: Context,
    ) -> TensorFieldBase:
        state = context.state
        if state is not None and not isinstance(state, VocabularyState):
            raise TypeError(
                f"cluster field at '{address}' requires VocabularyState context, got {type(state).__name__}"
            )

        def encode(field: RaggedField) -> torch.Tensor:
            if not len(field.values):
                return torch.zeros(field.shape, dtype=torch.int64)
            if state is None:
                raise RuntimeError(f"cluster field at '{address}' requires a vocabulary encoding context")
            tokens = state.indices(field.values, learn=False)
            return torch.from_numpy(field.place(tokens, fill=0))

        content = encode(input)
        target_content = encode(target)

        if state is not None and len(state) > (capacity := schema.requests[address].capacity):
            logger.bind(component="tensorfield", field_type="cluster", address=str(address)).warning(
                "vocabulary exceeds size={}", capacity
            )

        state_tensor = torch.from_numpy(input.dense)
        target_state = torch.from_numpy(target.dense)
        if strata == Strata.train:
            p_unavailable: float = schema.requests[address].p_unavailable
            unavailable_index: int = schema.requests[address].capacity

            if p_unavailable > 0.0:
                # Unavailable content never appears naturally during training, because the
                # train split is exactly where the vocabulary is built. We simulate a small
                # amount of OOV behavior so the content objective does not reward any real
                # class for valued inputs whose categorical content is unavailable.
                def regularize(values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
                    known = state.eq(Tokens.valued.value) & values.ne(unavailable_index)
                    selected = torch.rand_like(state, dtype=torch.float).lt(p_unavailable) & known
                    return values.masked_fill(selected, unavailable_index)

                content = regularize(content, state_tensor)
                target_content = regularize(target_content, target_state)

        return cls(
            state=state_tensor,
            content=content,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: target_state,
                    TensorKey.content: target_content,
                },
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
        )


@cluster.register
class Embedder(EmbedderBase):
    usage_ema: torch.Tensor
    committed: torch.Tensor
    adherence_ema: torch.Tensor

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address
        self.capacity: int = request.capacity
        self.size: int = request.size

        if address not in schema.reconstruct:
            # Cluster loss fires only for reconstructed rows; a plain-input
            # Cluster silently freezes n_committed at initialization.
            logger.bind(component="tensorfield", field_type="cluster", address=str(address)).warning(
                "Cluster field {address!s} has no reconstructing Mask; dynamic K-selection "
                "will not engage. Add Mask(reconstruct=True) to train the cluster head.",
                address=address,
            )

        self.vocab: OnlineVocabularyModel = OnlineVocabularyModel(size=self.capacity)
        self._override_depth: int = 0

        self.embeddings = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Embedding(
                    num_embeddings=len(Tokens),
                    embedding_dim=schema.d_model,
                ),
                TensorKey.cluster.name: torch.nn.Embedding(
                    num_embeddings=self.capacity + 1,
                    embedding_dim=self.size,
                ),
                TensorKey.content.name: torch.nn.Linear(
                    in_features=self.size,
                    out_features=schema.d_model,
                    bias=False,
                ),
            }
        )
        self.counters = torch.nn.ModuleDict(
            {
                TensorKey.state.name: Counter(address=address, size=len(Tokens)),
                TensorKey.content.name: Counter(address=address, size=self.capacity + 1),
            }
        )

        lower: int = request.n_clusters[0]
        usage_init = torch.zeros(self.size)
        usage_init[:lower] = 1.0 / lower
        committed_init = torch.zeros(self.size, dtype=torch.bool)
        committed_init[:lower] = True
        self.register_buffer("usage_ema", usage_init)
        self.register_buffer("committed", committed_init)
        self.register_buffer("adherence_ema", torch.zeros(()))

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N: int
        dims: list[int]

        N, *dims = inputs.state.shape
        state = inputs.state.reshape(-1)
        content = inputs.content.reshape(-1)
        valued = state.eq(Tokens.valued.value)

        if valued.any():
            valid_content = content.masked_select(valued)
            if ((valid_content < 0) | (valid_content > self.capacity)).any().item():
                raise ValueError(f"Token in address {self.origin} outside [0, {self.capacity}]")

        safe_content = content.masked_fill(~valued, 0)
        assign_logits = self.embeddings[TensorKey.cluster.name](safe_content)
        if self.training:
            cluster_assignments = torch.nn.functional.gumbel_softmax(assign_logits, hard=True, dim=-1)
        else:
            hard = torch.zeros_like(assign_logits)
            hard.scatter_(-1, assign_logits.argmax(dim=-1, keepdim=True), 1.0)
            cluster_assignments = hard
        cluster_assignments = cluster_assignments * valued.unsqueeze(-1)
        cluster_embeddings = self.embeddings[TensorKey.content.name](cluster_assignments)

        embeddings: torch.Tensor = (self.embeddings[TensorKey.state.name](state) + cluster_embeddings).reshape(
            N, *dims, -1
        )

        return Parcel(
            payload=embeddings,
            present=torch.ones(N, dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )

    @property
    def context(self) -> VocabularyState:
        return self.vocab.state

    def _resolve_assignment(self, assignment: _Assignment) -> torch.Tensor:
        weight = self.embeddings[TensorKey.cluster.name].weight
        if isinstance(assignment, bool):
            raise TypeError("assignment must be a cluster index or probability vector")
        if isinstance(assignment, int):
            if not 0 <= assignment < self.size:
                raise ValueError(f"cluster must be in [0, {self.size}); got {assignment}")
            distribution = torch.zeros(self.size, dtype=weight.dtype, device=weight.device)
            distribution[assignment] = 1.0
            return torch.log(distribution.clamp_min(1e-8))
        distribution = torch.as_tensor(assignment, dtype=weight.dtype, device=weight.device)
        if distribution.shape != (self.size,):
            raise ValueError(
                f"assignment probabilities must have shape ({self.size},); got {tuple(distribution.shape)}"
            )
        if bool((distribution < 0).any().item()):
            raise ValueError("assignment probabilities must be non-negative")
        total = float(distribution.sum().item())
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(f"assignment probabilities must sum to 1; got {total}")
        return torch.log(distribution.clamp_min(1e-8))

    def _assign(self, token: Any, assignment: _Assignment) -> int:
        logits = self._resolve_assignment(assignment)
        weight = self.embeddings[TensorKey.cluster.name].weight
        index = self.vocab.state.index.get(token)
        if index is None:
            if len(self.vocab.master) >= self.capacity:
                raise ValueError(f"cluster field {self.origin} at capacity ({self.capacity}); cannot assign {token!r}")
            self.vocab.master.append(token)
            index = len(self.vocab.master) - 1
        with torch.no_grad():
            weight[index].copy_(logits)
        return index

    def _override(self, assignments: Mapping[Any, _Assignment]):
        embedder = self

        class _Override:
            def __enter__(self):
                if embedder._override_depth:
                    raise RuntimeError("Cluster assignment overrides cannot be nested")
                embedder._override_depth += 1
                weight = embedder.embeddings[TensorKey.cluster.name].weight
                # (index, original_row, was_new_append) — LIFO on exit
                self.saved: list[tuple[int, torch.Tensor, bool]] = []
                try:
                    for token, assignment in assignments.items():
                        existing = embedder.vocab.state.index.get(token)
                        if existing is not None:
                            original = weight[existing].detach().clone()
                            embedder._assign(token, assignment)
                            self.saved.append((existing, original, False))
                        else:
                            if len(embedder.vocab.master) >= embedder.capacity:
                                raise ValueError(
                                    f"cluster field {embedder.origin} at capacity ({embedder.capacity}); "
                                    f"cannot override {token!r}"
                                )
                            new_index = len(embedder.vocab.master)
                            original = weight[new_index].detach().clone()
                            embedder._assign(token, assignment)
                            self.saved.append((new_index, original, True))
                except BaseException:
                    try:
                        self._rollback()
                    finally:
                        embedder._override_depth -= 1
                    raise
                return None

            def __exit__(self, exc_type, exc_val, exc_tb):
                try:
                    self._rollback()
                finally:
                    embedder._override_depth -= 1
                return False

            def _rollback(self):
                weight = embedder.embeddings[TensorKey.cluster.name].weight
                with torch.no_grad():
                    for index, original, was_new in reversed(self.saved):
                        weight[index].copy_(original)
                        if was_new and embedder.vocab.master:
                            embedder.vocab.master.pop()
                self.saved.clear()

        return _Override()

    def _save_to_state_dict(self, state_dict, prefix, keep_vars):  # ty:ignore[invalid-method-override]
        if self._override_depth:
            raise RuntimeError("cannot save or rebuild a model while Cluster assignment overrides are active")
        super()._save_to_state_dict(state_dict, prefix, keep_vars)


@cluster.register
def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine clustering counts to model-owned resources."""

    if strata != Strata.train:
        raise ValueError(f"cluster learner at '{address}' requires train strata, got {strata}")
    embedder: Embedder = module.nodes[address].embedder
    embedder.counters[TensorKey.state.name].learn(observation[TensorKey.state])
    embedder.counters[TensorKey.content.name].learn(observation[TensorKey.content])


def _resolve_cluster_embedder(model: "Model", address: Address | str) -> Embedder:
    """Resolve a live Cluster embedder without importing Model at module load."""
    from relflow.architecture.root import Model

    if not isinstance(model, Model):
        raise TypeError(f"Cluster binding model must be a Model, got {type(model).__name__}")

    normalized = Address(str(address))
    if normalized not in model.nodes:
        raise KeyError(f"no field at address {str(normalized)!r}")

    embedder = getattr(model.nodes[normalized], "embedder", None)
    if not isinstance(embedder, Embedder):
        raise TypeError(f"address {str(normalized)!r} is not a Cluster field (got {type(embedder).__name__})")
    return embedder


class _ClusterRuntime:
    """Resolve and mutate one Cluster field's model-owned runtime state."""

    def __init__(self, model: "Model", address: Address | str):
        self._model: ReferenceType[Model] = ref(model)
        self.address: Address = Address(str(address))
        self._embedder()

    def _resolve(self) -> tuple[Model, Embedder]:
        model = self._model()
        if model is None:
            raise RuntimeError("the model bound to this Cluster field no longer exists")
        if self.address not in model.nodes:
            raise KeyError(f"no field at address {str(self.address)!r}")

        embedder = getattr(model.nodes[self.address], "embedder", None)
        if not isinstance(embedder, Embedder):
            raise TypeError(f"address {str(self.address)!r} is not a Cluster field (got {type(embedder).__name__})")
        return model, embedder

    def _embedder(self) -> Embedder:
        return self._resolve()[1]

    @staticmethod
    def _assert_writable(model: Model, embedder: Embedder) -> None:
        if embedder.vocab.is_shared:
            raise RuntimeError("Cluster assignments cannot change while vocabulary state is shared")
        active = tuple(str(name) for name, count in model.locks.items() if count > 0)
        if active:
            raise RuntimeError(f"Cluster assignments cannot change while the model is active: {', '.join(active)}")

    def assign(self, token: Any, assignment: _Assignment) -> int:
        model, embedder = self._resolve()
        with embedder.vocab.lock:
            self._assert_writable(model, embedder)
            return embedder._assign(token, assignment)

    @contextmanager
    def override(self, assignments: Mapping[Any, _Assignment]) -> Iterator[None]:
        model, embedder = self._resolve()
        with embedder.vocab.lock:
            self._assert_writable(model, embedder)
            model.locks[_OVERRIDE_LOCK] += 1
            try:
                with embedder._override(assignments):
                    yield
            finally:
                if model.locks[_OVERRIDE_LOCK] <= 1:
                    model.locks.pop(_OVERRIDE_LOCK, None)
                else:
                    model.locks[_OVERRIDE_LOCK] -= 1


@cluster.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        n_clusters: int = request.size

        self.linears = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=len(Tokens),
                ),
                TensorKey.cluster.name: torch.nn.Linear(
                    in_features=schema.d_model, out_features=n_clusters, bias=False
                ),
            }
        )

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict[TensorKey, torch.Tensor]:
        return TensorDict(
            source={
                TensorKey.state: self.linears[TensorKey.state.name](pooled),
                TensorKey.cluster: self.linears[TensorKey.cluster.name](pooled),
            }
        )


@cluster.register
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    embedder: Embedder = module.nodes[prediction.address].embedder
    request: Request = cast(Request, module.schema.requests[prediction.address])
    N: int = batch.targets[TensorKey.state].numel()
    trainable = batch.trainable.reshape(N)

    state_inputs = prediction.payload[TensorKey.state].reshape(N, -1)
    state_targets = batch.targets[TensorKey.state].reshape(N)

    loss: torch.Tensor = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=state_inputs,
                target=state_targets,
                weight=cast(Counter, embedder.counters[TensorKey.state.name]).weight,
                reduction="none",
            )
            .masked_select(trainable)
            .mean()
        ),
    )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )
    module.track(
        (prediction.address, strata, "vocabulary", "size"),
        value=state_inputs.new_tensor(len(embedder.vocab.master), dtype=torch.float32),
    )

    valued: torch.Tensor = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        return loss

    cluster_logits: torch.Tensor = prediction.payload[TensorKey.cluster].reshape(N, -1)
    content_targets = batch.targets[TensorKey.content].reshape(N)

    assign_weight: torch.Tensor = embedder.embeddings[TensorKey.cluster.name].weight
    cluster_probs = torch.log_softmax(cluster_logits, dim=-1).exp()
    vocab_logits = cluster_probs @ assign_weight.T

    vocab_size = min(len(embedder.vocab.master), embedder.capacity)
    known = valued & content_targets.lt(vocab_size)
    if known.any():
        loss += module.track(
            (prediction.address, strata, Metric.loss, TensorKey.content),
            value=torch.nn.functional.cross_entropy(
                input=vocab_logits[known],
                target=content_targets[known],
                weight=cast(Counter, embedder.counters[TensorKey.content.name]).weight,
                reduction="mean",
            ),
        )
        module.track(
            (prediction.address, strata, Metric.accuracy, TensorKey.content),
            value=vocab_logits[:, :vocab_size].argmax(dim=1).eq(content_targets).masked_select(known).float().mean(),
        )

    if strata != Strata.train:
        return loss

    valued_logits = cluster_logits[valued]
    n_valued: int = valued_logits.shape[0]
    lower: int = request.n_clusters[0]

    if n_valued < max(lower, 2):
        return loss

    valued_probs = torch.softmax(valued_logits, dim=-1)
    upper: int = request.n_clusters[-1]

    with torch.no_grad():
        batch_usage = valued_probs.detach().mean(dim=0)
        embedder.usage_ema.mul_(request.ema_decay).add_(batch_usage * (1.0 - request.ema_decay))

        # Perplexity of natural usage tells us how many clusters the data is currently spending
        # mass on. Clamped into [lower, upper] this becomes the committed count so the balance
        # loss can grow the active set when content CE recruits more columns.
        usage_normalized = embedder.usage_ema / embedder.usage_ema.sum().clamp_min(1e-12)
        usage_safe = usage_normalized.clamp_min(1e-12)
        usage_entropy = -(usage_safe * usage_safe.log()).sum()
        perplexity = torch.exp(usage_entropy)
        n_committed: int = max(lower, min(upper, int(round(perplexity.item()))))

        _, committed_idx = embedder.usage_ema.topk(n_committed)
        embedder.committed.zero_()
        embedder.committed[committed_idx] = True

        sinkhorn_input = torch.log_softmax(valued_logits[:, committed_idx], dim=-1)
        scaled = sinkhorn_input / _BALANCE_EPSILON
        Q = torch.exp(scaled - scaled.max())
        Q = Q / Q.sum().clamp_min(1e-12)
        for _ in range(_BALANCE_ITERS):
            Q = Q / (n_valued * Q.sum(dim=1, keepdim=True).clamp_min(1e-12))
            Q = Q / (n_committed * Q.sum(dim=0, keepdim=True).clamp_min(1e-12))
        Q = Q * n_valued

    full_log_probs = torch.log_softmax(valued_logits, dim=-1)
    committed_log_probs = full_log_probs[:, committed_idx]
    balance_loss = -(Q * committed_log_probs).sum(dim=-1).mean()

    loss += module.track(
        (prediction.address, strata, Metric.loss, TensorKey.cluster),
        value=balance_loss,
    )
    module.track(
        (prediction.address, strata, "cluster", "active"),
        value=perplexity,
    )
    module.track(
        (prediction.address, strata, "cluster", "committed"),
        value=state_inputs.new_tensor(float(n_committed)),
    )
    module.track(
        (prediction.address, strata, "cluster", "usage_entropy"),
        value=usage_entropy,
    )
    module.track(
        (prediction.address, strata, "cluster", "sentinel_share"),
        value=content_targets.eq(embedder.capacity).masked_select(valued).float().mean(),
    )

    if lower != upper:
        uncommitted = ~embedder.committed
        if uncommitted.any():
            uncommitted_mass = valued_probs[:, uncommitted].mean(dim=0).sum()
            with torch.no_grad():
                embedder.adherence_ema.mul_(request.ema_decay).add_(
                    uncommitted_mass.detach() * (1.0 - request.ema_decay)
                )
    return loss


@cluster.register
def output(module: Model, address: Address) -> pa.StructType:
    cluster_type = pa.struct(
        [
            pa.field(TensorKey.value.name, pa.int32(), nullable=False),
            pa.field(TensorKey.probability.name, pa.float32(), nullable=False),
        ]
    )
    content_type = pa.struct(
        [
            pa.field(TensorKey.value.name, pa.large_string()),
            pa.field(TensorKey.probability.name, pa.float32(), nullable=False),
        ]
    )
    return pa.struct(
        [
            pa.field(TensorKey.cluster.name, cluster_type, nullable=False),
            pa.field(TensorKey.content.name, content_type, nullable=False),
        ]
    )


@cluster.register
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    cluster_type = datatype.field(TensorKey.cluster.name).type
    content_type = datatype.field(TensorKey.content.name).type
    embedder: Embedder = module.nodes[prediction.address].embedder
    logits = prediction.payload[TensorKey.cluster]
    cluster_probabilities = logits.reshape(-1, logits.shape[-1]).softmax(dim=-1)
    cluster_probability, cluster_ids = cluster_probabilities.max(dim=-1)
    cluster_values = struct(
        {
            TensorKey.value.name: array(cluster_ids, pa.int32()),
            TensorKey.probability.name: array(cluster_probability, pa.float32()),
        },
        cluster_type,
    )

    assign_weight: torch.Tensor = embedder.embeddings[TensorKey.cluster.name].weight
    vocabulary_logits = cluster_probabilities @ assign_weight.T
    vocabulary = labels(embedder.vocab)
    size = len(vocabulary)
    if size > vocabulary_logits.shape[-1]:
        raise ValueError(
            f"cluster vocabulary at {prediction.address!s} has {size} values but assignment width is "
            f"{vocabulary_logits.shape[-1]}"
        )

    count = vocabulary_logits.shape[0]
    content_labels: pa.Array = pa.nulls(count, type=pa.large_string())
    content_probability = torch.zeros(count, dtype=torch.float32, device=logits.device)
    if size:
        probabilities = vocabulary_logits[:, :size].softmax(dim=-1)
        content_probability, indices = probabilities.max(dim=-1)
        content_labels = pc.take(vocabulary, array(indices, pa.int64()))

    content_values = struct(
        {
            TensorKey.value.name: content_labels,
            TensorKey.probability.name: array(content_probability, pa.float32()),
        },
        content_type,
    )
    return struct(
        {
            TensorKey.cluster.name: cluster_values,
            TensorKey.content.name: content_values,
        },
        datatype,
    )


class ClusterReviveCallback(Callback):
    @torch.no_grad()
    def on_train_epoch_end(self, trainer: Trainer, pl_module: "Model") -> None:  # ty:ignore[invalid-method-override]
        epoch = int(trainer.current_epoch)
        targets: dict[Address, tuple[Embedder, Decoder, Request]] = {}
        for address, node in pl_module.nodes.items():
            request = pl_module.schema.requests.get(cast(Address, address))
            if not isinstance(request, Request) or request.revive_temperature <= 0.0:
                continue
            if request.n_clusters[0] == request.n_clusters[-1]:
                continue
            embedder = getattr(node, "embedder", None)
            decoder = getattr(node, "decoder", None)
            if isinstance(embedder, Embedder) and isinstance(decoder, Decoder):
                targets[cast(Address, address)] = (embedder, decoder, request)

        if not targets:
            return

        plans: dict[Address, dict[str, Any] | None] = {}
        if trainer.is_global_zero:
            for address, (embedder, _, request) in targets.items():
                plans[address] = self._plan(embedder, request, epoch=epoch)
        plans = broadcast_object(plans, src=0)

        for address, (embedder, decoder, request) in targets.items():
            self._apply(embedder, decoder, plans.get(address))

    @staticmethod
    def _plan(embedder: "Embedder", request: "Request", *, epoch: int) -> dict[str, Any] | None:
        dead = (~embedder.committed).nonzero(as_tuple=True)[0]
        n_dead = int(dead.numel())
        if n_dead == 0:
            return None

        committed_idx = embedder.committed.nonzero(as_tuple=True)[0]
        n_committed = int(committed_idx.numel())
        adherence = float(embedder.adherence_ema.item())
        warmup = epoch < _REVIVE_WARMUP

        signal = max(adherence, 1.0 if warmup else 0.0)
        if signal <= 0.0:
            return None

        base = math.exp(-epoch / request.revive_temperature)
        cap = 1.0 / n_dead
        p = min(base * signal, cap)
        if p <= 0.0:
            return None

        trials = torch.rand(n_dead)
        chosen_mask = trials < p
        if not bool(chosen_mask.any().item()):
            return None

        chosen = dead[chosen_mask].tolist()

        if n_committed > 0:
            committed_order = committed_idx[embedder.usage_ema[committed_idx].argsort(descending=True)]
            donors = [int(committed_order[i % n_committed].item()) for i in range(len(chosen))]
        else:
            donor = int(embedder.usage_ema.argmax().item())
            donors = [donor] * len(chosen)

        assign_w: torch.Tensor = embedder.embeddings[TensorKey.cluster.name].weight
        content_w: torch.Tensor = embedder.embeddings[TensorKey.content.name].weight
        v_plus_1 = int(assign_w.shape[0])
        d_model = int(content_w.shape[0])
        noise = _REVIVE_NOISE
        n_chosen = len(chosen)
        return {
            "dead_ids": chosen,
            "donors": donors,
            "noise_assign": noise * torch.randn(n_chosen, v_plus_1),
            "noise_content": noise * torch.randn(n_chosen, d_model),
            "noise_cluster": noise * torch.randn(n_chosen, d_model),
        }

    @staticmethod
    def _apply(embedder: "Embedder", decoder: "Decoder", plan: dict[str, Any] | None) -> None:
        if plan is None:
            return

        assign_w: torch.Tensor = embedder.embeddings[TensorKey.cluster.name].weight
        content_w: torch.Tensor = embedder.embeddings[TensorKey.content.name].weight
        cluster_w: torch.Tensor = decoder.linears[TensorKey.cluster.name].weight
        donors = plan["donors"]
        device = assign_w.device
        dtype = assign_w.dtype

        for i, k in enumerate(plan["dead_ids"]):
            donor = donors[i]
            assign_w.data[:, k] = assign_w.data[:, donor] + plan["noise_assign"][i].to(device=device, dtype=dtype)
            content_w.data[:, k] = content_w.data[:, donor] + plan["noise_content"][i].to(device=device, dtype=dtype)
            cluster_w.data[k, :] = cluster_w.data[donor, :] + plan["noise_cluster"][i].to(
                device=cluster_w.device, dtype=cluster_w.dtype
            )
            embedder.usage_ema[k] = embedder.usage_ema[donor] * 0.5
            embedder.usage_ema[donor] *= 0.5

        embedder.adherence_ema.zero_()


_MERGE_THRESHOLD: float = 0.95


class ClusterMergeCallback(Callback):
    @torch.no_grad()
    def on_train_epoch_end(self, trainer: Trainer, pl_module: "Model") -> None:  # ty:ignore[invalid-method-override]
        targets: dict[Address, tuple[Embedder, Decoder, Request]] = {}
        for address, node in pl_module.nodes.items():
            request = pl_module.schema.requests.get(cast(Address, address))
            if not isinstance(request, Request):
                continue
            if request.n_clusters[0] == request.n_clusters[-1]:
                continue
            embedder = getattr(node, "embedder", None)
            decoder = getattr(node, "decoder", None)
            if isinstance(embedder, Embedder) and isinstance(decoder, Decoder):
                targets[cast(Address, address)] = (embedder, decoder, request)

        if not targets:
            return

        plans: dict[Address, dict[str, int] | None] = {}
        if trainer.is_global_zero:
            for address, (embedder, decoder, request) in targets.items():
                plans[address] = self._plan(embedder, decoder, request)
        plans = broadcast_object(plans, src=0)

        for address, (embedder, decoder, _) in targets.items():
            self._apply(embedder, decoder, plans.get(address))

    @staticmethod
    def _plan(embedder: "Embedder", decoder: "Decoder", request: "Request") -> dict[str, int] | None:
        committed_idx = embedder.committed.nonzero(as_tuple=True)[0]
        n_committed = int(committed_idx.numel())
        lower = request.n_clusters[0]
        if n_committed <= lower or n_committed < 2:
            return None

        content_w = embedder.embeddings[TensorKey.content.name].weight  # (d_model, K)
        cluster_w = decoder.linears[TensorKey.cluster.name].weight  # (K, d_model)

        content_cols = content_w[:, committed_idx]
        cluster_rows = cluster_w[committed_idx, :]
        content_norm = content_cols / content_cols.norm(dim=0).clamp_min(1e-12)
        cluster_norm = cluster_rows / cluster_rows.norm(dim=1, keepdim=True).clamp_min(1e-12)

        content_sim = content_norm.T @ content_norm
        cluster_sim = cluster_norm @ cluster_norm.T
        joint_sim = torch.minimum(content_sim, cluster_sim)
        joint_sim.fill_diagonal_(-1.0)

        max_sim, max_j = joint_sim.max(dim=1)
        overall_max, overall_i = max_sim.max(dim=0)
        if float(overall_max.item()) <= _MERGE_THRESHOLD:
            return None

        i_local = int(overall_i.item())
        j_local = int(max_j[i_local].item())
        i_global = int(committed_idx[i_local].item())
        j_global = int(committed_idx[j_local].item())

        if embedder.usage_ema[i_global] < embedder.usage_ema[j_global]:
            loser, winner = i_global, j_global
        else:
            loser, winner = j_global, i_global
        return {"loser": loser, "winner": winner}

    @staticmethod
    def _apply(embedder: "Embedder", decoder: "Decoder", plan: dict[str, int] | None) -> None:
        if plan is None:
            return

        loser = plan["loser"]
        winner = plan["winner"]
        embedder.usage_ema[winner] = embedder.usage_ema[winner] + embedder.usage_ema[loser]
        embedder.usage_ema[loser] = 0.0
        embedder.committed[loser] = False
        decoder.linears[TensorKey.cluster.name].weight.data[loser, :].zero_()


cluster.callback(ClusterReviveCallback, ClusterMergeCallback)

# ty: ignore[invalid-method-override,unknown-argument]
from __future__ import annotations

import math
import warnings
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast
from weakref import ReferenceType, ref

import numpy as np
import pydantic
import torch
from beartype import beartype
from lightning.pytorch import Callback, Trainer
from tensordict import TensorDict, tensorclass

from relflow.data.nested import apply, extract_mask_literals, pad
from relflow.distributed import broadcast_object
from relflow.rich import console, record_incident
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
    apply_mask_policies,
)
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularyState, VocabularySyncCallback

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema

cluster: Plugin = Plugin(name="cluster")

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
@tensorclass
class TensorField(TensorFieldBase):
    state: torch.Tensor
    content: torch.Tensor
    trainable: torch.Tensor
    targets: TensorDict[TensorKey, torch.Tensor]

    @classmethod
    def new(
        cls,
        values: list,
        address: Address,
        schema: Schema,
        strata: Strata,
        interprocess_encoding_context: VocabularyState,
    ) -> TensorFieldBase:
        array_shape: tuple[int, ...] = schema.shapes[address]
        leading_shape: tuple[int, ...] = (len(values), *array_shape)
        values, literal_masks = extract_mask_literals(
            values,
            strata=strata,
            address=address,
            leaf_depth=len(leading_shape),
        )
        learn = strata == Strata.train

        rejected = interprocess_encoding_context.reserve(values, learn=learn)
        tokens = apply(values, interprocess_encoding_context.encode)

        if rejected and not interprocess_encoding_context.is_shared:
            rejected = interprocess_encoding_context.drain_rejections()
            capacity = schema.requests[address].capacity
            incident = record_incident(
                "vocabulary-capacity",
                "cluster",
                str(address),
                capacity,
                scope=schema,
            )
            if incident.emit and not incident.overflow:
                console.log(
                    "[relflow.warning]cluster vocabulary reached configured capacity; unseen values are unavailable[/]",
                    {"address": address, "capacity": capacity, "rejected": rejected},
                )
            elif incident.emit:
                console.log("[relflow.warning]additional vocabulary-capacity diagnostics are suppressed[/]")

        data, states = pad(
            nested=tokens,
            shape=leading_shape,
            dtype=np.int64,
            pad_value=0,
            overflows=schema.overflows(address),
            address=address,
        )
        literal_data, _ = pad(
            nested=literal_masks,
            shape=leading_shape,
            dtype=bool,
            pad_value=False,
            overflows=schema.overflows(address),
            address=address,
        )

        state_tensor = torch.tensor(states, dtype=torch.int64)
        literal_mask_tensor = torch.tensor(literal_data, dtype=torch.bool)
        state_tensor = state_tensor.masked_fill(literal_mask_tensor, Tokens.masked.value)
        content = torch.tensor(data=data, dtype=torch.int64)
        content = content.masked_fill(literal_mask_tensor, 0)
        if strata == Strata.train:
            p_unavailable: float = schema.requests[address].p_unavailable
            unavailable_index: int = schema.requests[address].capacity

            if p_unavailable > 0.0:
                # Unavailable content never appears naturally during training, because the
                # train split is exactly where the vocabulary is built. We simulate a small
                # amount of OOV behavior so the content objective does not reward any real
                # class for valued inputs whose categorical content is unavailable.
                is_known = state_tensor.eq(Tokens.valued.value) & content.ne(unavailable_index)
                if is_known.any():
                    simulated = (
                        torch.rand_like(input=state_tensor, dtype=torch.float).lt(other=p_unavailable) & is_known
                    )
                    if simulated.any():
                        content = content.masked_fill(simulated, unavailable_index)

        return cls(
            state=state_tensor,
            content=content,
            trainable=torch.zeros_like(input=state_tensor, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=len(values),
        )

    def hide(self, selected: torch.Tensor, *, cache_targets: bool = True, trainable: bool = True):
        selected = selected.to(device=self.state.device, dtype=torch.bool)
        mask_token = torch.full_like(input=self.state, fill_value=Tokens.masked.value)

        if cache_targets and TensorKey.state not in self.targets.keys():
            self.targets[TensorKey.state] = self.state.clone()

        if cache_targets and TensorKey.content not in self.targets.keys():
            self.targets[TensorKey.content] = self.content.clone()

        self.state = self.state.masked_scatter(selected, mask_token)
        self.content = self.content.masked_fill(selected, 0)

        if trainable:
            self.trainable |= selected

    def mask(self, p_mask: float = 0.0, **kwargs: Any):
        apply_mask_policies(self, p_mask=p_mask, **kwargs)

    def target(self, p_prune: float = 1.0):
        apply_mask_policies(self, p_prune=p_prune)

    @classmethod
    def empty(
        cls,
        batch_size: int,
        address: Address,
        schema: Schema,
    ):
        shape: tuple[int, ...] = (batch_size, *schema.shapes[address])

        state = torch.full(shape, Tokens.masked)
        content = torch.zeros(shape, dtype=torch.int64)

        return cls(
            state=state,
            content=content,
            trainable=torch.zeros_like(input=state, dtype=torch.bool),
            targets=TensorDict({}),
            batch_size=batch_size,
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

        if request.p_mask == 0.0 and request.p_prune == 0.0:
            # Cluster loss fires only for masked/pruned/target rows;
            # a plain-input Cluster silently freezes n_committed at init.
            warnings.warn(
                f"Cluster field {address!s} has p_mask=0 and p_prune=0; dynamic K-selection "
                "will not engage. Set p_mask, p_prune, or target=True to train the cluster head.",
                UserWarning,
                stacklevel=2,
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
    def forward(self, inputs: TensorFieldBase) -> Parcel:
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
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )

    @property
    def interprocess_encoding_context(self) -> VocabularyState:
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

    known = valued & content_targets.lt(embedder.capacity)
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
        vocab_size: int = len(embedder.vocab.master)
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
def write(module: Model, prediction: Prediction):
    node = module.nodes[prediction.address]
    embedder: Embedder = node.embedder
    state_logits: torch.Tensor = prediction.payload[TensorKey.state]
    cluster_logits: torch.Tensor = prediction.payload[TensorKey.cluster]

    tokens = np.fromiter((token.name for token in Tokens), dtype=object, count=len(Tokens))
    state_log_norm = state_logits.logsumexp(dim=-1, keepdim=True)
    state_distribution = (state_logits - state_log_norm).exp().detach().float().cpu().numpy()
    state_payload = {token: state_distribution[..., index] for index, token in enumerate(tokens.tolist())}

    cluster_log_probs = torch.log_softmax(cluster_logits, dim=-1)
    cluster_probs = cluster_log_probs.exp()
    cluster_max_logprobs, cluster_ids = cluster_log_probs.max(dim=-1)
    cluster_payload = {
        TensorKey.value.name: cluster_ids.detach().cpu().numpy().astype(np.int32),
        TensorKey.probability.name: cluster_max_logprobs.exp().detach().float().cpu().numpy(),
    }

    assign_weight: torch.Tensor = embedder.embeddings[TensorKey.cluster.name].weight
    vocab_logits: torch.Tensor = cluster_probs @ assign_weight.T

    vocab = np.array(embedder.vocab.snapshot(), dtype=object)
    content_shape = tuple(state_distribution.shape[:-1])
    content_labels = np.full(content_shape, None, dtype=object)
    content_probabilities = np.zeros(content_shape, dtype=np.float32)
    if len(vocab) > 0:
        candidate_indices = torch.arange(len(vocab), device=vocab_logits.device, dtype=torch.int64)
        candidate_logits = vocab_logits.index_select(dim=-1, index=candidate_indices)
        log_norm = candidate_logits.logsumexp(dim=-1, keepdim=True)
        max_logits, max_indices = candidate_logits.max(dim=-1)
        content_probabilities = (max_logits - log_norm.squeeze(-1)).exp().detach().float().cpu().numpy()
        max_indices_np: np.ndarray = max_indices.detach().cpu().numpy().astype(np.int32)
        content_labels = vocab[max_indices_np]

    return {
        TensorKey.state.name: state_payload,
        TensorKey.cluster.name: cluster_payload,
        TensorKey.content.name: {
            TensorKey.value.name: content_labels,
            TensorKey.probability.name: content_probabilities,
        },
    }


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

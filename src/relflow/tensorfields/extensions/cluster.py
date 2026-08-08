# ty: ignore[invalid-method-override,unknown-argument]
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

import numpy as np
import pydantic
import torch
from beartype import beartype
from loguru import logger
from tensordict import TensorDict, tensorclass

from relflow.data.nested import apply, extract_mask_literals, pad
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

    n_clusters: Annotated[
        int | tuple[int, int],
        pydantic.Field(alias="bounds", serialization_alias="bounds", description="Either a single size (int) or a tuple (lower_bound, upper_bound)")
    ]

    @property
    def size(self) -> int:
        if isinstance(self.n_clusters, tuple):
            return self.n_clusters[-1]
        return self.n_clusters

    @size.setter
    def size(self, value: int) -> None:
        self.n_clusters = value
        self.model_fields_set.add("n_clusters")

    @pydantic.model_validator(mode="before")
    @classmethod
    def reject_removed_options(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "max_vocab_size" in data:
            raise ValueError("max_vocab_size was removed; use size")

        return data

    @pydantic.model_validator(mode="after")
    def check_n_clusters(self):
        if isinstance(self.n_clusters, tuple):
            if len(self.n_clusters) != 2:
                raise ValueError("n_clusters tuple must have exactly 2 elements (lower_bound, upper_bound)")
            lower, upper = self.n_clusters
            if lower <= 0:
                raise ValueError("n_clusters lower bound must be greater than 0")
            if lower > upper:
                raise ValueError("n_clusters lower bound must be less than or equal to upper bound")
        elif isinstance(self.n_clusters, int):
            if self.n_clusters <= 0:
                raise ValueError("n_clusters must be greater than 0")
        else:
            raise ValueError("n_clusters must be an int or a tuple of two ints")
        
        return self

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

        interprocess_encoding_context.reserve(values, learn=learn)
        tokens = apply(values, interprocess_encoding_context.encode)

        if len(interprocess_encoding_context) > (size := schema.requests[address].size):
            logger.bind(component="tensorfield", field_type="cluster", address=str(address)).warning(
                "vocabulary exceeds size={}", size
            )

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
            unavailable_index: int = schema.requests[address].size

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
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]
        self.origin: Address = address
        self.destination: Address = request.parent.address
        self.capacity: int = request.capacity
        if isinstance(request.size, tuple):
            self.size: int = request.size[-1] #TODO consider whether to leave this as a tuple or instead a typed dictionary
        else: 
            self.size: int = request.size

        self.vocab: OnlineVocabularyModel = OnlineVocabularyModel(size=self.capacity)

        self.embeddings = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Embedding(
                    num_embeddings=len(Tokens),
                    embedding_dim=schema.d_model,
                ),
                TensorKey.cluster.name: torch.nn.Embedding(
                    num_embeddings=self.capacity, 
                    embedding_dim=self.size
                ),
                TensorKey.content.name: torch.nn.Linear(
                    in_features=self.size,
                    out_features=schema.d_model,
                ),
            }
        )
        self.counters = torch.nn.ModuleDict(
            {
                TensorKey.state.name: Counter(address=address, size=len(Tokens)),
                TensorKey.cluster.name: Counter(address=address, size=request.size),
            }
        )

    @beartype
    def forward(self, inputs: TensorFieldBase) -> Parcel:
        N: int
        dims: list[int]

        N, *dims = inputs.state.shape
        state = inputs.state.reshape(-1)
        content = inputs.content.reshape(-1)
        valued = state.eq(Tokens.valued.value)

        if valued.any() and (content.masked_select(valued) > self.capacity).any().item():
            raise ValueError(f"Token in address {self.origin} exceeds vocabulary size of {self.capacity}")

        known = valued & content.lt(self.capacity)
        safe_content = content.masked_fill(~known, 0)
        cluster_assignments = torch.softmax(self.embeddings[TensorKey.cluster.name](safe_content), dim=-1) * known.unsqueeze(-1) 
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


@cluster.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = schema.requests[address]

        if isinstance(request.size, tuple):
            n_clusters: int = request.size[-1]
        else: 
            n_clusters: int = request.size

        self.linears = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=len(Tokens),
                ),
                TensorKey.cluster.name: torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=n_clusters
                )
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
    """
    The loss function should:
    1. Encourage content (the input categories) to be moved into a cluster of best fit
    2. Encourage the cluster to be embedded as a meaningful representation into latent space

    (1) In order to encourage the content to be moved into a cluster of best fit, we choose to enforce that the embedding table outputs cluster assignments in probability space via softmax. However, as this probability space is used to then feed a content embedding of dim-model, we must enforce that we only feed-forward the arg-max of the probability space--we only feed-forward the cluster assignment itself. This, in itself as a requirement, then raises some concerning questions:
        - If we use a linear map from a vector with only one (1) non-zero value, and a non-zero value of value 1 explicitly, how is this different from two stacked embedding tables? I suppose differentiability is the primary difference, but I also wonder how the arg-max function will operate under differentiation--perhaps this will be implemented as a value translation followed by ReLU. 
            - We should use the torch.nn.functional.gumbel_softmax function in order to implement differentiable arg-max as an operation. 
        - Do we want a hard cluster, or soft clusters? Soft clusters aren't really clusters as much as they are information condensed embeddings
            - We must use hard-cluster definitions for downstream mapping. This is tied to the above--only use the single most predicted cluster assignment from the cluster mapping
        - How shall we enforce cluster sparsity in the case of `isinstance(request.size, tuple)`
            - We must implement a form of Lasso regression in order to enforce cluster sparsity (and we must now define and expose a new hyper-parameter which toggles the weight with which we penalize for sparsity)
        - How will OOV work?
            - It seems that the appropriate solution would be either the mode or the average cluster assignment as selected from the embedding table itself... perhaps I will consult with Grantham on this... 

    (2) In order to encourage that each input is moved into a cluster of best fit, we should seek to recreate the cluster from the d_model embedding given by the encoders' ultimate linear map. This is just a matter of building the decoder correctly, and ensuring that as information gets pooled via cross-attention up through the root and through decoder trees, that all information from the tree agrees with the cluster assignment. This is, in most part, given by cross-entropy of the cluster assignment much the same way that the `Category` data-type operates.
    """
    embedder: Embedder = module.nodes[prediction.address].embedder
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
        value=state_inputs.new_tensor(len(embedder.vocab.snapshot()), dtype=torch.float32),
    )

    valued: torch.Tensor = trainable & state_targets.eq(Tokens.valued.value)
    if not valued.any():
        return loss

    cluster_inputs = prediction.payload[TensorKey.cluster].reshape(N, -1)
    cluster_targets = batch.targets[TensorKey.cluster].reshape(N)

    cluster_inputs 

    n_content_tokens = cluster_inputs.shape[-1]
    invalid = valued & cluster_targets.gt(n_content_tokens)
    if invalid.any():
        raise ValueError(f"Token in address {prediction.address} exceeds vocabulary size")

    known = valued & cluster_targets.lt(n_content_tokens)
    unavailable = valued & cluster_targets.eq(n_content_tokens)

    cluster_loss_sum: torch.Tensor = cluster_inputs.new_zeros(())
    if known.any():
        known_losses = torch.nn.functional.cross_entropy(
            input=cluster_inputs[known],
            target=cluster_targets[known],
            weight=cast(Counter, embedder.counters[TensorKey.cluster.name]).weight,
            reduction="none",
        )
        cluster_loss_sum = cluster_loss_sum + known_losses.sum()

    if unavailable.any():
        unavailable_losses = -torch.nn.functional.log_softmax(cluster_inputs[unavailable], dim=1).mean(dim=1)
        cluster_loss_sum = cluster_loss_sum + unavailable_losses.sum()

    cluster_loss = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.cluster), #TODO: TensorKey.cluster is public facing via logs--consider renaming to cluster
        value=cluster_loss_sum / valued.float().sum().clamp_min(1.0),
    )
    loss += cluster_loss

    if not known.any():
        return loss

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.content),
        value=cluster_inputs.argmax(dim=1).eq(cluster_targets).masked_select(known).float().mean(),
    )

    return loss


@cluster.register
def write(module: Model, prediction: Prediction):
    state_logits: torch.Tensor = prediction.payload[TensorKey.state]
    cluster_logits: torch.Tensor = prediction.payload[TensorKey.cluster]

    tokens = np.fromiter((token.name for token in Tokens), dtype=object, count=len(Tokens))
    state_log_norm = state_logits.logsumexp(dim=-1, keepdim=True)
    state_distribution = (state_logits - state_log_norm).exp().detach().float().cpu().numpy()
    state_payload = {token: state_distribution[..., index] for index, token in enumerate(tokens.tolist())}

    cluster_log_norm = cluster_logits.logsumexp(dim=-1, keepdim=True)
    cluster_distribution = (cluster_logits - cluster_log_norm).exp().detach().float().cpu().numpy()
    cluster_payload = {TensorKey.probability.name: cluster_distribution}

    return {
        TensorKey.state.name: state_payload,
        TensorKey.cluster.name: cluster_payload,
    }

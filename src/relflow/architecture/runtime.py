"""Forward, loss, encoding, and Arrow output for RelFlow models."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypeAlias, TypedDict, cast

import pyarrow as pa
import torch
from loguru import logger
from tensordict import TensorDict

from relflow.architecture.contracts import sanitize
from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.node import NodeModule
from relflow.data.arrow import Batch, Encoded, mappings
from relflow.data.datasets.base import EncodedInput
from relflow.data.iterables import encode as encode_batch
from relflow.data.processors import Postprocessor, Preprocessor
from relflow.distributed import all_reduce_max
from relflow.structs.enums import Component, Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
)
from relflow.tensorfields.output import STATE, embedding, inferred, shape, state, struct

if TYPE_CHECKING:
    from relflow.architecture.root import Model

Retain = tuple[str, ...] | Literal["*"]
PredictionInput: TypeAlias = Batch | pa.Table | pa.RecordBatch | Sequence[Mapping[str, Any]]
RESERVED = frozenset({TensorKey.state.name, TensorKey.inferred.name, TensorKey.embedding.name})


class Output(TypedDict):
    loss: NotRequired[torch.Tensor]


def participation(
    module: Model,
    inputs: TensorDict[Address, TensorFieldBase],
    strata: Strata,
) -> set[Address]:
    """Resolve objective addresses selected on at least one distributed rank."""

    objectives = tuple(module.schema.objectives)
    if strata == Strata.predict or not objectives:
        return set()

    local = torch.stack([inputs[address].trainable.any() for address in objectives]).to(dtype=torch.uint8)
    selected = all_reduce_max(local).tolist()
    return {address for address, active in zip(objectives, selected, strict=True) if active}


@dataclass(frozen=True, slots=True)
class OutputEntry:
    """Frozen output contract for one expected forward address."""

    address: Address
    axes: tuple[int, ...]
    decoded: pa.StructType | None
    writer: Callable[..., pa.StructArray | None] | None
    embed: bool
    extension: str | None
    coordinate: pa.StructType | None
    output: pa.DataType | None


@dataclass(frozen=True, slots=True)
class OutputPlan:
    """Ordered prediction contract compiled for one model generation."""

    generation: int
    retain: Retain
    expected: frozenset[Address]
    entries: tuple[OutputEntry, ...]


def ingress(source: PredictionInput) -> Batch | pa.Table | pa.RecordBatch:
    """Adapt one small Python prediction collection to Arrow exactly once."""

    if isinstance(source, (Batch, pa.Table, pa.RecordBatch)):
        return source
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
        raise TypeError(
            "prediction input must be an rf.Batch, pyarrow Table, pyarrow RecordBatch, "
            f"or a sequence of mappings; got {type(source).__name__}"
        )
    if not source:
        raise ValueError("an empty Python prediction sequence has no Arrow schema; pass a typed Arrow table")
    if all(isinstance(value, Mapping) and not value for value in source):
        from relflow.data.datasets.arrow import identity

        return Batch(
            data=pa.table({}),
            identity=identity(len(source), namespace="direct:predict", offset=0),
        )
    try:
        return pa.Table.from_pylist(mappings(source, context="Python prediction observations"))
    except (pa.ArrowException, TypeError, ValueError) as error:
        raise TypeError(f"Python prediction observations are not Arrow-compatible: {error}") from error


def retention(names: Retain) -> Retain:
    """Validate and normalize one retained-input selection."""

    if names == "*":
        return names
    if not isinstance(names, tuple):
        raise TypeError("retain must be a tuple of column names or '*'")
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError("retain entries must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("retain entries must be unique")
    return names


def project(source: Batch, names: Retain) -> pa.Array:
    """Build the canonical retained-input value for each source row."""

    normalized = retention(names)
    selected = tuple(source.data.column_names) if normalized == "*" else normalized

    missing = [name for name in selected if name not in source.data.column_names]
    if missing:
        formatted = ", ".join(repr(name) for name in missing)
        raise KeyError(f"retained column(s) are absent after preprocessing: {formatted}")
    if not selected:
        return pa.nulls(len(source))

    fields = [source.data.schema.field(name) for name in selected]
    arrays = [source.data[name].combine_chunks() for name in selected]
    return pa.StructArray.from_arrays(arrays, fields=fields)


def axes(module: Model, address: Address) -> tuple[int, ...]:
    """Return repeated model axes represented inside one output row."""

    if address in module.schema.requests:
        return tuple(module.schema.requests[address].shape[1:])
    if address in module.schema.branches:
        lengths = tuple(
            node.length for node in module.schema.branches[address].path if getattr(node, "type", None) == "branch"
        )
        return lengths[1:-1]
    raise KeyError(f"prediction address {str(address)!r} is absent from the model schema")


def plan(module: Model, retain: Retain = (), *, refresh: bool = False) -> OutputPlan:
    """Compile and cache the exact prediction contract for this model generation."""

    generation = int(getattr(module, "_contract_generation", 0))
    retained = retention(retain)
    plans = getattr(module, "output_plans", None)
    if not isinstance(plans, dict):
        plans = {}
        module.output_plans = plans
    key = (generation, retained)
    cached = plans.get(key)
    if not refresh and isinstance(cached, OutputPlan):
        return cached

    expected = frozenset(Address(str(address)) for address in (*module.schema.decodes, *module.schema.embed))
    entries: list[OutputEntry] = []
    visited: set[Address] = set()
    for node in (module.schema.fields, *module.schema.fields.descendants):
        address = Address(str(node.address))
        if address not in expected:
            continue

        visited.add(address)
        decoded: pa.StructType | None = None
        writer: Callable[..., pa.StructArray | None] | None = None
        extension_name: str | None = None
        if address in module.schema.requests:
            extension = TENSORFIELDS[module.schema.requests[address].type]
            extension_name = extension.name
            declared = extension.output(module=module, address=address)
            if declared is not None and not isinstance(declared, pa.StructType):
                raise TypeError(f"extension output for {address!s} must return a pyarrow StructType or None")
            if declared is not None:
                conflicts = RESERVED.intersection(field.name for field in declared)
                if conflicts:
                    names = ", ".join(sorted(conflicts))
                    raise ValueError(f"extension output for {address!s} uses reserved field(s): {names}")
                decoded = declared
                writer = extension.write

        coordinate_fields: list[pa.Field] = []
        if decoded is not None:
            coordinate_fields.append(pa.field(TensorKey.state.name, STATE, nullable=False))
            coordinate_fields.extend(decoded)
            coordinate_fields.append(pa.field(TensorKey.inferred.name, pa.bool_(), nullable=False))
        if address in module.schema.embed:
            coordinate_fields.append(
                pa.field(
                    TensorKey.embedding.name,
                    pa.list_(pa.float32(), module.schema.d_model),
                    nullable=False,
                )
            )
        coordinate_type = pa.struct(coordinate_fields) if coordinate_fields else None
        model_axes = axes(module, address)
        output_type: pa.DataType | None = coordinate_type
        if output_type is not None:
            for size in reversed(model_axes):
                output_type = pa.list_(output_type, size)

        entries.append(
            OutputEntry(
                address=address,
                axes=model_axes,
                decoded=decoded,
                writer=writer,
                embed=address in module.schema.embed,
                extension=extension_name,
                coordinate=coordinate_type,
                output=output_type,
            )
        )

    absent = expected - visited
    if absent:
        formatted = ", ".join(str(address) for address in sorted(absent, key=str))
        raise ValueError(f"configured prediction address(es) are absent from schema traversal: {formatted}")

    compiled = OutputPlan(generation=generation, retain=retained, expected=expected, entries=tuple(entries))
    plans[key] = compiled
    return compiled


def coordinate(module: Model, entry: OutputEntry, prediction: Prediction, rows: int) -> pa.Array | None:
    """Assemble and shape one address-level prediction array."""

    address = entry.address
    fields: list[pa.Field] = []
    arrays: dict[str, pa.Array] = {}
    expected = rows * math.prod(entry.axes)

    if address in module.schema.requests:
        if TensorKey.state not in prediction.payload or TensorKey.inferred not in prediction.payload:
            raise ValueError(f"decoded prediction for {address!s} is missing shared state or inferred tensors")
        state_tensor = prediction.payload[TensorKey.state]
        inferred_tensor = prediction.payload[TensorKey.inferred]
        if state_tensor.ndim == 0 or state_tensor.shape[-1] != len(Tokens):
            raise ValueError(f"prediction state at {address!s} must end with {len(Tokens)} logits")
        if state_tensor.numel() != expected * len(Tokens) or inferred_tensor.numel() != expected:
            raise ValueError(
                f"prediction at {address!s} contains the wrong coordinate count; expected {expected} "
                f"for {rows} rows and model axes {entry.axes}"
            )

        if entry.decoded is not None:
            if entry.writer is None:
                raise RuntimeError(f"compiled output for {address!s} has no extension writer")
            written = entry.writer(module=module, prediction=prediction, datatype=entry.decoded)
            if not isinstance(written, pa.StructArray):
                raise TypeError(f"extension write for {address!s} must return a pyarrow StructArray")
            if written.type != entry.decoded:
                raise TypeError(f"extension write for {address!s} returned {written.type}; expected {entry.decoded}")

            state_values = state(state_tensor)
            fields.append(pa.field(TensorKey.state.name, state_values.type, nullable=False))
            arrays[TensorKey.state.name] = state_values
            for index, field in enumerate(entry.decoded):
                fields.append(field)
                arrays[field.name] = written.field(index)

            inferred_values = inferred(inferred_tensor)
            fields.append(pa.field(TensorKey.inferred.name, inferred_values.type, nullable=False))
            arrays[TensorKey.inferred.name] = inferred_values

    if entry.embed and TensorKey.embedding not in prediction.payload:
        raise ValueError(f"prediction for configured embedding address {address!s} has no embedding tensor")
    if not entry.embed and TensorKey.embedding in prediction.payload:
        raise ValueError(f"prediction for {address!s} returned an unplanned embedding tensor")
    if entry.embed:
        embedding_tensor = prediction.payload[TensorKey.embedding]
        if embedding_tensor.ndim == 0 or embedding_tensor.shape[-1] != module.schema.d_model:
            raise ValueError(f"prediction embedding at {address!s} must end with model width {module.schema.d_model}")
        embedding_values = embedding(embedding_tensor)
        fields.append(pa.field(TensorKey.embedding.name, embedding_values.type, nullable=False))
        arrays[TensorKey.embedding.name] = embedding_values

    if not fields:
        return None

    dtype = pa.struct(fields)
    if dtype != entry.coordinate:
        raise TypeError(f"prediction at {address!s} produced coordinate type {dtype}; expected {entry.coordinate}")
    values = struct(arrays, dtype)
    if len(values) != expected:
        raise ValueError(
            f"prediction at {address!s} wrote {len(values)} coordinates; expected {expected} "
            f"for {rows} rows and model axes {entry.axes}"
        )

    shaped = shape(values, entry.axes)
    if shaped.type != entry.output:
        raise TypeError(f"prediction at {address!s} produced output type {shaped.type}; expected {entry.output}")
    if len(shaped) != rows:
        raise ValueError(f"prediction at {address!s} produced {len(shaped)} output rows; expected {rows}")
    return shaped


def envelope(module: Model, predictions: list[Prediction], rows: int, compiled: OutputPlan | None = None) -> pa.Array:
    """Assemble address arrays in schema traversal order."""

    active = plan(module, refresh=True) if compiled is None else compiled
    indexed: dict[Address, Prediction] = {}
    for prediction in predictions:
        address = Address(str(prediction.address))
        if address in indexed:
            raise ValueError(f"forward returned duplicate prediction address {address!s}")
        indexed[address] = prediction

    missing = active.expected - indexed.keys()
    if missing:
        formatted = ", ".join(str(address) for address in sorted(missing, key=str))
        raise ValueError(f"forward omitted configured prediction address(es): {formatted}")

    unplanned = indexed.keys() - active.expected
    if unplanned:
        formatted = ", ".join(str(address) for address in sorted(unplanned, key=str))
        raise ValueError(f"forward returned unplanned prediction address(es): {formatted}")

    fields: list[pa.Field] = []
    arrays: list[pa.Array] = []
    for entry in active.entries:
        values = coordinate(module, entry, indexed[entry.address], rows)
        if values is None:
            continue
        fields.append(pa.field(str(entry.address), values.type, nullable=False))
        arrays.append(values)

    if not arrays:
        return pa.nulls(rows)
    return pa.StructArray.from_arrays(arrays, fields=fields)


def vacant(compiled: OutputPlan) -> pa.Array:
    """Build the planned public prediction type for a zero-row source."""

    entries = [entry for entry in compiled.entries if entry.output is not None]
    if not entries:
        return pa.nulls(0)
    fields = [pa.field(str(entry.address), entry.output, nullable=False) for entry in entries]
    arrays = [pa.array([], type=entry.output) for entry in entries]
    return pa.StructArray.from_arrays(arrays, fields=fields)


class ModelRuntime:
    """Own runtime behavior that depends on an already-built model graph."""

    @staticmethod
    def forward(
        module: Model,
        inputs: TensorDict[Address, TensorFieldBase],
        *,
        strata: Strata | str,
        dataloader_idx: int = 0,
        participating: set[Address] | None = None,
    ) -> list[Prediction]:
        strata = Strata.normalize(strata)
        sanitize(module, inputs, strata=strata, dataloader_idx=dataloader_idx)

        processed: dict[Address, list[Parcel]] = defaultdict(list)
        outgoing: dict[Address, Parcel] = {}
        predictions: list[Prediction] = []

        participating = participation(module, inputs, strata) if participating is None else participating

        for address in module.schema.active_requests:
            tensorfield: TensorFieldBase = inputs[address]
            node_module = cast(NodeModule, module.nodes[address])
            embedder: EmbedderBase = node_module.embedder
            embedded = embedder.embed(tensorfield)
            if embedded.destination is None:
                raise ValueError(f"parcel from '{embedded.origin}' has no destination")
            processed[embedded.destination].append(embedded)
            outgoing[embedded.origin] = embedded

        for depth in reversed(module.schema.depthwise):
            for address in depth:
                if not processed[address]:
                    continue

                node_module = cast(NodeModule, module.nodes[address])
                encoder: BranchEncoder = node_module.encoder
                encoded: Parcel = encoder(processed[address])
                if encoded.destination is None:
                    raise ValueError(f"parcel from '{encoded.origin}' has no destination")
                processed[encoded.destination].append(encoded)
                outgoing[encoded.origin] = encoded

                if address in module.schema.embed:
                    predictions.append(
                        Prediction(
                            address=encoded.origin,
                            payload=TensorDict(
                                {TensorKey.embedding: encoded.payload},
                                batch_size=encoded.payload.shape[0],
                            ),
                            batch_size=encoded.payload.shape[0],
                        )
                    )

        decodes = set(module.schema.decodes)
        embeds = set(module.schema.embed)
        for address in module.schema.active_requests:
            tensorfield = inputs[address]
            selected = (
                (strata == Strata.predict and address in decodes)
                or (strata != Strata.predict and address in participating)
                or address in embeds
            )
            if not selected:
                continue

            heritage: list[Address] = module.schema.requests[address].heritage
            parcels = [outgoing[item] for item in heritage if item in outgoing]

            node_module = cast(NodeModule, module.nodes[address])
            decoder: DecoderBase = node_module.decoder
            prediction = decoder(
                parcels,
                batch_size=tensorfield.state.shape[0],
                device=tensorfield.state.device,
                embed=address in embeds,
            )
            prediction.payload[TensorKey.inferred] = (
                tensorfield.inferred if strata == Strata.predict else tensorfield.trainable
            )
            predictions.append(prediction)

        return predictions

    @staticmethod
    def step(
        module: Model,
        batch: Encoded | TensorDict[Address, TensorFieldBase],
        batch_idx: int,
        dataloader_idx: int = 0,
        *,
        strata: Strata,
    ) -> Output | Batch | None:
        inputs = batch.tensors if isinstance(batch, Encoded) else batch
        if isinstance(batch, Encoded):
            ModelRuntime.learn(module, batch.observations, strata=strata)
        if strata == Strata.predict and not isinstance(batch, Encoded):
            raise TypeError("prediction batches must retain their Arrow source")
        compiled = plan(module, batch.retain) if isinstance(batch, Encoded) and strata == Strata.predict else None
        participating = participation(module, inputs, strata)
        predictions = ModelRuntime.forward(
            module,
            inputs,
            strata=strata,
            dataloader_idx=dataloader_idx,
            participating=participating,
        )

        if strata == Strata.predict:
            assert isinstance(batch, Encoded)
            return ModelRuntime.write(
                module,
                predictions,
                source=batch.source,
                retain=batch.retain,
                compiled=compiled,
            )

        if not participating:
            logger.warning("no reconstruction objective was selected on any rank, skipping batch")
            if strata == Strata.train:
                return None
            return Output(loss=torch.tensor(0.0, device=inputs.device))

        losses: list[torch.Tensor] = []
        anchors: list[torch.Tensor] = []
        for prediction in predictions:
            if prediction.address not in module.schema.requests:
                continue
            if prediction.address not in module.schema.objectives:
                continue
            if strata == Strata.train and prediction.address in participating:
                anchors.extend(
                    value.sum() * 0.0
                    for value in prediction.payload.values()
                    if torch.is_tensor(value) and value.requires_grad
                )
            # Decoder participation is coordinated across ranks in forward.
            # Extension losses and scalar metrics remain local to ranks carrying
            # selected targets; the zero-loss path below anchors every module
            # into backward when only a peer has an objective.
            if not inputs[prediction.address].trainable.any():
                continue
            if set(prediction.payload.keys()) <= {TensorKey.embedding, TensorKey.inferred}:
                continue

            address = Address(str(prediction.address))
            request: RequestBase = module.schema.requests[address]
            extension: Extension = TENSORFIELDS[request.type]
            loss_fn = cast(Callable[..., torch.Tensor], extension.loss)
            loss = loss_fn(module=module, prediction=prediction, batch=inputs[address], strata=strata)
            losses.append(loss * torch.tensor(request.weight))

        if strata == Strata.train:
            if not anchors:
                raise RuntimeError("a distributed reconstruction objective produced no gradient-bearing decoder output")
            anchor = torch.stack(anchors).sum()
        else:
            anchor = torch.zeros((), device=inputs.device)
        if not losses:
            suffix = "anchored zero loss" if strata == Strata.train else "zero loss"
            logger.warning(f"reconstruction targets are present only on peer ranks, returning {suffix}")
            return Output(loss=anchor)

        loss = module.track((Metric.loss, strata), value=torch.stack(losses).sum() + anchor)
        return Output(loss=cast(torch.Tensor, loss))

    @staticmethod
    def write(
        module: Model,
        predictions: list[Prediction],
        *,
        source: Batch,
        retain: Retain = (),
        compiled: OutputPlan | None = None,
    ) -> Batch:
        """Convert tensor predictions to one canonical Arrow output batch."""

        if not isinstance(source, Batch):
            raise TypeError(f"source must be an rf.Batch, got {type(source).__name__}")
        active = plan(module, retain, refresh=True) if compiled is None else compiled
        if active.retain != retention(retain):
            raise ValueError(f"compiled output plan retain {active.retain!r} does not match write retain {retain!r}")
        inputs = project(source, active.retain)
        outputs = (
            vacant(active)
            if not len(source) and not predictions
            else envelope(module, predictions, len(source), active)
        )
        data = pa.Table.from_arrays(
            [inputs, outputs],
            names=["inputs", "predictions"],
        )
        if len(inputs) != len(source):
            raise ValueError("retained input output is not aligned with source identity")
        return source.replace(data)

    @staticmethod
    def prepare(
        module: Model,
        batch: Batch | pa.Table | pa.RecordBatch,
        *,
        preprocess: Preprocessor | None,
        strata: Strata,
        seed: int = 0,
        epoch: int = 0,
    ) -> Encoded:
        """Normalize Arrow input, preprocess it, and encode one carrier."""

        from relflow.data.datasets.arrow import convert, merge, process

        source = batch if isinstance(batch, Batch) else convert(batch, namespace=f"direct:{strata}", offset=0)
        processor = Preprocessor.normalize(preprocess)
        if processor is not None:
            source = merge(
                process(
                    (source,),
                    preprocessor=processor,
                    strata=strata,
                    schema=module.schema,
                    encoding_context=module.interprocess_encoding_context,
                )
            )
            if source is None:
                raise ValueError(f"preprocessor '{processor.name}' returned no observations")

        return encode_batch(
            batch=source,
            schema=module.schema,
            strata=strata,
            interprocess_encoding_context=module.interprocess_encoding_context,
            seed=seed,
            epoch=epoch,
        )

    @staticmethod
    def learn(
        module: Model,
        observations: Mapping[Address, TensorDict],
        *,
        strata: Strata | str,
    ) -> None:
        """Apply every model-owned pristine observation exactly once."""

        normalized = Strata.normalize(strata)
        if normalized != Strata.train:
            if observations:
                raise ValueError(f"{normalized} input cannot carry learnable observations")
            return

        learners = {
            address: TENSORFIELDS[request.type]
            for address, request in module.schema.active_requests.items()
            if Component.learn in TENSORFIELDS[request.type].components
        }
        missing = set(learners) - set(observations)
        extra = set(observations) - set(learners)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append("missing " + ", ".join(map(str, sorted(missing, key=str))))
            if extra:
                details.append("unexpected " + ", ".join(map(str, sorted(extra, key=str))))
            raise ValueError("training observations do not match registered learners: " + "; ".join(details))

        for address in module.schema.active_requests:
            if address not in learners:
                continue
            learners[address].learn(
                module=module,
                observation=observations[address].to(module.device),
                address=address,
                strata=normalized,
            )

    @staticmethod
    def encode(
        module: Model,
        batch: Batch | pa.Table | pa.RecordBatch,
        preprocess: Preprocessor | None = None,
        strata: Strata | str = Strata.predict,
        seed: int = 0,
        epoch: int = 0,
    ) -> EncodedInput:
        """Encode one Arrow input unit to tensorfields."""

        normalized = Strata.normalize(strata)
        encoded = ModelRuntime.prepare(
            module,
            batch,
            preprocess=preprocess,
            strata=normalized,
            seed=seed,
            epoch=epoch,
        )
        ModelRuntime.learn(module, encoded.observations, strata=normalized)
        return cast(EncodedInput, encoded.tensors)

    @staticmethod
    def predict(
        module: Model,
        batch: PredictionInput,
        preprocess: Preprocessor | None = None,
        postprocess: Postprocessor | None = None,
        retain: Retain = (),
    ) -> pa.Table:
        """Predict one Arrow input unit and return an Arrow table."""

        encoded = ModelRuntime.prepare(
            module,
            ingress(batch),
            preprocess=preprocess,
            strata=Strata.predict,
        )
        source = encoded.source
        inputs = encoded.tensors.to(module.device)
        compiled = plan(module, retain, refresh=True)
        raw: list[Prediction] = []
        if len(source):
            was_training = module.training
            module.eval()
            try:
                with torch.inference_mode():
                    raw = module(inputs, strata=Strata.predict)
            finally:
                if was_training:
                    module.train()

        written = ModelRuntime.write(module, raw, source=source, retain=retain, compiled=compiled)
        processor = Postprocessor.normalize(postprocess)
        if processor is not None:
            written = processor.run(written)
        return written.data


step = ModelRuntime.step

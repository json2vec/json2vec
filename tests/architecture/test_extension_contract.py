import uuid
from typing import Literal

import numpy as np
import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict, tensorclass

import relflow as rf
from relflow.data.ragged import RaggedField
from relflow.structs.enums import Strata, TensorKey
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
    TensorInput,
)


def build_extension(
    *,
    decoder: bool = True,
    loss: bool = True,
    explode: bool = False,
    empty: bool = False,
    state: object | None = None,
    contexts: list[rf.Context] | None = None,
):
    """Register one late-bound extension with nested, trailing-axis tensor content."""

    name = f"extension_{uuid.uuid4().hex}"
    extension = Extension(name=name, types=(str, bytes))

    @extension.register
    class Request(RequestBase):
        type: Literal[name] = name
        family: Literal["text", "bytes"] = "text"

    @extension.register
    @tensorclass
    class TensorField(TensorFieldBase):
        content: TensorDict
        state: torch.Tensor
        present: torch.Tensor
        trainable: torch.Tensor
        inferred: torch.Tensor
        targets: TensorDict

        @classmethod
        def new(
            cls,
            input: RaggedField,
            target: RaggedField,
            present: torch.Tensor,
            trainable: torch.Tensor,
            inferred: torch.Tensor,
            address: Address,
            schema: rf.Schema,
            strata: Strata,
            context: rf.Context,
        ) -> TensorFieldBase:
            if contexts is not None:
                contexts.append(context)

            def encode(field: RaggedField) -> TensorDict:
                lengths = np.asarray([len(value) for value in field.values.to_pylist()], dtype=np.float32)
                matrix = lengths[:, None, None] + np.arange(6, dtype=np.float32).reshape(1, 2, 3)
                cube = lengths[:, None, None, None] + np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
                return TensorDict(
                    {
                        "matrix": torch.from_numpy(field.place(matrix, fill=0.0, value_shape=(2, 3))),
                        "nested": TensorDict(
                            {"cube": torch.from_numpy(field.place(cube, fill=0.0, value_shape=(2, 2, 2)))},
                            batch_size=field.shape,
                        ),
                    },
                    batch_size=field.shape,
                )

            return cls(
                content=encode(input),
                state=torch.from_numpy(input.dense),
                present=present,
                trainable=trainable,
                inferred=inferred,
                targets=TensorDict(
                    {
                        TensorKey.state: torch.from_numpy(target.dense),
                        TensorKey.content: encode(target),
                    },
                    batch_size=target.shape,
                ),
                batch_size=input.batch_size,
            )

    @extension.register
    class Embedder(EmbedderBase):
        def __init__(self, schema: rf.Schema, address: Address):
            super().__init__(schema=schema, address=address)
            self.projection = torch.nn.Linear(14, schema.d_model, bias=False)
            if empty:
                self.empty = torch.nn.Parameter(torch.empty(0))

        @property
        def context(self) -> object | None:
            return state

        def forward(self, inputs: TensorInput) -> Parcel:
            values = torch.cat(
                [
                    inputs.content["matrix"].reshape(inputs.batch_size[0], -1),
                    inputs.content["nested", "cube"].reshape(inputs.batch_size[0], -1),
                ],
                dim=-1,
            )
            payload = self.projection(values)
            return Parcel(
                payload=payload,
                present=torch.ones(len(payload), dtype=torch.bool, device=payload.device),
                origin=self.address,
                destination=self.destination,
                batch_size=len(payload),
            )

    if decoder:

        @extension.register
        class Decoder(DecoderBase):
            def __init__(self, schema: rf.Schema, address: Address):
                if explode:
                    raise RuntimeError("decoder exploded")
                super().__init__(schema=schema, address=address)
                self.state = torch.nn.Linear(schema.d_model, 5)

            def decode(self, pooled: torch.Tensor) -> TensorDict:
                return TensorDict(
                    {TensorKey.state: self.state(pooled)},
                    batch_size=pooled.shape[:-1],
                )

    if loss:

        @extension.register
        def loss(module: object, prediction: Prediction, batch: object, strata: Strata) -> torch.Tensor:
            return prediction.payload[TensorKey.state].sum() * 0.0

    return extension, Request


def test_third_party_extension_receives_one_explicit_context_contract():
    marker = object()
    contexts: list[rf.Context] = []
    extension, Request = build_extension(decoder=False, loss=False, state=marker, contexts=contexts)
    try:
        model = rf.Model(value=Request(), d_model=8, n_layers=1, n_heads=2)

        model.encode(pa.table({"value": ["abc"]}))

        assert model.interprocess_encoding_context == {Address("record/value"): marker}
        assert len(contexts) == 2  # mocked graph input, then the real batch
        assert contexts[-1] == rf.Context(state=marker, salt=0)
    finally:
        TENSORFIELDS.pop(extension.name, None)


@pytest.mark.parametrize("branch", [False, True])
def test_reconstruction_validates_custom_leaf_and_branch_capabilities(branch):
    extension, Request = build_extension(decoder=False, loss=False)
    try:
        field = Request(name="value")
        fields = (
            {"items": rf.Branch(field, length=2, mask=True)}
            if branch
            else {"value": field.model_copy(update={"mask": True})}
        )

        with pytest.raises(ValueError, match=rf"extension '{extension.name}'.*Decoder, loss"):
            rf.Model(d_model=8, n_layers=1, n_heads=2, **fields)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_reconstruction_requires_decoder_and_loss_as_one_capability():
    extension, Request = build_extension(decoder=True, loss=False)
    try:
        with pytest.raises(ValueError, match=rf"extension '{extension.name}'.*required component\(s\): loss"):
            rf.Model(value=Request(mask=True), d_model=8, n_layers=1, n_heads=2)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_leaf_embedding_requires_a_decoder_but_branch_embedding_does_not():
    extension, Request = build_extension(decoder=False, loss=False)
    try:
        with pytest.raises(ValueError, match=rf"embedded leaf 'record/value'.*extension '{extension.name}'.*Decoder"):
            rf.Model(value=Request(embed=True), d_model=8, n_layers=1, n_heads=2)

        model = rf.Model(value=Request(), d_model=8, n_layers=1, n_heads=2, embed=True)
        assert "record" in model.schema.embed
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_failed_capability_mutation_restores_schema_and_runtime_graph():
    extension, Request = build_extension(decoder=False, loss=False)
    try:
        model = rf.Model(value=Request(), d_model=8, n_layers=1, n_heads=2)
        nodes = model.nodes

        with pytest.raises(ValueError, match=rf"extension '{extension.name}'.*Decoder, loss"):
            model.update(rf.where("name") == "value", mask=True)

        assert model.schema.requests["record/value"].mask == ()
        assert model.schema.objectives == []
        assert model.nodes is nodes
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_failed_runtime_rebuild_restores_schema_and_runtime_graph():
    extension, Request = build_extension(explode=True)
    try:
        model = rf.Model(value=Request(), d_model=8, n_layers=1, n_heads=2)
        nodes = model.nodes

        with pytest.raises(RuntimeError, match="decoder exploded"):
            model.update(rf.where("name") == "value", mask=True)

        assert model.schema.requests["record/value"].mask == ()
        assert model.schema.objectives == []
        assert model.nodes is nodes
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_late_extension_schema_round_trip_uses_the_live_registry():
    extension, Request = build_extension(decoder=False, loss=False)
    try:
        schema = rf.Schema.from_tree(
            rf.Branch(Request(name="value", family="bytes"), name="items", length=2),
            d_model=8,
            n_layers=1,
            n_heads=2,
        )

        restored = rf.Schema.model_validate(schema.model_dump(mode="python", round_trip=True))
        restored_json = rf.Schema.model_validate_json(schema.model_dump_json())

        assert isinstance(restored.requests["record/items/value"], Request)
        assert restored.requests["record/items/value"].family == "bytes"
        assert isinstance(restored_json.requests["record/items/value"], Request)
        assert restored_json.requests["record/items/value"].family == "bytes"
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_custom_tensordict_trailing_axes_gather_embed_and_scatter():
    extension, Request = build_extension(decoder=False, loss=False)
    try:
        model = rf.Model(
            value=Request(mask=rf.Mask(query="skip", skip=True, dropout=False)),
            d_model=8,
            n_layers=1,
            n_heads=2,
            embed=True,
        )
        inputs = model.encode(
            pa.table({"value": ["a", "bb", "ccc"], "skip": [False, True, False]}),
            strata=Strata.train,
        )
        seen = []
        embedder = model.nodes["record/value"].embedder
        handle = embedder.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        try:
            predictions = model(inputs, strata=Strata.train)
        finally:
            handle.remove()

        assert seen[0].content["matrix"].shape == (2, 2, 3)
        assert seen[0].content["nested", "cube"].shape == (2, 2, 2, 2)
        root = next(prediction for prediction in predictions if prediction.address == "record")
        assert root.payload[TensorKey.embedding].shape == (3, 8)
        assert torch.equal(root.payload[TensorKey.embedding][1], torch.zeros(8))
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_compact_extension_presence_is_scattered_without_routing_its_payload(monkeypatch):
    extension, Request = build_extension(decoder=False, loss=False)
    try:
        model = rf.Model(value=Request(), d_model=8, n_layers=1, n_heads=2)
        field = model.encode(pa.table({"value": ["a", "bb"]}), strata=Strata.train)["record/value"]
        embedder = model.nodes["record/value"].embedder
        original = embedder.forward

        def omit(inputs):
            compact = original(inputs)
            return Parcel(
                payload=torch.ones_like(compact.payload),
                present=torch.tensor([True, False]),
                origin=compact.origin,
                destination=compact.destination,
                batch_size=compact.batch_size,
            )

        monkeypatch.setattr(embedder, "forward", omit)

        parcel = embedder.embed(field)

        assert parcel.present.tolist() == [[True], [False]]
        assert torch.equal(parcel.payload[1], torch.zeros(1, 8))
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_source_less_multifamily_extension_uses_the_generic_vacancy():
    extension, Request = build_extension()
    try:
        model = rf.Model(label=Request(mask=True, family="bytes"), d_model=8, n_layers=1, n_heads=2)

        predictions = model.predict([{}, {}])

        assert len(predictions) == 2
        assert predictions["predictions"].type == pa.null()
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_all_skipped_custom_embedder_accepts_zero_sized_parameters():
    extension, Request = build_extension(decoder=False, loss=False, empty=True)
    try:
        model = rf.Model(
            value=Request(mask=rf.Mask(skip=True, dropout=False)),
            d_model=8,
            n_layers=1,
            n_heads=2,
        )
        field = model.encode(pa.table({"value": ["a"]}), strata=Strata.train)["record/value"]
        embedder = model.nodes["record/value"].embedder

        parcel = embedder.embed(field)
        parcel.payload.sum().backward()

        assert embedder.empty.grad is not None
        assert embedder.empty.grad.numel() == 0
    finally:
        TENSORFIELDS.pop(extension.name, None)

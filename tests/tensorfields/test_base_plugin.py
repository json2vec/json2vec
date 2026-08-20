import uuid

import pytest
from lightning.pytorch import Callback

from relflow.metrics import Trait
from relflow.structs.enums import Component, Strata
from relflow.structs.tree import Node
from relflow.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Plugin,
    RequestBase,
    TensorFieldBase,
)


def _plugin_name(prefix: str = "plug") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _build_plugin() -> Plugin:
    plugin = Plugin(name=_plugin_name())

    class Request(RequestBase):
        pass

    class TensorField(TensorFieldBase):
        @classmethod
        def new(cls, values, address, schema, strata):
            return object()

        def mask(self, p_mask: float):
            return None

        def target(self, p_prune: float):
            return None

    class Embedder(EmbedderBase):
        def __init__(self, schema: object, address: object):
            super().__init__(schema=schema, address=address)

    class Decoder(DecoderBase):
        def __init__(self, schema: object, address: object):
            super().__init__(schema=schema, address=address)

    plugin.register(Request)
    plugin.register(TensorField)
    plugin.register(Embedder)
    plugin.register(Decoder)

    def loss(module: object, prediction: object, batch: object, strata: Strata):
        return 3.14

    def write(module: object, prediction: object):
        return None

    plugin.register(loss)
    plugin.register(write)

    return plugin


def test_plugin_rejects_invalid_name():
    with pytest.raises(ValueError, match="lowercase letters"):
        Plugin(name="Bad-Name")


def test_plugin_stores_immutable_traits():
    plugin = Plugin(name=_plugin_name("traits"), traits=(Trait.classification,))
    try:
        assert plugin.traits == frozenset({Trait.classification})
        assert isinstance(plugin.traits, frozenset)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_rejects_invalid_traits():
    with pytest.raises(TypeError, match="Trait members"):
        Plugin(name=_plugin_name("badtraits"), traits=("classification",))


def test_plugin_request_must_inherit_request_base():
    plugin = Plugin(name=_plugin_name("badrequest"))

    class Request(Node):
        pass

    try:
        with pytest.raises(TypeError, match="subclass of RequestBase"):
            plugin.register(Request)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_warns_and_overwrites_duplicate_name():
    name = _plugin_name("duplicate")
    first = Plugin(name=name)
    try:
        with pytest.warns(UserWarning, match="overriding existing tensorfield plugin"):
            second = Plugin(name=name)

        assert TENSORFIELDS[name] is second
        assert TENSORFIELDS[name] is not first
    finally:
        TENSORFIELDS.pop(name, None)


def test_plugin_registers_components_and_wraps_loss():
    plugin = _build_plugin()
    try:
        assert Component.Request in plugin.components
        assert Component.TensorField in plugin.components
        assert Component.Embedder in plugin.components
        assert Component.Decoder in plugin.components
        assert Component.loss in plugin.components
        assert Component.write in plugin.components

        class DummyModule:
            def __init__(self):
                self.calls: list[tuple] = []

            def track(self, key: tuple, value: float):
                self.calls.append((key, value))
                return value

        class DummyPrediction:
            address = "root/field"

        module = DummyModule()
        value = plugin.loss(module, prediction=DummyPrediction(), batch=object(), strata=Strata.train)
        assert value == 3.14
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_write_defaults_to_no_output_when_unregistered():
    plugin = Plugin(name=_plugin_name("defaultwrite"))
    try:
        assert plugin.write(module=object(), prediction=object()) is None
        assert Component.write not in plugin.components
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_accepts_explicit_none_write():
    plugin = Plugin(name=_plugin_name("nonehooks"))
    try:
        plugin.register(None, component=Component.write)

        assert plugin.write(module=object(), prediction=object()) is None
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_registers_multiple_callback_factories():
    class FirstCallback(Callback):
        pass

    class SecondCallback(Callback):
        pass

    plugin = Plugin(name=_plugin_name("callbacks"))
    try:
        result = plugin.callback(FirstCallback, SecondCallback)

        assert result == (FirstCallback, SecondCallback)
        assert plugin.callback_factories == [FirstCallback, SecondCallback]
        assert [type(callback) for callback in plugin.callbacks] == [FirstCallback, SecondCallback]
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_rejects_invalid_callback_factory_without_registering():
    class InvalidCallback:
        pass

    plugin = Plugin(name=_plugin_name("badcallbacks"))
    try:
        with pytest.raises(TypeError, match="must produce a Lightning Callback"):
            plugin.callback(InvalidCallback)

        assert plugin.callback_factories == []
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_rejects_embedder_with_missing_address_param():
    class Embedder(EmbedderBase):
        def __init__(self, schema: object):
            super().__init__(schema=schema, address=None)

    plugin = Plugin(name=_plugin_name("badembedder"))
    try:
        with pytest.raises(TypeError, match="must accept 'schema' and 'address'"):
            plugin.register(Embedder)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_getattr_rejects_invalid_component_key():
    plugin = Plugin(name=_plugin_name("badattr"))
    try:
        with pytest.raises(ValueError, match="is not a valid Component enum value"):
            plugin.__getattr__("not_real")
    finally:
        TENSORFIELDS.pop(plugin.name, None)

import uuid

import pytest

from json2vec.structs.enums import Component, Strata
from json2vec.structs.tree import Node
from json2vec.tensorfields.base import (
    TENSORFIELDS,
    DecoderBase,
    EmbedderBase,
    Plugin,
    TensorFieldBase,
)


def _plugin_name(prefix: str = "plug") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _build_plugin() -> Plugin:
    plugin = Plugin(name=_plugin_name())

    class Request(Node):
        pass

    class TensorField(TensorFieldBase):
        @classmethod
        def new(cls, values, address, session, strata, state):
            return object()

        def mask(self, p_mask: float):
            return None

        def prune(self, p_prune: float):
            return None

    class Embedder(EmbedderBase):
        def __init__(self, structure: object, address: object):
            super().__init__(structure=structure, address=address)

    class Decoder(DecoderBase):
        def __init__(self, structure: object, address: object):
            super().__init__(structure=structure, address=address)

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


def test_plugin_rejects_duplicate_name():
    name = _plugin_name("duplicate")
    plugin = Plugin(name=name)
    try:
        with pytest.raises(ValueError, match="already registered"):
            Plugin(name=name)
    finally:
        TENSORFIELDS.pop(plugin.name, None)


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
        assert module.calls and module.calls[0][1] == 3.14
    finally:
        TENSORFIELDS.pop(plugin.name, None)


def test_plugin_rejects_embedder_with_missing_address_param():
    class Embedder(EmbedderBase):
        def __init__(self, structure: object):
            super().__init__(structure=structure, address=None)

    plugin = Plugin(name=_plugin_name("badembedder"))
    try:
        with pytest.raises(TypeError, match="must accept 'structure' and 'address'"):
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

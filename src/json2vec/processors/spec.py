import pluggy

hookspec = pluggy.HookspecMarker("processors")


class PluginSpec:
    @hookspec
    def plugin_class(self) -> None: ...

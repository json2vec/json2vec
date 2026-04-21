import pluggy

hookspec = pluggy.HookspecMarker("tensorfields")


class PluginSpec:
    @hookspec
    def plugin_class(self) -> None: ...

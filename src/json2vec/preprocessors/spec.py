import pluggy

hookspec = pluggy.HookspecMarker("preprocessors")


class PluginSpec:
    @hookspec
    def plugin_class(self) -> None: ...

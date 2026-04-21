from __future__ import annotations

import importlib
import pkgutil

subpkg = importlib.import_module(".extensions", __name__)
for _, fullname, _ in pkgutil.iter_modules(subpkg.__path__, subpkg.__name__ + "."):
    importlib.import_module(fullname)

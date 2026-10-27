"""Import the published (unmodified) resonance_network.py from vendor/.

CACHE_DIR in that module is read from the env var RN_CACHE_DIR at import time, so callers that need
a specific cache must call `set_cache_dir()` before the first `rn()` call.
"""
import importlib.util
import os
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "vendor" / "resonance_network.py"
_mod = None


def set_cache_dir(path):
    global _mod
    os.environ["RN_CACHE_DIR"] = str(path)
    if _mod is not None:
        from pathlib import Path as P
        _mod.CACHE_DIR = P(str(path))


def rn():
    global _mod
    if _mod is None:
        spec = importlib.util.spec_from_file_location("resonance_network_vendor", _PATH)
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
    return _mod

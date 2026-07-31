"""Load the Mojo shared library and declare its C ABI."""

from __future__ import annotations

import ctypes
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJO_SKMULTILEARN_LIB") or os.path.join(
    ROOT, "dist", "libmojo-scikit-multilearn.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "msml_knn_query": ([I] * 10, None),
    "msml_neighbor_label_counts": ([I] * 6, None),
    "msml_take_labelsets": ([I] * 5, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_SKMULTILEARN_LIB") and os.path.exists(LIB):
        return LIB
    sources = [
        os.path.join(path, name)
        for path, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    stale = not os.path.exists(LIB) or (
        sources and os.path.getmtime(LIB) < max(map(os.path.getmtime, sources))
    )
    if force or stale:
        proc = subprocess.run(
            ["bash", os.path.join(ROOT, "build", "build.sh")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0 or not os.path.exists(LIB):
            raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array) -> int:
    return array.ctypes.data

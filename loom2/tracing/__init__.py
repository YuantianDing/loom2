"""Trace Python tensor callbacks directly into the native Rust ``TensorFn``.

There is intentionally no Python-side expression AST. ``Vec(bool)`` values
remain packed bit-vectors, selections cross the native boundary as flat
source-bit positions, and each read or assignment becomes one ``BVGather`` or
``BVScatter`` node.
"""

from .ast import JaggedTensorParts, NativeTensorFnBuilder, ReducedExpr
from .tracer import TraceValue, TracedVec, is_tracing, trace

__all__ = [
    "JaggedTensorParts",
    "NativeTensorFnBuilder",
    "ReducedExpr",
    "TraceValue",
    "TracedVec",
    "is_tracing",
    "trace",
]

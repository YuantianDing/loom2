"""Python value schemas understood by the current tensor tracer.

The earlier ``BinaryAssign`` and finite-domain frontend is not part of the
active Rust ``TensorFn`` design. ``Vec`` is the sole schema boundary: Boolean
leaves are runtime bit-vectors and integer leaves are static gather indices.
"""

from .vec import Vec, VecIndex, VecValue

__all__ = [
    "Vec",
    "VecIndex",
    "VecValue",
]

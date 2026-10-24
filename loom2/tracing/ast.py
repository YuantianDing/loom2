"""Data-transfer contracts for tracing directly into the Rust compile IR.

This module intentionally contains no Python expression AST. During tracing,
Python values hold opaque expression handles owned by a Rust
``TensorFnBuilder``. Finishing the builder returns the native ``TensorFn``;
there is no ``TensorAst`` or per-bit ``TraceNode`` conversion step.

``JaggedTensorParts`` is only a short-lived representation of the arguments to
the native ``JaggedTensor`` constructor. The Rust binding must validate the
offsets before placing them in the compile IR.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, Tuple, TypeVar, runtime_checkable


T = TypeVar("T")
ExprHandle = object


@dataclass(frozen=True)
class JaggedTensorParts(Generic[T]):
    """Flat payload and offsets passed to a native ``JaggedTensor<T>``.

    A one-dimensional tensor has no ``jagged_dims``. For a jagged layer, its
    offsets start at zero, are nondecreasing, and include the terminal offset.
    For example, ``[[2], [3, 5], []]`` is transported as data ``(2, 3, 5)``
    and offsets ``((0, 1, 3, 3),)``.

    The native builder remains responsible for validating nested layers and
    converting this immutable Python transport object into the generic Rust
    container.
    """

    data: Tuple[T, ...]
    tensor_dims: Tuple[int, ...]
    jagged_dims: Tuple[Tuple[int, ...], ...] = ()

    @classmethod
    def flat(cls, data: Sequence[T]) -> "JaggedTensorParts[T]":
        """Create a one-dimensional, non-jagged tensor."""
        values = tuple(data)
        return cls(values, (len(values),))

    @classmethod
    def rows(cls, rows: Sequence[Sequence[T]]) -> "JaggedTensorParts[T]":
        """Create a two-dimensional tensor whose last dimension is jagged."""
        data = []
        offsets = [0]
        for row in rows:
            data.extend(row)
            offsets.append(len(data))
        return cls(tuple(data), (len(rows),), (tuple(offsets),))


@dataclass(frozen=True)
class ReducedExpr:
    """One native expression and the bit-tensor axes reduced from it.

    A reduction always folds the jagged (lane) dimension of the expression's
    packed layout; ``reduce_dims`` names that logical axis.
    """

    expression: ExprHandle
    reduce_dims: Tuple[int, ...] = ()


@runtime_checkable
class NativeTensorFnBuilder(Protocol):
    """Python-visible contract required from the Rust tracing bridge.

    The eventual PyO3 ``TensorFnBuilder`` owns all ``TensorExpr`` nodes. Each
    method below returns an opaque handle into that builder; Python must never
    inspect or serialize the expression graph.
    """

    def input_bit_vector(
        self,
        name: str,
        mask: JaggedTensorParts[int],
    ) -> ExprHandle:
        """Create one ``Input`` over a one-dimensional packed bit-vector.

        ``mask.data`` contains packed ``u64`` validity words and
        ``mask.jagged_dims`` must be empty. The resulting ``TensorType`` uses
        ``rest = false``.
        """
        ...

    def constant_bit_vector(
        self,
        value: JaggedTensorParts[int],
        mask: JaggedTensorParts[int],
    ) -> ExprHandle:
        """Create one packed ``Constant`` with a bit-vector ``TensorType``."""
        ...

    def bv_logical(
        self,
        operation: str,
        operands: Sequence[ReducedExpr],
    ) -> ExprHandle:
        """Create a flattened ``BVLogical`` node and its ``JTReduced`` inputs."""
        ...

    def bt_shift(self, source: ExprHandle, operation: str, amount: int) -> ExprHandle:
        """Create one ``BTShift`` (``logical``/``arithmetic``/``rotate``)."""
        ...

    def bt_expand_reduce(
        self, source: ExprHandle, dims: Sequence[Tuple[str, int]], operation: str
    ) -> ExprHandle:
        """Create one ``BTExpandReduce`` from ``(kind, extent)`` dimensions
        (``batch``/``expand``/``reduce``) folding with ``operation``."""
        ...

    def bv_jagged_permute(
        self,
        source: ExprHandle,
        indices: JaggedTensorParts[int],
    ) -> ExprHandle:
        """Create one ``BVJaggedPermute`` from ``JaggedTensor<usize>`` indices."""
        ...

    def finish(self, result: ExprHandle) -> object:
        """Consume the builder and return the native Rust ``TensorFn``."""
        ...

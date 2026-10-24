"""Execute Python callbacks while constructing a native Rust ``TensorFn``.

The tracing boundary is deliberately tensor-wide:

* ``Vec(bool)`` is one packed bit-vector. Its ``TensorType`` mask is a
  one-dimensional ``JaggedTensor<u64>``; tracing does not create one ``Input``
  for every Boolean.
* Labels, ``Vec.pytree`` indices, and ``Vec(int)`` values used for indexing
  resolve to static source-bit positions, not runtime values.
* Reading a selection asks the native builder for one ``BVJaggedPermute``.
  Rows are the groups of the selection's last dimension and row elements are
  lanes, so ``a[Vec.pytree({k: [...]})]`` keeps its jagged structure. Rank-1
  selections have one lane per row, which is the flat bit-vector layout.
* ``forall``/``exists``/``xorsum`` become one ``BTExpandReduce`` with explicit
  extents, e.g. ``expand_reduce.or %a [15, reduce(4)]``;
  the fusion pass turns a reduced permutation into ``BVGather``/``BVGather2``.
* Assignment ``a[k] = v`` is ``a ^ ((a ^ v) & mask)``: one masked update, no
  dedicated node. Tracing never emits one node per selected bit.

Logical operations and reductions are also sent to the native builder as they
are encountered. Python retains only opaque expression handles, and ``trace``
returns the builder's Rust ``TensorFn`` directly.

The default builder is the PyO3 ``TensorFnBuilder`` implemented in Rust. Tests
and alternate emitters may inject another implementation of the same protocol.
"""

from contextvars import ContextVar
from inspect import Parameter, signature
from typing import Any, Callable, Iterable, Optional, Tuple, get_type_hints

from loom2.lang.vec import Vec, VecIndex, VecValue

from .ast import (
    ExprHandle,
    JaggedTensorParts,
    NativeTensorFnBuilder,
    ReducedExpr,
)


_ACTIVE_TRACE: ContextVar[Optional["TraceSession"]] = ContextVar(
    "loom2_active_trace", default=None
)


def is_tracing() -> bool:
    """Return whether the current context is executing a traced callback."""
    return _ACTIVE_TRACE.get() is not None


def _session() -> "TraceSession":
    session = _ACTIVE_TRACE.get()
    if session is None:
        raise RuntimeError("symbolic Vec values can only be created while tracing")
    return session


def _load_native_builder(name: str) -> NativeTensorFnBuilder:
    """Load the PyO3 builder without falling back to a Python expression AST."""
    try:
        from loom2._loom2 import TensorFnBuilder
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "direct tracing requires the native TensorFnBuilder binding; "
            "build the extension with `maturin develop`"
        ) from error
    return TensorFnBuilder(name)


def _word_count(bit_width: int) -> int:
    if bit_width <= 0:
        raise ValueError("bit-vector widths must be positive")
    return (bit_width + 63) // 64


def _bit_vector_mask(bit_width: int) -> JaggedTensorParts[int]:
    """Build the one-dimensional ``JaggedTensor<u64>`` mask for ``Vec(bool)``."""
    words = [((1 << 64) - 1)] * _word_count(bit_width)
    tail = bit_width % 64
    if tail:
        words[-1] = (1 << tail) - 1
    return JaggedTensorParts.flat(words)


def _pack_bools(values: Iterable[bool]) -> Tuple[JaggedTensorParts[int], int]:
    """Pack Boolean leaves least-significant-bit first into ``u64`` words."""
    values = tuple(values)
    words = [0] * _word_count(len(values))
    for index, value in enumerate(values):
        if not isinstance(value, bool):
            raise TypeError("a traced Vec(bool) constant may contain only booleans")
        if value:
            words[index // 64] |= 1 << (index % 64)
    return JaggedTensorParts.flat(words), len(values)


def _require_leaf_type(schema: Vec, leaf_type: type, description: str) -> None:
    if any(value is not leaf_type for _, value in schema._leaf_entries()):
        raise TypeError(
            "{} must contain only {} leaves".format(description, leaf_type.__name__)
        )


def _with_bool_leaves(schema: Vec) -> Vec:
    """Preserve a static index shape while changing its leaves to Boolean."""
    children = []
    for label, child in schema._children:
        children.append(
            (label, _with_bool_leaves(child) if isinstance(child, Vec) else bool)
        )
    return Vec._from_children(children)


def _selection_indices(schema: Vec, key: Any) -> Tuple[Optional[Vec], Tuple[int, ...]]:
    """Resolve a ``Vec`` selection to its schema and flat source-bit positions.

    Labels, label paths, ``Vec.pytree`` indices, and static ``Vec(int)``
    values all become one tuple of integers, which crosses the native boundary
    as the index list of a single ``BVGather`` or ``BVScatter``.
    """
    if isinstance(key, VecIndex):
        return key.schema, tuple(schema.index(value) for value in key.values)
    if isinstance(key, VecValue):
        _require_leaf_type(key.schema, int, "a Vec index")
        if any(
            type(index) is not int or not 0 <= index < schema.bit_width()
            for index in key.values
        ):
            raise IndexError("Vec(int) contains an out-of-range source bit")
        return _with_bool_leaves(key.schema), key.values
    return schema._selection(key)


class TraceSession:
    """Own the native builder used by one callback execution."""

    __slots__ = ("builder",)

    def __init__(self, builder: NativeTensorFnBuilder) -> None:
        self.builder = builder

    def input(self, name: str, schema: Vec) -> ExprHandle:
        _require_leaf_type(schema, bool, "a traced input Vec")
        return self.builder.input_bit_vector(name, _bit_vector_mask(schema.bit_width()))

    def constant(self, schema: Vec, values: Iterable[bool]) -> ExprHandle:
        _require_leaf_type(schema, bool, "a traced constant Vec")
        packed, width = _pack_bools(values)
        if width != schema.bit_width():
            raise ValueError("Vec constant width does not match its schema")
        return self.builder.constant_bit_vector(packed, _bit_vector_mask(width))

    def logical(
        self, operation: str, operands: Iterable[ReducedExpr]
    ) -> ExprHandle:
        if operation not in ("and", "or", "xor"):
            raise ValueError("unknown Boolean operation: {!r}".format(operation))
        return self.builder.bv_logical(operation, tuple(operands))

    def shift(self, source: ExprHandle, amount: int) -> ExprHandle:
        """One logical ``BTShift``: ``output[d] = source[d + amount]``."""
        return self.builder.bt_shift(source, "logical", amount)

    def expand_reduce(
        self, source: ExprHandle, dims: Iterable[Tuple[str, int]], operation: str
    ) -> ExprHandle:
        """One ``BTExpandReduce``: ``(kind, extent)`` per dimension, ``kind`` in
        ``batch``/``expand``/``reduce``; ``operation`` folds reduced ones."""
        return self.builder.bt_expand_reduce(source, list(dims), operation)

    def permute(self, source: ExprHandle, rows: Iterable[Iterable[int]]) -> ExprHandle:
        """One ``BVJaggedPermute``: row ``d`` has one lane per index in ``rows[d]``."""
        return self.builder.bv_jagged_permute(
            source, JaggedTensorParts.rows(tuple(tuple(row) for row in rows))
        )


class TraceValue:
    """A symbolic one-bit value backed by a native ``TensorExpr`` handle."""

    __slots__ = ("_expression", "_session")

    def __init__(self, session: TraceSession, expression: ExprHandle) -> None:
        self._session = session
        self._expression = expression

    @classmethod
    def constant(cls, value: bool) -> "TraceValue":
        schema = Vec(bool)
        session = _session()
        return cls(session, session.constant(schema, (value,)))

    @staticmethod
    def coerce(value: Any) -> "TraceValue":
        if isinstance(value, TraceValue):
            return value
        if isinstance(value, bool):
            return TraceValue.constant(value)
        raise TypeError("traced Boolean expressions can only combine booleans")

    def _logical(self, operation: str, other: Any) -> "TraceValue":
        right = self.coerce(other)
        if right._session is not self._session:
            raise ValueError("cannot combine values from different trace sessions")
        expression = self._session.logical(
            operation,
            (ReducedExpr(self._expression), ReducedExpr(right._expression)),
        )
        return TraceValue(self._session, expression)

    def __and__(self, other: Any) -> "TraceValue":
        return self._logical("and", other)

    def __rand__(self, other: Any) -> "TraceValue":
        return self.coerce(other)._logical("and", self)

    def __or__(self, other: Any) -> "TraceValue":
        return self._logical("or", other)

    def __ror__(self, other: Any) -> "TraceValue":
        return self.coerce(other)._logical("or", self)

    def __xor__(self, other: Any) -> "TraceValue":
        return self._logical("xor", other)

    def __rxor__(self, other: Any) -> "TraceValue":
        return self.coerce(other)._logical("xor", self)

    def __invert__(self) -> "TraceValue":
        return self ^ True

    def __bool__(self) -> bool:
        raise TypeError(
            "traced values cannot control Python flow; use &, |, ~, forall, "
            "exists, or xorsum"
        )


class TracedVec:
    """A whole symbolic ``Vec(bool)`` represented by one native expression.

    ``_jagged`` records the packed layout of the expression: flat (one lane per
    leaf) or jagged (one row per group of the schema's last dimension, one lane
    per element of that group).
    """

    __slots__ = ("schema", "_expression", "_session", "_jagged")

    def __init__(self, schema: Vec, values: Iterable[Any]) -> None:
        """Create a packed constant when ``Vec(...)`` is called while tracing."""
        session = _session()
        values = tuple(values)
        if any(isinstance(value, (TraceValue, TracedVec)) for value in values):
            raise NotImplementedError(
                "assembling a Vec from symbolic pieces needs a native concatenate node"
            )
        self.schema = schema
        self._session = session
        self._expression = session.constant(schema, values)
        self._jagged = False

    @classmethod
    def _from_expression(
        cls, schema: Vec, session: TraceSession, expression: ExprHandle, jagged: bool = False
    ) -> "TracedVec":
        value = object.__new__(cls)
        value.schema = schema
        value._session = session
        value._expression = expression
        value._jagged = jagged
        return value

    def _flat(self, action: str) -> ExprHandle:
        if self._jagged:
            raise NotImplementedError(
                f"{action} a jagged value; reduce its last dimension first"
            )
        return self._expression

    def _as_jagged(self) -> ExprHandle:
        """This value regrouped by the last dimension of its schema."""
        if self._jagged:
            return self._expression
        return self._session.permute(self._expression, self.schema._last_dimension_groups())

    def __getitem__(self, key: Any) -> Any:
        source = self._flat("indexing")
        schema, positions = _selection_indices(self.schema, key)
        if schema is None or schema.rank == 1:
            expression = self._session.permute(source, ((index,) for index in positions))
            if schema is None:
                return TraceValue(self._session, expression)
            return self._from_expression(schema, self._session, expression)
        rows = (
            tuple(positions[offset] for offset in group)
            for group in schema._last_dimension_groups()
        )
        expression = self._session.permute(source, rows)
        return self._from_expression(schema, self._session, expression, jagged=True)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Overwrite selected bits in place as ``a ^ ((a ^ v) & mask)``.

        ``v`` is all ones for ``True``, zero for ``False``, or a symbolic value
        permuted into the selected positions.
        """
        current = self._flat("assigning into")
        _, positions = _selection_indices(self.schema, key)
        width = self.schema.bit_width()
        selected = set(positions)
        if len(selected) != len(positions):
            raise ValueError("a Vec assignment selects the same bit twice")
        mask = self._session.constant(
            self.schema, tuple(index in selected for index in range(width))
        )
        if isinstance(value, bool):
            if not value:
                cleared = self._session.logical(
                    "and", (ReducedExpr(current), ReducedExpr(mask))
                )
                self._expression = self._session.logical(
                    "xor", (ReducedExpr(current), ReducedExpr(cleared))
                )
                return
            replacement = self._session.constant(self.schema, (True,) * width)
        else:
            if isinstance(value, VecValue):
                value = TracedVec(value.schema, value.values)
            if not isinstance(value, (TracedVec, TraceValue)):
                raise TypeError("Vec assignment requires a Boolean or Vec(bool) value")
            if value._session is not self._session:
                raise ValueError("cannot combine values from different trace sessions")
            source = value._flat("assigning") if isinstance(value, TracedVec) else value._expression
            source_width = value.schema.bit_width() if isinstance(value, TracedVec) else 1
            if source_width != len(positions):
                raise ValueError("assigned value width does not match the selection")
            offset_of = {position: offset for offset, position in enumerate(positions)}
            replacement = self._session.permute(
                source,
                ((offset_of[index],) if index in offset_of else () for index in range(width)),
            )
        difference = self._session.logical(
            "xor", (ReducedExpr(current), ReducedExpr(replacement))
        )
        selected_difference = self._session.logical(
            "and", (ReducedExpr(difference), ReducedExpr(mask))
        )
        self._expression = self._session.logical(
            "xor", (ReducedExpr(current), ReducedExpr(selected_difference))
        )

    def _logical(self, operation: str, other: Any) -> "TracedVec":
        if isinstance(other, bool):
            other = TracedVec(self.schema, (other,) * self.schema.bit_width())
        elif isinstance(other, VecValue):
            if other.schema != self.schema:
                raise TypeError("Vec logical operands must have matching schemas")
            other = TracedVec(other.schema, other.values)
        if not isinstance(other, TracedVec) or other.schema != self.schema:
            raise TypeError("Vec logical operands must have matching schemas")
        if other._session is not self._session:
            raise ValueError("cannot combine values from different trace sessions")
        jagged = self._jagged or other._jagged
        left = self._as_jagged() if jagged else self._expression
        right = other._as_jagged() if jagged else other._expression
        expression = self._session.logical(operation, (ReducedExpr(left), ReducedExpr(right)))
        return self._from_expression(self.schema, self._session, expression, jagged)

    def __and__(self, other: Any) -> "TracedVec":
        return self._logical("and", other)

    def __rand__(self, other: Any) -> "TracedVec":
        return self._logical("and", other)

    def __or__(self, other: Any) -> "TracedVec":
        return self._logical("or", other)

    def __ror__(self, other: Any) -> "TracedVec":
        return self._logical("or", other)

    def __xor__(self, other: Any) -> "TracedVec":
        return self._logical("xor", other)

    def __rxor__(self, other: Any) -> "TracedVec":
        return self._logical("xor", other)

    def __invert__(self) -> "TracedVec":
        return self ^ True

    def __rshift__(self, amount: int) -> "TracedVec":
        """Logical shift toward lower bits: ``(v >> k)[d] = v[d + k]``, zero fill."""
        if not isinstance(amount, int) or amount < 0:
            raise TypeError("Vec shifts take a non-negative int")
        expression = self._session.shift(self._flat("shifting"), amount)
        return self._from_expression(self.schema, self._session, expression)

    def _reduce(self, operation: str) -> Any:
        """Fold the last dimension with one ``BTExpandReduce``.

        Regular flat values use the dense form ``[rows, reduce(n)]``; jagged
        values (and ragged flat ones, after regrouping into lanes) use
        ``[rows, reduce(max lanes)]`` over their lane layout.
        """
        groups = self.schema._last_dimension_groups()
        rows = len(groups)
        lengths = {len(group) for group in groups}
        if not self._jagged and len(lengths) == 1:
            (length,) = lengths
            dims = [("reduce", length)] if self.schema.rank == 1 else [("batch", rows), ("reduce", length)]
            source = self._expression
        else:
            dims = [("batch", rows), ("reduce", max(lengths, default=0))]
            source = self._as_jagged()
        expression = self._session.expand_reduce(source, dims, operation)
        schema = self.schema._without_last_dimension()
        if schema is None:
            return TraceValue(self._session, expression)
        return self._from_expression(schema, self._session, expression)

    def forall(self) -> Any:
        return self._reduce("and")

    def exists(self) -> Any:
        return self._reduce("or")

    def xorsum(self) -> Any:
        return self._reduce("xor")

    def __iter__(self):
        raise TypeError(
            "a traced Vec is one tensor expression and cannot be iterated bit by bit"
        )


def input_vec(name: str, schema: Vec) -> TracedVec:
    """Create one native ``Input`` for an entire ``Vec(bool)`` argument."""
    session = _session()
    expression = session.input(name, schema)
    return TracedVec._from_expression(schema, session, expression)


def recursive_vec(schema: Vec, function: Callable[[TracedVec], Any]) -> TracedVec:
    """Trace a finite least fixed point by monotone Kleene expansion.

    A Boolean lattice with ``n`` bits reaches its least fixed point after at
    most ``n`` strict improvements. Unrolling here keeps the native IR acyclic,
    which lets the ordinary packed-expression lowering handle recursion without
    introducing a second circuit-function representation.
    """
    current = TracedVec(schema, (False,) * schema.bit_width())
    for _ in range(schema.bit_width()):
        updated = function(current)
        if not isinstance(updated, TracedVec) or updated.schema != schema:
            raise TypeError("Vec.rec callback must return a value with the same schema")
        if updated._session is not current._session:
            raise ValueError("Vec.rec callback returned a value from another trace session")
        current = updated
    return current


def reduce_values(operation: str, values: Iterable[Any], identity: bool) -> Any:
    """Reduce scalar values without constructing a Python expression AST."""
    values = tuple(values)
    if not any(isinstance(value, TraceValue) for value in values):
        if operation == "and":
            return all(values)
        if operation == "or":
            return any(values)
        result = identity
        for value in values:
            result ^= bool(value)
        return result
    symbolic = tuple(TraceValue.coerce(value) for value in values)
    session = symbolic[0]._session
    if any(value._session is not session for value in symbolic):
        raise ValueError("cannot reduce values from different trace sessions")
    expression = session.logical(
        operation, tuple(ReducedExpr(value._expression) for value in symbolic)
    )
    return TraceValue(session, expression)


def trace(
    function: Callable[..., Any],
    *,
    builder_factory: Callable[[str], NativeTensorFnBuilder] = _load_native_builder
) -> object:
    """Trace an annotated callback and return its native Rust ``TensorFn``.

    Parameters must currently be annotated with ``Vec`` schemas containing
    Boolean leaves. The injected ``builder_factory`` exists for bridge tests;
    production callers use the PyO3 ``TensorFnBuilder`` automatically.
    """
    try:
        hints = get_type_hints(function)
    except (NameError, TypeError):
        hints = getattr(function, "__annotations__", {})

    session = TraceSession(builder_factory(function.__name__))
    arguments = []
    for parameter in signature(function).parameters.values():
        if parameter.kind not in (
            Parameter.POSITIONAL_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise NotImplementedError(
                "tracing variadic or keyword-only parameters is unsupported"
            )
        annotation = hints.get(parameter.name, parameter.annotation)
        if not isinstance(annotation, Vec):
            raise TypeError(
                "direct tracing requires every parameter to be annotated as Vec(bool)"
            )
        _require_leaf_type(annotation, bool, "a traced parameter Vec")
        arguments.append((parameter.name, annotation))

    token = _ACTIVE_TRACE.set(session)
    try:
        result = function(*(input_vec(name, schema) for name, schema in arguments))
    finally:
        _ACTIVE_TRACE.reset(token)

    if isinstance(result, VecValue):
        result = TracedVec._from_expression(
            result.schema, session, session.constant(result.schema, result.values)
        )
    elif isinstance(result, bool):
        result = TraceValue(session, session.constant(Vec(bool), (result,)))

    if isinstance(result, TracedVec) and result._jagged:
        raise TypeError("a traced function must return a flat Vec; reduce its last dimension")
    if isinstance(result, (TraceValue, TracedVec)):
        if result._session is not session:
            raise ValueError("traced function returned a value from another session")
        return session.builder.finish(result._expression)
    raise TypeError("a traced function must return a Boolean or Vec(bool) value")

"""Schema-driven bit tensors used by Loom tracing.

``Vec`` represents a bit array. ``Vec[2, 3](bool)`` corresponds to an MLIR
``tensor<2x3xi1>``. Positional children permit jagged schemas, keyword children
attach labels used as indices, and :meth:`Vec.pytree` converts ordinary Python
containers into tensor-shaped gather indices. During direct Rust tracing,
``Vec(bool)`` is a runtime packed bit-vector while ``Vec(int)`` is static
``JaggedTensor<usize>`` permutation metadata.
"""

from collections.abc import Mapping
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


class _ShapeFactory:
    def __init__(self, dimensions: Tuple[int, ...]) -> None:
        self.dimensions = dimensions

    def __call__(self, *args: Any, **kwargs: Any) -> "Vec":
        if args and kwargs:
            raise TypeError("Vec shape contents must be positional or labeled, not both")
        if kwargs:
            template: Any = Vec(**kwargs)
        elif len(args) == 1:
            template = args[0]
        elif args:
            template = Vec(*args)
        else:
            raise TypeError("Vec shape requires an element schema")
        for dimension in reversed(self.dimensions):
            template = Vec(*(template for _ in range(dimension)))
        if not isinstance(template, Vec):
            raise TypeError("Vec shape did not produce a tensor schema")
        return template


class Vec:
    """A possibly labeled or jagged tensor schema."""

    def __class_getitem__(cls, dimensions: Any) -> _ShapeFactory:
        if not isinstance(dimensions, tuple):
            dimensions = (dimensions,)
        if not dimensions or not all(isinstance(value, int) and value > 0 for value in dimensions):
            raise TypeError("Vec dimensions must be positive integers")
        return _ShapeFactory(dimensions)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args and kwargs:
            raise TypeError("Vec children must be positional or labeled, not both")
        if kwargs:
            self._children = tuple((name, child) for name, child in kwargs.items())
        else:
            self._children = tuple((None, child) for child in args)
        # An empty Vec is a zero-width rank-1 row, e.g. an empty gather row
        # in ``Vec.pytree({"k": []})``. Reductions over it yield the identity.
        ranks = tuple(child.rank if isinstance(child, Vec) else 0 for _, child in self._children)
        if len(set(ranks)) > 1:
            raise TypeError("all Vec children must have the same rank")
        self._rank = (ranks[0] if ranks else 0) + 1

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        nested = [child.shape for _, child in self._children if isinstance(child, Vec)]
        if not nested:
            return (len(self._children),)
        tail = nested[0] if all(shape == nested[0] for shape in nested) else tuple(
            None for _ in range(self.rank - 1)
        )
        return (len(self._children),) + tail

    def bit_width(self) -> int:
        return sum(child.bit_width() if isinstance(child, Vec) else 1 for _, child in self._children)

    def index(self, label: Any) -> int:
        """Return the stable flattened bit position of one leaf label or path."""
        return self._leaf_index(label)

    def _leaf_entries(self, prefix: Tuple[Any, ...] = ()) -> List[Tuple[Tuple[Any, ...], Any]]:
        result: List[Tuple[Tuple[Any, ...], Any]] = []
        for position, (label, child) in enumerate(self._children):
            index = position if label is None else label
            path = prefix + (index,)
            if isinstance(child, Vec):
                result.extend(child._leaf_entries(path))
            else:
                result.append((path, child))
        return result

    def _leaf_index(self, key: Any) -> int:
        # Fast path: a label that names exactly one leaf.
        cache = self.__dict__.get("_label_positions")
        if cache is None:
            cache = {}
            for position, (path, _) in enumerate(self._leaf_entries()):
                label = path[-1]
                cache[label] = None if label in cache else position
            self._label_positions = cache
        if not isinstance(key, tuple) and cache.get(key) is not None:
            return cache[key]
        entries = self._leaf_entries()
        if isinstance(key, tuple):
            matches = [index for index, (path, _) in enumerate(entries) if path == key]
        else:
            matches = [index for index, (path, _) in enumerate(entries) if path[-1] == key]
            if not matches and isinstance(key, int) and self.rank == 1 and 0 <= key < len(entries):
                matches = [key]
        if len(matches) != 1:
            raise KeyError(key)
        return matches[0]

    def _selection(self, key: Any) -> Tuple[Optional["Vec"], Tuple[int, ...]]:
        offset = 0
        for position, (label, child) in enumerate(self._children):
            width = child.bit_width() if isinstance(child, Vec) else 1
            selector = position if label is None else label
            if selector == key:
                return (
                    child if isinstance(child, Vec) else None,
                    tuple(range(offset, offset + width)),
                )
            offset += width
        index = self._leaf_index(key)
        return None, (index,)

    def _flatten_value(self, value: Any) -> Tuple[Any, ...]:
        if isinstance(value, VecValue):
            if value.schema != self:
                raise TypeError("Vec value has a different schema")
            return value.values
        if isinstance(value, Mapping):
            supplied = value
            child_values = []
            for position, (label, _) in enumerate(self._children):
                key = position if label is None else label
                child_values.append(supplied[key])
        else:
            child_values = list(value)
            if len(child_values) != len(self._children):
                raise ValueError(
                    "expected {} Vec children, got {}".format(
                        len(self._children), len(child_values)
                    )
                )
        flattened: List[Any] = []
        for (_, child), supplied in zip(self._children, child_values):
            if isinstance(child, Vec):
                flattened.extend(child._flatten_value(supplied))
            else:
                flattened.append(supplied)
        return tuple(flattened)

    def __call__(self, *values: Any, **labels: Any) -> Any:
        if values and labels:
            raise TypeError("Vec values must be positional or labeled, not both")
        supplied: Any
        if labels:
            supplied = labels
        elif len(values) == 1 and isinstance(values[0], (Mapping, list, tuple, VecValue)):
            supplied = values[0]
        else:
            supplied = values
        flattened = self._flatten_value(supplied)
        # Import lazily to avoid a module cycle. This lets nullary constructors
        # and callbacks assemble symbolic vectors through the ordinary public
        # Vec constructor while tracing.
        from loom2.tracing.tracer import TracedVec, is_tracing

        if is_tracing():
            return TracedVec(self, flattened)
        return VecValue(self, flattened)

    @staticmethod
    def pytree(tree: Any = None, **kwargs: Any) -> "VecIndex":
        """Convert a nested mapping/sequence into tensor-shaped gather indices."""
        if kwargs:
            if tree is not None:
                raise TypeError("Vec.pytree accepts either one value or keyword children")
            tree = kwargs
        if tree is None:
            raise TypeError("Vec.pytree requires a Python structure")
        return VecIndex(tree)

    def rec(self, function: Any) -> Any:
        """Trace or evaluate a least fixed point over this tensor schema."""
        from loom2.tracing.tracer import is_tracing, recursive_vec

        if is_tracing():
            return recursive_vec(self, function)
        current = VecValue(self, (False for _ in range(self.bit_width())))
        for _ in range(self.bit_width() + 1):
            updated = function(current)
            if not isinstance(updated, VecValue) or updated.schema != self:
                raise TypeError("Vec.rec callback must return a value with the same schema")
            if updated.values == current.values:
                return updated
            current = updated
        raise ValueError("recursive Vec did not reach a fixed point")

    def _last_dimension_groups(self) -> List[List[int]]:
        groups: List[List[int]] = []

        def visit(schema: "Vec", offset: int) -> int:
            if schema.rank == 1:
                width = schema.bit_width()
                groups.append(list(range(offset, offset + width)))
                return offset + width
            for _, child in schema._children:
                if not isinstance(child, Vec):
                    raise TypeError("invalid mixed-rank Vec schema")
                offset = visit(child, offset)
            return offset

        visit(self, 0)
        return groups

    def _positions(self, key: Any) -> Tuple[int, ...]:
        """Flattened leaf positions addressed by ``key`` (label, path, or VecIndex)."""
        if isinstance(key, VecIndex):
            return tuple(self.index(value) for value in key.values)
        return self._selection(key)[1]

    def _without_last_dimension(self) -> Optional["Vec"]:
        if self.rank == 1:
            return None
        children = []
        for label, child in self._children:
            if not isinstance(child, Vec):
                raise TypeError("invalid mixed-rank Vec schema")
            reduced: Any = bool if child.rank == 1 else child._without_last_dimension()
            children.append((label, reduced))
        return Vec._from_children(children)

    @classmethod
    def _from_children(cls, children: Sequence[Tuple[Optional[str], Any]]) -> "Vec":
        if all(label is not None for label, _ in children):
            return cls(**{str(label): child for label, child in children})
        return cls(*(child for _, child in children))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vec) and self._children == other._children

    def __repr__(self) -> str:
        values = ", ".join(
            repr(child) if label is None else "{}={!r}".format(label, child)
            for label, child in self._children
        )
        return "Vec({})".format(values)


class VecIndex:
    """Nested gather indices produced by :meth:`Vec.pytree`."""

    def __init__(self, tree: Any) -> None:
        self.tree = tree
        self.schema, self.values = self._convert(tree)

    @classmethod
    def _convert(cls, value: Any) -> Tuple[Optional[Vec], Tuple[Any, ...]]:
        if isinstance(value, Mapping):
            converted = [(str(key), cls._convert(child)) for key, child in value.items()]
            schema = Vec(**{key: child_schema or bool for key, (child_schema, _) in converted})
            values = tuple(item for _, (_, leaves) in converted for item in leaves)
            return schema, values
        if isinstance(value, (list, tuple)):
            converted = [cls._convert(child) for child in value]
            schema = Vec(*(child_schema or bool for child_schema, _ in converted))
            values = tuple(item for _, leaves in converted for item in leaves)
            return schema, values
        return None, (value,)


class VecValue:
    """Runtime value associated with a :class:`Vec` schema."""

    def __init__(self, schema: Vec, values: Iterable[Any]) -> None:
        self.schema = schema
        self.values = tuple(values)
        if len(self.values) != schema.bit_width():
            raise ValueError("Vec value width does not match its schema")

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, VecIndex):
            if key.schema is None:
                return self[key.values[0]]
            return VecValue(key.schema, (self[index] for index in key.values))
        schema, indices = self.schema._selection(key)
        if schema is None:
            return self.values[indices[0]]
        return VecValue(schema, (self.values[index] for index in indices))

    def __setitem__(self, key: Any, value: Any) -> None:
        positions = self.schema._positions(key)
        if isinstance(value, bool):
            replacement: Tuple[Any, ...] = (value,) * len(positions)
        elif isinstance(value, VecValue):
            replacement = value.values
        else:
            replacement = tuple(value)
        if len(replacement) != len(positions):
            raise ValueError("assigned value width does not match the selection")
        values = list(self.values)
        for position, bit in zip(positions, replacement):
            values[position] = bit
        self.values = tuple(values)

    def _logical(self, operation: str, other: Any) -> "VecValue":
        if isinstance(other, bool):
            right = (other,) * len(self.values)
        elif isinstance(other, VecValue) and other.schema == self.schema:
            right = other.values
        else:
            raise TypeError("Vec logical operands must have matching schemas")
        operators = {
            "and": lambda left, value: bool(left) and bool(value),
            "or": lambda left, value: bool(left) or bool(value),
            "xor": lambda left, value: bool(left) ^ bool(value),
        }
        return VecValue(self.schema, (operators[operation](left, value) for left, value in zip(self.values, right)))

    def __and__(self, other: Any) -> "VecValue":
        return self._logical("and", other)

    def __rand__(self, other: Any) -> "VecValue":
        return self._logical("and", other)

    def __or__(self, other: Any) -> "VecValue":
        return self._logical("or", other)

    def __ror__(self, other: Any) -> "VecValue":
        return self._logical("or", other)

    def __xor__(self, other: Any) -> "VecValue":
        return self._logical("xor", other)

    def __rxor__(self, other: Any) -> "VecValue":
        return self._logical("xor", other)

    def __invert__(self) -> "VecValue":
        return self ^ True

    def __rshift__(self, amount: int) -> "VecValue":
        """Logical shift toward lower bits: ``(v >> k)[d] = v[d + k]``, zero fill."""
        values = self.values[amount:] + (False,) * min(amount, len(self.values))
        return VecValue(self.schema, values)

    def _reduce(self, operation: str) -> Any:
        identities = {"and": True, "or": False, "xor": False}
        reduced = []
        for group in self.schema._last_dimension_groups():
            value = identities[operation]
            for index in group:
                if operation == "and":
                    value = value and bool(self.values[index])
                elif operation == "or":
                    value = value or bool(self.values[index])
                else:
                    value ^= bool(self.values[index])
            reduced.append(value)
        schema = self.schema._without_last_dimension()
        return reduced[0] if schema is None else VecValue(schema, reduced)

    def forall(self) -> Any:
        return self._reduce("and")

    def exists(self) -> Any:
        return self._reduce("or")

    def xorsum(self) -> Any:
        return self._reduce("xor")

    def __iter__(self) -> Iterator[Any]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VecValue) and self.schema == other.schema and self.values == other.values

    def __repr__(self) -> str:
        return "{!r}({!r})".format(self.schema, self.values)


__all__ = ["Vec", "VecIndex", "VecValue"]

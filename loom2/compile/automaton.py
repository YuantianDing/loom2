"""Deterministic bottom-up tree automata over constructor names.

A :class:`TreeAutomaton` gives every tree a state from the states of its
children, or rejects it::

    automaton.step("re.plus", (state,))   # the state of re.plus(t), or None

A :class:`~loom2.Cfg` is a grammar together with a series of automata; the
synthesizer only builds trees that every automaton accepts, keeping each
stored value's state so that a candidate is checked from its children's
states before its constructor runs.

:meth:`TreeAutomaton.grammar` builds the automaton of a regular tree
grammar's language, ``~automaton`` its complement and ``a & b`` the
intersection, so any regular tree language can be subtracted.
:meth:`TreeAutomaton.avoiding` builds the automaton of the trees that
contain none of a set of linear patterns::

    TreeAutomaton.avoiding((plus, (plus, ...)), (union, (union, ..., ...), ...))

A pattern is ``...`` (any subtree), a nullary function (that leaf), or a
tuple ``(function, pattern, ...)``. Patterns are linear: each ``...``
matches independently.
"""

from __future__ import annotations

from typing import Callable, Dict, FrozenSet, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from .function import CompiledFunction

State = Hashable


class _Sink:
    """The state of the trees a complemented automaton's original rejects
    (one per complement, so complements nest)."""

    def __repr__(self) -> str:
        return f"SINK@{id(self):x}"
Pattern = Union["ellipsis", CompiledFunction, Tuple[object, ...]]
# A pattern node: a constructor name and its children, `None` for `...`.
_Node = Tuple[str, Tuple[Optional["_Node"], ...]]


def _pattern(pattern: object) -> Optional[_Node]:
    if pattern is Ellipsis:
        return None
    if isinstance(pattern, CompiledFunction):
        pattern = (pattern,)
    if not (isinstance(pattern, tuple) and pattern and isinstance(pattern[0], CompiledFunction)):
        raise TypeError(f"a pattern is ..., a function, or (function, pattern, ...), not {pattern!r}")
    function, children = pattern[0], pattern[1:]
    if function.arity != len(children):
        raise ValueError(f"pattern {function.__name__} has arity {function.arity}, not {len(children)}")
    return function.__name__, tuple(_pattern(child) for child in children)


def _nodes(node: Optional[_Node], into: List[_Node]) -> None:
    if node is None:
        return
    if node not in into:
        into.append(node)
    for child in node[1]:
        _nodes(child, into)


class TreeAutomaton:
    """A deterministic bottom-up tree automaton.

    ``step(name, states)`` is the state of a tree ``name(t1, ..., tn)`` whose
    children are in ``states``, or ``None`` if the tree is rejected; states
    are hashable values. ``accepting(state)`` (default: every state) says
    whether a whole solution may be in ``state``.

    With ``merge_values``, two trees of one value are interchangeable
    whatever their states, so the synthesizer keeps only the first: true
    when every rejected tree has an accepted tree of the same value and no
    greater size, as for patterns that are left-hand sides of laws. By
    default the synthesizer keeps one tree per value and state.
    """

    def __init__(
        self,
        step: Callable[[str, Tuple[State, ...]], Optional[State]],
        accepting: Optional[Callable[[State], bool]] = None,
        *,
        merge_values: bool = False,
    ) -> None:
        self.step = step
        self.accepting = accepting or (lambda state: True)
        self.merge_values = merge_values

    def state(self, tree: Tuple[str, Sequence[object]]) -> Optional[State]:
        """The state of ``tree``, a ``(name, children)`` tuple, or ``None``."""
        name, children = tree
        states = []
        for child in children:
            state = self.state(child)
            if state is None:
                return None
            states.append(state)
        return self.step(name, tuple(states))

    def accepts(self, tree: Tuple[str, Sequence[object]]) -> bool:
        state = self.state(tree)
        return state is not None and self.accepting(state)

    def __invert__(self) -> "TreeAutomaton":
        """The complement: the trees this automaton does not accept.

        A rejected tree goes to a sink state, so the complement rejects no
        tree and accepts those in the sink or in a non-accepting state. It
        never merges values: the trees it removes need not have an equal
        accepted tree.
        """

        sink = _Sink()

        def step(name: str, states: Tuple[State, ...]) -> Optional[State]:
            if any(state is sink for state in states):
                return sink
            state = self.step(name, states)
            return sink if state is None else state

        return TreeAutomaton(step, lambda state: state is sink or not self.accepting(state))

    def __and__(self, other: "TreeAutomaton") -> "TreeAutomaton":
        """The product automaton: the trees both accept."""

        def step(name: str, states: Tuple[State, ...]) -> Optional[State]:
            left = self.step(name, tuple(state[0] for state in states))
            right = None if left is None else other.step(name, tuple(state[1] for state in states))
            return None if right is None else (left, right)

        return TreeAutomaton(
            step,
            lambda state: self.accepting(state[0]) and other.accepting(state[1]),
            merge_values=self.merge_values and other.merge_values,
        )

    @classmethod
    def grammar(
        cls, productions: Mapping[str, Sequence[Tuple[str, Tuple[str, ...]]]], starts: Union[str, Sequence[str]]
    ) -> "TreeAutomaton":
        """The trees a regular tree grammar derives from one of ``starts``,
        by the subset construction: a tree's state is the set of nonterminals
        that derive it. ``productions`` maps nonterminals to ``(constructor
        name, children)``. Every tree has a state, so the complement is exact."""
        starts = frozenset([starts] if isinstance(starts, str) else starts)
        by_name: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {}
        for symbol, entries in productions.items():
            for name, children in entries:
                by_name.setdefault(name, []).append((symbol, children))

        def step(name: str, states: Tuple[FrozenSet[str], ...]) -> FrozenSet[str]:
            return frozenset(
                symbol
                for symbol, children in by_name.get(name, ())
                if len(children) == len(states) and all(child in state for child, state in zip(children, states))
            )

        return cls(step, lambda state: not starts.isdisjoint(state))

    @classmethod
    def avoiding(cls, *patterns: Pattern) -> "TreeAutomaton":
        """The trees that contain none of ``patterns`` as a subtree.

        A state is the set of pattern nodes below a pattern root that the
        tree matches; a tree matching a whole pattern is rejected. Patterns
        are taken to be left-hand sides of laws, so the automaton merges
        values (see the class documentation).
        """
        roots = [_pattern(pattern) for pattern in patterns]
        if any(root is None for root in roots):
            raise ValueError("the pattern ... matches every tree")
        nodes: List[_Node] = []
        for root in roots:
            _nodes(root, nodes)
        root_set = set(roots)
        tracked = {child for node in nodes for child in node[1] if child is not None}
        by_name = {}
        for node in nodes:
            by_name.setdefault(node[0], []).append(node)

        def step(name: str, states: Tuple[FrozenSet[_Node], ...]) -> Optional[FrozenSet[_Node]]:
            matched = {
                node
                for node in by_name.get(name, ())
                if len(node[1]) == len(states) and all(want is None or want in state for want, state in zip(node[1], states))
            }
            if matched & root_set:
                return None
            return frozenset(matched & tracked)

        return cls(step, merge_values=True)


__all__ = ["Pattern", "State", "TreeAutomaton"]

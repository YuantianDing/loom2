"""Regular tree grammars over ``loom2.compile`` constructors.

A :class:`Cfg` maps nonterminals to productions ``(function, child, ...)``;
the first nonterminal is the start symbol::

    cfg = loom2.Cfg({"R": [char_a, (plus, "R"), (union, "R", "R")]})
    cfg = cfg.subtract((plus, (plus, ...)), (union, (union, ..., ...), ...))

:meth:`Cfg.subtract` removes a tree language, as a tree automaton
difference:

1. the grammar is made a deterministic bottom-up tree automaton (subset
   construction): a tree's state is the set of nonterminals deriving it;
2. the removed language's automaton is complemented and multiplied in;
3. the product is turned back into a grammar: a reachable state becomes one
   nonterminal per original nonterminal in its set, its *label*, so every
   nonterminal still traces one original nonterminal (:meth:`Cfg.origin`).

The copies of one original nonterminal denote subsets of its values: they
take its language in ``synth_fun``, and copies of the start symbol in an
accepting state are the *solutions*. When only law patterns are removed,
every removed tree has a kept tree of the same value, so the copies share
the original's hash group (:meth:`Cfg.group`) and values are deduplicated
across them; otherwise each copy keeps its own.
"""

from __future__ import annotations

from itertools import product
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .automaton import Pattern, State, TreeAutomaton
from .function import CompiledFunction

Production = Tuple[CompiledFunction, Tuple[str, ...]]
Tree = Tuple[str, Sequence[object]]


def _production(entry: object) -> Production:
    if isinstance(entry, CompiledFunction):
        function, children = entry, ()
    elif isinstance(entry, tuple) and entry:
        function, children = entry[0], tuple(entry[1:])
    else:
        raise TypeError(f"a production is a function or (function, child, ...), not {entry!r}")
    if not isinstance(function, CompiledFunction):
        raise TypeError("productions must start with a loom2.compile function")
    if function.arity != len(children):
        raise ValueError(f"{function.__name__} has arity {function.arity}, not {len(children)}")
    if not all(isinstance(child, str) for child in children):
        raise TypeError(f"children of {function.__name__} must name nonterminals")
    return function, children


class Cfg:
    """A regular tree grammar.

    ``productions`` maps each nonterminal to its productions, each a nullary
    :class:`~loom2.CompiledFunction` or a tuple ``(function, child, ...)``;
    the first nonterminal is the start symbol. ``commutative`` lists the
    binary constructors with ``f(x, y) = f(y, x)``. ``solutions`` are the
    nonterminals whose trees are solutions (default: the start symbol),
    ``groups`` their hash groups (default: their own name; see
    ``synth_fun``) and ``origins`` the nonterminals of the original grammar
    they trace (default: themselves).
    """

    def __init__(
        self,
        productions: Mapping[str, Sequence[object]],
        *,
        solutions: Optional[Iterable[str]] = None,
        groups: Optional[Mapping[str, str]] = None,
        origins: Optional[Mapping[str, str]] = None,
        commutative: Iterable[CompiledFunction] = (),
    ) -> None:
        if not productions:
            raise ValueError("a grammar needs a nonterminal")
        self._productions: Dict[str, Tuple[Production, ...]] = {}
        for symbol, entries in productions.items():
            if not entries:
                raise ValueError(f"nonterminal {symbol!r} has no productions")
            self._productions[symbol] = tuple(_production(entry) for entry in entries)
        for symbol, entries in self._productions.items():
            for function, children in entries:
                unknown = [child for child in children if child not in self._productions]
                if unknown:
                    raise ValueError(f"{function.__name__} in {symbol!r} references unknown nonterminal {unknown[0]!r}")
        self._solutions = tuple(solutions) if solutions is not None else (self.start,)
        unknown = [symbol for symbol in self._solutions if symbol not in self._productions]
        if unknown:
            raise ValueError(f"solution {unknown[0]!r} is not a nonterminal")
        self._commutative = frozenset(commutative)
        for function in self._commutative:
            if not isinstance(function, CompiledFunction) or function.arity != 2:
                raise ValueError(f"commutative constructors must be binary loom2.compile functions, not {function!r}")
        groups, origins = dict(groups or {}), dict(origins or {})
        self._groups = {symbol: groups.get(symbol, symbol) for symbol in self._productions}
        self._origins = {symbol: origins.get(symbol, symbol) for symbol in self._productions}

    @property
    def start(self) -> str:
        """The start symbol."""
        return next(iter(self._productions))

    @property
    def nonterminals(self) -> Tuple[str, ...]:
        return tuple(self._productions)

    @property
    def solutions(self) -> Tuple[str, ...]:
        """The nonterminals whose trees are solutions."""
        return self._solutions

    @property
    def commutative(self) -> FrozenSet[CompiledFunction]:
        """The constructors with ``f(x, y) = f(y, x)``: the synthesizer
        builds only one orientation of their applications (see ``synth_fun``)."""
        return self._commutative

    def productions(self, symbol: str) -> Tuple[Production, ...]:
        """The productions of ``symbol`` as ``(function, children)``."""
        return self._productions[symbol]

    def group(self, symbol: str) -> str:
        """The hash group of ``symbol``."""
        return self._groups[symbol]

    def origin(self, symbol: str) -> str:
        """The nonterminal of the original grammar that ``symbol`` traces."""
        return self._origins[symbol]

    def items(self) -> Iterable[Tuple[str, Tuple[Production, ...]]]:
        return self._productions.items()

    def labels(self) -> TreeAutomaton:
        """This grammar as a deterministic bottom-up automaton: a tree's state
        is the set of nonterminals deriving it, accepted when one of them is a
        solution. Every tree has a state, so its complement is exact."""
        names = {symbol: [(function.__name__, children) for function, children in entries] for symbol, entries in self.items()}
        return TreeAutomaton.grammar(names, self._solutions)

    def accepts(self, tree: Tree) -> bool:
        """Whether a solution nonterminal derives ``tree``, a ``(name,
        children)`` tuple."""
        return self.labels().accepts(tree)

    def subtract(self, *removed: Union[Pattern, TreeAutomaton, "Cfg"]) -> "Cfg":
        """This grammar without the trees of ``removed``, each one of:

        - a linear pattern: the trees that *contain* it anywhere (see
          :meth:`TreeAutomaton.avoiding`; consecutive patterns share one
          automaton);
        - a :class:`TreeAutomaton`: the trees it accepts;
        - a :class:`Cfg`: the trees its solution nonterminals derive.

        See the module documentation for the construction.
        """
        automata: List[TreeAutomaton] = []
        patterns: List[object] = []
        for item in [*removed, None]:
            if item is not None and not isinstance(item, (TreeAutomaton, Cfg)):
                patterns.append(item)
                continue
            if patterns:
                automata.append(TreeAutomaton.avoiding(*patterns))
                patterns = []
            if isinstance(item, TreeAutomaton):
                automata.append(~item)
            elif isinstance(item, Cfg):
                automata.append(~item.labels())
        return self._intersect(automata)

    def _intersect(self, automata: Sequence[TreeAutomaton]) -> "Cfg":
        """The grammar of the trees of ``self`` that every automaton accepts,
        split by label (steps 1 and 3 of the module documentation)."""
        constructors: Dict[str, Tuple[CompiledFunction, int]] = {}
        # by_name[name]: the (symbol, children) of every production `name(...)`.
        by_name: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {}
        for symbol, entries in self.items():
            for function, children in entries:
                constructors[function.__name__] = (function, len(children))
                by_name.setdefault(function.__name__, []).append((symbol, children))

        # A state: (the nonterminals deriving the tree, each automaton's state).
        Key = Tuple[FrozenSet[str], Tuple[State, ...]]

        def step(name: str, children: Tuple[Key, ...]) -> Optional[Key]:
            labels = frozenset(
                symbol
                for symbol, wanted in by_name[name]
                if all(want in child[0] for want, child in zip(wanted, children))
            )
            if not labels:
                return None  # no nonterminal derives the tree: never useful
            states = []
            for position, automaton in enumerate(automata):
                state = automaton.step(name, tuple(child[1][position] for child in children))
                if state is None:
                    return None
                states.append(state)
            return labels, tuple(states)

        # Reachable states and their transitions, bottom-up to a fixpoint.
        reached: List[Key] = []
        transitions: Dict[Tuple[str, Tuple[Key, ...]], Key] = {}
        changed = True
        while changed:
            changed = False
            for name, (_, arity) in constructors.items():
                for combination in product(tuple(reached), repeat=arity):
                    if (name, combination) in transitions:
                        continue
                    state = step(name, combination)
                    if state is None:
                        continue
                    transitions[(name, combination)] = state
                    if state not in reached:
                        reached.append(state)
                        changed = True

        def accepting(state: Key) -> bool:
            return all(automaton.accepting(component) for automaton, component in zip(automata, state[1]))

        # Step 3: a nonterminal per (state, label), with the productions of
        # its label whose children carry the children's labels.
        rules: Dict[Tuple[Key, str], List[Tuple[CompiledFunction, Tuple[Tuple[Key, str], ...]]]] = {}
        for (name, combination), state in transitions.items():
            function = constructors[name][0]
            for symbol, wanted in by_name[name]:
                if symbol in state[0] and all(want in child[0] for want, child in zip(wanted, combination)):
                    rules.setdefault((state, symbol), []).append(
                        (function, tuple((child, want) for child, want in zip(combination, wanted)))
                    )

        solutions = [(state, symbol) for state in reached if accepting(state) for symbol in self._solutions if symbol in state[0]]
        if not solutions:
            raise ValueError("subtracting leaves no solution")
        # Keep the nonterminals that solutions use.
        useful: Set[Tuple[Key, str]] = set(solutions)
        frontier = list(solutions)
        while frontier:
            for _, children in rules[frontier.pop()]:
                for child in children:
                    if child not in useful:
                        useful.add(child)
                        frontier.append(child)

        # Names: the label, with `#k` for its further copies; the start
        # symbol's first solution copy comes first, so it stays the start.
        order = {symbol: position for position, symbol in enumerate(self._productions)}
        kept = sorted(useful, key=lambda key: (key not in solutions[:1], order[key[1]], reached.index(key[0])))
        names: Dict[Tuple[Key, str], str] = {}
        copies: Dict[str, int] = {}
        for key in kept:
            copy = copies.get(key[1], 0)
            copies[key[1]] = copy + 1
            names[key] = key[1] if copy == 0 else f"{key[1]}#{copy}"
        merge = all(automaton.merge_values for automaton in automata)
        return Cfg(
            {
                names[key]: [(function, *(names[child] for child in children)) for function, children in rules[key]]
                for key in kept
            },
            solutions=[names[key] for key in solutions],
            groups={names[key]: self._groups[key[1]] if merge else names[key] for key in kept},
            origins={names[key]: self._origins[key[1]] for key in kept},
            commutative=self._commutative,
        )

    def __str__(self) -> str:
        lines = []
        for symbol, entries in self._productions.items():
            alternatives = " | ".join(
                function.__name__ + (f"({', '.join(children)})" if children else "") for function, children in entries
            )
            notes = []
            if self._origins[symbol] != symbol:
                notes.append(f"origin {self._origins[symbol]}")
            if self._groups[symbol] != symbol:
                notes.append(f"group {self._groups[symbol]}")
            if symbol in self._solutions:
                notes.append("solution")
            lines.append(f"{symbol} -> {alternatives}" + (f"    # {', '.join(notes)}" if notes else ""))
        return "\n".join(lines)

    def __repr__(self) -> str:
        size = sum(len(entries) for entries in self._productions.values())
        return f"Cfg({len(self._productions)} nonterminals, {size} productions, start={self.start!r})"


__all__ = ["Cfg"]

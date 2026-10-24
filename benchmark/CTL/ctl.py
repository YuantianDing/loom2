"""Depth-bounded CTL over Kripke structures, described by ``theories/<name>.json``.

A theory JSON lists ``[constructor, kind, proposition?]`` with kinds ``ap``,
``nap`` (negated proposition), ``top``, ``not``, ``and``, ``or``, ``AX``,
``EX``, ``AF``, ``EF``, ``AG``, ``EG``, ``AU``, ``EU``. A model lists
``states``, ``successors``, ``counters``, ``czero``/``cbot``, ``bound``,
``next_depth`` (one row shared by every state, or one row per state), the
states where each proposition holds, and ``initial``.

Every formula is one bit per `(model, state, counter)` in a shared ``Vec``.
Following Loom's semantics, the recursive operators unfold along
`(next, next_depth(next, d))` and stop at `czero` and `cbot`, e.g.

```text
AF x (s, d) = x(s, d) or (d not in {czero, cbot} and forall next in R(s). AF x (next, d'))
```

Rows are grouped into dependency levels (stop rows are level 0; a row's level
is one more than the highest level it reads), and each level is filled by one
fold over the already-final lower levels, the same length-staging as `re.*`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loom2  # noqa: E402


class Kripke:
    def __init__(self, name: str, data: Mapping[str, object]) -> None:
        self.name = name
        self.states: int = data["states"]
        self.counters: int = data["counters"]
        self.initial: int = data["initial"]
        self.czero: int = data["czero"]
        self.cbot: int = data["cbot"]
        self.successors: List[List[int]] = data["successors"]
        self.bound: List[int] = data["bound"]
        depth = data["next_depth"]
        self.depth_rows = depth if depth and isinstance(depth[0], list) else None
        self.depth_shared = None if self.depth_rows else depth
        self.props: Dict[str, set] = {prop: set(states) for prop, states in data["props"].items()}

    def next_depth(self, state: int, counter: int) -> int:
        return self.depth_rows[state][counter] if self.depth_rows else self.depth_shared[counter]

    def stops(self, counter: int) -> bool:
        return counter in (self.czero, self.cbot)

    def label(self, state: int, counter: int) -> str:
        return f"{self.name}|{state}|{counter}"


@dataclass(frozen=True)
class CtlTheory:
    language: loom2.Vec
    constructors: Mapping[str, loom2.CompiledFunction]
    grammar: Mapping[str, Sequence[object]]
    structures: Mapping[str, Kripke]

    def label(self, model: str) -> str:
        """The bit of `model` at `(initial, bound(initial))`."""
        kripke = self.structures[model]
        return kripke.label(kripke.initial, kripke.bound[kripke.initial])


def _compile(name: str, language: loom2.Vec, implementation: Callable[..., object], arity: int):
    parameters = implementation.__code__.co_varnames[:arity]
    implementation.__annotations__ = {**{p: language for p in parameters}, "return": language}
    return loom2.compile(implementation, name=name)


def _levels(structures: Sequence[Kripke]) -> Dict[Tuple[str, int, int], int]:
    """Dependency level of every `(model, state, counter)` row."""
    levels: Dict[Tuple[str, int, int], int] = {}
    for kripke in structures:
        for counter in range(kripke.counters):
            for state in range(kripke.states):
                stack = [(state, counter)]
                while stack:
                    s, d = stack[-1]
                    key = (kripke.name, s, d)
                    if key in levels:
                        stack.pop()
                        continue
                    if kripke.stops(d):
                        levels[key] = 0
                        stack.pop()
                        continue
                    deps = [(n, kripke.next_depth(n, d)) for n in kripke.successors[s]]
                    pending = [dep for dep in deps if (kripke.name, *dep) not in levels]
                    if pending:
                        if len(stack) > kripke.states * kripke.counters:
                            raise ValueError(f"{kripke.name}: cyclic next_depth dependencies")
                        stack.extend(pending)
                        continue
                    levels[key] = 1 + max((levels[(kripke.name, *dep)] for dep in deps), default=0)
                    stack.pop()
    return levels


def theory(definition: Mapping[str, object], models: Mapping[str, Mapping[str, object]]) -> CtlTheory:
    structures = [Kripke(name, data) for name, data in models.items()]
    rows = [(k, s, d) for k in structures for s in range(k.states) for d in range(k.counters)]
    language = loom2.Vec(**{k.label(s, d): bool for k, s, d in rows})

    # Same-depth successors (AX/EX).
    step = loom2.Vec.pytree(**{k.label(s, d): [k.label(n, d) for n in k.successors[s]] for k, s, d in rows})
    # Staged recursion: per level, rows of that level read their unfolded successors.
    levels = _levels(structures)
    top = max(levels.values(), default=0)
    stages = []
    for level in range(1, top + 1):
        index = loom2.Vec.pytree(
            **{
                k.label(s, d): [k.label(n, k.next_depth(n, d)) for n in k.successors[s]]
                if levels[(k.name, s, d)] == level
                else []
                for k, s, d in rows
            }
        )
        mask = language({k.label(s, d): levels[(k.name, s, d)] == level for k, s, d in rows})
        others = language({k.label(s, d): levels[(k.name, s, d)] != level for k, s, d in rows})
        stages.append((index, mask, others))

    def proposition(name: str, negated: bool):
        values = {k.label(s, d): (s in k.props[name]) != negated for k, s, d in rows}
        return lambda: language(values)

    def top_():
        return language({k.label(s, d): True for k, s, d in rows})

    def not_(x):
        return ~x

    def and_(x, y):
        return x & y

    def or_(x, y):
        return x | y

    def ax(x):
        return x[step].forall()

    def ex(x):
        return x[step].exists()

    def af(x):
        r = x
        for index, mask, _ in stages:
            r = r | (r[index].forall() & mask)
        return r

    def ef(x):
        r = x
        for index, _, _ in stages:
            r = r | r[index].exists()
        return r

    def ag(x):
        r = x
        for index, _, _ in stages:
            r = r & r[index].forall()  # rows of other levels fold nothing: true
        return r

    def eg(x):
        r = x
        for index, _, others in stages:
            r = r & (r[index].exists() | others)
        return r

    def au(x, y):
        r = y
        for index, mask, _ in stages:
            r = r | (x & r[index].forall() & mask)
        return r

    def eu(x, y):
        r = y
        for index, _, _ in stages:
            r = r | (x & r[index].exists())
        return r

    kinds = {
        "not": (not_, 1), "and": (and_, 2), "or": (or_, 2),
        "AX": (ax, 1), "EX": (ex, 1), "AF": (af, 1), "EF": (ef, 1),
        "AG": (ag, 1), "EG": (eg, 1), "AU": (au, 2), "EU": (eu, 2),
    }
    constructors: Dict[str, loom2.CompiledFunction] = {}
    start: List[object] = []
    for entry in definition["constructors"]:
        name, kind = entry[0], entry[1]
        if kind in ("ap", "nap"):
            implementation, arity = proposition(entry[2], kind == "nap"), 0
        elif kind == "top":
            implementation, arity = top_, 0
        else:
            implementation, arity = kinds[kind]
        implementation = implementation.__class__(
            implementation.__code__, implementation.__globals__, name, None, implementation.__closure__
        )
        constructors[name] = _compile(name, language, implementation, arity)
        start.append(constructors[name] if arity == 0 else (constructors[name],) + ("CTL",) * arity)
    return CtlTheory(language, constructors, {"CTL": tuple(start)}, {k.name: k for k in structures})


def synth_fun(out_file, definition, models, positives, negatives, **options) -> loom2.SynthesisProgram:
    """Synthesize a formula true on every positive and false on every negative model."""
    positives, negatives = list(dict.fromkeys(positives)), list(dict.fromkeys(negatives))
    if set(positives) & set(negatives):
        raise ValueError(f"models are both positive and negative: {sorted(set(positives) & set(negatives))}")
    instance = theory(definition, {name: models[name] for name in positives + negatives})
    constraints = {instance.label(model): True for model in positives}
    constraints.update({instance.label(model): False for model in negatives})
    return loom2.synth_fun(
        out_file,
        language=instance.language,
        grammar=instance.grammar,
        constraints=constraints,
        name="phi",
        **options,
    )


__all__ = ["CtlTheory", "Kripke", "synth_fun", "theory"]

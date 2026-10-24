"""LTL over lasso traces, described by ``theories/<name>.json``.

A theory JSON lists its constructors as ``[constructor, kind, proposition?]``
with kinds ``prop``, ``not``, ``X``, ``G``, ``F``, ``and``, ``or``, ``=>``,
``U``. A model (``models/<id>.json``) is a lasso: ``length`` positions,
``start``, ``next``, and the positions where each proposition holds.

Every formula is one bit per `(trace, offset)` in a shared flat ``Vec``. Each
lasso (`next(t) = t + 1` except at its end, which loops back to `loop`) is
unrolled into a segment of its `length` positions followed by a tail of `W`
copies continuing around the loop, so `next` is a shift by one:

```text
X x    x >> 1
F x    cs = x;  cs |= cs >> 1;  cs |= cs >> 2;  cs |= cs >> 4;  ...
G x    cs = x;  cs &= cs >> 1;  cs &= cs >> 2;  ...
x U y  cs1, cs2 = x, y
       cs2 |= cs1 & (cs2 >> 1);  cs1 &= cs1 >> 1
       cs2 |= cs1 & (cs2 >> 2);  cs1 &= cs1 >> 2;  ...
```

With shifts `1, 2, ..., 2^m` the window is `W = 2^(m+1) - 1 >= length - 1`
offsets, which reaches every future position, so real positions are exact
and never read past their own segment. Tail bits can go stale, so every
temporal operator ends with one permute that refreshes the tail from the
real positions (the unrolled sequence is periodic).
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


class Lasso:
    """One trace: positions `0..length`, linear until `next[length - 1]`."""

    def __init__(self, name: str, data: Mapping[str, object]) -> None:
        self.name = name
        self.length: int = data["length"]
        self.start: int = data["start"]
        self.next: List[int] = list(data["next"])
        self.props: Dict[str, set] = {prop: set(positions) for prop, positions in data["props"].items()}
        if "future" in data or any(self.next[t] != t + 1 for t in range(self.length - 1)):
            raise ValueError(f"{name}: shift-based LTL needs a lasso with next(t) = t + 1")
        self.loop: int = self.next[self.length - 1]

    def position(self, offset: int) -> int:
        """The real position at unrolled `offset`."""
        if offset < self.length:
            return offset
        period = self.length - self.loop
        return self.loop + (offset - self.length) % period

    def label(self, offset: int) -> str:
        return f"{self.name}|{offset}" if offset < self.length else f"{self.name}|~{offset}"


@dataclass(frozen=True)
class LtlTheory:
    language: loom2.Vec
    constructors: Mapping[str, loom2.CompiledFunction]
    grammar: Mapping[str, Sequence[object]]
    traces: Mapping[str, Lasso]

    def label(self, model: str) -> str:
        """The bit of `model` at its start position."""
        trace = self.traces[model]
        return trace.label(trace.start)


def _compile(name: str, language: loom2.Vec, implementation: Callable[..., object], arity: int):
    parameters = implementation.__code__.co_varnames[:arity]
    implementation.__annotations__ = {**{p: language for p in parameters}, "return": language}
    return loom2.compile(implementation, name=name)


def _window_steps(longest: int) -> List[int]:
    """Shifts `1, 2, ..., 2^m` with `2^(m+1) - 1 >= longest - 1`."""
    steps = [1]
    while 2 * steps[-1] - 1 < longest - 1:
        steps.append(2 * steps[-1])
    return steps


def theory(definition: Mapping[str, object], models: Mapping[str, Mapping[str, object]]) -> LtlTheory:
    traces = {name: Lasso(name, data) for name, data in models.items()}
    steps = _window_steps(max((trace.length for trace in traces.values()), default=1))
    window = 2 * steps[-1] - 1
    rows = [(trace, e) for trace in traces.values() for e in range(trace.length + window)]
    language = loom2.Vec(**{trace.label(e): bool for trace, e in rows})
    # Refresh the tail from the real positions.
    periodic = loom2.Vec.pytree(**{trace.label(e): trace.label(trace.position(e)) for trace, e in rows})

    def proposition(name: str):
        values = {trace.label(e): trace.position(e) in trace.props[name] for trace, e in rows}
        return lambda: language(values)

    def not_(x):
        return ~x

    def next_(x):
        return (x >> 1)[periodic]

    def globally(x):
        for k in steps:
            x = x & (x >> k)
        return x[periodic]

    def finally_(x):
        for k in steps:
            x = x | (x >> k)
        return x[periodic]

    def and_(x, y):
        return x & y

    def or_(x, y):
        return x | y

    def implies(x, y):
        return ~x | y

    def until(x, y):
        for k in steps:
            y = y | (x & (y >> k))
            x = x & (x >> k)
        return y[periodic]

    kinds = {
        "not": (not_, 1), "X": (next_, 1), "G": (globally, 1), "F": (finally_, 1),
        "and": (and_, 2), "or": (or_, 2), "=>": (implies, 2), "U": (until, 2),
    }
    constructors: Dict[str, loom2.CompiledFunction] = {}
    start: List[object] = []
    for entry in definition["constructors"]:
        name, kind = entry[0], entry[1]
        if kind == "prop":
            implementation, arity = proposition(entry[2]), 0
        else:
            implementation, arity = kinds[kind]
            # A fresh function object per constructor (names and annotations differ).
            implementation = implementation.__class__(
                implementation.__code__, implementation.__globals__, name, None, implementation.__closure__
            )
        constructors[name] = _compile(name, language, implementation, arity)
        start.append(constructors[name] if arity == 0 else (constructors[name],) + ("LTL",) * arity)
    return LtlTheory(language, constructors, {"LTL": tuple(start)}, traces)


def synth_fun(out_file, definition, models, positives, negatives, **options) -> loom2.SynthesisProgram:
    """Synthesize a formula true on every positive and false on every negative trace."""
    positives, negatives = list(dict.fromkeys(positives)), list(dict.fromkeys(negatives))
    if set(positives) & set(negatives):
        raise ValueError(f"traces are both positive and negative: {sorted(set(positives) & set(negatives))}")
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


__all__ = ["Lasso", "LtlTheory", "synth_fun", "theory"]

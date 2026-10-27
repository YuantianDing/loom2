"""Regular-expression theory builder shared by the modules in ``theories/``.

One theory is built per benchmark:

* every substring of every model string is interned once, in one shared
  ``Vec`` (the ``Literal``), so bits common to several models are shared;
* character constructors test the predicates Loom's models assign to each
  character (a theory module's ``CHARACTERS``); and
* operators are written with the public ``Vec`` API. ``re.*`` and ``re.+``
  are length-staged: strings of length ``i`` only read finished shorter
  strings, so no recursion survives tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loom2  # noqa: E402


def substrings(strings: Iterable[str]) -> Tuple[str, ...]:
    """Every substring of every string, once, in width/lexical order."""
    values = {
        value[start:end]
        for value in strings
        for start in range(len(value) + 1)
        for end in range(start, len(value) + 1)
    }
    return tuple(sorted(values, key=lambda value: (len(value), value)))


@dataclass(frozen=True)
class RegexTheory:
    substrings: Tuple[str, ...]
    language: loom2.Vec
    constructors: Mapping[str, loom2.CompiledFunction]
    grammar: Mapping[str, Sequence[object]]

    @property
    def commutative(self) -> Tuple[loom2.CompiledFunction, ...]:
        """Union and intersection may exchange their two operands."""
        return tuple(self.constructors[name] for name in ("re.union", "re.inter") if name in self.constructors)


def _compile(name: str, language: loom2.Vec, implementation: Callable[..., object], arity: int):
    parameters = implementation.__code__.co_varnames[:arity]
    implementation.__annotations__ = {
        **{parameter: language for parameter in parameters},
        "return": language,
    }
    return loom2.compile(implementation, name=name)


def theory(
    constructors: Sequence[Tuple[str, object]],
    character_classes: Mapping[str, Sequence[str]],
    strings: Iterable[str],
) -> RegexTheory:
    """Build the language and every constructor of one regex theory."""
    literal = substrings(strings)
    language = loom2.Vec(**{value: bool for value in literal})
    empty = language({value: value == "" for value in literal})

    def splits(prefix: bool, length=None):
        return loom2.Vec.pytree(
            **{
                k: [k[:i] if prefix else k[i:] for i in range(len(k) + 1)]
                if length is None or len(k) == length
                else []
                for k in literal
            }
        )

    left, right = splits(True), splits(False)
    longest = max(len(value) for value in literal)
    staged = [(splits(True, n), splits(False, n)) for n in range(1, longest + 1)]

    def character(predicate: str):
        def constant():
            return language(
                {
                    k: len(k) == 1
                    and (predicate == "*" or predicate in character_classes[k])
                    for k in literal
                }
            )

        return constant

    def opt(a):
        return a | empty

    def concat(a, b):
        return (a[left] & b[right]).exists()

    def union(a, b):
        return a | b

    def inter(a, b):
        return a & b

    def star(a):
        a[""] = True
        for prefixes, suffixes in staged:
            a |= (a[prefixes] & a[suffixes]).exists()
        return a

    def plus(a):
        # result[k] = a[k] or (a[k[:j]] and result[k[j:]]) for a nonempty prefix.
        result = a
        for prefixes, suffixes in staged:
            result |= (a[prefixes] & result[suffixes]).exists()
        return result

    operators = {
        "re.opt": (opt, 1),
        "re.plus": (plus, 1),
        "re.++": (concat, 2),
        "re.union": (union, 2),
        "re.inter": (inter, 2),
        "re.*": (star, 1),
    }
    compiled: Dict[str, loom2.CompiledFunction] = {}
    start = []
    for name, predicate in constructors:
        if predicate is None:
            implementation, arity = operators[name]
        else:
            implementation, arity = character(predicate), 0
        compiled[name] = _compile(name, language, implementation, arity)
        start.append(compiled[name] if arity == 0 else (compiled[name],) + ("Start",) * arity)
    return RegexTheory(literal, language, compiled, {"Start": tuple(start)})


__all__ = ["RegexTheory", "substrings", "theory"]

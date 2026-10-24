"""First-order (FO²) theories, described by ``theories/<name>.json``.

A theory JSON lists the signature and, for every grammar nonterminal, its
variable context, its sort, and its productions by kind:

```text
"sorts":     ["node", ...]
"functions": {"holds_lock": [["node"], "Bool"], "idn": [["node"], "id"], ...}
"start":     "FO0"
"nonterminals": {
  "Formula.x0.node": {"variables": [["x0", "node"]], "sort": "Bool",
                      "productions": [[constructor, kind, ...arguments, ...children]]},
  "Term.x0.node.node": {"variables": [["x0", "node"]], "sort": "node", ...}}
```

Production kinds:

| kind               | arguments                     | meaning                                |
| ------------------ | ----------------------------- | -------------------------------------- |
| `true` / `false`   |                               | constant formula                       |
| `not`/`and`/`or`   | formula children              | Boolean connective                     |
| `forall`/`exists`  | bound sort, field map, child  | quantify; the child's variables come   |
|                    |                               | from this context (`"@"` = bound one)  |
| `eq`               | two term children             | `t1 = t2`                              |
| `atom`             | predicate, term children      | `P(t1, ..., tk)`                       |
| `var`              | variable                      | term: the variable's value             |
| `apply`            | function, term children       | term: `f(t1, ..., tk)`                 |

Every nonterminal shares one labeled ``Vec`` (its ``Literal``) over all models
of a benchmark: a formula has one bit per `(model, assignment)`, and a term is
one-hot over `(model, assignment, element)`. Quantifiers, equality, atoms,
and function terms are reductions over gathered lanes, which fuse into
``gather``/``gather2`` nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
import sys
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loom2  # noqa: E402

Model = Mapping[str, object]


class Structure:
    """One finite model: elements per sort and function tables."""

    def __init__(self, name: str, data: Model) -> None:
        self.name = name
        self.elements: Dict[str, List[str]] = dict(data["sorts"])
        self.tables: Dict[str, object] = {}
        for function, table in data["functions"].items():
            if isinstance(table, list):
                self.tables[function] = {tuple(row[:-1]): row[-1] for row in table}
            else:
                self.tables[function] = {(): table}

    def apply(self, function: str, arguments: Tuple[str, ...]) -> object:
        return self.tables[function][arguments]


@dataclass(frozen=True)
class FirstOrderTheory:
    """A theory instantiated over one benchmark's models."""

    languages: Mapping[str, loom2.Vec]
    constructors: Mapping[str, loom2.CompiledFunction]
    grammar: Mapping[str, Sequence[object]]
    start: str
    models: Tuple[str, ...]

    @property
    def language(self) -> loom2.Vec:
        return self.languages[self.start]

    def label(self, model: str) -> str:
        """Label of the start symbol's bit for `model` (empty assignment)."""
        return f"{model}|"


def _arity(count: int, implementation: Callable[..., object]) -> Callable[..., object]:
    """A function of exactly `count` positional parameters (for tracing)."""
    wrappers = {
        0: lambda: implementation(),
        1: lambda a0: implementation(a0),
        2: lambda a0, a1: implementation(a0, a1),
        3: lambda a0, a1, a2: implementation(a0, a1, a2),
        4: lambda a0, a1, a2, a3: implementation(a0, a1, a2, a3),
        5: lambda a0, a1, a2, a3, a4: implementation(a0, a1, a2, a3, a4),
        6: lambda a0, a1, a2, a3, a4, a5: implementation(a0, a1, a2, a3, a4, a5),
    }
    return wrappers[count]


def theory(definition: Mapping[str, object], models: Mapping[str, Model]) -> FirstOrderTheory:
    """Build every nonterminal's language and every constructor."""
    structures = [Structure(name, data) for name, data in models.items()]
    nonterminals: Mapping[str, Mapping[str, object]] = definition["nonterminals"]

    def assignments(variables: Sequence[Sequence[str]], structure: Structure) -> Iterable[Tuple[str, ...]]:
        return itertools.product(*(structure.elements[sort] for _, sort in variables))

    def formula_label(model: str, assignment: Tuple[str, ...]) -> str:
        return f"{model}|{','.join(assignment)}"

    def term_label(model: str, assignment: Tuple[str, ...], element: str) -> str:
        return f"{model}|{','.join(assignment)}|{element}"

    # Rows of every nonterminal, as (structure, assignment, element or None).
    rows: Dict[str, List[Tuple[Structure, Tuple[str, ...], object]]] = {}
    languages: Dict[str, loom2.Vec] = {}
    for name, nonterminal in nonterminals.items():
        sort = nonterminal["sort"]
        entries = []
        for structure in structures:
            for assignment in assignments(nonterminal["variables"], structure):
                if sort == "Bool":
                    entries.append((structure, assignment, None))
                else:
                    entries.extend((structure, assignment, element) for element in structure.elements[sort])
        rows[name] = entries
        labels = [
            formula_label(s.name, a) if e is None else term_label(s.name, a, e)
            for s, a, e in entries
        ]
        languages[name] = loom2.Vec(**{label: bool for label in labels})

    def build(name: str, production: Sequence[object]):
        nonterminal = nonterminals[name]
        variables = nonterminal["variables"]
        language = languages[name]
        constructor, kind = production[0], production[1]
        arguments = list(production[2:])

        def constant(value: Callable[[Structure, Tuple[str, ...], object], bool]):
            values = {}
            for structure, assignment, element in rows[name]:
                label = formula_label(structure.name, assignment) if element is None else term_label(
                    structure.name, assignment, element
                )
                values[label] = bool(value(structure, assignment, element))
            return lambda: language(values)

        def reduce_children(children: Sequence[str], tuples_of, combine: str):
            """`op` over lanes: each output row folds AND-combined child lanes."""
            indices = [{} for _ in children]
            for structure, assignment, element in rows[name]:
                label = formula_label(structure.name, assignment) if element is None else term_label(
                    structure.name, assignment, element
                )
                lane_tuples = tuples_of(structure, assignment, element)
                for position, child in enumerate(children):
                    child_sort = nonterminals[child]["sort"]
                    indices[position][label] = [
                        term_label(structure.name, assignment, values[position]) if child_sort != "Bool"
                        else formula_label(structure.name, values[position])
                        for values in lane_tuples
                    ]
            pytrees = [loom2.Vec.pytree(**index) for index in indices]

            def implementation(*values):
                selected = values[0][pytrees[0]]
                for value, pytree in zip(values[1:], pytrees[1:]):
                    selected = selected & value[pytree]
                return selected.forall() if combine == "and" else selected.exists()

            return implementation

        if kind == "true":
            return constant(lambda s, a, e: True), ()
        if kind == "false":
            return constant(lambda s, a, e: False), ()
        if kind == "not":
            return (lambda p: ~p), tuple(arguments)
        if kind == "and":
            return (lambda p, q: p & q), tuple(arguments)
        if kind == "or":
            return (lambda p, q: p | q), tuple(arguments)
        if kind in ("forall", "exists"):
            sort, mapping, child = arguments
            names = [variable for variable, _ in variables]

            def child_assignments(structure, assignment, element):
                return [
                    (tuple(bound if field == "@" else assignment[names.index(field)] for field in mapping),)
                    for bound in structure.elements[sort]
                ]

            return reduce_children([child], child_assignments, "and" if kind == "forall" else "or"), (child,)
        if kind == "eq":
            children = arguments
            sort = nonterminals[children[0]]["sort"]
            return reduce_children(
                children, lambda s, a, e: [(v, v) for v in s.elements[sort]], "or"
            ), tuple(children)
        if kind == "atom":
            predicate, children = arguments[0], arguments[1:]
            if not children:
                return constant(lambda s, a, e: s.apply(predicate, ())), ()
            sorts = definition["functions"][predicate][0]

            def satisfying(structure, assignment, element):
                return [
                    values
                    for values in itertools.product(*(structure.elements[sort] for sort in sorts))
                    if structure.apply(predicate, values)
                ]

            return reduce_children(children, satisfying, "or"), tuple(children)
        if kind == "var":
            (variable,) = arguments
            position = [v for v, _ in variables].index(variable)
            return constant(lambda s, a, e: a[position] == e), ()
        if kind == "apply":
            function, children = arguments[0], arguments[1:]
            if not children:
                return constant(lambda s, a, e: s.apply(function, ()) == e), ()
            sorts = definition["functions"][function][0]

            def mapping_to(structure, assignment, element):
                return [
                    values
                    for values in itertools.product(*(structure.elements[sort] for sort in sorts))
                    if structure.apply(function, values) == element
                ]

            return reduce_children(children, mapping_to, "or"), tuple(children)
        raise ValueError(f"{constructor}: unknown production kind {kind!r}")

    constructors: Dict[str, loom2.CompiledFunction] = {}
    grammar: Dict[str, List[object]] = {}
    for name, nonterminal in nonterminals.items():
        entries = []
        for production in nonterminal["productions"]:
            implementation, children = build(name, production)
            function = _arity(len(children), implementation)
            parameters = function.__code__.co_varnames[: len(children)]
            function.__annotations__ = {
                **{parameter: languages[child] for parameter, child in zip(parameters, children)},
                "return": languages[name],
            }
            compiled = loom2.compile(function, name=production[0])
            constructors[production[0]] = compiled
            entries.append(compiled if not children else (compiled, *children))
        grammar[name] = entries
    start = definition["start"]
    ordered = {start: grammar[start], **{name: value for name, value in grammar.items() if name != start}}
    return FirstOrderTheory(languages, constructors, ordered, start, tuple(models))


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
        languages=instance.languages,
        grammar=instance.grammar,
        constraints=constraints,
        name="phi",
        **options,
    )


__all__ = ["FirstOrderTheory", "Structure", "synth_fun", "theory"]

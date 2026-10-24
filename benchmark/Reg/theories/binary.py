"""Loom theories.sl (regular expressions over {0,1}) as a ``Vec`` theory.

``CONSTRUCTORS`` lists the grammar in Loom's order: character constructors
with the predicate they test (``"*"`` matches any single character), then the
regex operators. ``CHARACTERS`` gives the predicates Loom's models assign to
every character that occurs in this theory's model strings.
"""

from __future__ import annotations

from typing import Dict, Iterable

import loom2

from ..regex import RegexTheory, theory as regex_theory

CONSTRUCTORS = [('char.0', 'zero'),
 ('char.1', 'one'),
 ('re.allchar', '*'),
 ('re.opt', None),
 ('re.++', None),
 ('re.union', None),
 ('re.*', None)]

CHARACTERS = {'0': ('zero',), '1': ('one',), '?': ('one', 'zero'), '_': ()}


def theory(positives: Iterable[str], negatives: Iterable[str]) -> RegexTheory:
    """The shared substring language and every constructor."""
    return regex_theory(CONSTRUCTORS, CHARACTERS, set(positives) | set(negatives))


def functions(positives: Iterable[str], negatives: Iterable[str]) -> Dict[str, loom2.CompiledFunction]:
    """Every grammar constructor, compiled with ``loom2.compile``."""
    return dict(theory(positives, negatives).constructors)


def synth_fun(out_file, positives: Iterable[str], negatives: Iterable[str], **options) -> loom2.SynthesisProgram:
    """Synthesize a regex accepting ``positives`` and rejecting ``negatives``."""
    positives, negatives = set(positives), set(negatives)
    if positives & negatives:
        raise ValueError(f"examples are both accepted and rejected: {sorted(positives & negatives)}")
    definition = theory(positives, negatives)
    constraints = {value: True for value in positives}
    constraints.update({value: False for value in negatives})
    return loom2.synth_fun(
        out_file,
        language=definition.language,
        grammar=definition.grammar,
        constraints=constraints,
        **options,
    )

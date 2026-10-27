"""Loom theories.stackoverflow.regfix.sl (Regel StackOverflow, regfix) as a ``Vec`` theory.

``CONSTRUCTORS`` lists the grammar in Loom's order: character constructors
with the predicate they test (``"*"`` matches any single character), then the
regex operators. ``CHARACTERS`` gives the predicates Loom's models assign to
every character that occurs in this theory's model strings.
"""

from __future__ import annotations

from typing import Dict, Iterable

import loom2

from ..regex import RegexTheory, theory as regex_theory

CONSTRUCTORS = [('char.any', 'is.any'),
 ('char.digit', 'is.digit'),
 ('char.lower', 'is.lower'),
 ('char.upper', 'is.upper'),
 ('char.alpha', 'is.alpha'),
 ('char.alnum', 'is.alnum'),
 ('char.word', 'is.word'),
 ('char.space', 'is.space'),
 ('char.hex', 'is.hex'),
 ('char.x20', 'is.x20'),
 ('char.x21', 'is.x21'),
 ('char.x23', 'is.x23'),
 ('char.x25', 'is.x25'),
 ('char.x2C', 'is.x2C'),
 ('char.x2D', 'is.x2D'),
 ('char.x2E', 'is.x2E'),
 ('char.x30', 'is.x30'),
 ('char.x31', 'is.x31'),
 ('char.x32', 'is.x32'),
 ('char.x33', 'is.x33'),
 ('char.x34', 'is.x34'),
 ('char.x35', 'is.x35'),
 ('char.x36', 'is.x36'),
 ('char.x37', 'is.x37'),
 ('char.x38', 'is.x38'),
 ('char.x39', 'is.x39'),
 ('char.x40', 'is.x40'),
 ('char.x41', 'is.x41'),
 ('char.x42', 'is.x42'),
 ('char.x45', 'is.x45'),
 ('char.x48', 'is.x48'),
 ('char.x4C', 'is.x4C'),
 ('char.x4F', 'is.x4F'),
 ('char.x5A', 'is.x5A'),
 ('char.x61', 'is.x61'),
 ('char.x62', 'is.x62'),
 ('char.x63', 'is.x63'),
 ('char.x64', 'is.x64'),
 ('char.x65', 'is.x65'),
 ('char.x67', 'is.x67'),
 ('char.x68', 'is.x68'),
 ('char.x69', 'is.x69'),
 ('char.x6E', 'is.x6E'),
 ('char.x6F', 'is.x6F'),
 ('char.x74', 'is.x74'),
 ('char.x79', 'is.x79'),
 ('char.x39B', 'is.x39B'),
 ('re.opt', None),
 ('re.plus', None),
 ('re.++', None),
 ('re.union', None),
 ('re.inter', None),
 ('re.*', None)]

CHARACTERS = {' ': ('is.any', 'is.space', 'is.x20'),
 '!': ('is.any', 'is.x21'),
 '#': ('is.any', 'is.x23'),
 '%': ('is.any', 'is.x25'),
 ',': ('is.any', 'is.x2C'),
 '-': ('is.any', 'is.x2D'),
 '.': ('is.any', 'is.x2E'),
 '0': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word', 'is.x30'),
 '1': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word', 'is.x31'),
 '2': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word', 'is.x32'),
 '3': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word', 'is.x33'),
 '4': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word', 'is.x34'),
 '5': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word', 'is.x35'),
 '6': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word', 'is.x36'),
 '7': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word', 'is.x37'),
 '8': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word', 'is.x38'),
 '9': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word', 'is.x39'),
 '@': ('is.any', 'is.x40'),
 'A': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.upper', 'is.word', 'is.x41'),
 'B': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.upper', 'is.word', 'is.x42'),
 'E': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.upper', 'is.word', 'is.x45'),
 'H': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word', 'is.x48'),
 'L': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word', 'is.x4C'),
 'O': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word', 'is.x4F'),
 'Z': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word', 'is.x5A'),
 'a': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.lower', 'is.word', 'is.x61'),
 'b': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.lower', 'is.word', 'is.x62'),
 'c': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.lower', 'is.word', 'is.x63'),
 'd': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.lower', 'is.word', 'is.x64'),
 'e': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.lower', 'is.word', 'is.x65'),
 'g': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word', 'is.x67'),
 'h': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word', 'is.x68'),
 'i': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word', 'is.x69'),
 'n': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word', 'is.x6E'),
 'o': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word', 'is.x6F'),
 't': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word', 'is.x74'),
 'y': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word', 'is.x79'),
 'Λ': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word', 'is.x39B')}


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
    options.setdefault("commutative", definition.commutative)
    return loom2.synth_fun(
        out_file,
        language=definition.language,
        grammar=definition.grammar,
        constraints=constraints,
        **options,
    )

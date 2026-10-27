"""Loom theories.stackoverflow.sl (Regel StackOverflow) as a ``Vec`` theory.

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
 ('char.x21', 'is.x21'),
 ('char.x23', 'is.x23'),
 ('char.x24', 'is.x24'),
 ('char.x25', 'is.x25'),
 ('char.x26', 'is.x26'),
 ('char.x27', 'is.x27'),
 ('char.x28', 'is.x28'),
 ('char.x29', 'is.x29'),
 ('char.x2A', 'is.x2A'),
 ('char.x2B', 'is.x2B'),
 ('char.x2C', 'is.x2C'),
 ('char.x2D', 'is.x2D'),
 ('char.x2E', 'is.x2E'),
 ('char.x2F', 'is.x2F'),
 ('char.x3A', 'is.x3A'),
 ('char.x3B', 'is.x3B'),
 ('char.x3C', 'is.x3C'),
 ('char.x3D', 'is.x3D'),
 ('char.x3E', 'is.x3E'),
 ('char.x3F', 'is.x3F'),
 ('char.x40', 'is.x40'),
 ('char.x5C', 'is.x5C'),
 ('char.x5E', 'is.x5E'),
 ('char.x5F', 'is.x5F'),
 ('char.x7B', 'is.x7B'),
 ('char.x7C', 'is.x7C'),
 ('char.x7D', 'is.x7D'),
 ('re.opt', None),
 ('re.plus', None),
 ('re.++', None),
 ('re.union', None),
 ('re.inter', None),
 ('re.*', None)]

CHARACTERS = {' ': ('is.any', 'is.space'),
 '!': ('is.any', 'is.x21'),
 '#': ('is.any', 'is.x23'),
 '$': ('is.any', 'is.x24'),
 '%': ('is.any', 'is.x25'),
 '&': ('is.any', 'is.x26'),
 "'": ('is.any', 'is.x27'),
 '(': ('is.any', 'is.x28'),
 ')': ('is.any', 'is.x29'),
 '*': ('is.any', 'is.x2A'),
 '+': ('is.any', 'is.x2B'),
 ',': ('is.any', 'is.x2C'),
 '-': ('is.any', 'is.x2D'),
 '.': ('is.any', 'is.x2E'),
 '/': ('is.any', 'is.x2F'),
 '0': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word'),
 '1': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word'),
 '2': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word'),
 '3': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word'),
 '4': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word'),
 '5': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word'),
 '6': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word'),
 '7': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word'),
 '8': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word'),
 '9': ('is.alnum', 'is.any', 'is.digit', 'is.hex', 'is.word'),
 ':': ('is.any', 'is.x3A'),
 ';': ('is.any', 'is.x3B'),
 '<': ('is.any', 'is.x3C'),
 '=': ('is.any', 'is.x3D'),
 '>': ('is.any', 'is.x3E'),
 '?': ('is.any', 'is.x3F'),
 '@': ('is.any', 'is.x40'),
 'A': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.upper', 'is.word'),
 'B': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.upper', 'is.word'),
 'C': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.upper', 'is.word'),
 'D': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.upper', 'is.word'),
 'E': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.upper', 'is.word'),
 'F': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.upper', 'is.word'),
 'G': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'H': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'I': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'J': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'K': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'L': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'M': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'N': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'O': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'P': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'Q': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'R': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'S': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'T': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'U': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'W': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'X': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'Y': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 'Z': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word'),
 '\\': ('is.any', 'is.x5C'),
 '^': ('is.any', 'is.x5E'),
 '_': ('is.any', 'is.word', 'is.x5F'),
 'a': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.lower', 'is.word'),
 'b': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.lower', 'is.word'),
 'c': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.lower', 'is.word'),
 'd': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.lower', 'is.word'),
 'e': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.lower', 'is.word'),
 'f': ('is.alnum', 'is.alpha', 'is.any', 'is.hex', 'is.lower', 'is.word'),
 'g': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'h': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'i': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'j': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'k': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'l': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'm': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'n': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'o': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'p': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'q': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'r': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 's': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 't': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'u': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'v': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'w': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'x': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'y': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 'z': ('is.alnum', 'is.alpha', 'is.any', 'is.lower', 'is.word'),
 '{': ('is.any', 'is.x7B'),
 '|': ('is.any', 'is.x7C'),
 '}': ('is.any', 'is.x7D'),
 'Λ': ('is.alnum', 'is.alpha', 'is.any', 'is.upper', 'is.word')}


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

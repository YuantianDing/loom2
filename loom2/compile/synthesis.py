"""Trace a ``Vec`` grammar and compile it to a native synthesizer.

The whole pipeline after tracing is native Rust:

1. every constructor and the constraint are traced into ``TensorFn`` graphs;
2. permutes and reductions are fused into ``gather``/``gather2`` nodes;
3. each graph is lowered to a small vector loop program;
4. Melior builds the synthesis module from upstream MLIR dialects only;
5. upstream MLIR passes and LLVM produce CPU (or, with ``device="cuda"``,
   GPU) MLIR, LLVM-dialect MLIR, LLVM IR, and a native object; and
6. the object is linked with the ``cpp/cpu`` (or ``cpp/cuda``) runtime.

Every stage is written to ``artifact_dir``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from loom2.lang import Vec

from .cfg import Cfg
from .function import CompiledFunction, compile as compile_function


# Largest candidate, in constructor tokens, when no ``max_size`` is given.
# The benchmark runners and Rust's ``SynthesisOptions::default`` share it.
DEFAULT_MAX_SIZE = 20


class SynthesisError(RuntimeError):
    """A malformed grammar or failed native compilation stage."""


@dataclass(frozen=True)
class SynthesisProgram:
    """Artifacts produced for one synthesized executable."""

    executable: Path
    artifact_dir: Path
    stages: Tuple[Path, ...]

    def run(self) -> str:
        process = subprocess.run(
            (str(self.executable),), check=True, stdout=subprocess.PIPE, text=True
        )
        return process.stdout.strip()


GrammarEntry = Union[CompiledFunction, Tuple[object, ...]]


def _native():
    try:
        from loom2 import _loom2
    except ImportError as error:
        raise SynthesisError(
            "loom2.synth_fun needs the native extension; build it with "
            "`maturin develop --release --features llvm`"
        ) from error
    if not hasattr(_loom2, "build_synthesizer"):
        raise SynthesisError(
            "the native extension was built without the `llvm` feature; rebuild "
            "with `maturin develop --release --features llvm`"
        )
    return _loom2


def _constraint_function(language: Vec, constraints: Mapping[str, bool]) -> CompiledFunction:
    """Trace example constraints as one tensor-wide Boolean function.

    Positive labels are reduced with one ``forall`` and negative labels with
    one ``exists``, so the constraint stays two gathers wide regardless of the
    number of examples.
    """
    positives = [label for label, expected in constraints.items() if expected]
    negatives = [label for label, expected in constraints.items() if not expected]

    def constraint(candidate):
        result = True
        if positives:
            result = candidate[Vec.pytree(positives)].forall()
        if negatives:
            result = result & ~candidate[Vec.pytree(negatives)].exists()
        return result

    constraint.__annotations__ = {"candidate": language, "return": Vec(bool)}
    return compile_function(constraint, name="constraint")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synth_fun(
    output: Union[str, "os.PathLike[str]"],
    *,
    language: Vec,
    grammar: Union[Mapping[str, Sequence[GrammarEntry]], Cfg],
    constraints: Mapping[str, bool],
    languages: Optional[Mapping[str, Vec]] = None,
    groups: Optional[Mapping[str, str]] = None,
    commutative: Sequence[CompiledFunction] = (),
    name: str = "r",
    artifact_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
    max_size: int = DEFAULT_MAX_SIZE,
    max_pool_bytes: int = 64 * 1024 * 1024,
    initial_hash_capacity: int = 4096,
    optimization_level: int = 1,
    openmp: bool = True,
    debug_info: bool = True,
    device: str = "cpu",
    cuda_chip: Optional[str] = None,
    cuda_block_size: int = 256,
) -> SynthesisProgram:
    """Trace a ``Vec`` grammar and generate a native synthesis executable.

    Grammar entries are either nullary :class:`CompiledFunction` objects or a
    tuple ``(function, child_nonterminal, ...)``. The first nonterminal is the
    start symbol. A nonterminal denotes values of ``languages[symbol]`` when
    given, and of ``language`` otherwise; ``language`` must be the start
    symbol's language. Constraints map labels in ``language`` to their
    required Boolean value.

    ``grammar`` may also be a :class:`~loom2.Cfg`, for example one with
    patterns or tree languages subtracted: its copies of a nonterminal take
    the language of the nonterminal they trace (``Cfg.origin``), its hash
    groups, and its ``solutions`` are the candidates checked against the
    constraints.

    ``groups`` maps nonterminals to hash groups (default: their own name,
    or the ``Cfg``'s groups). Nonterminals of one group share one hash set,
    so a value is kept only the first time any of them produces it: group
    only nonterminals that denote the same values, of the same width.
    Without a ``Cfg``, every nonterminal in the start symbol's group is a
    candidate solution, checked against the constraints.

    ``commutative`` lists binary constructors with ``f(x, y) = f(y, x)``
    (added to the ``Cfg``'s): of two candidates that only swap the children
    of such a production, only the one whose first child was stored first
    is built, when both children share a hash group and the grammar also
    has the swapped production.

    ``device="cuda"`` runs every production filter as a CUDA kernel
    (``cuda_block_size`` threads per block, compiled for ``cuda_chip``, such
    as ``"sm_75"``; default: the GPU found when loom2 was built). It needs
    loom2 built with the ``cuda`` feature.

    ``debug_info`` gives the executable DWARF line tables; the generated
    code's lines point at the Rust code in loom2 that emitted it.
    """
    languages = dict(languages or {})
    if not grammar:
        raise ValueError("the synthesis grammar must not be empty")
    solutions: Tuple[str, ...] = ()
    commutative = set(commutative)
    if isinstance(grammar, Cfg):
        cfg = grammar
        groups = {**{symbol: cfg.group(symbol) for symbol in cfg.nonterminals}, **dict(groups or {})}
        languages = {symbol: languages.get(symbol, languages.get(cfg.origin(symbol), language)) for symbol in cfg.nonterminals}
        solutions = cfg.solutions
        commutative |= cfg.commutative
        grammar = {symbol: [(function, *children) for function, children in cfg.productions(symbol)] for symbol in cfg.nonterminals}
    groups = dict(groups or {})
    unknown_groups = [symbol for symbol in groups if symbol not in grammar]
    if unknown_groups:
        raise ValueError(f"groups names unknown nonterminal {unknown_groups[0]!r}")
    functions: Dict[str, CompiledFunction] = {}
    specification: List[Tuple[str, int, str, bool, List[Tuple[str, List[str], bool]]]] = []
    for symbol, entries in grammar.items():
        width = languages.get(symbol, language).bit_width()
        if not entries:
            raise ValueError(f"nonterminal {symbol!r} has no productions")
        productions = []
        for entry in entries:
            if isinstance(entry, CompiledFunction):
                function, children = entry, ()
            else:
                function, children = entry[0], tuple(entry[1:])
            if not isinstance(function, CompiledFunction):
                raise TypeError("grammar entries must start with a loom2.compile function")
            if function.arity != len(children):
                raise ValueError(
                    f"{function.__name__} has arity {function.arity}, not {len(children)}"
                )
            unknown = [child for child in children if child not in grammar]
            if unknown:
                raise ValueError(f"production references unknown nonterminal {unknown[0]!r}")
            functions.setdefault(function.__name__, function)
            productions.append((function.__name__, list(children), function in commutative))
        specification.append((symbol, width, groups.get(symbol, symbol), symbol in solutions, productions))
    for label in constraints:
        language.index(label)

    native = _native()
    traced = [function.trace() for function in functions.values()]
    constraint = _constraint_function(language, constraints).trace()

    output_path = Path(output).resolve()
    artifacts = Path(artifact_dir).resolve() if artifact_dir else output_path.parent
    artifacts.mkdir(parents=True, exist_ok=True)
    try:
        stages = native.build_synthesizer(
            name,
            traced,
            specification,
            constraint,
            str(artifacts),
            str(output_path),
            max_size=max_size,
            max_pool_bytes=max_pool_bytes,
            initial_hash_capacity=initial_hash_capacity,
            optimization_level=optimization_level,
            openmp=openmp,
            debug_info=debug_info,
            device=device,
            cuda_chip=cuda_chip,
            cuda_block_size=cuda_block_size,
        )
    except RuntimeError as error:
        raise SynthesisError(str(error)) from error
    stages = tuple(Path(stage) for stage in stages)

    manifest = {
        "frontend": "loom2 Vec -> TensorFn -> packed i64 words -> Melior MLIR",
        "executable": str(output_path),
        "options": {
            "max_expression_size": max_size,
            "max_pool_bytes": max_pool_bytes,
            "initial_hash_capacity": initial_hash_capacity,
            "optimization_level": optimization_level,
            "openmp": openmp,
            "debug_info": debug_info,
            "device": device,
            "cuda_chip": cuda_chip,
            "cuda_block_size": cuda_block_size,
        },
        "artifacts": [
            {"file": path.name, "bytes": path.stat().st_size, "sha256": _digest(path)}
            for path in sorted(artifacts.iterdir())
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return SynthesisProgram(output_path, artifacts, stages)


__all__ = ["DEFAULT_MAX_SIZE", "SynthesisError", "SynthesisProgram", "synth_fun"]

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

from .function import CompiledFunction, compile as compile_function


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
    grammar: Mapping[str, Sequence[GrammarEntry]],
    constraints: Mapping[str, bool],
    languages: Optional[Mapping[str, Vec]] = None,
    name: str = "r",
    artifact_dir: Optional[Union[str, "os.PathLike[str]"]] = None,
    max_size: int = 8,
    max_pool_bytes: int = 64 * 1024 * 1024,
    initial_hash_capacity: int = 4096,
    optimization_level: int = 1,
    openmp: bool = True,
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

    ``device="cuda"`` runs every production filter as a CUDA kernel
    (``cuda_block_size`` threads per block, compiled for ``cuda_chip``, such
    as ``"sm_75"``; default: the GPU found when loom2 was built). It needs
    loom2 built with the ``cuda`` feature.
    """
    languages = dict(languages or {})
    if not grammar:
        raise ValueError("the synthesis grammar must not be empty")
    functions: Dict[str, CompiledFunction] = {}
    specification: List[Tuple[str, int, List[Tuple[str, List[str]]]]] = []
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
            productions.append((function.__name__, list(children)))
        specification.append((symbol, width, productions))
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


__all__ = ["SynthesisError", "SynthesisProgram", "synth_fun"]

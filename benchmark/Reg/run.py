"""Build (and optionally run) one benchmark of this family, keeping every stage.

    python benchmark/Reg/run.py 0001 --run

Stages are written to ``<id>-ir/`` next to this file:

    01-tensor-fn.ir     traced TensorFn graphs (constructors and constraint)
    02-fused.ir         after permute/reduce fusion (gather, gather2)
    03-loops.ir         vector loop programs
    04-frontend.mlir    Melior-built synthesis module (upstream dialects only)
    05-cpu.mlir         inlined, canonicalized, OpenMP CPU MLIR
    06-llvm.mlir        LLVM-dialect MLIR
    07-llvm.ll          LLVM IR
    08-program.o        optimized native object
    09-<name>           executable linked with cpp/cpu
    10-result.txt       synthesized expression (with --run)
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def compile_benchmark(args: argparse.Namespace) -> Path:
    metadata = json.loads((HERE / "benchmark.json").read_text())[args.benchmark_id]
    models = json.loads((HERE / f"models/{metadata['model']}.json").read_text())
    positives = {models[name] for name in metadata["positive"]}
    negatives = {models[name] for name in metadata["negative"]}
    theory = importlib.import_module(f"benchmark.{HERE.name}.theories.{metadata['theory']}")

    output = (args.output or HERE / f"{args.benchmark_id}-ir").resolve()
    output.mkdir(parents=True, exist_ok=True)
    slug = HERE.name.lower().replace("+", "-plus")
    program = theory.synth_fun(
        output / f"09-{slug}-{args.benchmark_id}",
        positives,
        negatives,
        artifact_dir=output,
        max_size=args.max_size,
        max_pool_bytes=args.max_pool_bytes,
        initial_hash_capacity=args.initial_hash_capacity,
        optimization_level=args.opt_level,
        openmp=args.openmp,
    )
    for stage in program.stages:
        print(f"wrote {stage}")
    if args.run:
        result = program.run()
        (output / "10-result.txt").write_text(result + "\n")
        print(result)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("benchmark_id", nargs="?", default="0030" if HERE.name == "Reg+" else "0001")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-size", type=int, default=8)
    parser.add_argument("--max-pool-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--initial-hash-capacity", type=int, default=4096)
    parser.add_argument("-O", "--opt-level", type=int, choices=range(4), default=1)
    parser.add_argument("--openmp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        compile_benchmark(_parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

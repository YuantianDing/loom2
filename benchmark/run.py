"""Build (and optionally run) one benchmark of any family.

    python benchmark/run.py Reg+ 0030 --run
    python benchmark/run.py LTL 0001 --stages benchmark/LTL/0001-ir

The executable goes to ``--output`` (default ``benchmark/build/<family>-<id>``).
Intermediate stages are discarded unless ``--stages DIR`` keeps them:

    01-tensor-fn.ir     traced TensorFn graphs (constructors and constraint)
    02-fused.ir         after permute/reduce fusion (gather, gather2)
    03-frontend.mlir    high-mlir-built synthesis module (upstream dialects only)
    04-cpu.mlir         inlined, canonicalized, OpenMP CPU MLIR
                        (04-gpu.mlir with outlined kernels for --device cuda)
    05-llvm.mlir        LLVM-dialect MLIR
    06-llvm.ll          LLVM IR
    07-program.o        optimized native object
    08-<family>-<id>    executable linked with cpp/cpu (or cpp/cuda), unless --output
    09-result.txt       synthesized expression (with --run)
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loom2  # noqa: E402  (needs ROOT on sys.path)

FAMILIES = {path.parent.name: path.parent for path in sorted(HERE.glob("*/benchmark.json"))}


def family_directory(name: str) -> Path:
    """A family by directory name (``Reg+``), case-insensitively, or slug (``reg-plus``)."""
    for family, directory in FAMILIES.items():
        if name.lower() in (family.lower(), slug(family)):
            return directory
    raise ValueError(f"unknown benchmark family {name!r}; expected one of {', '.join(FAMILIES)}")


def slug(family: str) -> str:
    return family.lower().replace("+", "-plus")


def load_models(directory: Path, metadata: dict) -> dict:
    """A benchmark's models: one file per benchmark, or shared model files."""
    files = metadata["models"] if "models" in metadata else [metadata["model"]]
    models = {}
    for file in files:
        models.update(json.loads((directory / f"models/{file}.json").read_text()))
    return models


def synth_fun(directory: Path, benchmark_id: str, executable: Path, **options) -> loom2.SynthesisProgram:
    """Call the family's ``synth_fun`` on one benchmark."""
    metadata = json.loads((directory / "benchmark.json").read_text())[benchmark_id]
    models = load_models(directory, metadata)
    interpreter = directory / f"{directory.name.lower()}.py"
    if interpreter.is_file():
        # FO/LTL/CTL: a JSON theory interpreted over named models.
        family = importlib.import_module(f"benchmark.{directory.name}.{interpreter.stem}")
        definition = json.loads((directory / f"theories/{metadata['theory']}.json").read_text())
        return family.synth_fun(
            executable, definition, models, metadata["positive"], metadata["negative"], **options
        )
    # Reg/Reg+: a Python theory over example strings.
    theory = importlib.import_module(f"benchmark.{directory.name}.theories.{metadata['theory']}")
    positives = {models[name] for name in metadata["positive"]}
    negatives = {models[name] for name in metadata["negative"]}
    return theory.synth_fun(executable, positives, negatives, **options)


def build(args: argparse.Namespace, stages: Path) -> None:
    directory = family_directory(args.family)
    name = f"{slug(directory.name)}-{args.benchmark_id}"
    if args.output:
        executable = args.output.resolve()
    elif args.stages:
        executable = stages / f"08-{name}"
    else:
        executable = HERE / "build" / name
    program = synth_fun(
        directory,
        args.benchmark_id,
        executable,
        artifact_dir=stages,
        max_size=args.max_size,
        max_pool_bytes=args.max_pool_bytes,
        initial_hash_capacity=args.initial_hash_capacity,
        optimization_level=args.opt_level,
        openmp=args.openmp,
        debug_info=args.debug_info,
        device=args.device,
        cuda_block_size=args.cuda_block_size,
    )
    for stage in program.stages if args.stages else [program.executable]:
        print(f"wrote {stage}")
    if args.run:
        result = program.run()
        if args.stages:
            (stages / "09-result.txt").write_text(result + "\n")
        print(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("family", help=f"one of {', '.join(FAMILIES)}")
    parser.add_argument("benchmark_id")
    parser.add_argument("-o", "--output", type=Path, help="executable path")
    parser.add_argument("--stages", type=Path, help="keep every intermediate stage in this directory")
    parser.add_argument("--max-size", type=int, default=loom2.DEFAULT_MAX_SIZE)
    parser.add_argument("--max-pool-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--initial-hash-capacity", type=int, default=4096)
    parser.add_argument("-O", "--opt-level", type=int, choices=range(4), default=2)
    parser.add_argument("--openmp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-info", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cuda-block-size", type=int, default=256)
    parser.add_argument("--run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.stages:
            args.stages = args.stages.resolve()
            args.stages.mkdir(parents=True, exist_ok=True)
            build(args, args.stages)
        else:
            with tempfile.TemporaryDirectory(prefix="loom2-stages-") as temporary:
                build(args, Path(temporary))
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

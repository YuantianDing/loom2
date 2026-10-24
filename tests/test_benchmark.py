import importlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import loom2  # noqa: E402

BENCHMARK = ROOT / "benchmark"


def family(name):
    return json.loads((BENCHMARK / name / "benchmark.json").read_text())


def benchmark_theory(name, benchmark_id):
    """The theory of one benchmark and its constraints over strings."""
    metadata = family(name)[benchmark_id]
    models = json.loads((BENCHMARK / name / f"models/{metadata['model']}.json").read_text())
    module = importlib.import_module(f"benchmark.{name}.theories.{metadata['theory']}")
    positives = {models[model] for model in metadata["positive"]}
    negatives = {models[model] for model in metadata["negative"]}
    constraints = {value: True for value in positives}
    constraints.update({value: False for value in negatives})
    return module.theory(positives, negatives), constraints

try:
    from loom2 import _loom2

    NATIVE_BACKEND = hasattr(_loom2, "build_synthesizer")
    CUDA_BACKEND = getattr(_loom2, "CUDA_BACKEND", False)
except ImportError:
    NATIVE_BACKEND = CUDA_BACKEND = False


def has_gpu():
    if not shutil.which("nvidia-smi"):
        return False
    return subprocess.run(("nvidia-smi", "-L"), capture_output=True).returncode == 0


def parse_expression(text):
    tokens = re.findall(r"[^\s(),]+|[(),]", text)
    position = 0

    def expression():
        nonlocal position
        name = tokens[position]
        position += 1
        arguments = []
        if position < len(tokens) and tokens[position] == "(":
            position += 1
            while True:
                arguments.append(expression())
                position += 1
                if tokens[position - 1] == ")":
                    break
        return name, arguments

    return expression()


def evaluate(tree, theory):
    name, arguments = tree
    return theory.constructors[name](*(evaluate(argument, theory) for argument in arguments))


class BenchmarkTests(unittest.TestCase):
    def test_suites_are_complete_and_consistent(self):
        reg, reg_plus = family("Reg"), family("Reg+")
        self.assertEqual(len(reg), 57)
        self.assertEqual(len(reg_plus), 322)
        self.assertEqual(reg_plus["0030"]["theory"], "deepregex")
        self.assertIn("w00079", reg_plus["0030"]["negative"])
        for name, entries in (("Reg", reg), ("Reg+", reg_plus)):
            for benchmark_id, metadata in entries.items():
                self.assertEqual(set(metadata), {"theory", "model", "positive", "negative"})
                self.assertTrue((BENCHMARK / name / f"theories/{metadata['theory']}.py").is_file())
                models = json.loads((BENCHMARK / name / f"models/{metadata['model']}.json").read_text())
                for model in metadata["positive"] + metadata["negative"]:
                    self.assertIn(model, models, (name, benchmark_id))
        deepregex = importlib.import_module("benchmark.Reg+.theories.deepregex")
        self.assertIn(("re.inter", None), deepregex.CONSTRUCTORS)
        self.assertEqual(deepregex.CHARACTERS["!"], ("is.any", "is.x21"))

    def test_models_share_one_substring_literal(self):
        theory, constraints = benchmark_theory("Reg+", "0030")
        self.assertEqual(
            theory.substrings,
            ("", "!", "0", "@", "A", "a", "!!", "!0", "!@", "!a", "AA", "aa", "!!!", "!!a", "AAA"),
        )
        self.assertEqual(len(constraints), 13)
        self.assertEqual(theory.language.bit_width(), 15)

    def test_theory_agrees_with_python_regular_expressions(self):
        theory, _ = benchmark_theory("Reg+", "0030")
        language, constructors = theory.language, theory.constructors

        def value(pattern):
            return language({k: re.fullmatch(pattern, k) is not None for k in theory.substrings})

        for pattern in ("a", "!", "a|!", "!a", "[A0]"):
            for name, wrapped in (("re.*", "(?:{})*"), ("re.plus", "(?:{})+"), ("re.opt", "(?:{})?")):
                self.assertEqual(
                    constructors[name](value(pattern)).values,
                    value(wrapped.format(pattern)).values,
                    (name, pattern),
                )
            self.assertEqual(
                constructors["re.++"](value(pattern), value("!|a")).values,
                value(f"(?:{pattern})(?:!|a)").values,
            )
        self.assertEqual(constructors["char.x21"]().values, value("!").values)
        self.assertEqual(constructors["char.alnum"]().values, value("[A-Za-z0-9]").values)

    @unittest.skipUnless(NATIVE_BACKEND, "native extension built without the llvm feature")
    def test_native_pipeline_synthesizes_a_verified_regex(self):
        theory, constraints = benchmark_theory("Reg+", "0030")
        with tempfile.TemporaryDirectory() as directory:
            program = loom2.synth_fun(
                Path(directory) / "synth",
                language=theory.language,
                grammar=theory.grammar,
                constraints=constraints,
                artifact_dir=directory,
                max_size=4,
            )
            names = [stage.name for stage in program.stages]
            self.assertEqual(names[:8], [
                "01-tensor-fn.ir", "02-fused.ir", "03-loops.ir", "04-frontend.mlir",
                "05-cpu.mlir", "06-llvm.mlir", "07-llvm.ll", "08-program.o",
            ])
            fused = (Path(directory) / "02-fused.ir").read_text()
            self.assertIn("gather2.or.and", fused)
            frontend = (Path(directory) / "04-frontend.mlir").read_text()
            # Operation names start a statement or follow `=`; `loom.*` names
            # that appear are attributes only.
            operations = re.findall(r"(?:^\s*|= )([a-z_]+)\.[a-z_]+", frontend, re.MULTILINE)
            self.assertLessEqual(set(operations), {"func", "arith", "scf", "memref", "vector"})
            self.assertIn("vector.transfer_read", frontend)
            cpu = (Path(directory) / "05-cpu.mlir").read_text()
            self.assertIn("omp.parallel", cpu)
            self.assertNotIn("xi1>", cpu)

            output = program.run()
            self.assertTrue(output.startswith("r = "), output)
            solution = evaluate(parse_expression(output[len("r = "):]), theory)
            for label, expected in constraints.items():
                self.assertIs(solution[label], expected, label)
            self.assertEqual(len(re.findall(r"[^\s(),]+", output[len("r = "):])), 4)

    @unittest.skipUnless(CUDA_BACKEND and has_gpu(), "needs the cuda feature and an NVIDIA GPU")
    def test_cuda_pipeline_synthesizes_a_verified_regex(self):
        theory, constraints = benchmark_theory("Reg+", "0030")
        with tempfile.TemporaryDirectory() as directory:
            program = loom2.synth_fun(
                Path(directory) / "synth",
                language=theory.language,
                grammar=theory.grammar,
                constraints=constraints,
                artifact_dir=directory,
                max_size=4,
                device="cuda",
            )
            names = [stage.name for stage in program.stages]
            self.assertEqual(names[3:5], ["04-frontend.mlir", "05-gpu.mlir"])
            gpu = (Path(directory) / "05-gpu.mlir").read_text()
            self.assertIn("gpu.launch_func", gpu)
            self.assertIn("gpu.module", gpu)
            self.assertIn("gpu.binary", (Path(directory) / "06-llvm.mlir").read_text())

            output = program.run()
            self.assertTrue(output.startswith("r = "), output)
            solution = evaluate(parse_expression(output[len("r = "):]), theory)
            for label, expected in constraints.items():
                self.assertIs(solution[label], expected, label)
            self.assertEqual(len(re.findall(r"[^\s(),]+", output[len("r = "):])), 4)


if __name__ == "__main__":
    unittest.main()

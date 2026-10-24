"""LLVM-JIT correctness: every compiled function must agree with its Python body.

The Python body evaluated on ``VecValue`` arguments is an independent
reference (it never touches the tracer, the TensorFn IR, fusion, loop
programs, MLIR, or LLVM). Each check runs the JIT and the Python body on
all-false, all-true, and random inputs at several densities, compares every
result bit, and checks that the padding bits of the last result word are 0.

Covered: every operator of the tracing frontend in isolation and combined,
widths around the 32/64-bit word boundaries, and every constructor of one
benchmark per theory in Reg, Reg+, FO, LTL, and CTL.
"""

import importlib
import json
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import loom2  # noqa: E402
from loom2 import Vec, VecValue  # noqa: E402

function_module = importlib.import_module("loom2.compile.function")

BENCHMARK = ROOT / "benchmark"

try:
    from loom2._loom2 import JitFunction  # noqa: F401

    HAVE_JIT = True
except ImportError:
    HAVE_JIT = False


def bits_of(value):
    if isinstance(value, VecValue):
        return value.values
    if isinstance(value, bool):
        return (value,)
    raise TypeError(f"unexpected result {value!r}")


def random_value(schema, rng, sample):
    width = schema.bit_width()
    if sample == 0:
        values = [False] * width
    elif sample == 1:
        values = [True] * width
    else:
        density = (0.5, 0.1, 0.9, 0.3)[sample % 4]
        values = [rng.random() < density for _ in range(width)]
    return VecValue(schema, values)


def check_function(test, function, samples=10, seed=0, context=""):
    """JIT ``function`` and compare it with its Python body on ``samples`` inputs."""
    compiled = function.compiled()
    test.assertIsNotNone(compiled, f"{context}{function.__name__} did not JIT")
    schemas = function._parameters
    rng = random.Random(f"{seed}:{function.__name__}")
    for sample in range(samples):
        arguments = [random_value(schema, rng, sample) for schema in schemas]
        expected = bits_of(function.function(*arguments))
        got = bits_of(function(*arguments))
        if got != expected:
            wrong = [index for index, (a, b) in enumerate(zip(got, expected)) if a != b]
            test.fail(
                f"{context}{function.__name__}: sample {sample}: {len(wrong)} wrong bits, "
                f"first at {wrong[:8]} of {len(expected)}"
            )
        # Padding bits beyond the result width must be zero (hashing relies on it).
        words = []
        for value, bits in zip(arguments, compiled.input_bits):
            words.extend(function_module._pack(value.values, bits))
        result = compiled(words)
        test.assertEqual(len(result), (compiled.result_bits + 63) // 64)
        if compiled.result_bits % 64:
            test.assertEqual(result[-1] >> (compiled.result_bits % 64), 0, f"{context}{function.__name__} padding")
    return compiled


def typed(name, schema_in, schema_out, body, arity):
    parameters = body.__code__.co_varnames[:arity]
    body.__annotations__ = {**{parameter: schema_in for parameter in parameters}, "return": schema_out}
    return loom2.compile(body, name=name, jit=True)


def shift_by(amount):
    return lambda x: x >> amount


def flat(width, prefix="b"):
    return Vec(**{f"{prefix}{index}": bool for index in range(width)})


WIDTHS = (1, 2, 31, 32, 33, 63, 64, 65, 100, 128, 129, 257)


@unittest.skipUnless(HAVE_JIT, "loom2 was built without the llvm feature")
class OperatorTests(unittest.TestCase):
    """Each tracing operator, in isolation and combined, across word boundaries."""

    def test_pointwise_logic_and_constants(self):
        for width in WIDTHS:
            L = flat(width)
            pattern = L([index % 3 == 1 for index in range(width)])
            cases = {
                "and": (lambda x, y: x & y, 2),
                "or": (lambda x, y: x | y, 2),
                "xor": (lambda x, y: x ^ y, 2),
                "not": (lambda x: ~x, 1),
                "and_true": (lambda x: x & True, 1),
                "and_false": (lambda x: x & False, 1),
                "or_true": (lambda x: x | True, 1),
                "xor_true": (lambda x: x ^ True, 1),
                "with_constant": (lambda x, y: (x & pattern) | (y ^ pattern), 2),
                "nested": (lambda x, y: ~((x | y) & (x ^ ~y)) ^ (x & ~y), 2),
                "shared": (lambda x, y: ((x & y) | (x & y)) ^ (x & y), 2),
                "three": (lambda x, y, z: (x & y) | (y & z) | (x & z), 3),
            }
            for name, (body, arity) in cases.items():
                with self.subTest(width=width, case=name):
                    check_function(self, typed(name, L, L, body, arity))

    def test_nullary_constants(self):
        for width in WIDTHS:
            L = flat(width)
            values = [index % 5 in (0, 3) for index in range(width)]
            with self.subTest(width=width):
                constant = typed("constant", L, L, lambda: L(values), 0)
                check_function(self, constant, samples=1)
                self.assertEqual(constant().values, tuple(values))
                check_function(self, typed("ones", L, L, lambda: L([True] * width), 0), samples=1)
                check_function(self, typed("zeros", L, L, lambda: L([False] * width), 0), samples=1)

    def test_shifts(self):
        for width in WIDTHS:
            L = flat(width)
            for amount in sorted({0, 1, 2, 7, 31, 32, 33, 63, 64, 65, width - 1, width, width + 3}):
                if amount < 0:
                    continue
                with self.subTest(width=width, amount=amount):
                    check_function(self, typed(f"shr{amount}", L, L, shift_by(amount), 1))
            with self.subTest(width=width, case="doubling scan"):
                # The LTL F/U schemes: cs |= cs >> 1, 2, 4, 8, ...
                def scan(x, y):
                    for k in (1, 2, 4, 8, 16, 32, 64):
                        y = y | (x & (y >> k))
                        x = x & (x >> k)
                    return y

                check_function(self, typed("scan", L, L, scan, 2))

    def test_permutes(self):
        rng = random.Random(7)
        for width in WIDTHS:
            L = flat(width)
            labels = [f"b{index}" for index in range(width)]
            with self.subTest(width=width, case="reverse"):
                index = Vec.pytree(**{label: labels[width - 1 - i] for i, label in enumerate(labels)})
                check_function(self, typed("reverse", L, L, lambda x: x[index], 1))
            with self.subTest(width=width, case="random"):
                index = Vec.pytree(**{label: rng.choice(labels) for label in labels})
                check_function(self, typed("random", L, L, lambda x: x[index], 1))
            with self.subTest(width=width, case="broadcast"):
                index = Vec.pytree(**{label: labels[0] for label in labels})
                check_function(self, typed("broadcast", L, L, lambda x: x[index], 1))
            with self.subTest(width=width, case="composed"):
                first = Vec.pytree(**{label: rng.choice(labels) for label in labels})
                second = Vec.pytree(**{label: rng.choice(labels) for label in labels})
                check_function(self, typed("composed", L, L, lambda x, y: x[first][second] ^ y[second], 2))

    def test_jagged_reductions(self):
        rng = random.Random(11)
        for width in WIDTHS:
            L = flat(width)
            labels = [f"b{index}" for index in range(width)]
            # Rows of 0..6 lanes (including empty rows, which fold to the identity).
            rows = {label: [rng.choice(labels) for _ in range(rng.randrange(7))] for label in labels}
            other = {label: [rng.choice(labels) for _ in range(len(rows[label]))] for label in labels}
            index, partner = Vec.pytree(**rows), Vec.pytree(**other)
            dense = Vec.pytree(**{label: labels for label in labels})
            cases = {
                "forall": lambda x: x[index].forall(),
                "exists": lambda x: x[index].exists(),
                "xorsum": lambda x: x[index].xorsum(),
                # reduce(op2(permute, permute)) -> gather2
                "gather2_and_or": lambda x, y: (x[index] & y[partner]).exists(),
                "gather2_or_and": lambda x, y: (x[index] | y[partner]).forall(),
                "gather2_xor_xor": lambda x, y: (x[index] ^ y[partner]).xorsum(),
                # three permutes do not fuse
                "unfused": lambda x, y: (x[index] & y[partner] & ~x[partner]).exists(),
                "dense_all": lambda x: x[dense].forall(),
                "dense_any": lambda x, y: (x[dense] & y[dense]).exists(),
                "mixed": lambda x, y: x[index].exists() ^ (y & x[partner].forall()),
            }
            for name, body in cases.items():
                with self.subTest(width=width, case=name):
                    check_function(self, typed(name, L, L, body, body.__code__.co_argcount))

    def test_scalar_results(self):
        for width in (1, 40, 64, 65, 200):
            L = flat(width)
            labels = [f"b{index}" for index in range(width)]
            everything = Vec.pytree(labels)
            for name, body in {
                "all": lambda x: x[everything].forall(),
                "any": lambda x: x[everything].exists(),
                "parity": lambda x: x[everything].xorsum(),
                "bit": lambda x: x[labels[-1]],
            }.items():
                with self.subTest(width=width, case=name):
                    compiled = check_function(self, typed(name, L, bool, body, 1))
                    self.assertEqual(compiled.result_bits, 1)

    def test_masked_updates(self):
        for width in WIDTHS:
            L = flat(width)
            labels = [f"b{index}" for index in range(width)]
            chosen = Vec.pytree(labels[::3])

            def set_true(x):
                x[labels[0]] = True
                return x

            def set_false(x):
                x[labels[-1]] = False
                return x

            def set_many(x, y):
                x[chosen] = y[chosen]
                return x

            for name, body in {"set_true": set_true, "set_false": set_false, "set_many": set_many}.items():
                with self.subTest(width=width, case=name):
                    check_function(self, typed(name, L, L, body, body.__code__.co_argcount))

    def test_two_dimensional_parameters(self):
        rows, lanes = 9, 13
        grid = Vec(**{f"r{row}": Vec(**{f"c{lane}": bool for lane in range(lanes)}) for row in range(rows)})
        column = Vec(**{f"r{row}": bool for row in range(rows)})
        for name, body in {
            "row_all": lambda x: x.forall(),
            "row_any": lambda x, y: (x & ~y).exists(),
            "row_parity": lambda x: x.xorsum(),
        }.items():
            with self.subTest(case=name):
                check_function(self, typed(name, grid, column, body, body.__code__.co_argcount))

    def test_different_input_widths(self):
        A, B, C = flat(3, "a"), flat(70, "b"), flat(130, "c")
        index_a = Vec.pytree(**{f"c{index}": f"a{index % 3}" for index in range(130)})
        index_b = Vec.pytree(**{f"c{index}": f"b{(index * 7) % 70}" for index in range(130)})

        def mix(a: A, b: B, c: C) -> C:
            return a[index_a] ^ (b[index_b] & c)

        check_function(self, loom2.compile(mix, jit=True))


@unittest.skipUnless(HAVE_JIT, "loom2 was built without the llvm feature")
class CallBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.L = flat(70)

    def test_results_keep_the_return_schema_and_compilation_is_cached(self):
        L = self.L

        def double(x: L) -> L:
            return x ^ (x >> 1)

        function = loom2.compile(double)
        value = random_value(L, random.Random(1), 2)
        result = function(value)
        self.assertIsInstance(result, VecValue)
        self.assertIs(result.schema, L)
        first = function.compiled()
        function(value)
        self.assertIs(function.compiled(), first)

    def test_calls_while_tracing_inline_the_python_body(self):
        L = self.L

        def inner(x: L) -> L:
            return x >> 2

        inner = loom2.compile(inner)

        def outer(x: L, y: L) -> L:
            return inner(x) | inner(y)

        outer = loom2.compile(outer, jit=True)
        outer.compiled()
        self.assertIsNone(inner._compiled, "the inner call ran under tracing, not the JIT")
        check_function(self, outer)

    def test_fallbacks_and_errors(self):
        L = self.L

        def plain(x: L) -> L:
            return ~x

        auto = loom2.compile(plain)
        # Non-VecValue arguments run the Python body in auto mode, raise with jit=True.
        with self.assertRaises(TypeError):
            loom2.compile(plain, jit=True)("not a vec")
        off = loom2.compile(plain, jit=False)
        off(L([False] * 70))
        self.assertIsNone(off._compiled)
        # Untraceable (unannotated) functions fall back in auto mode.
        untyped = loom2.compile(lambda x: ~x)
        self.assertEqual(untyped(L([False] * 70)).values, (True,) * 70)
        self.assertIsNone(untyped.compiled())
        with self.assertRaises(TypeError):
            loom2.compile(lambda x: ~x, jit=True)(L([False] * 70))
        # A value of another schema is rejected.
        with self.assertRaises(TypeError):
            auto(VecValue(flat(70, "z"), [False] * 70))

    def test_environment_switch(self):
        previous = os.environ.get("LOOM2_JIT")
        os.environ["LOOM2_JIT"] = "0"
        try:
            self.assertIs(loom2.compile(lambda x: x).jit, False)
        finally:
            if previous is None:
                del os.environ["LOOM2_JIT"]
            else:
                os.environ["LOOM2_JIT"] = previous
        self.assertEqual(loom2.compile(lambda x: x).jit, "auto")


def _index(family):
    return json.loads((BENCHMARK / family / "benchmark.json").read_text())


def _models(family, metadata):
    files = metadata["models"] if "models" in metadata else [metadata["model"]]
    models = {}
    for file in files:
        models.update(json.loads((BENCHMARK / family / f"models/{file}.json").read_text()))
    return models


def _first_per_theory(family, size=None):
    """One benchmark per theory: the smallest by ``size`` (default: first)."""
    picks = {}
    for bench, metadata in _index(family).items():
        key = size(metadata) if size else 0
        if metadata["theory"] not in picks or key < picks[metadata["theory"]][0]:
            picks[metadata["theory"]] = (key, bench)
    return sorted((theory, bench) for theory, (_, bench) in picks.items())


@unittest.skipUnless(HAVE_JIT, "loom2 was built without the llvm feature")
class BenchmarkConstructorTests(unittest.TestCase):
    """Every constructor of one benchmark per theory, JIT vs. Python."""

    samples = 4

    def check_theory(self, family, bench, constructors):
        for name, function in constructors.items():
            # Fresh wrappers so each test compiles with jit=True.
            compiled = loom2.compile(function.function, name=function.__name__, jit=True)
            with self.subTest(family=family, benchmark=bench, constructor=name):
                check_function(self, compiled, samples=self.samples, seed=bench, context=f"{family}/{bench}: ")

    def test_reg_and_reg_plus(self):
        for family in ("Reg", "Reg+"):
            for theory, bench in _first_per_theory(family):
                metadata = _index(family)[bench]
                models = _models(family, metadata)
                module = importlib.import_module(f"benchmark.{family}.theories.{theory}")
                positives = {models[m] for m in metadata["positive"]}
                negatives = {models[m] for m in metadata["negative"]}
                self.check_theory(family, bench, module.theory(positives, negatives).constructors)

    def test_fo(self):
        fo = importlib.import_module("benchmark.FO.fo")
        for theory, bench in _first_per_theory("FO"):
            definition = json.loads((BENCHMARK / "FO" / f"theories/{theory}.json").read_text())
            models = _models("FO", _index("FO")[bench])
            self.check_theory("FO", bench, fo.theory(definition, models).constructors)

    def _temporal(self, family, size, max_size=None):
        module = importlib.import_module(f"benchmark.{family}.{family.lower()}")
        for theory, bench in _first_per_theory(family, size):
            metadata = _index(family)[bench]
            if max_size is not None and size(metadata) > max_size:
                with self.subTest(family=family, benchmark=bench, theory=theory):
                    self.skipTest(f"smallest {theory} benchmark has {size(metadata)} rows")
                continue
            definition = json.loads((BENCHMARK / family / f"theories/{theory}.json").read_text())
            models = _models(family, metadata)
            names = list(dict.fromkeys(metadata["positive"] + metadata["negative"]))
            instance = module.theory(definition, {name: models[name] for name in names})
            self.check_theory(family, bench, instance.constructors)

    def test_ltl(self):
        self._temporal("LTL", lambda metadata: len(metadata["positive"]) + len(metadata["negative"]))

    def test_ctl(self):
        def size(metadata):
            models = _models("CTL", metadata)
            names = set(metadata["positive"] + metadata["negative"])
            return sum(models[name]["states"] * models[name]["counters"] for name in names)

        # Six theories only have 564,992-row models; their staged AF/EG/AU
        # bodies are too large to JIT within a test's time and memory budget.
        self._temporal("CTL", size, max_size=10_000)


if __name__ == "__main__":
    unittest.main()

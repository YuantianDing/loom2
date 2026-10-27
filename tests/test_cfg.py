import tempfile
from collections import Counter
import unittest
from itertools import product
from pathlib import Path

import loom2
from test_benchmark import benchmark_theory, evaluate, parse_expression

CAN_SYNTHESIZE = hasattr(loom2._loom2, "build_synthesizer")


def compositions(total, parts):
    if parts == 1:
        return [(total,)] if total >= 1 else []
    return [(head, *rest) for head in range(1, total) for rest in compositions(total - head, parts - 1)]


def trees(cfg, symbol, size, memo):
    """Every tree of exactly `size` constructors that `symbol` derives,
    ignoring the grammar's automata, as `(name, children)` tuples."""
    key = (symbol, size)
    if key not in memo:
        found = set()
        for function, children in cfg.productions(symbol):
            if not children:
                if size == 1:
                    found.add((function.__name__, ()))
                continue
            for split in compositions(size - 1, len(children)):
                for parts in product(*(trees(cfg, child, part, memo) for child, part in zip(children, split))):
                    found.add((function.__name__, parts))
        memo[key] = found
    return memo[key]


def matches(tree, pattern):
    if pattern is Ellipsis:
        return True
    if isinstance(pattern, loom2.CompiledFunction):
        pattern = (pattern,)
    name, children = tree
    return name == pattern[0].__name__ and len(children) == len(pattern) - 1 and all(
        matches(child, sub) for child, sub in zip(children, pattern[1:])
    )


def contains(tree, pattern):
    return matches(tree, pattern) or any(contains(child, pattern) for child in tree[1])


def node_count(tree):
    return 1 + sum(node_count(child) for child in tree[1])


class CfgTests(unittest.TestCase):
    def setUp(self):
        self.theory, self.constraints = benchmark_theory("Reg", "0001")
        f = self.f = self.theory.constructors
        self.cfg = loom2.Cfg(self.theory.grammar)
        self.patterns = [
            (f["re.*"], (f["re.*"], ...)),
            (f["re.opt"], (f["re.*"], ...)),
            (f["re.++"], (f["re.++"], ..., ...), ...),
            (f["re.union"], (f["re.union"], ..., ...), ...),
            (f["re.union"], f["char.0"], ...),
        ]
        self.memo = {}

    def all_trees(self, size):
        return trees(self.cfg, self.cfg.start, size, self.memo)

    def test_pattern_subtraction_removes_exactly_the_trees_containing_a_pattern(self):
        subtracted = self.cfg.subtract(*self.patterns)
        for size in range(1, 7):
            for tree in self.all_trees(size):
                expected = not any(contains(tree, pattern) for pattern in self.patterns)
                self.assertEqual(subtracted.accepts(tree), expected, tree)

    def test_subtracting_a_grammar_removes_exactly_its_language(self):
        f = self.f
        # The regexes that are a star or a concatenation ending in a star.
        removed = loom2.Cfg(
            {
                "E": [(f["re.*"], "Start"), (f["re.++"], "Start", "E")],
                "Start": self.theory.grammar["Start"],
            }
        )
        language = removed.labels()
        subtracted = self.cfg.subtract(removed)
        for size in range(1, 7):
            for tree in self.all_trees(size):
                in_removed = tree[0] == "re.*" or (tree[0] == "re.++" and language.accepts(tree[1][1]) and tree[1][1][0] in ("re.*", "re.++"))
                self.assertEqual(language.accepts(tree), in_removed, tree)
                self.assertEqual(subtracted.accepts(tree), not in_removed, tree)

    def test_complement_and_intersection_compose(self):
        avoid = loom2.TreeAutomaton.avoiding(*self.patterns)
        twice = ~~avoid
        both = avoid & ~loom2.TreeAutomaton.avoiding(self.patterns[0])
        for size in range(1, 6):
            for tree in self.all_trees(size):
                self.assertEqual(twice.accepts(tree), avoid.accepts(tree))
                self.assertEqual((~avoid).accepts(tree), not avoid.accepts(tree))
                self.assertEqual(both.accepts(tree), avoid.accepts(tree) and contains(tree, self.patterns[0]))

    def derived(self, cfg, size):
        """The trees of `size` its solution nonterminals derive, and how often."""
        memo = {}
        found = Counter()
        for symbol in cfg.solutions:
            for tree in trees(cfg, symbol, size, memo):
                found[tree] += 1
        return found

    def test_subtraction_splits_by_label_and_derives_each_tree_once(self):
        f = self.f
        # Two nonterminals deriving overlapping trees: char.0 is both an R and an A.
        grammar = loom2.Cfg(
            {
                "R": [f["char.0"], f["char.1"], (f["re.++"], "R", "A"), (f["re.*"], "A")],
                "A": [f["char.0"], (f["re.opt"], "R")],
            }
        )
        removed = loom2.Cfg({"E": [(f["re.*"], "Start")], "Start": self.theory.grammar["Start"]})
        subtracted = grammar.subtract(*self.patterns[:2], removed)
        # Every nonterminal traces one original nonterminal and keeps its
        # productions' shape.
        self.assertEqual({subtracted.origin(symbol) for symbol in subtracted.nonterminals}, {"R", "A"})
        for symbol in subtracted.nonterminals:
            original = {(function.__name__, tuple(grammar.origin(child) for child in children)) for function, children in grammar.productions(subtracted.origin(symbol))}
            for function, children in subtracted.productions(symbol):
                self.assertIn((function.__name__, tuple(subtracted.origin(child) for child in children)), original)
        # The complement keeps copies apart; their hash groups are their own.
        self.assertEqual({subtracted.group(symbol) for symbol in subtracted.nonterminals}, set(subtracted.nonterminals))
        self.assertTrue(all(subtracted.origin(symbol) == "R" for symbol in subtracted.solutions))
        memo = {}
        for size in range(1, 7):
            kept = self.derived(subtracted, size)
            self.assertTrue(all(count == 1 for count in kept.values()), "one derivation per tree")
            expected = {
                tree
                for tree in trees(grammar, "R", size, memo)
                if not any(contains(tree, pattern) for pattern in self.patterns[:2]) and tree[0] != "re.*"
            }
            self.assertEqual(set(kept), expected, f"size {size}")

    def test_pattern_subtraction_shares_hash_groups_and_composes(self):
        f = self.f
        subtracted = self.cfg.subtract(*self.patterns)
        self.assertGreater(len(subtracted.nonterminals), 1)
        self.assertEqual({subtracted.group(symbol) for symbol in subtracted.nonterminals}, {"Start"})
        again = subtracted.subtract((f["re.opt"], (f["re.opt"], ...)))
        self.assertEqual({again.origin(symbol) for symbol in again.nonterminals}, {"Start"})
        self.assertEqual({again.group(symbol) for symbol in again.nonterminals}, {"Start"})
        for size in range(1, 6):
            for tree in self.all_trees(size):
                expected = subtracted.accepts(tree) and not contains(tree, (f["re.opt"], (f["re.opt"], ...)))
                self.assertEqual(again.accepts(tree), expected, tree)

    def test_patterns_are_validated(self):
        with self.assertRaises(ValueError):
            self.cfg.subtract(...)
        with self.assertRaises(ValueError):
            self.cfg.subtract((self.f["re.*"], ..., ...))
        everything = [(function, *([...] * function.arity)) for function in self.f.values()]
        with self.assertRaises(ValueError):
            self.cfg.subtract(*everything)

    def synthesize(self, grammar, theory=None, constraints=None, **options):
        theory = theory or self.theory
        constraints = constraints or self.constraints
        with tempfile.TemporaryDirectory() as directory:
            program = loom2.synth_fun(
                Path(directory) / "synth",
                language=theory.language,
                grammar=grammar,
                constraints=constraints,
                **{"max_size": 6, **options},
            )
            output = program.run()
        self.assertTrue(output.startswith("r = "), output)
        tree = parse_expression(output[len("r = "):])
        solution = evaluate(tree, theory)
        for label, expected in constraints.items():
            self.assertIs(solution[label], expected, label)
        return tree

    def test_commutative_constructors_survive_subtraction_and_are_validated(self):
        f = self.f
        cfg = loom2.Cfg(self.theory.grammar, commutative=[f["re.union"]])
        self.assertEqual(cfg.subtract(*self.patterns).commutative, {f["re.union"]})
        with self.assertRaises(ValueError):
            loom2.Cfg(self.theory.grammar, commutative=[f["re.*"]])

    def test_synth_fun_validates_commutative_constructors_before_compilation(self):
        for function in (self.f["re.*"], self.f["char.0"], object()):
            with self.subTest(function=function), self.assertRaisesRegex(ValueError, "must be binary"):
                loom2.synth_fun(
                    "unused",
                    language=self.theory.language,
                    grammar=self.theory.grammar,
                    constraints=self.constraints,
                    commutative=[function],
                )

    @unittest.skipUnless(CAN_SYNTHESIZE, "needs the llvm feature")
    def test_commutativity_keeps_solutions_as_small_packed_and_transposed(self):
        # Reg+ builds re.union transposed and Reg packed; both stay complete.
        for family, benchmark_id in (("Reg", "0001"), ("Reg+", "0030")):
            theory, constraints = benchmark_theory(family, benchmark_id)
            f = theory.constructors
            symmetric = [f[name] for name in ("re.union", "re.inter") if name in f]
            plain = self.synthesize(theory.grammar, theory, constraints, max_size=5)
            commuted = self.synthesize(theory.grammar, theory, constraints, max_size=5, commutative=symmetric)
            cfg = loom2.Cfg(theory.grammar, commutative=symmetric).subtract(*self.laws(f))
            both = self.synthesize(cfg, theory, constraints, max_size=5)
            self.assertEqual(node_count(commuted), node_count(plain), family)
            self.assertEqual(node_count(both), node_count(plain), family)

    @unittest.skipUnless(CAN_SYNTHESIZE, "needs the llvm feature")
    def test_commutative_filters_cover_full_and_partial_blocks(self):
        labels = tuple(f"b{i}" for i in range(66))
        language = loom2.Vec(**dict.fromkeys(labels, bool))
        indices = loom2.Vec.pytree(**{label: [label] for label in labels})

        def leaf(i):
            @loom2.compile(name=f"c{i}", jit=False)
            def constant() -> language:
                return language({label: j == i for j, label in enumerate(labels)})
            return constant

        leaves = [leaf(i) for i in range(len(labels))]
        for transposed, target, distinct in ((False, (0, 65), False), (True, (0, 65), False),
                                         (True, (64, 65), False), (True, (0, 65), True)):
            with self.subTest(transposed=transposed, target=target, distinct=distinct):
                @loom2.compile(jit=False)
                def union(a: language, b: language):
                    return (a[indices] | b[indices]).exists() if transposed else a | b

                if distinct:
                    # Equal counts do not mean identical sets. Both orientations
                    # are legal, but only the earlier pool goes on the left.
                    grammar = {"Start": [(union, "B", "A"), (union, "A", "B")],
                               "A": leaves[:33], "B": leaves[33:]}
                    groups = {"A": "Leaves", "B": "Leaves"}
                else:
                    grammar = {"Start": [(union, "A", "A")], "A": leaves}
                    groups = {}
                with tempfile.TemporaryDirectory() as directory:
                    program = loom2.synth_fun(
                        Path(directory) / "synth", language=language, grammar=grammar,
                        groups=groups, commutative=[union], max_size=3, openmp=False,
                        constraints={label: i in target for i, label in enumerate(labels)},
                    )
                    output = program.run()
                    self.assertTrue(output.startswith("r = "), output)
                    tree = parse_expression(output[len("r = "):])
                    self.assertEqual(tree[0], "union", output)
                    self.assertEqual({child[0] for child in tree[1]}, {f"c{i}" for i in target}, output)
                    frontend = (Path(directory) / "03-frontend.mlir").read_text()
                    if not distinct:
                        self.assertIn("call @commutative_hash_filter_Start_union_rule0", frontend)
                        signature = next(line for line in frontend.splitlines()
                                         if "func.func private @commutative_hash_filter_Start_union_rule0" in line)
                        self.assertIn(f"loom.transposed = {str(transposed).lower()}", signature)
                    else:
                        self.assertNotIn("call @commutative_hash_filter_", frontend)

    @staticmethod
    def laws(f):
        """Associativity of the binary constructors, as left-nesting patterns."""
        return [(f[name], (f[name], ..., ...), ...) for name in ("re.union", "re.inter", "re.++") if name in f]

    @unittest.skipUnless(CAN_SYNTHESIZE, "needs the llvm feature")
    def test_synthesis_with_subtracted_patterns_finds_a_solution_as_small(self):
        plain = self.synthesize(self.theory.grammar)
        subtracted = self.synthesize(self.cfg.subtract(*self.patterns[:4]))
        self.assertEqual(node_count(subtracted), node_count(plain))

    @unittest.skipUnless(CAN_SYNTHESIZE, "needs the llvm feature")
    def test_synthesis_avoids_a_subtracted_language(self):
        f = self.f
        plain = self.synthesize(self.theory.grammar)
        # Remove the solution found without subtraction (and everything with
        # its root constructor at the root): another one is found instead.
        root = f[plain[0]]
        removed = loom2.Cfg({"E": [(root, *(["Start"] * root.arity))], "Start": self.theory.grammar["Start"]})
        other = self.synthesize(self.cfg.subtract(removed))
        self.assertNotEqual(other[0], plain[0])


if __name__ == "__main__":
    unittest.main()

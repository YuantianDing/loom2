"""Focused tests for the Python-to-Rust tracing boundary."""

import unittest

from loom2 import Vec
from loom2.tracing import JaggedTensorParts, trace


class RecordingBuilder:
    """Test double implementing the native ``TensorFnBuilder`` protocol."""

    def __init__(self, name):
        self.name = name
        self.events = []

    def input_bit_vector(self, name, mask):
        self.events.append(("input", name, mask))
        return ("input", name)

    def constant_bit_vector(self, value, mask):
        self.events.append(("constant", value, mask))
        return ("constant", value.data)

    def bv_logical(self, operation, operands):
        self.events.append(("logical", operation, tuple(operands)))
        return (operation, len(operands))

    def bt_expand_reduce(self, source, dims, operation):
        self.events.append(("reduce", source, tuple(dims), operation))
        return ("reduce", source)

    def bv_jagged_permute(self, source, indices):
        self.events.append(("permute", source, indices))
        return ("permute", source)

    def finish(self, result):
        return self.name, result, tuple(self.events)


class TracingScaffoldTests(unittest.TestCase):
    def test_vec_bool_and_vec_int_cross_as_whole_tensors(self):
        bits = Vec[3](bool)
        indices = Vec[2](int)(2, 0)

        def selected(value: bits):
            return (value[indices] | True).exists()

        name, _, events = trace(selected, builder_factory=RecordingBuilder)
        self.assertEqual(name, "selected")
        self.assertEqual(events[0][0:2], ("input", "value"))
        self.assertEqual(events[0][2], JaggedTensorParts.flat((0b111,)))

        permutes = [event[2] for event in events if event[0] == "permute"]
        # The rank-1 selection is flat; exists is a dense fold of its 2 bits.
        self.assertEqual(permutes, [JaggedTensorParts.rows(((2,), (0,)))])
        reduction = next(event for event in events if event[0] == "reduce")
        self.assertEqual(reduction[2:], ((("reduce", 2),), "or"))

    def test_pytree_selection_is_one_jagged_permute(self):
        strings = Vec(**{"": bool, "a": bool, "aa": bool})
        prefixes = Vec.pytree(**{"": [], "a": ["", "a"], "aa": ["", "a", "aa"]})

        def staged(value: strings):
            return value[prefixes].exists()

        _, _, events = trace(staged, builder_factory=RecordingBuilder)
        permutes = [event[2] for event in events if event[0] == "permute"]
        self.assertEqual(permutes, [JaggedTensorParts.rows(((), (0, 1), (0, 1, 2)))])
        reduction = next(event for event in events if event[0] == "reduce")
        # Jagged rows of 0, 2, and 3 lanes: keep 3 rows, fold up to 3 lanes.
        self.assertEqual(reduction[2:], ((("batch", 3), ("reduce", 3)), "or"))

    def test_assignment_is_a_masked_update(self):
        strings = Vec(**{"": bool, "a": bool, "aa": bool})

        def star_seed(value: strings):
            value[""] = True
            return value

        _, result, events = trace(star_seed, builder_factory=RecordingBuilder)
        constants = [event[1].data for event in events if event[0] == "constant"]
        self.assertEqual(constants, [(0b001,), (0b111,)])  # mask, then ones
        operations = [event[1] for event in events if event[0] == "logical"]
        self.assertEqual(operations, ["xor", "and", "xor"])  # a ^ ((a ^ v) & mask)
        self.assertEqual(result, ("xor", 2))

    def test_runtime_parameters_require_boolean_leaves(self):
        indices = Vec[2](int)

        def invalid(value: indices):
            return value

        with self.assertRaisesRegex(TypeError, "bool leaves"):
            trace(invalid, builder_factory=RecordingBuilder)


if __name__ == "__main__":
    unittest.main()

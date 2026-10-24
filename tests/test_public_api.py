import unittest

import loom2


class PublicApiTests(unittest.TestCase):
    def test_vec_schemas_support_shapes_labels_and_runtime_logic(self):
        rectangular = loom2.Vec[2, 3](bool)
        positional = loom2.Vec(loom2.Vec[3](bool), loom2.Vec[3](bool))
        labeled = loom2.Vec[2](a=bool, b=bool, c=bool)

        self.assertEqual(rectangular.shape, (2, 3))
        self.assertEqual(positional.shape, rectangular.shape)
        self.assertEqual(labeled.shape, rectangular.shape)
        labeled_value = labeled((True, False, True), (False, True, False))
        self.assertEqual(tuple(labeled_value[0]), (True, False, True))

        strings = loom2.Vec(empty=bool, zero=bool, one=bool)
        left = strings(empty=True, zero=False, one=True)
        right = strings(empty=True, zero=True, one=False)
        self.assertEqual(tuple(left & right), (True, False, False))
        self.assertIs((left | right)["zero"], True)

    def test_vec_pytree_reductions_and_fixed_point_evaluation(self):
        strings = loom2.Vec(empty=bool, zero=bool, one=bool)
        indices = loom2.Vec.pytree(
            empty=["empty"],
            zero=["empty", "zero"],
            one=["one", "empty"],
        )
        selected = strings(empty=True, zero=False, one=True)[indices]

        self.assertEqual(tuple(selected.exists()), (True, True, True))
        self.assertEqual(tuple(selected.forall()), (True, False, True))

        seed = strings(empty=True, zero=False, one=False)
        closure = strings.rec(
            lambda result: strings(
                empty=result["empty"],
                zero=result["zero"] | result["empty"],
                one=result["one"] | result["zero"],
            )
            | seed
        )
        self.assertEqual(tuple(closure), (True, True, True))

    def test_assignment_and_empty_rows_follow_the_summary_star_example(self):
        strings = loom2.Vec(**{"": bool, "a": bool, "aa": bool})
        value = strings({"": False, "a": True, "aa": False})
        value[""] = True
        self.assertEqual(tuple(value), (True, True, False))

        # One length-staged step of re.*: "aa" splits into "a" + "a".
        prefixes = loom2.Vec.pytree(**{"": [], "a": [], "aa": ["", "a", "aa"]})
        suffixes = loom2.Vec.pytree(**{"": [], "a": [], "aa": ["aa", "a", ""]})
        value |= (value[prefixes] & value[suffixes]).exists()
        self.assertEqual(tuple(value), (True, True, True))

    def test_native_version_is_optional_for_the_pure_python_surface(self):
        version = loom2.native_version()
        self.assertTrue(version is None or isinstance(version, str))


if __name__ == "__main__":
    unittest.main()

"""Supplied HW 3 checks. Copy alongside hw3.py and leave this file unchanged."""

import sys
import unittest
from unittest.mock import Mock, patch

import hw3


class MemoizationTests(unittest.TestCase):
    def setUp(self):
        # A controlled clock makes the boundary checks fast and deterministic.
        clock_patch = patch.object(hw3.time, "monotonic", return_value=100.0)
        self.clock = clock_patch.start()
        self.addCleanup(clock_patch.stop)

    def test_repeated_argument_uses_cached_result(self):
        func = Mock(side_effect=lambda x: x * 2)
        cached = hw3.memorize_3s(func)
        self.assertEqual(cached(4), 8)
        self.assertEqual(cached(4), 8)
        func.assert_called_once_with(4)

    def test_expiry_boundary_and_refresh(self):
        func = Mock(side_effect=["first", "second", "third"])
        cached = hw3.memorize_3s(func)
        self.assertEqual(cached("x"), "first")
        self.clock.return_value = 102.999
        self.assertEqual(cached("x"), "first")
        self.clock.return_value = 103.0
        self.assertEqual(cached("x"), "second")
        self.clock.return_value = 105.999
        self.assertEqual(cached("x"), "second")
        self.clock.return_value = 106.5
        self.assertEqual(cached("x"), "third")
        self.assertEqual(func.call_count, 3)

    def test_arguments_have_independent_results_and_expiry(self):
        func = Mock(side_effect=[10, 20, 11, 21])
        cached = hw3.memorize_3s(func)
        self.assertEqual(cached(1), 10)
        self.clock.return_value = 101.0
        self.assertEqual(cached(2), 20)
        self.clock.return_value = 103.0
        self.assertEqual(cached(1), 11)
        self.assertEqual(cached(2), 20)
        self.clock.return_value = 104.0
        self.assertEqual(cached(2), 21)
        self.assertEqual(cached(1), 11)
        self.assertEqual(func.call_count, 4)

    def test_cache_hits_do_not_extend_expiry(self):
        func = Mock(side_effect=["old", "new"])
        cached = hw3.memorize_3s(func)
        self.assertEqual(cached("x"), "old")
        for instant in (101.0, 102.0, 102.5):
            self.clock.return_value = instant
            self.assertEqual(cached("x"), "old")
        self.clock.return_value = 103.0
        self.assertEqual(cached("x"), "new")
        self.assertEqual(func.call_count, 2)

    def test_falsey_results_are_cached(self):
        for value in (0, False, None, "", []):
            with self.subTest(value=value):
                func = Mock(return_value=value)
                cached = hw3.memorize_3s(func)
                self.assertEqual(cached(("hashable", 1)), value)
                self.assertEqual(cached(("hashable", 1)), value)
                self.assertEqual(func.call_count, 1)

    def test_decorated_functions_have_separate_caches(self):
        first = Mock(return_value="first")
        second = Mock(return_value="second")
        cached_first = hw3.memorize_3s(first)
        cached_second = hw3.memorize_3s(second)
        for _ in range(2):
            self.assertEqual(cached_first(1), "first")
            self.assertEqual(cached_second(1), "second")
        first.assert_called_once_with(1)
        second.assert_called_once_with(1)

    def test_expiry_starts_after_computation_finishes(self):
        def slow_function(x):
            self.clock.return_value += 2.0
            return x * 2

        func = Mock(side_effect=slow_function)
        cached = hw3.memorize_3s(func)
        self.assertEqual(cached(3), 6)  # Starts at 100, finishes at 102.
        self.clock.return_value = 104.999
        self.assertEqual(cached(3), 6)
        self.assertEqual(func.call_count, 1)
        self.clock.return_value = 105.0
        self.assertEqual(cached(3), 6)
        self.assertEqual(func.call_count, 2)

    def test_exceptions_propagate_and_are_not_cached(self):
        func = Mock(side_effect=[ValueError("try again"), 42])
        cached = hw3.memorize_3s(func)
        with self.assertRaisesRegex(ValueError, "try again"):
            cached("x")
        self.assertEqual(cached("x"), 42)
        self.assertEqual(cached("x"), 42)
        self.assertEqual(func.call_count, 2)


class TraversalTests(unittest.TestCase):
    def check_orders(self, root, pre, inorder, post):
        # Capture the actual input links/values, not just the traversal result.
        original_nodes = []
        pending = [root] if root is not None else []
        while pending:
            node = pending.pop()
            original_nodes.append((node, node.val, node.left, node.right))
            if node.left is not None:
                pending.append(node.left)
            if node.right is not None:
                pending.append(node.right)

        for order, expected in (("pre", pre), ("in", inorder), ("post", post)):
            for approach in ("recursion", "iteration"):
                name = f"{order}_order_{approach}"
                with self.subTest(function=name):
                    traverse = getattr(hw3, name)
                    first = traverse(root)
                    self.assertIsInstance(first, list)
                    self.assertEqual(first, expected)
                    for node, value, left, right in original_nodes:
                        self.assertEqual(node.val, value)
                        self.assertIs(node.left, left)
                        self.assertIs(node.right, right)
                    first.append("not part of the tree")
                    self.assertEqual(traverse(root), expected)
                    self.assertEqual(traverse(None), [])

    def test_example_tree(self):
        node = hw3.Node
        root = node("F", node("B", node("A"), node("D", node("C"), node("E"))),
                    node("G", right=node("I", left=node("H"))))
        self.check_orders(root, list("FBADCEGIH"), list("ABCDEFGHI"), list("ACEDBHIGF"))

    def test_empty_tree(self):
        self.check_orders(None, [], [], [])

    def test_single_node(self):
        self.check_orders(hw3.Node("only"), ["only"], ["only"], ["only"])

    def test_left_chain(self):
        root = hw3.Node("A", left=hw3.Node("B", left=hw3.Node("C")))
        self.check_orders(root, list("ABC"), list("CBA"), list("CBA"))

    def test_right_chain(self):
        root = hw3.Node("A", right=hw3.Node("B", right=hw3.Node("C")))
        self.check_orders(root, list("ABC"), list("ABC"), list("CBA"))

    def test_tree_is_not_necessarily_sorted(self):
        root = hw3.Node("M", hw3.Node("Z"), hw3.Node("A"))
        self.check_orders(root, list("MZA"), list("ZMA"), list("ZAM"))

    def test_duplicate_values_are_separate_nodes(self):
        root = hw3.Node("X", hw3.Node("X", right=hw3.Node("Y")), hw3.Node("X"))
        self.check_orders(root, list("XXYX"), list("XYXX"), list("YXXX"))

    def test_iterative_versions_handle_deep_trees(self):
        original_limit = sys.getrecursionlimit()
        values = [str(i) for i in range(original_limit + 100)]
        for side in ("left", "right"):
            root = None
            for value in reversed(values):
                root = hw3.Node(value, **{side: root})
            expected_inorder = values[::-1] if side == "left" else values
            for order, expected in (("pre", values), ("in", expected_inorder),
                                    ("post", values[::-1])):
                with self.subTest(side=side, order=order):
                    self.assertEqual(getattr(hw3, f"{order}_order_iteration")(root), expected)
                    self.assertEqual(sys.getrecursionlimit(), original_limit)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU structural proofs complement the GPU numerical and dispatch checks."""

import ast
import random
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "vllm/model_executor/layers/quantization"


class MappingProof(unittest.TestCase):
    def test_permutation_and_group_slices(self):
        rng = random.Random(73)
        for size in (8, 32, 101, 256):
            for fraction in (0, 0.01, 0.49, 0.5, 0.75, 1):
                keys = [
                    rng.randrange(7) if rng.random() < fraction else -1
                    for _ in range(size)
                ]
                cold = [i for i, key in enumerate(keys) if key >= 0]
                sorted_cold = sorted(cold, key=lambda i: keys[i])
                permutation = sorted_cold + [i for i, k in enumerate(keys) if k < 0]
                inverse = [-1] * size
                for position, old in enumerate(sorted_cold):
                    inverse[old] = position
                for old in range(size):
                    if keys[old] < 0:
                        inverse[old] = len(cold) + old - sum(c < old for c in cold)
                self.assertEqual(sorted(inverse), list(range(size)))
                self.assertEqual(
                    [permutation[inverse[i]] for i in range(size)], list(range(size))
                )
                start = 0
                for expert in range(7):
                    members = [i for i in cold if keys[i] == expert]
                    self.assertEqual(permutation[start : start + len(members)], members)
                    start += len(members)
                # Native, tiny cold, grouped cold and invalid routes all share map.
                self.assertEqual(
                    [permutation[inverse[i]] for i in reversed(range(size))],
                    list(reversed(range(size))),
                )

    def test_sum_only_addresses_changed(self):
        base = (ROOT / "nvfp4_arvq_route_sum.py").read_text()
        candidate = (ROOT / "nvfp4_arvq_direct_output.py").read_text()

        def arithmetic(source):
            tree = ast.parse(source)
            fn = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "kernel"
            )
            return [
                ast.dump(n)
                for n in ast.walk(fn)
                if isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name)
                    and t.id in ("a0", "a1", "a2", "a3", "result")
                    for t in n.targets
                )
            ]

        self.assertEqual(arithmetic(base), arithmetic(candidate))
        self.assertEqual(candidate.count("mul_rn("), base.count("mul_rn("))

    def test_inverse_reaches_every_routed_load(self):
        tree = ast.parse((ROOT / "nvfp4_arvq_direct_output.py").read_text())
        kernel = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "kernel"
        )
        self.assertEqual(
            [a.arg for a in kernel.args.args][:4], ["R", "W", "INV", "OUT"]
        )
        routed_loads = [
            n
            for n in ast.walk(kernel)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "load"
            and any(
                isinstance(x, ast.Name) and x.id == "R" for x in ast.walk(n.args[0])
            )
        ]
        self.assertEqual(len(routed_loads), 4)
        for load in routed_loads:
            self.assertTrue(
                any(
                    isinstance(x, ast.Name) and x.id == "INV"
                    for x in ast.walk(load.args[0])
                )
            )
            self.assertFalse(
                any(
                    isinstance(x, ast.Name) and x.id == "base"
                    for x in ast.walk(load.args[0])
                )
            )
        wrapper = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "sum_routes"
        )
        launch = next(
            n
            for n in ast.walk(wrapper)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)
        )
        self.assertEqual(
            [a.id for a in launch.args[:4] if isinstance(a, ast.Name)],
            ["routed", "weights", "inverse", "out"],
        )

    def test_default_off_and_current_paths_preserved(self):
        text = (ROOT / "nvfp4_arvq_prefill.py").read_text()
        tree = ast.parse(text)
        guard = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and "VLLM_ARVQ_DIRECT_COLD_OUTPUT" in ast.unparse(n.test)
        )
        guard_text = ast.unparse(guard.test)
        self.assertIn(
            "os.environ.get('VLLM_ARVQ_DIRECT_COLD_OUTPUT', '0') == '1'", guard_text
        )
        self.assertIn("tokens == 4096", guard_text)
        self.assertIn("n13 == 1024", guard_text)
        self.assertNotIn("2048", guard_text)
        self.assertIn("_compact_prefill_enabled()", guard_text)
        self.assertIn("cold_scatter.prepare(x, sorted_slots, routed)", text)
        self.assertIn("scatter_cold(out, route_slots)", text)

    def test_2048_uses_existing_scatter(self):
        tree = ast.parse((ROOT / "nvfp4_arvq_prefill.py").read_text())
        guard = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and "VLLM_ARVQ_DIRECT_COLD_OUTPUT" in ast.unparse(n.test)
        )
        comparisons = [
            n
            for n in ast.walk(guard.test)
            if isinstance(n, ast.Compare)
            and isinstance(n.left, ast.Name)
            and n.left.id in ("tokens", "n13")
        ]
        for tokens, n13, expected in [
            (2048, 1024, False),
            (4096, 1024, True),
            (4096, 2048, False),
            (2177, 1024, False),
        ]:
            checks = [
                eval(
                    compile(ast.Expression(n), "<guard>", "eval"),
                    {},
                    {"tokens": tokens, "n13": n13},
                )
                for n in comparisons
            ]
            self.assertEqual(all(checks), expected)

    def test_routing_counts_do_not_specialize(self):
        tree = ast.parse((ROOT / "nvfp4_arvq_direct_output.py").read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        for name, dynamic, static in (
            ("inverse_kernel", "C", ("S", "BLOCK")),
            ("scatter_kernel", "N", ("H", "BLOCK")),
        ):
            fn = functions[name]
            args = {arg.arg: arg.annotation for arg in fn.args.args}
            self.assertIsNone(args[dynamic])
            for key in static:
                annotation = args[key]
                assert annotation is not None
                self.assertEqual(ast.unparse(annotation), "tl.constexpr")
            decorator = fn.decorator_list[0]
            assert isinstance(decorator, ast.Call)
            suppressed = next(
                ast.literal_eval(kw.value)
                for kw in decorator.keywords
                if kw.arg == "do_not_specialize"
            )
            self.assertIn(dynamic, suppressed)

    def test_no_second_output_or_mapping_allocation(self):
        src = (ROOT / "nvfp4_arvq_prefill.py").read_text()
        self.assertIn("inverse = local_ids.view(torch.int32)[:slots]", src)
        self.assertIn("2 * cold_slots.numel() >= slots", src)
        self.assertIn("torch.ops.aten.mm.dtype_out(", src)
        self.assertIn("destination = routed[start:stop]", src)


if __name__ == "__main__":
    unittest.main()

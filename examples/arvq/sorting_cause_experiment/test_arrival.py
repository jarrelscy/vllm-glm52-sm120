# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest

from compare_arrival import split_choices


class ChoiceSplitTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            (1, {"choices": [{"index": 1, "token_ids": [1, 2]}]}),
            (2, {"choices": [{"index": 0, "token_ids": [4]}]}),
            (
                3,
                {
                    "choices": [
                        {"index": 0, "token_ids": [5, 6, 7]},
                        {"index": 1, "token_ids": [3, 4]},
                    ]
                },
            ),
            (3.1, {"usage": {"completion_tokens": 8}}),
        ]

    def test_interleaved_choices_preserve_first_token_batch(self):
        rows, usage = split_choices(self.events, 3.2, 2, 4)
        self.assertEqual(usage["completion_tokens"], 8)
        self.assertEqual([r["first_emission_token_count"] for r in rows], [1, 2])
        self.assertTrue(all(r["token_ids_match_usage"] for r in rows))

    def test_missing_usage_rejected(self):
        with self.assertRaises(ValueError):
            split_choices(self.events[:-1], 3.2, 2, 4)

    def test_per_choice_count_mismatch_rejected(self):
        self.events[1][1]["choices"][0]["token_ids"] = []
        with self.assertRaises(ValueError):
            split_choices(self.events, 3.2, 2, 4)


if __name__ == "__main__":
    unittest.main()

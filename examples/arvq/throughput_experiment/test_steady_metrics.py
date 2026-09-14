# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest

from steady_metrics import steady_decode_summary


def req(events, offset=0, usage=None):
    return {
        "token_event_timeline": events,
        "batch_start_offset_s": offset,
        "token_ids_match_usage": True,
        "completion_tokens": sum(n for _, n in events) if usage is None else usage,
    }


class SteadyMetricsTests(unittest.TestCase):
    def test_excludes_entire_first_mtp_batch(self):
        r = steady_decode_summary([req([(1, 4), (2, 3), (3, 2)])])
        self.assertEqual(r["tokens_per_stream"], [5])
        self.assertEqual(r["aggregate_decode_tps"], 2.5)

    def test_staggered_starts_and_only_common_interval(self):
        r = steady_decode_summary(
            [
                req([(1, 4), (2, 8), (3, 2), (4, 3)]),
                req([(1, 4), (2, 1), (3, 7), (5, 9)], offset=1),
            ]
        )
        self.assertEqual((r["start_offset_s"], r["end_offset_s"]), (2, 4))
        self.assertEqual(r["tokens_per_stream"], [5, 8])
        self.assertEqual(r["aggregate_decode_tps"], 6.5)

    def test_no_overlap_is_not_a_rate(self):
        r = steady_decode_summary([req([(1, 1), (2, 1)]), req([(3, 1), (4, 1)])])
        self.assertFalse(r["valid"])

    def test_counts_ids_not_emission_chunks(self):
        r = steady_decode_summary([req([(0, 1), (1, 4), (2, 4)])])
        self.assertEqual(r["aggregate_decode_tps"], 4)

    def test_usage_mismatch_rejected(self):
        self.assertFalse(
            steady_decode_summary([req([(0, 1), (1, 4)], usage=7)])["valid"]
        )


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import unittest

from bench_serving import (
    delta_metrics,
    normalized,
    parsed_metrics,
    speculative_summary,
    stream_result,
)


class Counting(unittest.TestCase):
    def test_mtp_batches_are_not_tokens(self):
        events = [
            (
                1.0,
                {
                    "choices": [
                        {"index": 0, "text": "first batch", "token_ids": [1, 2, 3]}
                    ]
                },
            ),
            (
                1.5,
                {
                    "choices": [
                        {"index": 0, "text": "second batch", "token_ids": [4, 5, 6, 7]}
                    ]
                },
            ),
            (
                1.6,
                {"choices": [], "usage": {"completion_tokens": 7, "prompt_tokens": 10}},
            ),
        ]
        r = stream_result(events, 1.7)
        self.assertEqual(r["completion_tokens"], 7)
        self.assertEqual(r["text_event_count_diagnostic_only"], 2)
        self.assertEqual(r["post_first_emission_tokens"], 4)
        self.assertEqual(r["decode_tps"], 8)
        self.assertEqual(r["decode_wall_s"], 0.5)

    def test_missing_usage_refuses_chunk_fallback(self):
        with self.assertRaises(ValueError):
            stream_result([(1, {"choices": [{"text": "words"}]})], 2)

    def test_usage_without_ids_retains_count_not_fake_precision(self):
        r = stream_result(
            [
                (1, {"choices": [{"text": "foo"}]}),
                (2, {"choices": [{"text": "bar"}], "usage": {"completion_tokens": 8}}),
            ],
            3,
        )
        self.assertEqual(r["completion_tokens"], 8)
        self.assertIsNone(r["decode_tps"])
        self.assertEqual(r["completion_tokens_over_decode_wall_tps"], 8)

    def test_single_batch_no_decode_rate(self):
        r = stream_result(
            [
                (
                    1,
                    {
                        "choices": [{"text": "abc", "token_ids": [1, 2, 3]}],
                        "usage": {"completion_tokens": 3},
                    },
                )
            ],
            2,
        )
        self.assertIsNone(r["decode_tps"])

    def test_token_id_mismatch_does_not_claim_exact_rate(self):
        r = stream_result(
            [
                (1, {"choices": [{"text": "a", "token_ids": [1]}]}),
                (
                    2,
                    {
                        "choices": [{"text": "b", "token_ids": [2]}],
                        "usage": {"completion_tokens": 3},
                    },
                ),
            ],
            3,
        )
        self.assertFalse(r["token_ids_match_usage"])
        self.assertIsNone(r["decode_tps"])


class Metrics(unittest.TestCase):
    def test_labels_and_created(self):
        s = parsed_metrics(
            'vllm:spec_decode_num_drafts_total{model_name="glm '
            '5.3",engine="0"} 8\n'
            'vllm:spec_decode_num_drafts_created{engine="0"} 123\n'
            "# comment\n"
        )
        self.assertEqual(len(s), 1)
        self.assertEqual(next(iter(s.values())), 8)

    def test_reset_rejected(self):
        with self.assertRaises(ValueError):
            delta_metrics({"a": 5}, {"a": 4})

    def test_exact_spec_counts_and_contamination(self):
        d = {
            "vllm:spec_decode_num_drafts_total": 4,
            "vllm:spec_decode_num_draft_tokens_total": 16,
            "vllm:spec_decode_num_accepted_tokens_total": 8,
            "vllm:request_decode_time_seconds_sum": 0.4,
            "vllm:request_decode_time_seconds_count": 1,
            "vllm:generation_tokens_total": 13,
        }
        r = speculative_summary(d, {"completion_tokens": 13, "decode_wall_s": 0.5})
        self.assertEqual(r["expected_emitted_per_draft_step"], 3)
        self.assertEqual(r["actual_completion_tokens_per_draft_step"], 3.25)
        self.assertEqual(r["server_decode_ms_per_draft_step_proxy"], 100)
        d["vllm:request_decode_time_seconds_count"] = 2
        r = speculative_summary(d, {"completion_tokens": 13, "decode_wall_s": 0.5})
        self.assertIsNone(r["server_decode_ms_per_draft_step_proxy"])

    def test_normalized_counter_estimate(self):
        n = {
            "target_prompt_tokens": 16,
            "speculative": {"server_decode_ms_per_draft_step_proxy": 20},
        }
        ref = {
            "label": "old_mtp",
            "runs": [
                {
                    "target_prompt_tokens": 16,
                    "speculative": {
                        "isolated_counters_match_request": True,
                        "expected_emitted_per_draft_step": 3,
                    },
                }
            ],
        }
        self.assertEqual(normalized(n, ref)["normalized_tps_estimate"], 150)


if __name__ == "__main__":
    unittest.main()

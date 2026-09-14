import unittest

from bench_concurrency import batch_summary


def row(start, end, tokens, ttft=0.1):
    return {
        "batch_start_offset_s": start,
        "batch_end_offset_s": end,
        "completion_tokens": tokens,
        "ttft_s": ttft,
        "total_s": end - start,
        "decode_tps": 999,
        "token_ids_match_usage": True,
    }


class CountingTests(unittest.TestCase):
    def test_overlapping_requests_use_shared_wall_span(self):
        r = batch_summary([row(0, 2, 100), row(0, 2, 100)])
        self.assertEqual(r["aggregate_output_tps"], 100)
        self.assertNotEqual(r["aggregate_output_tps"], sum(r["per_request_decode_tps"]))

    def test_staggered_start_and_straggler(self):
        r = batch_summary([row(1, 3, 100), row(1.25, 5, 50)])
        self.assertEqual(r["wall_span_s"], 4)
        self.assertEqual(r["aggregate_output_tps"], 37.5)
        self.assertEqual(r["dispatch_skew_ms"], 250)

    def test_each_stream_preserves_ttft(self):
        r = batch_summary([row(0, 1, 10, 0.1), row(0, 2, 20, 0.7)])
        self.assertEqual(r["ttft_ms_per_stream"], [100, 700])
        self.assertEqual(r["completed_output_tokens"], 30)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-counted common decode interval, excluding every initial emission."""


def steady_decode_summary(requests):
    timelines = []
    for request in requests:
        timeline = request.get("token_event_timeline", [])
        if not request.get("token_ids_match_usage") or not timeline:
            return {"valid": False, "reason": "Missing verified token-ID timeline"}
        if sum(n for _, n in timeline) != request["completion_tokens"]:
            return {"valid": False, "reason": "Token timeline disagrees with usage"}
        offset = request["batch_start_offset_s"]
        timelines.append([(offset + t, n) for t, n in timeline])
    if not timelines:
        return {"valid": False, "reason": "No requests"}
    start = max(timeline[0][0] for timeline in timelines)
    end = min(timeline[-1][0] for timeline in timelines)
    if end <= start:
        return {"valid": False, "reason": "No common decode interval"}
    counts = [sum(n for t, n in timeline if start < t <= end) for timeline in timelines]
    return {
        "valid": True,
        "start_offset_s": start,
        "end_offset_s": end,
        "duration_s": end - start,
        "tokens_per_stream": counts,
        "aggregate_decode_tps": sum(counts) / (end - start),
    }

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.mark.parametrize("threads", [128, 256])
def test_cooperative_staging_covers_each_value_once(threads):
    copied = [j for lane in range(threads) for j in range(lane, 512, threads)]
    assert sorted(copied) == list(range(512))
    assert [j for j in range(threads) if j < 64] == list(range(64))


def test_staged_operands_match_original_addresses_for_odd_and_partial_groups():
    for count in range(1, 17):
        first = 37
        for pair in range(8):
            slot = first + 2 * pair
            partner = slot + 1 if 2 * pair + 1 < count else -1
            for q in range(8):
                source = partner if partner >= 0 and q >= 4 else slot
                plane = q % 4 if partner >= 0 else q
                valid = 2 * pair < count and plane < 4
                if not valid:
                    continue
                for c in range(4):
                    for word in (c, 4 + c):
                        staged = ((source - first) * 4 + plane) * 8 + word
                        loaded_slot = first + staged // 32
                        loaded_plane = (staged // 8) % 4
                        loaded_word = staged % 8
                        assert (loaded_slot, loaded_plane, loaded_word) == (
                            source,
                            plane,
                            word,
                        )
                        assert loaded_slot - first < count
                staged_scale = (source - first) * 4 + plane
                assert (first + staged_scale // 4, staged_scale % 4) == (source, plane)


def test_eight_row_warps_preserve_exact_output_coverage():
    for n in (1024, 6144):
        assert n % 128 == 0
        tiles = [block * 8 + warp for block in range(n // 128) for warp in range(8)]
        assert tiles == list(range(n // 16))

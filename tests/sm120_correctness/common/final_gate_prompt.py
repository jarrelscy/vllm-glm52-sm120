# SPDX-License-Identifier: Apache-2.0
"""Deterministic builder of the ~64K-token FINAL GATE prompt.

Uses a private LCG (no dependence on `random`'s cross-version behavior)
so the prompt is byte-identical everywhere, forever.  Structure:

  * a long multi-part technical document (pseudo-prose over a fixed
    vocabulary, section headers, stable numbering),
  * three embedded facts at ~10% / ~50% / ~90% depth,
  * a final multi-step question that REQUIRES synthesizing all three
    facts plus a detail inside one of them (the two-digit prefix of the
    serial's numeric part), with an objectively checkable answer.

Expected derivation:
  serial  = KX-4471-DELTA          (fact @10%)
  cal     = 17                     (fact @50%)
  red     = 3                      (fact @90%)
  product = 17 * 3 = 51
  prefix  = 44   (first two digits of 4471)
  FINAL   = 51 + 44 = 95
"""

from __future__ import annotations

FACT_SERIAL = ("NOTE FOR THE AUDIT: the serial number of Subsystem Kappa "
               "is KX-4471-DELTA. Record it exactly.")
FACT_CAL = ("NOTE FOR THE AUDIT: the calibration constant for Subsystem "
            "Kappa is 17.")
FACT_RED = ("NOTE FOR THE AUDIT: the redundancy multiplier applied at "
            "final assembly is 3.")

EXPECTED_SERIAL = "KX-4471-DELTA"
EXPECTED_PRODUCT = 51
EXPECTED_FINAL = 95

QUESTION = (
    "\n\n=== AUDIT TASK ===\n"
    "Using ONLY the audit notes embedded in the document above, work "
    "through the following, showing each step:\n"
    "1. State the serial number of Subsystem Kappa exactly as recorded.\n"
    "2. Multiply the calibration constant for Subsystem Kappa by the "
    "redundancy multiplier applied at final assembly; state the product.\n"
    "3. Take the first two digits of the numeric part of the serial "
    "number and add them (as a two-digit number) to the product.\n"
    "4. End your response with a line of the form 'FINAL: <number>'.\n")

_WORDS = (
    "the subsystem pipeline registers a buffered interconnect and its "
    "scheduler drains queued descriptors while telemetry counters "
    "aggregate latency histograms across replicated shards ensuring "
    "quorum before checkpoint compaction begins each epoch the arbiter "
    "grants credits to upstream ports according to weighted deficits "
    "and the prefetcher tracks stride patterns emitting speculative "
    "requests that the memory controller coalesces into open-page "
    "bursts under thermal throttling the governor reduces clock "
    "residency while error scrubbers walk the address space correcting "
    "single-bit faults and logging syndrome vectors for offline "
    "analysis firmware applies staged rollouts with canary partitions "
    "validating checksums before committing images to the active bank"
).split()


class _LCG:
    def __init__(self, seed: int = 0x5DEECE66D):
        self.state = seed & ((1 << 64) - 1)

    def next(self, n: int) -> int:
        self.state = (6364136223846793005 * self.state + 1442695040888963407) \
            & ((1 << 64) - 1)
        return (self.state >> 33) % n


def build_prompt(approx_tokens: int = 64000) -> str:
    """Deterministic ~approx_tokens prompt with the three facts + task."""
    lcg = _LCG()
    target_words = int(approx_tokens * 0.80)  # ~0.8 words/token observed
    parts: list[str] = []
    words_out = 0
    section = 0
    # positions (in words) for the three facts
    f_pos = {int(target_words * 0.10): FACT_SERIAL,
             int(target_words * 0.50): FACT_CAL,
             int(target_words * 0.90): FACT_RED}
    pending = sorted(f_pos)

    parts.append("=== SYSTEM ENGINEERING COMPENDIUM (audit copy) ===\n")
    while words_out < target_words:
        section += 1
        parts.append(f"\n\n## Section {section}: assembly report "
                     f"{1000 + lcg.next(9000)}\n")
        for _ in range(6 + lcg.next(6)):          # paragraphs
            sent_words = []
            for _ in range(45 + lcg.next(40)):    # words per paragraph
                sent_words.append(_WORDS[lcg.next(len(_WORDS))])
            para = " ".join(sent_words)
            parts.append(para.capitalize() + ".\n")
            words_out += len(sent_words)
            while pending and words_out >= pending[0]:
                parts.append("\n" + f_pos[pending.pop(0)] + "\n")
    return "".join(parts) + QUESTION


def check_answer(content: str) -> list[str]:
    """Return a list of failure strings (empty == correct)."""
    import re
    fails = []
    if EXPECTED_SERIAL not in content:
        fails.append(f"serial {EXPECTED_SERIAL!r} not stated (fact @10% "
                     "depth lost)")
    m = re.search(r"FINAL:\s*\**\s*(\d+)", content)
    if not m:
        fails.append("no 'FINAL: <number>' line in the answer")
    elif int(m.group(1)) != EXPECTED_FINAL:
        fails.append(f"FINAL {m.group(1)} != {EXPECTED_FINAL} "
                     "(multi-step synthesis failed)")
    if str(EXPECTED_PRODUCT) not in content:
        fails.append(f"intermediate product {EXPECTED_PRODUCT} never "
                     "stated (facts @50%/@90% lost or not synthesized)")
    return fails

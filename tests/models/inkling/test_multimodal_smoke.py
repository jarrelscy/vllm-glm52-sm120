# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end API-level multimodal smoke test for Inkling-512k-NVFP4-AQLM-hybrid.

Unlike the other files in this directory (kernel/unit correctness tests that
import ``vllm.models.inkling`` directly and run on CPU or a bare GPU process),
this file drives an **already-running** OpenAI-compatible server over HTTP,
exactly the way the rest of this box's models are exercised (see
``/home/jarrelscy/homeassistant/CLAUDE.md``). It intentionally does NOT launch
its own ``vllm serve`` subprocess (unlike ``tests.utils.RemoteOpenAIServer``,
used by e.g. ``tests/entrypoints/multimodal/openai/chat_completion/test_audio.py``)
because Inkling is large (66-layer, 256-expert MoE, 512K context) and this
mission's plan is: bring the server up ONCE by hand, then run the base
text-only ``generate()`` smoke test, and only once THAT passes point this file
at the live instance.

Run it against a server already listening on :8001:

    pytest tests/models/inkling/test_multimodal_smoke.py -v

Point it elsewhere with ``INKLING_BASE_URL`` (default ``http://localhost:8001/v1``)
and ``INKLING_API_KEY`` / ``VLLM_API_KEY`` (default ``"EMPTY"``, i.e. no auth --
mirrors this box's ``.env`` convention of an empty ``VLLM_API_KEY``, see
``tests/sm120_correctness/common/server_client.py``). If no server answers
``GET /v1/models``, every test in this module is SKIPPED (not failed) with a
message naming the missing server -- this file is meant to sit inert in CI
until someone points it at a live box.

Suggested serving command (fill in once the base text-only smoke test is
confirmed working; adjust ``--max-model-len``/parallelism to taste -- this is
NOT the production 512K config, just enough to exercise all three modalities
quickly):

    vllm serve /data/huggingface/hub/models--jarrelscy--Inkling-512k-NVFP4-AQLM-hybrid/snapshots/db86e0aa27dc29c776894ab68c439edd622b2363 \\
        --served-model-name inkling \\
        --tensor-parallel-size 2 \\
        --max-model-len 8192 \\
        --max-num-seqs 4 \\
        --gpu-memory-utilization 0.90 \\
        --kv-cache-dtype fp8_e4m3 \\
        --limit-mm-per-prompt '{"image": 2, "audio": 2}' \\
        --port 8001

No ``--trust-remote-code`` is needed: ``InklingForConditionalGeneration`` is
registered natively in this fork's ``vllm/model_executor/models/registry.py``,
and the ``inkling_nvfp4_aqlm_hybrid`` quant method is auto-detected from the
checkpoint's ``quantization_config``/``hf_quant_config.json`` (see
``vllm/models/inkling/aqlm_hybrid.py``); it is NOT loadable on stock vLLM
(only on this SM120 fork), per the config's own quantization note.

============================================================================
Request-format grounding (what each test below actually sends over HTTP) --
cite the exact code each assumption is based on, since none of this was
verified against a running server:
============================================================================

Vision (image):
    Sent as a plain OpenAI ``image_url`` content part with a ``data:`` URI,
    e.g. ``{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}``.
    No special per-patch encoding is needed on the client side: the vendored
    ``InklingImageProcessor.preprocess`` (vllm/transformers_utils/processors/inkling.py)
    numba-patchifies whatever PIL image vLLM's standard image fetcher hands
    it into ``patch_size``-sized (default 40) tiles, and unconditionally
    ``.expand()``s each patch to ``temporal_patch_size=2`` "frames" (see
    ``_encode_image_bytes``, which does
    ``.view(num_patches, 1, patch_size, patch_size, 3).expand(num_patches, 2, ...)``)
    -- i.e. a single still image already satisfies the T=2 requirement; there
    is no separate multi-frame construction needed for a static image.
    ``InklingProcessingInfo.get_supported_mm_limits`` (vllm/models/inkling/common/mm_preprocess.py)
    only ever registers an ``"image"`` limit, never ``"video"`` -- so the
    ``vision_config.temporal_patch_size`` field is purely an internal HMLP
    tower detail, not a second client-facing modality. This test therefore
    does NOT attempt to construct a "video" input.

Audio:
    Sent as an OpenAI ``input_audio`` content part: ``{"type": "input_audio",
    "input_audio": {"data": "<base64 WAV>", "format": "wav"}}``. Confirmed
    parseable by this fork's ``vllm/entrypoints/chat_utils.py``
    (``_InputAudioParser`` / ``parse_input_audio``, chat_utils.py:904-1048),
    which builds a ``data:audio/{format};base64,{data}`` URL and hands it to
    ``MediaConnector.fetch_audio`` (vllm/multimodal/media/connector.py:445),
    which decodes via ``AudioMediaIO.load_bytes`` -> ``load_audio``
    (vllm/multimodal/media/audio.py) -- a real WAV decode, not a bespoke
    Inkling format. The repo's existing
    ``tests/entrypoints/multimodal/openai/chat_completion/test_audio.py``
    (a different, audio-native model) uses the sibling ``audio_url`` part
    instead; both are accepted by the same parser table
    (chat_utils.py ``_ContentPart`` handlers), so either works -- this file
    picks ``input_audio`` because it matches the vendor-neutral OpenAI
    Realtime/Audio API shape rather than a vLLM-only extension.
    Audio is decoded server-side to mono float32 and resampled to the dMel
    extractor's ``sample_rate`` (16 kHz; ``InklingAudioEncoderParams.sample_rate``,
    vllm/transformers_utils/processors/inkling.py), via the
    ``target_sr``/``target_channels=1`` passed to ``InklingMultiModalDataParser``
    in ``InklingProcessingInfo.get_data_parser``. This test therefore builds
    16 kHz mono WAV clips client-side (matching, not relying on, that resample
    path) using vLLM's own ``vllm.multimodal.utils.encode_audio_base64`` /
    ``encode_audio_url`` helpers (the same helpers
    ``tests/entrypoints/multimodal/openai/chat_completion/test_audio.py`` uses),
    so the WAV encoding matches an already-exercised code path.
    NOT independently verified: whether the "dmel" audio tower (a bag-of-bins
    embedding sum over 20 tokens/s, see ``InklingAudio.forward`` in
    vllm/models/inkling/common/towers.py) gives the *base* (non-instruction-tuned
    on audio-QA?) Inkling checkpoint enough signal to do open-ended audio
    description/pitch-judgment -- that is a capability question, not a
    plumbing question, and is called out per-test below.

Text:
    Plain ``content: str`` chat message, no multimodal fields -- this is the
    baseline "the API server itself, tokenizer, and sampler work" check and
    should already be covered by the base ``generate()`` smoke test; it is
    repeated here (with a stricter, MM-adjacent prompt) mainly as a fast
    canary so a multimodal-only regression doesn't get blamed on text.

============================================================================
Open questions to resolve against a live server before trusting these tests:
============================================================================
  1. Whether ``--limit-mm-per-prompt`` (or its absence) is required for this
     model to accept ANY image/audio content at all (vLLM raises a clear
     "multimodal support not enabled" style error if the launch config didn't
     enable image/audio) -- the suggested serve command above sets both to 2.
  2. Whether the served checkpoint has actually been instruction-tuned to
     *answer questions about* audio/image content, vs. only being able to
     encode it (i.e. whether ``test_audio_description_is_input_dependent``'s
     differential assertion is the right bar, or whether stronger semantic
     checks like the color-naming test are achievable for audio too).
  3. Exact ``MAX_AUDIO_TOKENS`` behavior at the API boundary: clips producing
     more than 12,000 dMel tokens (~10 min at 20 tok/s, see
     ``vllm/models/inkling/common/mm_preprocess.py``) raise ``ValueError``
     server-side; this file stays far under that (a few seconds of audio) so
     it isn't exercised here.
  4. Whether the box's actual boot command needs ``--trust-remote-code``
     after all (e.g. if the HF repo's own ``config.json``/``modeling_*.py``
     is loaded instead of this fork's native registry entry) -- the analysis
     above assumes the native-registry path since ``InklingForConditionalGeneration``
     already appears in ``vllm/model_executor/models/registry.py``.
"""

from __future__ import annotations

import base64
import io
import os
import urllib.error
import urllib.request

import numpy as np
import pytest

openai = pytest.importorskip("openai")


# ---------------------------------------------------------------------------
# Server connection config
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("INKLING_BASE_URL", "http://localhost:8001/v1")
API_KEY = os.environ.get("INKLING_API_KEY") or os.environ.get("VLLM_API_KEY") or "EMPTY"
MODEL_NAME = os.environ.get("INKLING_MODEL_NAME", "inkling")
REQUEST_TIMEOUT_S = float(os.environ.get("INKLING_REQUEST_TIMEOUT_S", "120"))


def _server_reachable() -> tuple[bool, str]:
    """GET /models; returns (ok, reason). Never raises."""
    url = BASE_URL.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                return False, f"GET {url} -> HTTP {resp.status}"
        return True, ""
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        return False, f"GET {url} failed: {e!r}"


@pytest.fixture(scope="module", autouse=True)
def _require_live_server():
    ok, reason = _server_reachable()
    if not ok:
        pytest.skip(
            "No Inkling server reachable at "
            f"{BASE_URL} ({reason}). This module is an end-to-end smoke test "
            "meant to run against a manually-launched `vllm serve` instance -- "
            "see the module docstring for the suggested command. Set "
            "INKLING_BASE_URL to point elsewhere."
        )


@pytest.fixture(scope="module")
def client() -> "openai.OpenAI":
    return openai.OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=REQUEST_TIMEOUT_S)


def _chat_text(client, messages, max_tokens=32, temperature=0.0, **extra):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        **extra,
    )
    assert len(resp.choices) == 1
    content = resp.choices[0].message.content
    return content or ""


# ---------------------------------------------------------------------------
# Synthetic input builders
# ---------------------------------------------------------------------------


def _make_solid_color_png_data_url(color: tuple[int, int, int], size: int = 160) -> str:
    """A ``size``x``size`` solid-color PNG as a base64 data URI.

    160 = patch_size(40) * 4, so InklingImageProcessor's patchifier
    (vllm/transformers_utils/processors/inkling.py:_encode_image_bytes) emits
    a clean 4x4 grid of whole patches with no partial-patch padding, keeping
    the dummy input as close as possible to a "normal" real image.
    """
    from PIL import Image

    img = Image.new("RGB", (size, size), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _make_wav_data_url(
    freq_hz: float | None,
    duration_s: float = 2.0,
    sample_rate: int = 16_000,
) -> str:
    """A mono WAV clip as a base64 data URI.

    ``freq_hz=None`` produces silence. ``sample_rate`` defaults to 16 kHz to
    match ``InklingAudioEncoderParams.sample_rate`` (the model's dMel
    extractor default, vllm/transformers_utils/processors/inkling.py) --
    matching this isn't strictly required (the server resamples via
    ``InklingMultiModalDataParser(target_sr=...)`` regardless) but avoids
    exercising the resampler as a confound in these tests.

    Uses vLLM's own ``encode_audio_url`` (same helper used by
    tests/entrypoints/multimodal/openai/chat_completion/test_audio.py) so the
    WAV encoding matches an already-exercised vLLM code path rather than a
    hand-rolled one.
    """
    from vllm.multimodal.utils import encode_audio_url

    n_samples = int(duration_s * sample_rate)
    if freq_hz is None:
        audio = np.zeros(n_samples, dtype=np.float32)
    else:
        t = np.arange(n_samples, dtype=np.float32) / sample_rate
        # Gentle fade in/out to avoid a click transient dominating the STFT
        # in the first/last analysis window.
        fade = min(int(0.05 * sample_rate), n_samples // 4)
        envelope = np.ones(n_samples, dtype=np.float32)
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            envelope[:fade] = ramp
            envelope[-fade:] = ramp[::-1]
        audio = 0.5 * np.sin(2 * np.pi * freq_hz * t).astype(np.float32) * envelope
    return encode_audio_url(audio, sample_rate, format="WAV")


# ---------------------------------------------------------------------------
# 1. Plain text generation coherence
# ---------------------------------------------------------------------------


class TestTextCoherence:
    """Baseline canary: no multimodal fields at all.

    PASS criterion: for a closed-form factual prompt with a single known
    correct token/word, the (temperature=0) response contains that word.
    This is stricter than "non-empty" -- it catches a server that's up and
    responding but generating garbage (e.g. wrong chat template applied,
    tokenizer/vocab mismatch after the SM120 port, EOS handling broken).
    """

    def test_arithmetic_sequence_completion(self, client):
        content = _chat_text(
            client,
            [
                {
                    "role": "user",
                    "content": (
                        "Complete the sequence with just the next number, "
                        "nothing else: 2, 4, 6, 8, "
                    ),
                }
            ],
            max_tokens=64,
        )
        assert content.strip(), "empty completion for a trivial text prompt"
        assert "10" in content, f"expected '10' in completion, got: {content!r}"

    def test_capital_city_fact(self, client):
        content = _chat_text(
            client,
            [
                {
                    "role": "user",
                    "content": (
                        "What is the capital of France? Answer with only the "
                        "city name, nothing else."
                    ),
                }
            ],
            max_tokens=64,
        )
        assert "paris" in content.lower(), f"expected 'Paris' in completion, got: {content!r}"


# ---------------------------------------------------------------------------
# 2. Audio input (dmel)
# ---------------------------------------------------------------------------


class TestAudioInput:
    """dMel audio-tower plumbing + (best-effort) capability checks.

    PASS criteria are split into two tiers because "does the audio path work
    at all" (plumbing) and "can the model reason about audio content"
    (capability) are different claims -- see open question #2 in the module
    docstring.
    """

    def test_audio_request_accepted_and_nonempty(self, client):
        """Tier 1 (plumbing, required): the server accepts an `input_audio`
        content part for this model and returns a non-empty completion.
        A crash, a 4xx/5xx, or an empty string here means the audio tower
        wiring (MAX_AUDIO_TOKENS bookkeeping, num_audio_tokens field config,
        AUDIO_MARKER_ID/AUDIO_TOKEN_ID prompt expansion in
        InklingMultiModalProcessor._get_prompt_updates) is broken, independent
        of whether the answer is *correct*.
        """
        audio_url = _make_wav_data_url(freq_hz=440.0, duration_s=2.0)
        content = _chat_text(
            client,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_url.split(",", 1)[1],
                                "format": "wav",
                            },
                        },
                        {
                            "type": "text",
                            "text": "In one short sentence, describe what you hear.",
                        },
                    ],
                }
            ],
            max_tokens=32,
        )
        assert content.strip(), "empty completion for an audio + text prompt"

    def test_audio_response_is_input_dependent(self, client):
        """Tier 1 (plumbing, required): responses to two acoustically very
        different clips (silence vs. a 440 Hz tone) must not be byte-identical
        under greedy decoding. If they *are* identical, the most likely
        explanation is that the audio embeddings are being dropped/zeroed
        somewhere in the port (e.g. a broken ``num_audio_tokens`` ->
        ``AUDIO_TOKEN_ID`` placeholder count, or the dmel bins clamping to a
        constant) and the model is silently falling back to text-only
        generation. This is a regression-style differential check, not a
        semantic-correctness check.
        """
        prompt = "In one short sentence, describe what you hear."
        silence_url = _make_wav_data_url(freq_hz=None, duration_s=2.0)
        tone_url = _make_wav_data_url(freq_hz=440.0, duration_s=2.0)

        def ask(data_url: str) -> str:
            return _chat_text(
                client,
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": data_url.split(",", 1)[1],
                                    "format": "wav",
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_tokens=32,
            )

        silence_answer = ask(silence_url)
        tone_answer = ask(tone_url)
        assert silence_answer.strip(), "empty completion for silent audio"
        assert tone_answer.strip(), "empty completion for tone audio"
        assert silence_answer != tone_answer, (
            "identical responses to silence vs. a 440 Hz tone -- audio "
            f"embeddings may not be reaching the model. silence={silence_answer!r} "
            f"tone={tone_answer!r}"
        )

    @pytest.mark.optional
    def test_audio_pitch_direction_capability(self, client):
        """Tier 2 (capability, optional -- run with ``--optional``): a
        stronger semantic check that the model can tell a low tone from a
        high tone. This asks more of the checkpoint than plumbing correctness
        (open question #2), so it's marked optional rather than gating the
        base multimodal smoke pass: a fresh SM120 port failing this while
        passing the two tests above should be triaged as a possible
        capability/precision issue in the audio tower forward pass, not
        necessarily a build-breaking bug.
        """
        low_answer = _chat_text(
            client,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": _make_wav_data_url(
                                    freq_hz=100.0, duration_s=2.0
                                ).split(",", 1)[1],
                                "format": "wav",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Is the pitch of this tone HIGH or LOW? "
                                "Answer with exactly one word: HIGH or LOW."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=4,
        )
        high_answer = _chat_text(
            client,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": _make_wav_data_url(
                                    freq_hz=8000.0, duration_s=2.0
                                ).split(",", 1)[1],
                                "format": "wav",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Is the pitch of this tone HIGH or LOW? "
                                "Answer with exactly one word: HIGH or LOW."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=4,
        )
        assert "low" in low_answer.lower(), f"100 Hz tone: got {low_answer!r}"
        assert "high" in high_answer.lower(), f"8000 Hz tone: got {high_answer!r}"


# ---------------------------------------------------------------------------
# 3. Vision input (hmlp)
# ---------------------------------------------------------------------------


class TestVisionInput:
    """HMLP vision-tower plumbing + capability check.

    PASS criterion: a solid-color image elicits the correct color name.
    This is both a plumbing check (a broken vision path most likely produces
    an empty/generic/off-topic answer, or a server error) and a real
    capability check, and is a much lower bar than object/scene recognition
    -- appropriate for a first vision smoke test.
    """

    @pytest.mark.parametrize(
        ("rgb", "color_word"),
        [
            ((255, 0, 0), "red"),
            ((0, 0, 255), "blue"),
        ],
    )
    def test_solid_color_image_is_named_correctly(self, client, rgb, color_word):
        data_url = _make_solid_color_png_data_url(rgb, size=160)
        content = _chat_text(
            client,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {
                            "type": "text",
                            "text": (
                                "What single color is this image? Answer with "
                                "just the color name."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=64,
        )
        assert content.strip(), "empty completion for an image + text prompt"
        assert color_word in content.lower(), (
            f"expected {color_word!r} in completion for a solid {rgb} image, "
            f"got: {content!r}"
        )

    def test_two_images_are_distinguished(self, client):
        """Plumbing check with two image items in one request: per-image
        ``num_patches``-based field splitting
        (``MultiModalFieldConfig.flat_from_sizes("image", num_patches)`` in
        ``InklingMultiModalProcessor._get_mm_fields_config``) must keep each
        image's patches separate rather than concatenating/misaligning them.
        PASS criterion: asked to compare two different solid colors, the
        response must at least mention both color words (in either order) --
        catches images being merged/overwritten/dropped when the item count
        goes from 1 to 2.
        """
        red_url = _make_solid_color_png_data_url((255, 0, 0), size=160)
        blue_url = _make_solid_color_png_data_url((0, 0, 255), size=160)
        content = _chat_text(
            client,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": red_url}},
                        {"type": "image_url", "image_url": {"url": blue_url}},
                        {
                            "type": "text",
                            "text": (
                                "Name the two colors shown, one per image, "
                                "separated by a comma."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=64,
        )
        assert content.strip(), "empty completion for a two-image prompt"
        low = content.lower()
        assert "red" in low, f"expected 'red' mentioned, got: {content!r}"
        assert "blue" in low, f"expected 'blue' mentioned, got: {content!r}"

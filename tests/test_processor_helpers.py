"""Unit tests for the PCM<->float32 helpers (pure, no model needed)."""
import numpy as np
from voicebot.pipeline.processors import _bytes_to_f32, _f32_to_bytes


def test_roundtrip_preserves_signal_within_quantization():
    original = np.array([0.0, 0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    restored = _bytes_to_f32(_f32_to_bytes(original))
    assert restored.shape == original.shape
    assert restored.dtype == np.float32
    # within one 16-bit quantization step
    assert np.max(np.abs(restored - original)) < 1.0 / 32767


def test_clipping_handles_out_of_range():
    loud = np.array([2.0, -3.0], dtype=np.float32)
    restored = _bytes_to_f32(_f32_to_bytes(loud))
    # clipped to [-1, 1] range (full scale)
    assert restored[0] <= 1.0 and restored[1] >= -1.0
    assert restored[0] > 0.99 and restored[1] < -0.99


def test_bytes_to_f32_scales_int16_to_unit_range():
    pcm = np.array([0, 16384, -16384], dtype=np.int16).tobytes()
    out = _bytes_to_f32(pcm)
    np.testing.assert_allclose(out, [0.0, 0.5, -0.5], atol=1e-4)

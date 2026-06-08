"""Noise reduction via noisereduce (spectral gating). Pure class, no Pipecat.
Works at 16kHz; non-stationary mode suits streaming mic input."""
import numpy as np
import noisereduce as nr
from voicebot.config import CONFIG


class Denoiser:
    def __init__(self, sample_rate: int = CONFIG.input_sample_rate):
        self._sr = sample_rate

    # Below this peak amplitude a frame is effectively silence. Spectral gating
    # then divides by a ~0 noise floor, producing NaNs — and there's nothing to
    # denoise anyway, so we pass such frames straight through.
    _SILENCE_PEAK = 1e-3

    def process(self, frame: np.ndarray) -> np.ndarray:
        """16kHz mono float32 in -> denoised 16kHz mono float32 out (same shape)."""
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        if x.size == 0 or float(np.max(np.abs(x))) < self._SILENCE_PEAK:
            return x
        out = nr.reduce_noise(y=x, sr=self._sr, stationary=False)
        out = np.nan_to_num(np.asarray(out, dtype=np.float32), nan=0.0,
                            posinf=0.0, neginf=0.0)
        return out.reshape(x.shape)

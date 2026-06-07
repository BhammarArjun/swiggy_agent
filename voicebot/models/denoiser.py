"""Noise reduction via noisereduce (spectral gating). Pure class, no Pipecat.
Works at 16kHz; non-stationary mode suits streaming mic input."""
import numpy as np
import noisereduce as nr
from voicebot.config import CONFIG


class Denoiser:
    def __init__(self, sample_rate: int = CONFIG.input_sample_rate):
        self._sr = sample_rate

    def process(self, frame: np.ndarray) -> np.ndarray:
        """16kHz mono float32 in -> denoised 16kHz mono float32 out (same shape)."""
        x = np.asarray(frame, dtype=np.float32).reshape(-1)
        out = nr.reduce_noise(y=x, sr=self._sr, stationary=False)
        return np.asarray(out, dtype=np.float32).reshape(x.shape)

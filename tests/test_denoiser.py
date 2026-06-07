import numpy as np
from voicebot.models.denoiser import Denoiser
from voicebot.config import CONFIG


def test_process_preserves_shape_and_dtype():
    dn = Denoiser()
    frame = (np.random.randn(CONFIG.input_sample_rate).astype(np.float32) * 0.01)
    out = dn.process(frame)
    assert out.dtype == np.float32
    assert out.shape == frame.shape

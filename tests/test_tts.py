import numpy as np
from voicebot.models.tts import KokoroTTS
from voicebot.config import CONFIG


def test_synthesize_yields_audio_chunks():
    tts = KokoroTTS()
    chunks = list(tts.synthesize("Two chicken biryanis added to your cart."))
    assert chunks, "should produce at least one chunk"
    audio = np.concatenate(chunks)
    assert audio.dtype == np.float32
    assert audio.size > CONFIG.tts_sample_rate * 0.3  # > ~0.3s of audio

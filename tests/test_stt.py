import os
import numpy as np
import soundfile as sf
import pytest
from voicebot.models.stt import ParakeetSTT

FIXTURE = "tests/fixtures/hello.wav"
pytestmark = pytest.mark.skipif(not os.path.exists(FIXTURE), reason="no fixture wav")


def test_transcribes_to_nonempty_text():
    audio, sr = sf.read(FIXTURE, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    stt = ParakeetSTT()
    text = stt.transcribe(audio)
    assert isinstance(text, str)
    assert len(text.strip()) > 0

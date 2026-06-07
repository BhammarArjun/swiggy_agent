"""End-to-end smoke test of the model stages on a pre-recorded WAV (no live
mic). Proves audio -> text -> agent (tools) -> reply -> audio works offline."""
import os
import numpy as np
import soundfile as sf
import pytest
from voicebot.config import CONFIG
from voicebot.models.stt import ParakeetSTT
from voicebot.models.llm import GemmaAgent
from voicebot.models.tts import KokoroTTS
from voicebot.tools import swiggy_stub
from voicebot.prompts.system import SWIGGY_SYSTEM_PROMPT

FIXTURE = "tests/fixtures/hello.wav"
pytestmark = pytest.mark.skipif(
    not (os.path.exists(FIXTURE) and os.path.exists(CONFIG.llm_model_path)),
    reason="fixture wav or model missing")


def test_audio_in_to_audio_out():
    swiggy_stub.reset_state()
    audio, _ = sf.read(FIXTURE, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    text = ParakeetSTT().transcribe(audio)
    assert text.strip()

    agent = GemmaAgent(CONFIG.llm_model_path, swiggy_stub.TOOLS, SWIGGY_SYSTEM_PROMPT)
    reply = "".join(agent.send(text))
    assert reply.strip()

    chunks = list(KokoroTTS().synthesize(reply))
    out = np.concatenate(chunks)
    assert out.size > 0

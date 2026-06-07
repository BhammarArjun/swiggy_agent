"""STTProcessor VAD-driven capture flow, with the transcriber faked so no
model loads. Verifies it captures audio between VAD start/stop and emits a
TranscriptionFrame."""
import numpy as np
import pytest
from pipecat.frames.frames import (
    AudioRawFrame, TranscriptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
import voicebot.pipeline.processors as P


class _FakeSTT:
    def __init__(self, *a, **k):
        self.seen_samples = 0

    def transcribe(self, audio: np.ndarray) -> str:
        self.seen_samples = audio.shape[0]
        return "add a chicken biryani"


@pytest.fixture
def proc(monkeypatch):
    monkeypatch.setattr(P, "ParakeetSTT", _FakeSTT)
    return P.STTProcessor()


async def _send(proc, frame):
    await proc.process_frame(frame, FrameDirection.DOWNSTREAM)


def _audio_frame(n=1600):
    pcm = (np.zeros(n, dtype=np.int16)).tobytes()
    return AudioRawFrame(pcm, 16000, 1)


@pytest.mark.asyncio
async def test_captures_between_vad_and_emits_transcription(proc):
    pushed = []
    async def fake_push(frame, direction=None, *a, **k):
        pushed.append(frame)
    proc.push_frame = fake_push

    await _send(proc, VADUserStartedSpeakingFrame())
    await _send(proc, _audio_frame(1600))
    await _send(proc, _audio_frame(1600))
    await _send(proc, VADUserStoppedSpeakingFrame())

    tx = [f for f in pushed if isinstance(f, TranscriptionFrame)]
    assert len(tx) == 1
    assert tx[0].text == "add a chicken biryani"
    assert proc._stt.seen_samples == 3200


@pytest.mark.asyncio
async def test_audio_outside_speech_is_ignored(proc):
    pushed = []
    async def fake_push(frame, direction=None, *a, **k):
        pushed.append(frame)
    proc.push_frame = fake_push

    # audio arriving with no preceding VAD-start must not be captured
    await _send(proc, _audio_frame(1600))
    await _send(proc, VADUserStoppedSpeakingFrame())

    assert not any(isinstance(f, TranscriptionFrame) for f in pushed)

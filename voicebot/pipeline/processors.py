"""Pipecat FrameProcessor adapters around the pure model classes.

Thin translation layer: each processor wraps a pure model class (built and
tested separately) and converts to/from Pipecat frames. Audio in the pipeline
is 16-bit PCM bytes; the pure models work in float32.
"""
import numpy as np
from pipecat.frames.frames import (
    Frame, AudioRawFrame, InputAudioRawFrame, TranscriptionFrame,
    TextFrame, TTSAudioRawFrame, InterruptionFrame, UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from voicebot.models.denoiser import Denoiser
from voicebot.models.stt import ParakeetSTT
from voicebot.models.llm import GemmaAgent
from voicebot.models.tts import KokoroTTS
from voicebot.config import CONFIG


def _bytes_to_f32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _f32_to_bytes(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


class DenoiserProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._dn = Denoiser()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            clean = self._dn.process(_bytes_to_f32(frame.audio))
            await self.push_frame(
                AudioRawFrame(_f32_to_bytes(clean), frame.sample_rate, frame.num_channels),
                direction)
        else:
            await self.push_frame(frame, direction)


class STTProcessor(FrameProcessor):
    """Buffers audio frames; on VAD-driven UserStoppedSpeaking, transcribes."""
    def __init__(self):
        super().__init__()
        self._stt = ParakeetSTT()
        self._buf: list[np.ndarray] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame) and not isinstance(frame, TTSAudioRawFrame):
            self._buf.append(_bytes_to_f32(frame.audio))
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame) and self._buf:
            audio = np.concatenate(self._buf)
            self._buf = []
            text = self._stt.transcribe(audio)
            await self.push_frame(frame, direction)
            if text:
                await self.push_frame(TranscriptionFrame(text, "user", ""), direction)
        else:
            await self.push_frame(frame, direction)


class GemmaLLMProcessor(FrameProcessor):
    def __init__(self, tools, system_prompt):
        super().__init__()
        self._agent = GemmaAgent(CONFIG.llm_model_path, tools, system_prompt)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self._agent.interrupt()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            for token in self._agent.send(frame.text):
                await self.push_frame(TextFrame(token), direction)
        else:
            await self.push_frame(frame, direction)


class KokoroTTSProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._tts = KokoroTTS()
        self._interrupted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            self._interrupted = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, TextFrame):
            self._interrupted = False
            for chunk in self._tts.synthesize(frame.text):
                if self._interrupted:
                    break
                await self.push_frame(
                    TTSAudioRawFrame(_f32_to_bytes(chunk), CONFIG.tts_sample_rate, 1),
                    direction)
        else:
            await self.push_frame(frame, direction)

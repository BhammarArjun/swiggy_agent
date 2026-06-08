"""Pipecat FrameProcessor adapters around the pure model classes.

Thin translation layer: each processor wraps a pure model class (built and
tested separately) and converts to/from Pipecat frames. Audio in the pipeline
is 16-bit PCM bytes; the pure models work in float32.

NOTE (barge-in latency): model inference in GemmaAgent.send() and
KokoroTTS.synthesize() is synchronous CPU work that runs on the event-loop
thread. Interruption is honored at token/chunk boundaries, so worst-case
barge-in latency is one chunk's synth time. If live testing shows laggy
barge-in, offload these sync loops to a thread (asyncio.to_thread) so
InterruptionFrames can preempt mid-chunk. Deferred until observed.
"""
import asyncio
import re
from typing import Callable, Iterator

import numpy as np
from pipecat.frames.frames import (
    Frame, AudioRawFrame, InputAudioRawFrame, TranscriptionFrame,
    TextFrame, TTSAudioRawFrame, InterruptionFrame,
    VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

from voicebot.models.denoiser import Denoiser
from voicebot.models.stt import ParakeetSTT
from voicebot.models.llm import GemmaAgent
from voicebot.models.tts import KokoroTTS
from voicebot.config import CONFIG


async def _stream_in_thread(
    gen_factory: Callable[[], Iterator],
    should_stop: Callable[[], bool],
):
    """Run a blocking generator on a worker thread and yield its items on the
    event loop, so synchronous model inference (LLM/TTS) doesn't block the loop
    (audio I/O + barge-in stay responsive). Stops early when should_stop() flips.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    sentinel = object()

    def worker():
        try:
            for item in gen_factory():
                if should_stop():
                    break
                loop.call_soon_threadsafe(queue.put_nowait, item)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, sentinel)

    fut = loop.run_in_executor(None, worker)
    try:
        while True:
            item = await queue.get()
            if item is sentinel:
                break
            yield item
    finally:
        await fut


def _bytes_to_f32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def _f32_to_bytes(audio: np.ndarray) -> bytes:
    # nan_to_num guards against NaN/inf (e.g. from upstream DSP) which would
    # otherwise cast to garbage int16 and emit RuntimeWarnings.
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()


class DenoiserProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._dn = Denoiser()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            clean = self._dn.process(_bytes_to_f32(frame.audio))
            # Emit a denoised InputAudioRawFrame (not a plain AudioRawFrame) so the
            # downstream VADProcessor — which only analyzes InputAudioRawFrame —
            # still sees it. This is why the denoiser sits BEFORE the VAD.
            await self.push_frame(
                InputAudioRawFrame(_f32_to_bytes(clean), frame.sample_rate,
                                   frame.num_channels),
                direction)
        else:
            await self.push_frame(frame, direction)


class STTProcessor(FrameProcessor):
    """Captures mic audio between VAD speech start/stop, then transcribes.

    pipecat 1.3.0's VADProcessor emits VADUserStartedSpeakingFrame /
    VADUserStoppedSpeakingFrame (the bare User*SpeakingFrame variants are
    produced by the turn-controller subsystem, which this custom pipeline does
    not use). We capture only while the VAD reports speech, so silence/noise
    between utterances is not transcribed.
    """

    # Hard cap so a stuck-open mic / missed VAD-stop can't grow the buffer
    # without bound. 30s of 16k mono float32 ~= 1.9 MB.
    _MAX_BUFFER_SAMPLES = CONFIG.input_sample_rate * 30

    def __init__(self):
        super().__init__()
        self._stt = ParakeetSTT()
        self._buf: list[np.ndarray] = []
        self._buffered_samples = 0
        self._capturing = False

    def _clear_buffer(self) -> None:
        self._buf = []
        self._buffered_samples = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            # New utterance begins (also a barge-in): start fresh capture.
            self._clear_buffer()
            self._capturing = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, InterruptionFrame):
            self._clear_buffer()
            self._capturing = False
            await self.push_frame(frame, direction)
        elif (self._capturing and isinstance(frame, AudioRawFrame)
              and not isinstance(frame, TTSAudioRawFrame)):
            chunk = _bytes_to_f32(frame.audio)
            self._buf.append(chunk)
            self._buffered_samples += chunk.shape[0]
            if self._buffered_samples > self._MAX_BUFFER_SAMPLES:
                # keep only the most recent window; oldest audio is least useful
                self._clear_buffer()
                self._buf.append(chunk)
                self._buffered_samples = chunk.shape[0]
            await self.push_frame(frame, direction)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._capturing = False
            await self.push_frame(frame, direction)
            if self._buf:
                audio = np.concatenate(self._buf)
                self._clear_buffer()
                # transcription is blocking CPU work — keep it off the event loop
                text = await asyncio.to_thread(self._stt.transcribe, audio)
                if text:
                    await self.push_frame(TranscriptionFrame(text, "user", ""), direction)
        else:
            await self.push_frame(frame, direction)


# Sentence-final punctuation; we flush accumulated tokens to TTS at these
# boundaries so Kokoro synthesizes whole phrases (not one word/token at a time).
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]?\s|[\n]+")


class GemmaLLMProcessor(FrameProcessor):
    def __init__(self, tools, system_prompt):
        super().__init__()
        self._agent = GemmaAgent(CONFIG.llm_model_path, tools, system_prompt)
        self._interrupted = False

    async def cleanup(self):
        # Release the LiteRT-LM conversation + engine on pipeline shutdown.
        await super().cleanup()
        self._agent.close()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, VADUserStartedSpeakingFrame)):
            # Barge-in: user started speaking (or explicit interruption).
            self._interrupted = True
            self._agent.interrupt()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            self._interrupted = False
            # The model streams sub-word tokens; we buffer them and only emit a
            # TextFrame once a full sentence is ready, so TTS speaks phrases
            # instead of spelling out tokens.
            buf = ""
            async for token in _stream_in_thread(
                lambda: self._agent.send(frame.text),
                lambda: self._interrupted,
            ):
                if self._interrupted:
                    break
                buf += token
                # flush every complete sentence found in the buffer
                while True:
                    m = _SENTENCE_END.search(buf)
                    if not m:
                        break
                    sentence, buf = buf[:m.end()], buf[m.end():]
                    sentence = sentence.strip()
                    if sentence:
                        await self.push_frame(TextFrame(sentence), direction)
            # flush any trailing text that didn't end with punctuation
            tail = buf.strip()
            if tail and not self._interrupted:
                await self.push_frame(TextFrame(tail), direction)
        else:
            await self.push_frame(frame, direction)


class KokoroTTSProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._tts = KokoroTTS()
        self._interrupted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, VADUserStartedSpeakingFrame)):
            # Barge-in: stop speaking as soon as the user starts.
            self._interrupted = True
            await self.push_frame(frame, direction)
        elif isinstance(frame, TextFrame):
            self._interrupted = False
            async for chunk in _stream_in_thread(
                lambda: self._tts.synthesize(frame.text),
                lambda: self._interrupted,
            ):
                if self._interrupted:
                    break
                await self.push_frame(
                    TTSAudioRawFrame(_f32_to_bytes(chunk), CONFIG.tts_sample_rate, 1),
                    direction)
        else:
            await self.push_frame(frame, direction)

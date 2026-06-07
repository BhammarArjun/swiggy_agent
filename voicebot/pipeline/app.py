"""Assemble and run the local voicebot Pipecat pipeline."""
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams

from voicebot.config import CONFIG
from voicebot.tools import swiggy_stub
from voicebot.prompts.system import SWIGGY_SYSTEM_PROMPT
from voicebot.pipeline.processors import (
    DenoiserProcessor, STTProcessor, GemmaLLMProcessor, KokoroTTSProcessor,
)


def build_task() -> PipelineTask:
    transport = LocalAudioTransport(LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_in_sample_rate=CONFIG.input_sample_rate,
        audio_out_sample_rate=CONFIG.tts_sample_rate,
    ))
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(
        params=VADParams(stop_secs=CONFIG.vad_stop_secs)))
    pipeline = Pipeline([
        transport.input(),
        vad,
        DenoiserProcessor(),
        STTProcessor(),
        GemmaLLMProcessor(swiggy_stub.TOOLS, SWIGGY_SYSTEM_PROMPT),
        KokoroTTSProcessor(),
        transport.output(),
    ])
    return PipelineTask(pipeline, params=PipelineParams(
        audio_in_sample_rate=CONFIG.input_sample_rate,
        audio_out_sample_rate=CONFIG.tts_sample_rate,
    ))


async def run() -> None:
    runner = PipelineRunner(handle_sigint=True)
    await runner.run(build_task())

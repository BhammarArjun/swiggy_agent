import numpy as np
import pytest
from pipecat.frames.frames import TextFrame, TranscriptionFrame
from voicebot.pipeline.processors import GemmaLLMProcessor
from voicebot.tools import swiggy_stub
from voicebot.prompts.system import SWIGGY_SYSTEM_PROMPT
from voicebot.config import CONFIG
import os

pytestmark = pytest.mark.skipif(
    not os.path.exists(CONFIG.llm_model_path), reason="model missing")


@pytest.mark.asyncio
async def test_llm_processor_emits_text_frames():
    pushed = []
    proc = GemmaLLMProcessor(swiggy_stub.TOOLS, SWIGGY_SYSTEM_PROMPT)

    async def fake_push(frame, direction=None, *args, **kwargs):
        pushed.append(frame)
    proc.push_frame = fake_push

    from pipecat.processors.frame_processor import FrameDirection
    await proc.process_frame(
        TranscriptionFrame("What addresses do I have?", "user", ""),
        FrameDirection.DOWNSTREAM,
    )
    assert any(isinstance(f, TextFrame) for f in pushed)

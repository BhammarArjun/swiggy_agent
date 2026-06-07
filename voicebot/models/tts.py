"""Kokoro text-to-speech. Pure class, no Pipecat. Streams audio per sentence."""
import numpy as np
from kokoro import KPipeline
from voicebot.config import CONFIG


class KokoroTTS:
    def __init__(self, voice: str = CONFIG.tts_voice, lang_code: str = CONFIG.tts_lang_code):
        self._pipeline = KPipeline(lang_code=lang_code)
        self._voice = voice

    def synthesize(self, text: str):
        """text -> iterator of float32 audio chunks at CONFIG.tts_sample_rate (24k)."""
        for _, _, audio in self._pipeline(text, voice=self._voice):
            if hasattr(audio, "detach"):  # torch.Tensor
                audio = audio.detach().cpu().numpy()
            yield np.asarray(audio, dtype=np.float32).reshape(-1)

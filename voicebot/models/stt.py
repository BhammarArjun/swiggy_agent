"""Parakeet-MLX speech-to-text. Pure class, no Pipecat."""
import tempfile
import numpy as np
import soundfile as sf
from parakeet_mlx import from_pretrained
from voicebot.config import CONFIG


class ParakeetSTT:
    def __init__(self, model_id: str = CONFIG.stt_model_id):
        self._model = from_pretrained(model_id)
        # Warm up: the first MLX transcribe compiles Metal kernels (~1.4s). Do it
        # now, at startup, so the user's first real utterance isn't penalised.
        try:
            self.transcribe(np.zeros(CONFIG.input_sample_rate, dtype=np.float32))
        except Exception:
            pass

    def transcribe(self, audio: np.ndarray) -> str:
        """float32 16kHz mono audio -> text. parakeet-mlx transcribes a file path,
        so we write the segment to a temp wav first."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, audio, CONFIG.input_sample_rate)
            result = self._model.transcribe(f.name)
        return getattr(result, "text", str(result)).strip()

"""Central configuration for the voicebot."""
import os
from pathlib import Path
from dataclasses import dataclass


def _load_env() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
os.environ.setdefault("GLOG_minloglevel", "3")


@dataclass(frozen=True)
class Config:
    # audio
    input_sample_rate: int = 16000      # mic + STT + VAD work at 16k
    tts_sample_rate: int = 24000        # Kokoro native output rate
    # models
    llm_model_path: str = os.path.expanduser("~/models/litert/gemma-4-E2B-it.litertlm")
    llm_cache_dir: str = os.path.expanduser("~/.cache/litert-lm")
    stt_model_id: str = "mlx-community/parakeet-tdt-0.6b-v2"
    tts_voice: str = "af_heart"         # Kokoro voice; revisit during TTS task
    tts_lang_code: str = "a"            # 'a' = American English
    # VAD
    vad_stop_secs: float = 0.5          # trailing silence to end an utterance
    # denoiser
    denoise_attenuation_db: float = 30.0


CONFIG = Config()

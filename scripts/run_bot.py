"""Entry point: talk to the bot (use headphones).

Runnable from anywhere, e.g.:
    python scripts/run_bot.py
    python run_bot.py   (from inside scripts/)

Console hygiene: LiteRT-LM links absl/glog C++ logging that writes straight to
file descriptor 2 (stderr) and ignores every GLOG_*/TF_* env knob in this
build. To get a clean console showing only Pipecat output, we point the OS-level
stderr fd at /dev/null (silencing the C++ libs) and rebind Python's sys.stderr
to the real terminal — so loguru (Pipecat) logs and Python tracebacks still show.
"""
import os
import sys

# --- redirect C-level stderr BEFORE importing anything that links LiteRT ---
_real_stderr_fd = os.dup(2)                      # keep a handle to the terminal
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 2)                           # C/C++ stderr -> /dev/null
os.close(_devnull_fd)
# Rebind Python's stderr to the real terminal so loguru/pipecat + tracebacks
# (which write via sys.stderr, not raw fd 2) remain visible. loguru binds its
# default sink to sys.stderr at import time, so this must happen before the
# voicebot/pipecat imports below.
sys.stderr = os.fdopen(_real_stderr_fd, "w", buffering=1)

import asyncio
import logging
import warnings
from pathlib import Path

# Third-party DSP/ML libs (noisereduce, torch, kokoro) emit noisy Runtime/Future
# warnings on every audio frame. Silence them so the console shows only the
# Pipecat lifecycle + our tool-call traces.
warnings.filterwarnings("ignore")

# ensure the repo root (which contains the `voicebot` package) is importable,
# regardless of the current working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Keep stdlib logging quiet except our own stub-tool tracing, which is useful
# for watching the Swiggy tool chain fire.
logging.basicConfig(level=logging.WARNING)
logging.getLogger("swiggy_stub").setLevel(logging.INFO)

from voicebot.pipeline.app import run

if __name__ == "__main__":
    print("Voicebot starting — wear headphones. Speak when ready. Ctrl-C to quit.")
    asyncio.run(run())

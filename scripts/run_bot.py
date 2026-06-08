"""Entry point: talk to the bot (use headphones).

Runnable from anywhere, e.g.:
    python scripts/run_bot.py
    python run_bot.py   (from inside scripts/)
"""
import asyncio
import sys
from pathlib import Path

# ensure the repo root (which contains the `voicebot` package) is importable,
# regardless of the current working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voicebot.pipeline.app import run

if __name__ == "__main__":
    print("Voicebot starting — wear headphones. Speak when ready. Ctrl-C to quit.")
    asyncio.run(run())

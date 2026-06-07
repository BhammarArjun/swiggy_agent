"""Entry point: python scripts/run_bot.py — talk to the bot (use headphones)."""
import asyncio
from voicebot.pipeline.app import run

if __name__ == "__main__":
    print("Voicebot starting — wear headphones. Speak when ready. Ctrl-C to quit.")
    asyncio.run(run())

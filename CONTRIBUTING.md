# Contributing

Thanks for helping out! This is a fully on-device voice agent, so the setup is a
bit more hands-on than a typical cloud app. Here's how to get running and how to
keep changes consistent.

## Prerequisites

- **Apple Silicon Mac** (M-series). The STT stage (Parakeet) runs on MLX/Metal,
  and the LLM (LiteRT-LM) and TTS (Kokoro) are tuned for on-device CPU/GPU.
- **Python 3.11**
- **A Hugging Face account + token** with access to the gated models (used only
  to download model weights the first time).
- **Headphones** for live testing — v1 has no echo cancellation, so speakers
  would let the bot hear itself.

## One-time setup

```bash
# 1. clone + virtualenv
git clone https://github.com/BhammarArjun/swiggy_agent.git
cd swiggy_agent
python3.11 -m venv .venv && source .venv/bin/activate

# 2. dependencies
pip install -r requirements.txt

# 3. Hugging Face token for gated model downloads
#    create a .env in the repo root (it is gitignored — never commit it):
cat > .env <<'EOF'
HF_TOKEN=hf_your_token_here
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
EOF
chmod 600 .env

# 4. the Gemma 4 E2B LiteRT-LM model must be present locally at:
#    ~/models/litert/gemma-4-E2B-it.litertlm
#    (download from the LiteRT community bundle; ~1.5 GB)
```

Parakeet and Kokoro weights download automatically on first run (cached under
`~/.cache/huggingface`). The first launch is slow — it loads all four models.

## Running

```bash
python scripts/run_bot.py     # talk to the bot — use headphones
```

The console logs the full turn flow (🎙️ speech start/stop → 📝 transcription →
🧠 LLM → 🔊 TTS → ✋ barge-in) plus each Swiggy tool call, so you can see exactly
what the agent is doing.

## Tests

```bash
pytest                 # full suite
pytest tests/test_swiggy_stub.py    # a single file
```

Some tests are **model-gated**: they `skip` automatically if the LiteRT-LM model
isn't present (so CI / a fresh checkout still passes). If you have the model
locally, they run for real. Please keep tests green before opening a PR.

## Project layout

```
voicebot/
  config.py            # central config (model paths, sample rates, VAD timing)
  models/              # pure, framework-free wrappers: stt, llm, tts, denoiser
  pipeline/
    processors.py      # Pipecat FrameProcessors (model glue + barge-in + buffering)
    app.py             # assembles and runs the pipeline
  prompts/system.py    # Swiggy system prompt
  tools/swiggy_stub.py # stub MCP tools + in-memory cart
scripts/run_bot.py     # entry point
tests/                 # unit + integration tests
```

**Architecture principle:** keep `voicebot/models/*` as **pure classes** with no
Pipecat dependency (easy to unit-test in isolation). All Pipecat-specific glue
lives in `voicebot/pipeline/processors.py`.

## Conventions for changes

- **Match the surrounding style** — small, focused modules; docstrings explaining
  *why*, not just *what*.
- **Add/adjust a test** for behavioural changes. Model-gate any test that needs
  real weights (see `tests/test_gemma_agent.py` for the pattern).
- **Never commit secrets or personal data.** `.env`, model binaries
  (`*.litertlm`, `*.gguf`), and images are gitignored — keep it that way.
- **The Swiggy tools are stubs** (`voicebot/tools/swiggy_stub.py`). They mirror
  the real MCP tool signatures so the live integration is a drop-in swap — keep
  signatures stable when editing them. v1 must **never place a real order**.
- Note platform constraints in code when relevant — e.g. Parakeet must run on the
  event-loop thread (MLX GPU streams are thread-local); the CPU LLM/TTS stages
  run on worker threads to keep the loop responsive.

## Submitting a PR

1. Branch off `main`.
2. Make the change + tests; run `pytest`.
3. Open a PR describing what changed and why. Mention if you tested live
   (mic/headphones) since that path can't be exercised in CI.

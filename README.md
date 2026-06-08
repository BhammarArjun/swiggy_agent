# Swiggy Voice Agent — a fully on-device food-ordering voicebot

A voice assistant for Swiggy-style food ordering that runs **entirely locally on
Apple Silicon** — no cloud APIs. Every stage (speech-to-text, reasoning,
tool-calling, text-to-speech) executes on-device, so audio and conversation
never leave the laptop. The user talks; the agent searches restaurants, builds a
cart, and places an order through tool calls, with **real-time barge-in** (you
can interrupt it mid-sentence).

> **Status:** working end-to-end prototype. The Swiggy actions currently run as
> **local stubs** with the exact signatures the real tools will expose, so
> swapping in the live integration requires no pipeline changes. **No real
> orders are placed.**

## Architecture

A single [Pipecat](https://github.com/pipecat-ai/pipecat) pipeline streams audio
through these stages:

```
 mic ─▶ VAD ─▶ denoiser ─▶ STT ─▶ LLM (+ tools) ─▶ TTS ─▶ speaker
        │        │          │        │                │
     Silero  noisereduce  Parakeet  Gemma 4 E2B     Kokoro
              (spectral    -MLX     via LiteRT-LM
               gating)              (reasoning +
                                     tool-calling)
```

| Stage | Component | Notes |
|-------|-----------|-------|
| Voice activity detection | Silero VAD | marks utterance start/stop, drives turn-taking + barge-in |
| Denoise | `noisereduce` | spectral gating so ambient sound doesn't trip the pipeline |
| Speech-to-text | Parakeet-MLX (`parakeet-tdt-0.6b-v2`) | runs on the Apple Metal GPU |
| Reasoning + tools | Gemma 4 E2B via **LiteRT-LM** | auto-generates tool schemas, chains tool calls |
| Text-to-speech | Kokoro (`af_heart`) | streamed at sentence/clause boundaries for low latency |
| Orchestration | Pipecat | custom `FrameProcessor`s wrap each local model |

The LLM is the single "brain": it holds the conversation, decides when an action
is needed, calls the tool, and uses the result to continue the dialogue.

## MCP integration

The Swiggy **MCP server is exposed to the LLM as its toolset** (search
restaurants, get menu, update cart, apply coupon, place order, …). During a
conversation the model decides — from the user's intent — when an action is
needed and **issues a tool call, which the system routes to the corresponding
MCP tool**; the result is fed back into the model's context so it can continue.
In other words, MCP is invoked **on demand, driven by the model's reasoning**,
not on a fixed script.

In this prototype those tools are local stubs (`voicebot/tools/swiggy_stub.py`)
with identical signatures, so connecting the real MCP client is a drop-in
replacement.

## Repository layout

```
voicebot/
  config.py            # central config (model paths, sample rates, VAD timing)
  models/              # pure, framework-free wrappers: stt, llm, tts, denoiser
  pipeline/
    processors.py      # Pipecat FrameProcessors (translation + barge-in + buffering)
    app.py             # assembles and runs the pipeline
  prompts/system.py    # Swiggy system prompt (confirm before ordering, COD only, …)
  tools/swiggy_stub.py # stub MCP tools + in-memory cart
scripts/run_bot.py     # entry point — `python scripts/run_bot.py`
tests/                 # unit + integration tests (model-gated where needed)
phase-1/               # model-evaluation notes that justified the model choices
docs/superpowers/      # design spec + implementation plan
```

## Running it

Requires an Apple Silicon Mac, the Gemma 4 E2B LiteRT-LM model at
`~/models/litert/gemma-4-E2B-it.litertlm`, and a Hugging Face token in a local
`.env` (gitignored) for the gated STT/TTS model downloads.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_bot.py        # use headphones (no echo cancellation in v1)
pytest                           # run the test suite
```

Say something like *"Order two chicken biryanis from a biryani place near my
home"* and confirm when asked. The console logs the full turn flow — speech
start/stop, transcription, the tool chain firing, each spoken sentence, and
barge-in events.

## Design choices & constraints

- **Why Gemma 4 E2B as the only brain:** Phase-1 evaluation (see `phase-1/`)
  showed it reliably chains multi-step tool calls with real IDs, whereas a
  smaller routing model (FunctionGemma-270M) could not.
- **Single user, no concurrency** — a personal on-device assistant.
- **Safety:** confirms before placing an order; COD only; cart capped; v1 never
  places a real order (stub tools).
- **Latency-aware:** CPU LLM/TTS run on worker threads to keep the event loop
  responsive; STT is pinned to the main thread (MLX GPU streams are
  thread-local); barge-in flushes the output audio buffer for instant stop.

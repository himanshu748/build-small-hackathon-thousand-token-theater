---
title: Thousand-Token Theater
emoji: 🎭
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: mit
suggested_hardware: zero-a10g
short_description: MiniCPM improv troupe with a hard 1,000-token memory
tags:
  - build-small-hackathon
  - thousand-token-wood
  - minicpm
  - openbmb
---

# 🎭 Thousand-Token Theater

A troupe of small-model actors improvises a one-act play that **you** direct.
The twist that gives the piece its name: the troupe's *entire* shared memory of
the play is hard-capped at **1,000 tokens**, measured by the model's own
tokenizer. As the script grows past the cap, the oldest beats are **forgotten** —
and the actors carry on with only what still fits.

The forgetting is the point. You watch the play drift: a plot the cast cared
about ten beats ago quietly falls out of memory, a character reappears with no
idea who they once were, a secret you planted dissolves into the dark. It is a
living demonstration of what a small context window does to a story.

Built for the **Build Small Hackathon** · track *Adventure in Thousand Token Wood*.

## How to play

1. Pick a **setting** (woodland fable, noir alley, derelict starship, royal banquet) and an optional **premise**.
2. **Raise the curtain** — the Narrator opens the scene.
3. Press **Next beat ▶** to let the next actor improvise, or **Twist 🎲** for a random shock.
4. Type a **stage direction** in the Director box to steer the play (it, too, can later be forgotten).
5. Watch the **Troupe memory** meter fill toward 1,000 tokens — and the **Forgotten** panel catch what slips away.

## What actually runs (no hand-waving)

- **Model:** [`openbmb/MiniCPM4.1-8B`](https://huggingface.co/openbmb/MiniCPM4.1-8B) — an OpenBMB small model (≤32B) — loaded with `trust_remote_code=True` and run **on the Space's ZeroGPU (A10G)** via `@spaces.GPU`. Every line on stage is generated live; nothing is pre-written.
- **The 1,000-token cap is real:** memory length is measured with MiniCPM's own tokenizer (`model.py::count_tokens`). When the running script exceeds the budget, the engine evicts the oldest beats (`theater.py::TheaterEngine._append_and_evict`). The actors are only ever shown what still fits, so the forgetting genuinely changes their behaviour.
- **Snappy lines:** short, in-character generation (no chain-of-thought); sampling temperature 0.7 / top_p 0.95.
- **Streamed live:** each line is streamed token-by-token (`TextIteratorStreamer`) so you watch the actors write in real time. Sampling follows MiniCPM's official no-think guidance (temperature 0.7, top_p 0.95).
## Architecture

| File | Role |
| --- | --- |
| `theater.py` | The improv engine: cast, turn order, prompt construction, and the bounded-memory / eviction logic. Model-agnostic (dependency-injected `generate_fn` + `count_tokens_fn`) so it is unit-testable without a GPU. |
| `model.py` | The only file that touches MiniCPM. Loads MiniCPM4.1-8B on ZeroGPU; exposes `generate()`, `generate_stream()`, and `count_tokens()`. No mock or fallback. |
| `app.py` | The Gradio stage: script view, the 1,000-token memory meter, and the "Forgotten to the Wood" panel. |
| `test_engine.py` | Local logic test using a deterministic stub (no model) to verify the budget is never exceeded, eviction works, and turns rotate. |

## Run locally

```bash
pip install -r requirements.txt
python app.py            # loads MiniCPM4.1-8B; needs a CUDA GPU (~16GB)
python test_engine.py    # engine logic only; no GPU/model required
```

## Credits

Model by **OpenBMB** (MiniCPM). Built with **Gradio** on **Hugging Face Spaces**
for the Build Small Hackathon. MIT-licensed.

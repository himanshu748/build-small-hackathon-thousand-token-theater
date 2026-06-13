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
  - zerogpu
  - tiny-titan
  - track:wood
  - sponsor:openbmb
  - achievement:offgrid
  - achievement:offbrand
  - achievement:fieldnotes
models:
  - openbmb/MiniCPM5-1B
  - openbmb/VoxCPM2
---

# 🎭 Thousand-Token Theater

A troupe of small-model actors improvises a one-act play that **you** direct —
and now you can **hear** it: every actor performs each line aloud in their own
voice. The twist that gives the piece its name: the troupe's *entire* shared
memory of the play is hard-capped at **1,000 tokens**, measured by the model's own
tokenizer. As the script grows past the cap, the oldest beats are **forgotten** —
and the actors carry on with only what still fits.

The forgetting is the point. You watch the play drift: a plot the cast cared
about ten beats ago quietly falls out of memory, a character reappears with no
idea who they once were, a secret you planted dissolves into the dark. It is a
living demonstration of what a small context window does to a story.

Built for the **Build Small Hackathon** · track *Adventure in Thousand Token Wood*.

## What's new

- **🔊 Every actor has a voice.** Each line is spoken aloud in a distinct,
  consistent voice via [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2).
  Voices are *designed* per character (grand badger baritone, breathless little
  wren, croaky toad impresario, velvet narrator) and then cloned per line so each
  actor sounds like themselves across the whole play. Toggle it with **🔊 Voices**.
- **🧠 The memory meter now drains.** It shows how much of the 1,000-token memory
  is still free, draining toward empty as the scene fills; when it hits zero the
  troupe is at capacity and every new line visibly buries an older one.
- **⭐ Save highlights.** Bookmark a favourite line or a whole scene — saved
  moments survive even after the troupe forgets them, and you can download them.
- **🪶 Tiny actors.** The whole troupe now runs on **`openbmb/MiniCPM5-1B`**, a
  1-billion-parameter model — small enough to leave the GPU free for the voice
  model, and proof that a *tiny* model can carry an entire improv show.

## Submission Evidence

- Live Space: https://huggingface.co/spaces/build-small-hackathon/thousand-token-theater
- Public GitHub evidence repo: https://github.com/himanshu748/build-small-hackathon-thousand-token-theater
- Demo video + social post: https://x.com/i/status/2064354192748110158
- Field Notes / build report: https://github.com/himanshu748/build-small-hackathon-thousand-token-theater/blob/main/docs/field-notes.md
No public traces are claimed for this project yet.

## Hackathon Fit

- Track: Adventure in Thousand Token Wood.
- Build surface: custom Gradio `Blocks` app hosted as a Hugging Face Space.
- Model rule: the actors run on `openbmb/MiniCPM5-1B` and the voices on `openbmb/VoxCPM2` — both OpenBMB small models under the `≤32B` limit.
- OpenBMB angle: two OpenBMB models drive the show end-to-end — MiniCPM5-1B writes every line and counts the 1,000-token memory, and VoxCPM2 speaks every line aloud.
- Tiny Titan angle: the entire improv troupe is powered by a single **1B** model (`MiniCPM5-1B`) — a genuinely tiny model carrying a whole live performance.
- Off-Brand angle: custom theater/playbill UI with stage, director controls, per-actor voices, a draining memory meter, a Forgotten panel, and saveable highlights.
- Off the Grid angle: the app avoids external cloud model APIs; both generation and speech run through models loaded on the Space runtime.

Not claimed: OpenAI Codex, Sharing is Caring, Llama Champion, Modal, Well-Tuned, or Best Agent.

## How to play

1. Pick a **setting** (woodland fable, noir alley, derelict starship, royal banquet) and an optional **premise**.
2. **Raise the curtain** — the Narrator opens the scene, and you hear it.
3. Press **Play it ▶** to let the next actor improvise (and speak), or **Surprise me 🎲** for a random shock.
4. Type a **stage direction** in the Director box to steer the play (it, too, can later be forgotten).
5. Watch the **Troupe memory** meter **drain** toward empty — and the **Forgotten** panel catch what slips away.
6. Tap **⭐ Save last line** or **🎬 Save this scene** to keep favourite moments; they survive the forgetting and can be downloaded. Use **🔊 Voices** to turn speech on or off.

## What actually runs (no hand-waving)

- **Actors:** [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) — an OpenBMB small model (≤32B) — loaded with `trust_remote_code=True` and run **on the Space's ZeroGPU (A10G)** via `@spaces.GPU`. Every line on stage is generated live; nothing is pre-written.
- **Voices:** [`openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2) — OpenBMB's tokenizer-free TTS — also on the Space's ZeroGPU. Each character's timbre is designed once from a text description, then every line is cloned from that reference so the voice stays consistent. If synthesis genuinely fails, the line still appears on stage and the app says so — **no stand-in/fake voice is ever played**.
- **The 1,000-token cap is real:** memory length is measured with MiniCPM's own tokenizer (`model.py::count_tokens`). When the running script exceeds the budget, the engine evicts the oldest beats (`theater.py::TheaterEngine._append_and_evict`). The actors are only ever shown what still fits, so the forgetting genuinely changes their behaviour — and the meter drains to reflect it.
- **Snappy lines:** short, in-character generation (no chain-of-thought); MiniCPM5 No-Think sampling (temperature 0.7 / top_p 0.95).
- **Live, then spoken in chunks:** each line streams onto the stage token-by-token as the model writes it, then it's **spoken aloud one sentence at a time** (autoplay) — the voice arrives in chunks as each sentence is synthesized, rather than one long clip after the line.

## Architecture

| File | Role |
| --- | --- |
| `theater.py` | The improv engine: cast, turn order, prompt construction, and the bounded-memory / eviction logic. Model-agnostic (dependency-injected `generate_fn` + `count_tokens_fn`) so it is unit-testable without a GPU. |
| `model.py` | The text-model boundary. Loads `MiniCPM5-1B` on ZeroGPU; exposes `generate()`, `generate_stream()`, and `count_tokens()`. No mock or fallback. |
| `voice.py` | The voice boundary. Loads `VoxCPM2` on ZeroGPU; designs one reference voice per character, then synthesizes each line (`synthesize()`). Honest failure surface — never a fake voice. |
| `app.py` | The Gradio stage: live script view, per-actor audio playback, the draining 1,000-token memory meter, the "Forgotten to the Wood" panel, and saveable highlights. |
| `test_engine.py` | Local logic test using a deterministic stub (no model) to verify the budget is never exceeded, eviction works, and turns rotate. |

## Run locally

```bash
pip install -r requirements.txt
python app.py            # loads MiniCPM5-1B + VoxCPM2; needs a CUDA GPU (~10GB)
python test_engine.py    # engine logic only; no GPU/model required
```

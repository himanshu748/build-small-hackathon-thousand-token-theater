# 🎭 Thousand-Token Theater — Field Report

*A troupe of tiny models improvises a play you direct — and forgets it as it goes.*

Built for the **Build Small Hackathon** · track *Adventure in Thousand Token Wood*.

- **Try it (live):** https://huggingface.co/spaces/build-small-hackathon/thousand-token-theater
- **Code:** https://github.com/himanshu748/build-small-hackathon-thousand-token-theater

---

The last year of AI has been a race to make models bigger. This is the opposite bet: what if the *smallness* is the story? In Thousand-Token Theater, a troupe of small-model actors performs an improvised one-act play on a tiny lantern-lit stage — and the entire performance has to live inside a memory of just **1,000 tokens**. When it runs out, the actors forget. You direct anyway.

## A memory you can watch forget

Most apps treat a small context window as a limitation to hide. Here it's the whole show. As your play grows, the running script is measured by the model's *own* tokenizer; the moment it crosses 1,000 tokens, the oldest beats are evicted into a panel called **Forgotten to the Wood**. The actors are only ever shown what still fits — so a secret you planted ten beats ago quietly drops out of memory, a character returns with no idea who they once were, and the story drifts and reinvents itself in real time.

> The forgetting isn't a bug to apologize for. It's the drama.

## How to play

You are the **Director**. Pick a setting — a moonlit clearing in the Thousand Token Wood, a rain-slicked noir alley, a derelict starship, a royal banquet — add an optional premise, and raise the curtain. Then steer: whisper a stage direction, drop a twist, or just let the play roll on. Every line is generated and **streamed live**, word by word.

The cast:
- **Bramblewhisker** — a grandiose badger tragedian who treats every moment as a climax
- **Pip** — a tiny, anxious wren who blurts the truth at the worst time
- **Maestro Croak** — a pompous toad impresario, forever stealing the scene

## Under the hood

The whole troupe is one model — [`openbmb/MiniCPM4.1-8B`](https://huggingface.co/openbmb/MiniCPM4.1-8B) — running live on a single Hugging Face **ZeroGPU** (A10G). A small, model-agnostic engine handles the bounded memory: it builds each prompt only from what still fits in the budget, evicts the oldest beats when the tokenizer says we're over, and rotates the cast turn by turn. Lines stream via `TextIteratorStreamer`; the engine's eviction logic is **unit-tested independently of the model**, so the 1,000-token cap is real and verifiable — not a number painted on a meter.

## Why "build small"

No frontier model. No API. A single ≤8B model improvising an entire cast of distinct voices in real time, where the tightest possible constraint becomes the creative engine. The point isn't that the model is impressive *despite* being small — it's that the smallness is exactly what makes the piece feel alive.

## Honest notes

This is the text edition. A spoken-voice version using OpenBMB's VoxCPM was prototyped but pulled from the shipped build for stability — this report claims only what actually runs. Every line on stage is generated on the fly; nothing is pre-written.

---

*Built with MiniCPM on ZeroGPU + Gradio. The limit isn't hidden — it's the show.* 🍄

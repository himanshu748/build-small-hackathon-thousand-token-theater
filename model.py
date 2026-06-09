"""
The real model boundary for Thousand-Token Theater.

Only file that touches MiniCPM. Runs openbmb/MiniCPM4.1-8B on the Space's
ZeroGPU (A10G). Exposes:

    MODEL_ID
    count_tokens(text) -> int          # real MiniCPM tokenizer length (the cap)
    generate(messages) -> str          # one full chat completion (blocking)
    generate_stream(messages) -> gen   # yields cumulative text token-by-token

ZeroGPU pattern (per HF docs): the model is placed on CUDA at MODULE level
(CUDA-emulation at startup makes this work and it persists across calls);
the GPU is actually attached only inside @spaces.GPU functions, which may be
generators that yield. No mock, no fallback: failures surface.
"""

from __future__ import annotations

import re
import threading

import numpy as np
import spaces
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "openbmb/MiniCPM4.1-8B"

print(f"[theater] loading tokenizer for {MODEL_ID} ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"[theater] loading {MODEL_ID} onto GPU ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda")
model.eval()
print("[theater] model ready.", flush=True)

# Official MiniCPM4.1 no-think sampling (model card): temperature 0.7, top_p 0.95.
GEN = dict(do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.05)


def count_tokens(text: str) -> int:
    """Exact token length under MiniCPM's own tokenizer — this defines the cap."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _input_ids(messages):
    """Apply the chat template with reasoning DISABLED (snappy stage lines)."""
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)


@spaces.GPU(duration=120)
def generate(messages, max_new_tokens: int = 220) -> str:
    """One full chat completion (used by the blocking path / tests)."""
    input_ids = _input_ids(messages)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id, **GEN)
    return tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


@spaces.GPU(duration=120)
def generate_stream(messages, max_new_tokens: int = 220):
    """Generator: yields the cumulative line as MiniCPM writes it (live theatre)."""
    input_ids = _input_ids(messages)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(input_ids=input_ids, streamer=streamer, max_new_tokens=max_new_tokens,
                  pad_token_id=tokenizer.eos_token_id, **GEN)

    def _run():
        with torch.no_grad():
            model.generate(**kwargs)

    threading.Thread(target=_run, daemon=True).start()
    acc = ""
    for piece in streamer:
        acc += piece
        yield acc


# --------------------------------------------------------------------------- #
# VoxCPM — OpenBMB text-to-speech (the troupe's voices)
# --------------------------------------------------------------------------- #
TTS_MODEL_ID = "openbmb/VoxCPM-0.5B"  # ~5GB VRAM; coexists with the 8B LLM on the 24GB A10G

# Distinct "Voice Design" descriptions per character (no reference audio needed).
VOICE = {
    "The Narrator": "a warm, theatrical storyteller's voice, measured and resonant",
    "Bramblewhisker": "a deep, grandiose old male voice, booming and dramatic",
    "Pip": "a tiny, fast, breathless and anxious young voice",
    "Maestro Croak": "a pompous, nasal, self-important male voice",
}

# Pre-cache TTS weights to disk at startup (no CUDA here) so the first spoken
# line doesn't pay a download inside the GPU window.
try:
    snapshot_download(TTS_MODEL_ID)
    print("[theater] voice weights cached.", flush=True)
except Exception as e:
    print("[theater] voice pre-cache skipped:", repr(e)[:160], flush=True)

_tts = None


def _get_tts():
    """Lazy-load VoxCPM the first time we speak (inside the GPU context)."""
    global _tts
    if _tts is None:
        from voxcpm import VoxCPM
        _tts = VoxCPM.from_pretrained(TTS_MODEL_ID, load_denoiser=False, device="cuda")
    return _tts


def _voice_text(speaker: str, line: str) -> str:
    spoken = re.sub(r"\*[^*]*\*", " ", line or "")          # drop *stage actions*
    spoken = re.sub(r"\s+", " ", spoken).strip()[:240] or "..."
    desc = VOICE.get(speaker)
    return f"({desc}) {spoken}" if desc else spoken


@spaces.GPU(duration=90)
def synthesize(speaker: str, line: str):
    """Speak a line in the character's voice -> (sample_rate, float32 waveform)."""
    tts = _get_tts()
    wav = tts.generate(text=_voice_text(speaker, line), cfg_value=2.0, inference_timesteps=10)
    sr = int(getattr(getattr(tts, "tts_model", None), "sample_rate", 16000))
    return sr, np.asarray(wav, dtype=np.float32)

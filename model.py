"""
The real model boundary for Thousand-Token Theater.

Only file that touches the models. Runs on the Space's ZeroGPU (A10G, 24GB):
  - openbmb/MiniCPM3-4B  (~8GB bf16) writes the script.
  - openbmb/VoxCPM2      (~8GB)      speaks each line, a distinct voice per actor.
Both are OpenBMB models, together ~16GB → comfortable on the 24GB card. (MiniCPM4.1-8B
is richer but 16GB + VoxCPM2's 8GB would exceed the card.)

Exposes: MODEL_ID, count_tokens, generate, generate_stream, synthesize.
ZeroGPU pattern: LLM placed on cuda at module level (CUDA emulation at startup);
VoxCPM lazy-loads inside its @spaces.GPU function. No mock, no fallback.
"""

from __future__ import annotations

import re
import threading

import numpy as np
import spaces
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "openbmb/MiniCPM3-4B"

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
print("[theater] script model ready.", flush=True)

GEN = dict(do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.05)


def count_tokens(text: str) -> int:
    """Exact token length under the script model's tokenizer — this defines the cap."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _input_ids(messages):
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)


@spaces.GPU(duration=180)
def generate(messages, max_new_tokens: int = 220) -> str:
    input_ids = _input_ids(messages)
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id, **GEN)
    return tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()


@spaces.GPU(duration=180)
def generate_stream(messages, max_new_tokens: int = 220):
    """Generator: yields the cumulative line as the model writes it (live theatre)."""
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
# VoxCPM2 — OpenBMB text-to-speech (a distinct voice per character)
# --------------------------------------------------------------------------- #
TTS_MODEL_ID = "openbmb/VoxCPM2"

# "Voice Design": describe each voice in parentheses (no reference audio needed).
VOICE = {
    "The Narrator": "a warm, theatrical storyteller's voice, measured and resonant, mid-pitch",
    "Bramblewhisker": "a deep, grandiose elderly male voice, booming and slow and dramatic",
    "Pip": "a tiny, high-pitched, fast and breathless anxious young voice",
    "Maestro Croak": "a pompous, nasal, gravelly self-important middle-aged male voice",
}
# Fixed seed per character so each voice stays consistent across lines (and distinct).
SEED = {"The Narrator": 11, "Bramblewhisker": 23, "Pip": 42, "Maestro Croak": 77}

# Pre-cache TTS weights to disk at startup (no CUDA here) so the first spoken line
# doesn't pay a download inside the GPU window.
try:
    snapshot_download(TTS_MODEL_ID)
    print("[theater] voice weights cached.", flush=True)
except Exception as e:
    print("[theater] voice pre-cache skipped:", repr(e)[:160], flush=True)

_tts = None


def _get_tts():
    """Lazy-load VoxCPM2 the first time we speak (inside the GPU context)."""
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


@spaces.GPU(duration=180)
def synthesize(speaker: str, line: str):
    """Speak a line in the character's voice -> (sample_rate, float32 waveform).
    duration=180 covers the one-time VoxCPM2 load + warm-up on the very first call."""
    tts = _get_tts()
    torch.manual_seed(SEED.get(speaker, 7))   # consistent, distinct voice per character
    wav = tts.generate(text=_voice_text(speaker, line), cfg_value=2.0, inference_timesteps=10)
    sr = int(getattr(getattr(tts, "tts_model", None), "sample_rate", 16000))
    wav = np.clip(np.asarray(wav, dtype=np.float32).squeeze(), -1.0, 1.0)   # 1-D float32
    print(f"[theater] synth ok: {speaker}, {wav.shape[0]} samples @ {sr}Hz", flush=True)
    return sr, wav

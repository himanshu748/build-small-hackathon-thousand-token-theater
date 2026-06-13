"""
The real text-model boundary for Thousand-Token Theater.

Runs openbmb/MiniCPM5-1B on the Space's ZeroGPU (A10G). Exposes:
    MODEL_ID, count_tokens(text), generate(messages), generate_stream(messages)

Why MiniCPM5-1B: it is OpenBMB's current-generation *tiny* model (1B params,
llama-architecture). At ~1B it loads fast, leaves the 24GB A10G almost entirely
free for the VoxCPM2 voice model to live alongside it, and is genuinely a "tiny
titan" — a small model carrying the whole show.

ZeroGPU pattern: the model is placed on cuda at module level (CUDA emulation at
startup); the GPU is actually attached only inside @spaces.GPU functions, which
may be generators that yield. No mock, no fallback.
"""

from __future__ import annotations

import os
# VoxCPM2 (loaded in voice.py, same process) torch.compiles a submodule that
# crashes TorchDynamo on this stack ("Cannot construct ConstantVariable for
# torch.device"). Disable compilation process-wide so everything runs eager.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import threading

import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL_ID = "openbmb/MiniCPM5-1B"

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

# Official MiniCPM5 "No-Think" sampling (model card): temperature 0.7, top_p 0.95.
# Reasoning is disabled per-call via the chat template (enable_thinking=False) so
# the actors fire off snappy stage lines instead of long deliberations.
GEN = dict(do_sample=True, temperature=0.7, top_p=0.95, repetition_penalty=1.05)


def count_tokens(text: str) -> int:
    """Exact token length under MiniCPM's own tokenizer — this defines the cap."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _model_inputs(messages):
    """Tokenize chat messages into model inputs.

    transformers 5.x `apply_chat_template(return_dict=True, return_tensors="pt")`
    returns a dict (input_ids + attention_mask) — matching MiniCPM5's official
    snippet — which is then splatted into `model.generate(**inputs, ...)`.
    """
    kw = dict(tokenize=True, add_generation_prompt=True,
              return_dict=True, return_tensors="pt")
    try:
        enc = tokenizer.apply_chat_template(messages, enable_thinking=False, **kw)
    except TypeError:
        enc = tokenizer.apply_chat_template(messages, **kw)
    return enc.to(model.device)


@spaces.GPU(duration=120)
def generate(messages, max_new_tokens: int = 140) -> str:
    """One full chat completion (used by the blocking path / tests)."""
    inputs = _model_inputs(messages)
    in_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id, **GEN)
    return tokenizer.decode(out[0][in_len:], skip_special_tokens=True).strip()


@spaces.GPU(duration=120)
def generate_stream(messages, max_new_tokens: int = 140):
    """Generator: yields the cumulative line as MiniCPM writes it (live theatre)."""
    inputs = _model_inputs(messages)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new_tokens,
                  pad_token_id=tokenizer.eos_token_id, **GEN)

    def _run():
        with torch.no_grad():
            model.generate(**kwargs)

    threading.Thread(target=_run, daemon=True).start()
    acc = ""
    for piece in streamer:
        acc += piece
        yield acc

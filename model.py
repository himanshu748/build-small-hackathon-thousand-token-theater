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

import threading

import spaces
import torch
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

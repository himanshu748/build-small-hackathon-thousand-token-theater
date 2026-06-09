"""
The real model boundary for Thousand-Token Theater.

This is the ONLY file that touches MiniCPM. It runs on the Hugging Face Space's
ZeroGPU (A10G) and is imported by app.py. It exposes exactly what the engine
needs:

    MODEL_ID                       # the OpenBMB model actually loaded
    count_tokens(text) -> int      # real MiniCPM tokenizer length (for the cap)
    generate(messages) -> str      # one chat completion on GPU

There is no fallback or mock here: if the model fails to load or generate, the
app surfaces the error rather than pretending. (Local logic tests use a separate
stub and never import this module.)
"""

from __future__ import annotations

import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Flagship OpenBMB small model (<=32B): hybrid-reasoning 8B, ~16GB in bf16,
# fits the 24GB A10G. Swap to "openbmb/MiniCPM3-4B" here if you want snappier
# turns; nothing else needs to change.
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


def count_tokens(text: str) -> int:
    """Exact token length under MiniCPM's own tokenizer — this defines the cap."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _build_inputs(messages: list):
    """Apply the chat template, disabling 'thinking' for snappy stage lines."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,  # MiniCPM4.1 hybrid-reasoning toggle
        )
    except TypeError:
        # Older templates don't accept enable_thinking.
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )


@spaces.GPU(duration=120)
def generate(messages: list, max_new_tokens: int = 160,
             temperature: float = 0.9, top_p: float = 0.95) -> str:
    """Run one chat completion. `messages` is OpenAI-style role/content dicts."""
    input_ids = _build_inputs(messages).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

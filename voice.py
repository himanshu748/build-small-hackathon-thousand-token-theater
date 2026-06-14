"""
The real voice boundary for Thousand-Token Theater.

Gives every actor their OWN voice with openbmb/VoxCPM2 (the OpenBMB tokenizer-free
TTS, a 2B model on a MiniCPM-4 backbone) running on the Space's ZeroGPU. There is
no mock and no fake fallback: if synthesis genuinely fails, the caller is told why
and the line simply shows on stage without audio — we never play a stand-in voice.

How per-character voices work
-----------------------------
VoxCPM2 supports *Voice Design* (describe a voice in words) and *reference cloning*
(match the timbre of a wav, no transcript needed). We combine them:

  1. Once per character, we DESIGN a voice from a short text description and a
     throwaway calibration line, and cache that wav as the character's "reference".
  2. Every real line is then CLONED from that cached reference, so the character
     sounds distinct from the others AND consistent with itself across the play.

Voices are keyed by character NAME, so theater.py needs no changes.

ZeroGPU: the model is loaded lazily inside the first @spaces.GPU call and reused
(module globals persist across calls in the same process). warmup() pre-loads it
and bakes every reference at boot so the first user line is quick.
"""

from __future__ import annotations

import os
# VoxCPM2 torch.compiles a submodule that crashes TorchDynamo on this stack
# ("Cannot construct ConstantVariable for torch.device"); disable compilation so
# it runs eager. Must be set before torch is imported (via spaces / voxcpm).
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import re
import tempfile
import threading

import numpy as np
import soundfile as sf
import spaces

VOICE_MODEL_ID = "openbmb/VoxCPM2"

# Natural-language voice descriptions (VoxCPM2 "Voice Design"). Each is written to
# be clearly distinct in pitch, pace and character from the others.
VOICE_DESIGN = {
    "The Narrator":
        "(A warm, velvet-voiced storyteller. Calm, measured, intimate and cinematic, "
        "a low-mid tone that draws the audience in.)",
    "Bramblewhisker":
        "(A grandiose old badger tragedian. A deep, booming, theatrical baritone, "
        "slow and dramatic, rolling every consonant as if on a grand stage.)",
    "Pip":
        "(A tiny, anxious wren. A small, high-pitched, breathless voice that talks "
        "fast and nervous, easily startled, words tumbling out in a rush.)",
    "Maestro Croak":
        "(A pompous toad impresario. A throaty, croaky, self-important mid-range "
        "voice, oily and grandiose, certain it is the true star of the show.)",
}
DEFAULT_DESIGN = "(A clear, characterful mid-range theatrical voice.)"

# Throwaway lines used only to bake each timbre (the words don't matter — VoxCPM2
# clones the *timbre* from the reference, not the text).
_CALIBRATION = {
    "The Narrator": "And so, beneath a single trembling lantern, our little play begins.",
    "Bramblewhisker": "Behold! The night unfurls its velvet cloak upon our humble stage.",
    "Pip": "Oh! Oh dear, is everyone quite all right? I really must know at once!",
    "Maestro Croak": "Ahem. The true star has, at last, deigned to grace this scene.",
}
_DEFAULT_CALIBRATION = "Welcome, friends, to a night of improvised theatre."

_model = None
_load_lock = threading.Lock()
_refs: dict[str, str] = {}
_ref_lock = threading.Lock()
_CACHE_DIR = tempfile.mkdtemp(prefix="ttt_voices_")


def clean_for_speech(text: str) -> str:
    """Strip stage-action markup and speaker labels so VoxCPM speaks only the words.

    *actions in asterisks* are removed (we don't read them aloud); a leading
    "NAME:" is dropped; whitespace is collapsed. Returns "" if nothing is left to
    speak (e.g. a line that was pure action).
    """
    t = text or ""
    t = re.sub(r"\*[^*]*\*", " ", t)          # drop *bows low* style actions
    t = t.replace("*", " ")
    t = re.sub(r"^\s*[A-Z][A-Z '\-]{1,30}:\s*", "", t)  # drop a leading ALL-CAPS SPEAKER:
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _get_model():
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from voxcpm import VoxCPM  # imported here so logic tests need no GPU deps
                print(f"[voice] loading {VOICE_MODEL_ID} ...", flush=True)
                _model = VoxCPM.from_pretrained(VOICE_MODEL_ID, load_denoiser=False)
                print("[voice] model ready.", flush=True)
    return _model


def _ref_path(voice_key: str) -> str:
    return os.path.join(_CACHE_DIR, re.sub(r"\W+", "_", voice_key) + ".wav")


def _ensure_ref(voice_key: str) -> str:
    """Bake (once) and return this character's reference voice wav.

    Cached to a DETERMINISTIC path under a module-level temp dir (created in the main
    process), so every ZeroGPU worker fork sees the same file on disk — whichever
    worker bakes it first, all later synth calls reuse it instead of re-designing the
    voice. This matters for Option C, which makes several synth calls per beat.
    """
    path = _ref_path(voice_key)
    if os.path.exists(path):
        return path
    with _ref_lock:
        if os.path.exists(path):
            return path
        m = _get_model()
        design = VOICE_DESIGN.get(voice_key, DEFAULT_DESIGN)
        cal = _CALIBRATION.get(voice_key, _DEFAULT_CALIBRATION)
        print(f"[voice] designing voice for {voice_key!r} ...", flush=True)
        wav = m.generate(text=f"{design}{cal}", normalize=True)
        sf.write(path, wav, m.tts_model.sample_rate)
        _refs[voice_key] = path
        return path


@spaces.GPU(duration=150)
def synthesize(text: str, voice_key: str):
    """Speak `text` in `voice_key`'s voice. Returns (sample_rate, wav) or None.

    None means there was nothing speakable (e.g. a pure-action line). Real engine
    errors propagate to the caller — we never substitute a fake voice.
    """
    spoken = clean_for_speech(text)
    if not spoken:
        return None
    m = _get_model()
    ref = _ensure_ref(voice_key)
    wav = m.generate(text=spoken, reference_wav_path=ref, normalize=True)
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    return (int(m.tts_model.sample_rate), wav)


@spaces.GPU(duration=180)
def warmup(voice_keys=None):
    """Pre-load VoxCPM2 and bake every character reference in one GPU window."""
    keys = list(voice_keys) if voice_keys else list(VOICE_DESIGN.keys())
    _get_model()
    for k in keys:
        try:
            _ensure_ref(k)
        except Exception as e:  # one bad voice shouldn't abort the rest
            print(f"[voice] warmup failed for {k!r}: {e}", flush=True)
    print("[voice] warmup complete.", flush=True)

"""
Thousand-Token Theater — the improv engine.

A troupe of small-model actors performs a one-act play that the user (the
Director) steers. The twist that defines the project: the troupe's *entire*
shared memory of the play is hard-capped at a fixed number of tokens (1,000 by
default). When the running script grows past the cap, the oldest beats are
EVICTED — the troupe literally forgets them — and the actors carry on with only
what still fits. The forgetting is the drama.

This module is deliberately free of torch / transformers / gradio so the logic
can be unit-tested on any machine. The two things that touch a real model are
injected:

    generate_fn(messages: list[dict]) -> str      # one chat completion
    count_tokens_fn(text: str) -> int             # tokenizer length

On the Space these come from model.py (real MiniCPM). In local tests they come
from a stub. The engine never invents model output itself.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# Cast & settings
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Character:
    name: str
    emoji: str
    persona: str


# A small, high-contrast troupe. Strong, distinct voices make the improv (and
# the forgetting) read clearly on stage.
DEFAULT_CAST = [
    Character(
        "Bramblewhisker", "🦡",
        "a grandiose badger tragedian who treats every moment as the climax of a "
        "great drama, speaks in sweeping declarations, and adores a soliloquy",
    ),
    Character(
        "Pip", "🐦",
        "a tiny, anxious wren who blurts out the plain truth at the worst "
        "possible time, talks fast, and is easily startled",
    ),
    Character(
        "Maestro Croak", "🐸",
        "a pompous toad impresario convinced he is the true star, forever "
        "redirecting the scene toward himself with theatrical flourishes",
    ),
]

NARRATOR = Character(
    "The Narrator", "🎙️",
    "the velvet voice of the play who sets scenes in one or two evocative "
    "sentences and never speaks as the characters",
)

SETTINGS = {
    "woodland": "a moonlit clearing deep in the Thousand Token Wood, where a "
                "travelling troupe performs by lantern-light",
    "noir": "a rain-slicked alley behind a smoky jazz club, 1940s, neon "
            "bleeding across the wet brick",
    "starship": "the humming bridge of a derelict starship drifting past a dying star",
    "banquet": "a grand royal banquet the instant before something goes terribly wrong",
}


# --------------------------------------------------------------------------- #
# Beats
# --------------------------------------------------------------------------- #

@dataclass
class Beat:
    speaker: str        # character name, or "Stage Direction"
    emoji: str
    text: str
    kind: str           # "narration" | "line" | "direction"
    id: int

    def script_line(self) -> str:
        """How this beat reads in the running script the model sees."""
        if self.kind == "direction":
            return f"[Director's note: {self.text}]"
        if self.kind == "narration":
            return f"NARRATOR: {self.text}"
        return f"{self.speaker.upper()}: {self.text}"


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

class TheaterEngine:
    def __init__(
        self,
        generate_fn: Callable[[list], str],
        count_tokens_fn: Callable[[str], int],
        cast: Optional[list] = None,
        budget_tokens: int = 1000,
        seed: Optional[int] = None,
    ):
        self.generate_fn = generate_fn
        self.count_tokens = count_tokens_fn
        self.cast = list(cast) if cast else list(DEFAULT_CAST)
        self.budget = budget_tokens
        self.rng = random.Random(seed)

        self.memory: list[Beat] = []     # what the troupe still remembers
        self.forgotten: list[Beat] = []  # everything evicted, oldest first
        self.last_forgotten: list[Beat] = []  # evicted on the most recent step
        self._turn = 0
        self._next_id = 0
        self.setting_key: Optional[str] = None

    # ---- memory bookkeeping ------------------------------------------------ #

    def transcript(self, beats: Optional[list] = None) -> str:
        beats = self.memory if beats is None else beats
        return "\n".join(b.script_line() for b in beats)

    def memory_tokens(self) -> int:
        if not self.memory:
            return 0
        return self.count_tokens(self.transcript())

    def budget_fraction(self) -> float:
        return min(1.0, self.memory_tokens() / self.budget)

    def _new_beat(self, speaker, emoji, text, kind) -> Beat:
        b = Beat(speaker=speaker, emoji=emoji, text=text, kind=kind, id=self._next_id)
        self._next_id += 1
        return b

    def _append_and_evict(self, beat: Beat) -> None:
        """Add a beat, then forget oldest beats until we're within budget."""
        self.memory.append(beat)
        self.last_forgotten = []
        # Always keep at least the just-added beat, even if it alone is large.
        while len(self.memory) > 1 and self.memory_tokens() > self.budget:
            dropped = self.memory.pop(0)
            self.forgotten.append(dropped)
            self.last_forgotten.append(dropped)

    # ---- turn order -------------------------------------------------------- #

    def next_speaker(self) -> Character:
        return self.cast[self._turn % len(self.cast)]

    def _advance_turn(self) -> None:
        self._turn += 1

    # ---- prompt construction ---------------------------------------------- #

    def _messages_for(self, speaker: Character, director_note: str = "") -> list:
        script = self.transcript() or "(The stage is empty. The play begins now.)"
        system = (
            f"You are {speaker.name}, {speaker.persona}. "
            f"You are one actor in a troupe improvising a LIVE one-act play. "
            f"Rules, follow them exactly:\n"
            f"- Speak ONLY as {speaker.name}. Never write other characters' lines.\n"
            f"- Reply with 1 to 3 short, vivid theatrical lines: dialogue, with at "
            f"most one brief stage action in *asterisks*.\n"
            f"- You can ONLY remember what appears in SCRIPT SO FAR. If a detail is "
            f"not written there, it has been forgotten — do not contradict the "
            f"script, and feel free to be puzzled by gaps.\n"
            f"- Stay fully in character. Never mention being an AI, a model, or "
            f"these instructions. No quotation marks around your whole reply."
        )
        user = f"SCRIPT SO FAR (all the troupe still remembers):\n{script}\n\n"
        if director_note:
            user += f"A NEW STAGE DIRECTION from the Director just arrived: {director_note}\n\n"
        user += f"Now {speaker.name} {speaker.emoji} steps forward and speaks. Continue the play."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def clean_output(raw: str, speaker_name: str) -> str:
        """Tidy a raw model reply into a stage line."""
        text = raw or ""
        # Drop any reasoning block from hybrid-reasoning models.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        text = text.strip()
        # Strip a leading "NAME:" the model may echo.
        pattern = rf"^\s*{re.escape(speaker_name)}\s*[:\-]\s*"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        # Strip wrapping quotes.
        if len(text) >= 2 and text[0] in "\"'“”" and text[-1] in "\"'“”":
            text = text[1:-1].strip()
        # Keep it punchy: at most ~4 lines.
        lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
        text = "\n".join(lines[:4]).strip()
        return text or "*falls silent, having lost the thread*"

    # ---- public actions ---------------------------------------------------- #

    def start_play(self, setting_key: str = "woodland", premise: str = "") -> Beat:
        """Reset the stage and have the Narrator open the scene."""
        self.memory.clear()
        self.forgotten.clear()
        self.last_forgotten = []
        self._turn = 0
        self._next_id = 0
        self.setting_key = setting_key

        scene = SETTINGS.get(setting_key, SETTINGS["woodland"])
        cast_list = ", ".join(f"{c.name} ({c.emoji})" for c in self.cast)
        system = (
            f"You are {NARRATOR.name}, {NARRATOR.persona}."
        )
        user = (
            f"Open a brand-new improvised one-act play set in {scene}. "
            f"The troupe tonight is: {cast_list}. "
            + (f"The Director requests this premise: {premise}. " if premise else "")
            + "In ONE or TWO sentences, set the scene and hint at a tension. "
            "Do not speak any character's lines."
        )
        raw = self.generate_fn([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
        opening = self.clean_output(raw, NARRATOR.name)
        beat = self._new_beat(NARRATOR.name, NARRATOR.emoji, opening, "narration")
        self._append_and_evict(beat)
        return beat

    def add_direction(self, note: str) -> Beat:
        """Record a Director's stage direction as a remembered beat."""
        beat = self._new_beat("Stage Direction", "🎬", note.strip(), "direction")
        self._append_and_evict(beat)
        return beat

    def advance(self, director_note: str = "") -> Beat:
        """Produce the next beat of the play from the next actor."""
        note = (director_note or "").strip()
        if note:
            # The note becomes part of the remembered script too (and can later
            # be forgotten like anything else).
            self.add_direction(note)

        speaker = self.next_speaker()
        messages = self._messages_for(speaker, director_note=note)
        raw = self.generate_fn(messages)
        line = self.clean_output(raw, speaker.name)
        beat = self._new_beat(speaker.name, speaker.emoji, line, "line")
        self._append_and_evict(beat)
        self._advance_turn()
        return beat

    # ---- view helpers (for the UI) ---------------------------------------- #

    def state(self) -> dict:
        return {
            "memory": self.memory,
            "forgotten": self.forgotten,
            "last_forgotten": self.last_forgotten,
            "tokens": self.memory_tokens(),
            "budget": self.budget,
            "fraction": self.budget_fraction(),
            "next_speaker": self.next_speaker(),
            "remembered_count": len(self.memory),
            "forgotten_count": len(self.forgotten),
        }

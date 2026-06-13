"""
Thousand-Token Theater — the improv engine.

A troupe of small-model actors performs a one-act play the user (the Director)
steers. The defining twist: the troupe's ENTIRE shared memory of the play is
hard-capped at 1,000 tokens (MiniCPM's own tokenizer). When the running script
grows past the cap, the oldest beats are EVICTED — the troupe forgets them — and
the actors carry on with only what still fits. The forgetting is the drama.

No torch / transformers / gradio here, so the logic is unit-testable anywhere.
The model is injected:

    generate_fn(messages) -> str        # one blocking completion (tests/blocking path)
    count_tokens_fn(text) -> int        # tokenizer length

For live streaming, the app calls prepare_beat()/prepare_opening() to get the
chat messages, streams tokens from the model itself, then calls
commit_beat()/commit_opening() to fold the finished line into memory. The engine
never invents model output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# Cast & settings
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Character:
    name: str
    emoji: str
    persona: str


DEFAULT_CAST = [
    Character(
        "Bramblewhisker", "🦡",
        "a grandiose badger tragedian who treats every moment as the climax of a "
        "great drama, speaks in sweeping declarations, and adores a soliloquy",
    ),
    Character(
        "Pip", "🐦",
        "a tiny, anxious wren who blurts out the plain truth at the worst possible "
        "moment, talks fast, and is easily startled",
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
    "sentences and never speaks as the characters.",
)

SETTINGS = {
    "woodland": "a moonlit clearing deep in the Thousand Token Wood, where a "
                "travelling troupe performs by lantern-light",
    "noir": "a rain-slicked alley behind a smoky jazz club, 1940s, neon bleeding "
            "across the wet brick",
    "starship": "the humming bridge of a derelict starship drifting past a dying star",
    "banquet": "a grand royal banquet the instant before something goes terribly wrong",
}


# --------------------------------------------------------------------------- #
# Beats
# --------------------------------------------------------------------------- #

@dataclass
class Beat:
    speaker: str
    emoji: str
    text: str
    kind: str            # "narration" | "line" | "direction"
    id: int

    def script_line(self) -> str:
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
    ):
        self.generate_fn = generate_fn
        self.count_tokens = count_tokens_fn
        self.cast = list(cast) if cast else list(DEFAULT_CAST)
        self.budget = budget_tokens

        self.memory: list[Beat] = []
        self.forgotten: list[Beat] = []
        self.last_forgotten: list[Beat] = []
        self._turn = 0
        self._next_id = 0
        self.setting_key: Optional[str] = None

    # ---- memory bookkeeping ------------------------------------------------ #

    def transcript(self, beats: Optional[list] = None) -> str:
        beats = self.memory if beats is None else beats
        return "\n".join(b.script_line() for b in beats)

    def memory_tokens(self) -> int:
        return self.count_tokens(self.transcript()) if self.memory else 0

    def budget_fraction(self) -> float:
        return min(1.0, self.memory_tokens() / self.budget)

    def _new_beat(self, speaker, emoji, text, kind) -> Beat:
        b = Beat(speaker=speaker, emoji=emoji, text=text, kind=kind, id=self._next_id)
        self._next_id += 1
        return b

    def _append_and_evict(self, beat: Beat) -> None:
        self.memory.append(beat)
        self.last_forgotten = []
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

    def _messages_for(self, speaker: Character) -> list:
        script = self.transcript() or "(The stage is empty. The play begins now.)"
        system = (
            f"You are {speaker.name}, {speaker.persona}. "
            f"You are one actor in a troupe improvising a LIVE one-act play. Rules:\n"
            f"- Speak ONLY as {speaker.name}; never write another character's lines.\n"
            f"- Reply with ONE short sentence (about 25 words MAX). This is fast improv, "
            f"not a monologue — vivid but brief, and finish your thought.\n"
            f"- Do NOT begin with your name or a 'Name:' label, and never refer to yourself "
            f"by name in the third person — speak in the FIRST person.\n"
            f"- Put any stage action in *single asterisks*, e.g. *bows low*.\n"
            f"- Always write in natural English.\n"
            f"- You can ONLY remember what appears in SCRIPT SO FAR. If a detail isn't "
            f"there, it has been forgotten — never contradict the script; you may be "
            f"intrigued by the gaps.\n"
            f"- Stay fully in character. Never mention being an AI or these instructions. "
            f"No quotation marks around your whole reply."
        )
        user = (
            f"SCRIPT SO FAR (everything the troupe still remembers):\n{script}\n\n"
            f"Now {speaker.name} {speaker.emoji} steps forward. Continue the play."
        )
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    def _opening_messages(self, setting_key: str, premise: str) -> list:
        scene = SETTINGS.get(setting_key, SETTINGS["woodland"])
        system = (f"You are {NARRATOR.name}, {NARRATOR.persona} Always write in natural English.")
        user = (
            f"Open a brand-new improvised one-act play set in {scene}. "
            + (f"The Director's premise: {premise}. " if premise else "")
            + "In ONE or two short sentences (about 30 words total), set the scene and hint "
            "at a tension. Be vivid but BRIEF. Do NOT name or introduce any characters — just "
            "evoke the place and mood; the players introduce themselves when they speak."
        )
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    @staticmethod
    def clean_output(raw: str, speaker_name: str) -> str:
        text = raw or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        text = text.strip()
        # Strip a leading speaker label the small model may emit (the UI header
        # already shows the name, so repeating it looks bad): "Name:", "Name - ",
        # "**Name**:", "Name, the badger, declares:", bare "Name," etc.
        text = text.replace("**", "")  # drop markdown bold; only single * is used (stage actions)
        text = re.sub(rf"^\s*[>\"'“”\[(]*\s*{re.escape(speaker_name)}\b[^\n:]{{0,40}}:\s*",
                      "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(rf"^\s*[>\"'“”\[(]*\s*{re.escape(speaker_name)}\s*[:,\-–—]\s*",
                      "", text, count=1, flags=re.IGNORECASE)
        if len(text) >= 2 and text[0] in "\"'“”" and text[-1] in "\"'“”":
            text = text[1:-1].strip()
        lines = [ln for ln in (l.strip() for l in text.splitlines()) if ln]
        text = "\n".join(lines[:4]).strip()
        # If the model was cut off mid-sentence, trim back to the last completed sentence.
        if text and text[-1] not in '.!?…"\'’”*)':
            cut = max((text.rfind(c) for c in '.!?…'), default=-1)
            if cut >= 40:
                text = text[:cut + 1].strip()
        return text or "*falls silent, having lost the thread*"

    # ---- streaming-friendly API (used by the app) -------------------------- #

    def prepare_opening(self, setting_key: str = "woodland", premise: str = "") -> list:
        """Reset the stage; return the Narrator's chat messages (no generation)."""
        self.memory.clear()
        self.forgotten.clear()
        self.last_forgotten = []
        self._turn = 0
        self._next_id = 0
        self.setting_key = setting_key
        return self._opening_messages(setting_key, (premise or "").strip())

    def commit_opening(self, raw_text: str) -> Beat:
        beat = self._new_beat(NARRATOR.name, NARRATOR.emoji,
                              self.clean_output(raw_text, NARRATOR.name), "narration")
        self._append_and_evict(beat)
        return beat

    def add_direction(self, note: str) -> Beat:
        beat = self._new_beat("Stage Direction", "🎬", note.strip(), "direction")
        self._append_and_evict(beat)
        return beat

    def prepare_beat(self, director_note: str = ""):
        """Record any Director note, choose the next speaker, return (speaker, messages)."""
        note = (director_note or "").strip()
        if note:
            self.add_direction(note)
        speaker = self.next_speaker()
        return speaker, self._messages_for(speaker)

    def commit_beat(self, speaker: Character, raw_text: str) -> Beat:
        beat = self._new_beat(speaker.name, speaker.emoji,
                              self.clean_output(raw_text, speaker.name), "line")
        self._append_and_evict(beat)
        self._advance_turn()
        return beat

    # ---- blocking API (tests / non-streaming) ------------------------------ #

    def start_play(self, setting_key: str = "woodland", premise: str = "") -> Beat:
        messages = self.prepare_opening(setting_key, premise)
        return self.commit_opening(self.generate_fn(messages))

    def advance(self, director_note: str = "") -> Beat:
        speaker, messages = self.prepare_beat(director_note)
        return self.commit_beat(speaker, self.generate_fn(messages))

    # ---- view helpers ------------------------------------------------------ #

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

"""
Local logic test for the Theater engine.

This does NOT use MiniCPM. It injects a deterministic stub at the model boundary
and a heuristic token counter, purely to verify the engine's behaviour:
  - the running script stays within the token budget,
  - eviction moves the oldest beats into `forgotten`,
  - turn order rotates through the cast,
  - Director notes are recorded and themselves become forgettable.

The shipped app uses the REAL model (model.py). This stub is test-only.
"""

import re
from theater import TheaterEngine, DEFAULT_CAST

# --- stubs ----------------------------------------------------------------- #

_CANNED = [
    "Hark! The lantern gutters, and with it my courage. *throws a paw to the sky*",
    "Um, actually, I think the bridge is on fire. Just saying. Quickly.",
    "Silence! The true protagonist of this tale has, at last, arrived. *bows*",
    "*peers into the dark* Did anyone else hear that, or is it just my heart?",
    "I shall deliver a monologue so vast the stars themselves will weep.",
    "There's no time! The river's rising and I left the door unlatched!",
]
_i = {"n": 0}


def fake_generate(messages):
    """Return a deterministic in-character-ish line, ignoring real content."""
    line = _CANNED[_i["n"] % len(_CANNED)]
    _i["n"] += 1
    return line


def heuristic_tokens(text):
    # ~1.4 'tokens' per whitespace word — a stand-in for a real tokenizer.
    words = re.findall(r"\S+", text)
    return max(0, round(len(words) * 1.4))


# --- test ------------------------------------------------------------------ #

def main():
    eng = TheaterEngine(
        generate_fn=fake_generate,
        count_tokens_fn=heuristic_tokens,
        budget_tokens=120,   # small cap so eviction triggers fast in the test
        seed=7,
    )

    eng.start_play("woodland", premise="someone in the troupe is secretly a king")
    notes = ["", "", "A storm rolls in.", "", "", "Reveal Pip is the lost prince.",
             "", "", "", "The lantern dies.", "", "", "", "", ""]

    seen_speakers = set()
    for note in notes:
        beat = eng.advance(note)
        seen_speakers.add(beat.speaker)
        st = eng.state()
        assert st["tokens"] <= eng.budget, (
            f"BUDGET BROKEN: {st['tokens']} > {eng.budget}")

    st = eng.state()
    print("== final state ==")
    print("remembered beats:", st["remembered_count"], "| tokens:", st["tokens"], "/", st["budget"])
    print("forgotten beats :", st["forgotten_count"])
    print("turn rotated through:", sorted(seen_speakers))
    print("next up          :", st["next_speaker"].name)
    print()
    print("-- currently remembered --")
    print(eng.transcript())
    print()
    print("-- recently forgotten --")
    for b in st["last_forgotten"]:
        print("  forgot:", b.script_line()[:70])

    # assertions
    assert st["forgotten_count"] > 0, "nothing was ever forgotten — eviction not working"
    assert st["remembered_count"] >= 1
    assert len({c.name for c in DEFAULT_CAST} & seen_speakers) == len(DEFAULT_CAST), \
        "not all cast members got a turn"
    print("\nALL CHECKS PASSED ✅")


if __name__ == "__main__":
    main()

"""
Thousand-Token Theater — a Gradio Space for the Build Small Hackathon
(track: Adventure in Thousand Token Wood).

A troupe of small-model actors (MiniCPM4.1-8B, running on the Space's ZeroGPU)
improvises a one-act play that YOU direct. The catch that gives the piece its
name: the troupe's entire shared memory of the play is hard-capped at 1,000
tokens, measured by MiniCPM's own tokenizer. As the script grows, the oldest
beats are forgotten — and the actors carry on with only what still fits. You
watch the play remember, drift, and forget in real time.

The model is genuinely invoked (see model.py). Nothing on stage is pre-written.
"""

from __future__ import annotations

import html
import random

import gradio as gr

import model  # real MiniCPM boundary (loads on the Space's GPU)
from theater import TheaterEngine, SETTINGS

BUDGET = 1000

SETTING_LABELS = {
    "🌲 The Thousand Token Wood (woodland fable)": "woodland",
    "🌧️ Rain-soaked noir alley": "noir",
    "🚀 Derelict starship": "starship",
    "👑 A royal banquet gone wrong": "banquet",
}

TWISTS = [
    "A storm rolls in.",
    "A stranger arrives, uninvited.",
    "Someone confesses a secret they've kept for years.",
    "The lights flicker and die.",
    "An old enemy returns.",
    "A letter arrives bearing terrible news.",
    "One of you is revealed to be a liar.",
    "The ground begins to tremble.",
    "A song drifts in from somewhere offstage.",
    "Everyone suddenly forgets why they came.",
]

CHAR_COLOR = {
    "The Narrator": "#e6b15c",
    "Bramblewhisker": "#d08a4a",
    "Pip": "#7fc1d6",
    "Maestro Croak": "#8fb15a",
    "Stage Direction": "#9a8f86",
}


def new_engine() -> TheaterEngine:
    return TheaterEngine(
        generate_fn=model.generate,
        count_tokens_fn=model.count_tokens,
        budget_tokens=BUDGET,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt_text(text: str) -> str:
    safe = html.escape(text)
    # *stage action* -> italic
    out, em = [], False
    for chunk in safe.split("*"):
        out.append(f"<em>{chunk}</em>" if em and chunk else chunk)
        em = not em
    return "".join(out).replace("\n", "<br>")


def render_stage(engine: TheaterEngine) -> str:
    if engine is None or not engine.memory:
        return ("<div class='stage empty'>The stage is dark. "
                "Choose a setting and <b>raise the curtain</b>.</div>")
    rows = []
    for b in engine.memory:
        color = CHAR_COLOR.get(b.speaker, "#cfc4b8")
        if b.kind == "direction":
            rows.append(
                f"<div class='beat direction'>🎬 <span>{_fmt_text(b.text)}</span></div>"
            )
        else:
            rows.append(
                f"<div class='beat'>"
                f"<div class='who' style='color:{color}'>{b.emoji} {html.escape(b.speaker)}</div>"
                f"<div class='said'>{_fmt_text(b.text)}</div>"
                f"</div>"
            )
    return f"<div class='stage'>{''.join(rows)}</div>"


def render_meter(engine: TheaterEngine) -> str:
    tokens = engine.memory_tokens() if engine else 0
    frac = (tokens / BUDGET) if engine else 0.0
    pct = min(100, round(frac * 100))
    color = "#5fb56a" if frac < 0.7 else ("#e0a23c" if frac < 0.9 else "#d8553f")
    remembered = len(engine.memory) if engine else 0
    forgotten = len(engine.forgotten) if engine else 0
    edge = ("<div class='meter-note'>The oldest lines are slipping into the dark…</div>"
            if frac >= 0.85 else "")
    return f"""
    <div class='meter-wrap'>
      <div class='meter-top'>
        <span>🧠 Troupe memory</span>
        <span class='meter-num'>{tokens} / {BUDGET} tokens</span>
      </div>
      <div class='meter-bar'><div class='meter-fill' style='width:{pct}%;background:{color}'></div></div>
      <div class='meter-bottom'>
        <span>🎭 {remembered} beats on stage</span>
        <span>🍂 {forgotten} forgotten</span>
      </div>
      {edge}
    </div>
    """


def render_forgotten(engine: TheaterEngine) -> str:
    if engine is None or not engine.forgotten:
        return ("<div class='forgotten empty'>Nothing forgotten yet. "
                "Keep the play going — at 1,000 tokens, the troupe starts to forget.</div>")
    just = engine.last_forgotten
    blocks = []
    if just:
        items = "".join(f"<li>{_fmt_text(b.script_line())}</li>" for b in just)
        blocks.append(f"<div class='just-forgot'><div class='ff-head'>Just slipped away…</div><ul>{items}</ul></div>")
    # a faint echo of older forgotten lines
    older = engine.forgotten[:-len(just)] if just else engine.forgotten
    if older:
        tail = older[-4:]
        items = "".join(f"<li>{_fmt_text(b.script_line())}</li>" for b in tail)
        blocks.append(f"<div class='old-forgot'><div class='ff-head'>Lost earlier ({len(older)} total)</div><ul>{items}</ul></div>")
    return f"<div class='forgotten'>{''.join(blocks)}</div>"


def render_status(engine: TheaterEngine, msg: str = "") -> str:
    if engine is None:
        return msg or "Raise the curtain to begin."
    nxt = engine.next_speaker()
    base = f"**Next to speak:** {nxt.emoji} {nxt.name}"
    return f"{base} — {msg}" if msg else base


def _views(engine: TheaterEngine, msg: str = ""):
    return (
        render_stage(engine),
        render_meter(engine),
        render_forgotten(engine),
        render_status(engine, msg),
        engine,
    )


# --------------------------------------------------------------------------- #
# Event handlers
# --------------------------------------------------------------------------- #

def on_start(setting_label, premise, _engine):
    engine = new_engine()
    key = SETTING_LABELS.get(setting_label, "woodland")
    engine.start_play(key, premise=(premise or "").strip())
    return _views(engine, "The curtain rises. Press ▶ to let the troupe play on.")


def on_next(note, engine):
    if engine is None:
        return _views(None, "Raise the curtain first 🎭")
    engine.advance(note or "")
    return render_stage(engine), render_meter(engine), render_forgotten(engine), \
        render_status(engine), engine, ""  # clear the director box


def on_twist(engine):
    if engine is None:
        return _views(None, "Raise the curtain first 🎭") + ("",)
    twist = random.choice(TWISTS)
    engine.advance(twist)
    return render_stage(engine), render_meter(engine), render_forgotten(engine), \
        render_status(engine, f"Twist thrown: *{twist}*"), engine, ""


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

CSS = """
.gradio-container {max-width: 1080px !important;}
#title {text-align:center; font-family: 'Georgia', serif;}
#title h1 {font-size: 2.4rem; margin-bottom: .2rem; color:#f0e6d2;}
#title p {color:#bdae99; margin-top:0;}
.stage {background:#171310; border:1px solid #2e2722; border-radius:12px; padding:18px 20px;
        min-height:320px; max-height:460px; overflow-y:auto; font-family:'Georgia',serif;
        box-shadow: inset 0 0 60px rgba(0,0,0,.55);}
.stage.empty {display:flex; align-items:center; justify-content:center; color:#7d7064; text-align:center;}
.beat {margin:0 0 14px 0;}
.beat .who {font-weight:700; letter-spacing:.03em; font-size:.95rem; margin-bottom:2px;}
.beat .said {color:#e9ded0; line-height:1.5;}
.beat .said em {color:#c9b89f;}
.beat.direction {color:#9a8f86; font-style:italic; border-left:2px solid #3a322b; padding-left:10px; margin:10px 0;}
.meter-wrap {background:#15120f; border:1px solid #2e2722; border-radius:10px; padding:12px 14px; margin-top:8px;}
.meter-top, .meter-bottom {display:flex; justify-content:space-between; font-size:.85rem; color:#bdae99;}
.meter-num {font-variant-numeric: tabular-nums; color:#e9ded0;}
.meter-bar {height:12px; background:#241f1a; border-radius:6px; overflow:hidden; margin:8px 0;}
.meter-fill {height:100%; border-radius:6px; transition:width .4s ease;}
.meter-note {color:#d8893f; font-size:.8rem; margin-top:6px; font-style:italic;}
.forgotten {background:#13100e; border:1px dashed #36302a; border-radius:10px; padding:10px 14px; font-size:.86rem;}
.forgotten.empty {color:#6f6458; font-style:italic;}
.forgotten .ff-head {color:#a06a3a; font-weight:700; margin:4px 0; text-transform:uppercase; letter-spacing:.08em; font-size:.72rem;}
.forgotten ul {margin:2px 0 10px 0; padding-left:16px;}
.just-forgot li {color:#caa37e;}
.old-forgot li {color:#6f6458;}
"""

with gr.Blocks(title="Thousand-Token Theater") as demo:
    engine_state = gr.State(value=None)

    gr.HTML(
        "<div id='title'><h1>🎭 Thousand-Token Theater</h1>"
        "<p>A troupe of tiny MiniCPM actors improvises a play you direct — "
        "but their whole memory is capped at 1,000 tokens. What they forget becomes the story.</p></div>"
    )

    with gr.Row():
        with gr.Column(scale=3):
            stage = gr.HTML(render_stage(None))
            with gr.Row():
                director_box = gr.Textbox(
                    placeholder="Whisper a stage direction… (e.g. 'a stranger knocks') — or leave empty",
                    label="Director", scale=4, lines=1,
                )
                next_btn = gr.Button("Next beat ▶", scale=1, variant="primary")
                twist_btn = gr.Button("Twist 🎲", scale=1)
            status = gr.Markdown(render_status(None))
        with gr.Column(scale=2):
            setting = gr.Dropdown(
                choices=list(SETTING_LABELS.keys()),
                value=list(SETTING_LABELS.keys())[0],
                label="Setting",
            )
            premise = gr.Textbox(label="Premise (optional)",
                                 placeholder="e.g. someone here is secretly a king")
            start_btn = gr.Button("🎬 Raise the curtain", variant="primary")
            meter = gr.HTML(render_meter(None))
            forgotten = gr.HTML(render_forgotten(None))

    gr.Markdown(
        "Built for the **Build Small Hackathon** · *Adventure in Thousand Token Wood*. "
        "Runs **openbmb/MiniCPM4.1-8B** on ZeroGPU. The 1,000-token cap is enforced by "
        "MiniCPM's own tokenizer — the forgetting is real, not scripted."
    )

    start_btn.click(on_start, [setting, premise, engine_state],
                    [stage, meter, forgotten, status, engine_state])
    next_btn.click(on_next, [director_box, engine_state],
                   [stage, meter, forgotten, status, engine_state, director_box])
    twist_btn.click(on_twist, [engine_state],
                    [stage, meter, forgotten, status, engine_state, director_box])

if __name__ == "__main__":
    # Gradio 6: theme and css are passed to launch() (not the Blocks constructor).
    demo.queue(max_size=24).launch(theme=gr.themes.Base(), css=CSS)

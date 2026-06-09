"""
Thousand-Token Theater — Gradio 6 Space for the Build Small Hackathon
(track: Adventure in Thousand Token Wood).

A troupe of small-model actors improvises a one-act play you direct, performed
LIVE: each line is streamed onto the stage as openbmb/MiniCPM4.1-8B writes it,
then SPOKEN aloud by openbmb/VoxCPM-0.5B in a distinct voice per character. The
troupe's whole shared memory is hard-capped at 1,000 tokens (MiniCPM's own
tokenizer); past the cap the oldest beats are forgotten — and the play drifts.
Both models are genuinely invoked on ZeroGPU (see model.py); nothing is pre-written.
"""

from __future__ import annotations

import html
import random
import re

import gradio as gr

import model  # real MiniCPM + VoxCPM boundary (loads on the Space's GPU)
from theater import TheaterEngine, NARRATOR

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
    "The Narrator": "#e8c275",
    "Bramblewhisker": "#e0975a",
    "Pip": "#86c9dd",
    "Maestro Croak": "#9cc05f",
    "Stage Direction": "#a89a8c",
}


def new_engine() -> TheaterEngine:
    return TheaterEngine(generate_fn=model.generate,
                         count_tokens_fn=model.count_tokens, budget_tokens=BUDGET)


def _light_clean(text: str, name: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    t = re.sub(r"</?think>", "", t, flags=re.IGNORECASE)
    t = re.sub(rf"^\s*{re.escape(name)}\s*[:\-]\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def _speak(speaker_name: str, line: str):
    """Synthesize a spoken line; never let TTS failure break the play."""
    try:
        return model.synthesize(speaker_name, line)
    except Exception as e:
        print("[theater] TTS error:", repr(e)[:200], flush=True)
        return None


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt_text(text: str) -> str:
    safe = html.escape(text)
    out, em = [], False
    for chunk in safe.split("*"):
        out.append(f"<em>{chunk}</em>" if em and chunk else chunk)
        em = not em
    return "".join(out).replace("\n", "<br>")


def _beat_html(speaker, emoji, text, kind, live=False):
    if kind == "direction":
        return f"<div class='beat direction'>🎬 <span>{_fmt_text(text)}</span></div>"
    color = CHAR_COLOR.get(speaker, "#cfc4b8")
    cursor = "<span class='cursor'>▌</span>" if live else ""
    return (f"<div class='beat'><div class='who' style='color:{color}'>{emoji} "
            f"{html.escape(speaker)}</div><div class='said'>{_fmt_text(text)}{cursor}</div></div>")


def render_stage(engine, live=None) -> str:
    if engine is None or (not engine.memory and not live):
        return ("<div class='stage'><div class='stage-empty'>"
                "<div class='masklogo'>🎭</div>"
                "<p>The stage is dark.</p>"
                "<p class='sub'>Choose a setting and <b>raise the curtain</b>.</p>"
                "</div></div>")
    rows = [_beat_html(b.speaker, b.emoji, b.text, b.kind) for b in engine.memory]
    if live:
        rows.append(_beat_html(live[0], live[1], live[2] or "…", "line", live=True))
    return f"<div class='stage'>{''.join(rows)}</div>"


def render_meter(engine) -> str:
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
      <div class='meter-top'><span>🧠 Troupe memory</span>
        <span class='meter-num'>{tokens} / {BUDGET} tokens</span></div>
      <div class='meter-bar'><div class='meter-fill' style='width:{pct}%;background:{color}'></div></div>
      <div class='meter-bottom'><span>🎭 {remembered} on stage</span>
        <span>🍂 {forgotten} forgotten</span></div>
      {edge}
      <div class='meter-cap'>live count from the model's own tokenizer</div>
    </div>"""


def render_forgotten(engine) -> str:
    if engine is None or not engine.forgotten:
        return ("<div class='forgotten empty'>Nothing forgotten yet. Keep the play going — "
                "at 1,000 tokens the troupe starts to forget its own story.</div>")
    blocks = []
    just = engine.last_forgotten
    if just:
        items = "".join(f"<li>{_fmt_text(b.script_line())}</li>" for b in just)
        blocks.append(f"<div class='just-forgot'><div class='ff-head'>Just slipped away…</div><ul>{items}</ul></div>")
    older = engine.forgotten[:-len(just)] if just else engine.forgotten
    if older:
        items = "".join(f"<li>{_fmt_text(b.script_line())}</li>" for b in older[-4:])
        blocks.append(f"<div class='old-forgot'><div class='ff-head'>Lost earlier ({len(older)} total)</div><ul>{items}</ul></div>")
    return f"<div class='forgotten'>{''.join(blocks)}</div>"


def render_status(engine, msg: str = "") -> str:
    if engine is None:
        return msg or "Raise the curtain to begin."
    nxt = engine.next_speaker()
    base = f"**Next to speak:** {nxt.emoji} {nxt.name}"
    return f"{base} — {msg}" if msg else base


# --------------------------------------------------------------------------- #
# Event handlers (streaming generators; outputs end with the spoken-audio slot)
# --------------------------------------------------------------------------- #

def on_start(setting_label, premise, _engine):
    engine = new_engine()
    key = SETTING_LABELS.get(setting_label, "woodland")
    messages = engine.prepare_opening(key, premise=(premise or "").strip())
    partial = ""
    for partial in model.generate_stream(messages):
        live = (NARRATOR.name, NARRATOR.emoji, _light_clean(partial, NARRATOR.name))
        yield (render_stage(engine, live=live), render_meter(engine),
               render_forgotten(engine), "🎙️ The Narrator sets the scene…", engine, None)
    beat = engine.commit_opening(partial)
    yield (render_stage(engine), render_meter(engine), render_forgotten(engine),
           render_status(engine, "The curtain rises. Press ▶ to let the troupe play on."),
           engine, _speak(NARRATOR.name, beat.text))


def _run_beat(engine, note, status_suffix):
    speaker, messages = engine.prepare_beat(note)
    partial = ""
    for partial in model.generate_stream(messages):
        live = (speaker.name, speaker.emoji, _light_clean(partial, speaker.name))
        yield (render_stage(engine, live=live), render_meter(engine),
               render_forgotten(engine), f"{speaker.emoji} **{speaker.name}** is speaking…",
               engine, "", None)
    beat = engine.commit_beat(speaker, partial)
    yield (render_stage(engine), render_meter(engine), render_forgotten(engine),
           render_status(engine, status_suffix), engine, "", _speak(speaker.name, beat.text))


def on_next(note, engine):
    if engine is None:
        yield (render_stage(None), render_meter(None), render_forgotten(None),
               "Raise the curtain first 🎭", None, "", None)
        return
    yield from _run_beat(engine, note or "", "")


def on_twist(engine):
    if engine is None:
        yield (render_stage(None), render_meter(None), render_forgotten(None),
               "Raise the curtain first 🎭", None, "", None)
        return
    twist = random.choice(TWISTS)
    yield from _run_beat(engine, twist, f"Twist thrown: *{twist}*")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=EB+Garamond:ital@0;1&display=swap');
.gradio-container {max-width: 1100px !important;
  background:
    radial-gradient(1200px 500px at 50% -10%, rgba(224,151,90,0.10), transparent 60%),
    radial-gradient(900px 600px at 50% 120%, rgba(120,60,30,0.16), transparent 60%),
    #120e0c !important;}
#playbill {text-align:center; padding:18px 0 6px; font-family:'Playfair Display',Georgia,serif;}
#playbill .ribbon {color:#caa46a; letter-spacing:.42em; font-size:.72rem; text-transform:uppercase; font-family:'EB Garamond',serif;}
#playbill h1 {font-size:2.9rem; font-weight:800; margin:.12em 0 .05em; color:#f3e7cf;
  text-shadow:0 2px 18px rgba(224,151,90,.25); letter-spacing:.01em;}
#playbill .rule {width:220px; height:1px; margin:8px auto 10px;
  background:linear-gradient(90deg,transparent,#caa46a,transparent);}
#playbill p.tag {color:#c3b39a; font-family:'EB Garamond',serif; font-size:1.02rem; max-width:780px; margin:0 auto;}
.stage {background:
    linear-gradient(180deg, rgba(0,0,0,.35), rgba(0,0,0,.0) 18%),
    radial-gradient(120% 80% at 50% 120%, rgba(232,165,90,.10), transparent 55%),
    #15100d;
  border:1px solid #34291f; border-radius:14px; padding:22px 24px;
  min-height:360px; max-height:500px; overflow-y:auto;
  font-family:'EB Garamond',Georgia,serif; font-size:1.08rem;
  box-shadow:inset 0 0 80px rgba(0,0,0,.6), 0 8px 30px rgba(0,0,0,.35);}
.stage-empty {height:320px; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#7d7064; text-align:center;}
.stage-empty .masklogo {font-size:3rem; opacity:.7; margin-bottom:.3em;}
.stage-empty .sub {font-size:.95rem;}
.beat {margin:0 0 15px 0; animation:rise .35s ease;}
@keyframes rise {from{opacity:0; transform:translateY(4px)} to{opacity:1; transform:none}}
.beat .who {font-family:'Playfair Display',serif; font-weight:700; letter-spacing:.02em; font-size:.98rem; margin-bottom:2px;}
.beat .said {color:#ece1cf; line-height:1.6;}
.beat .said em {color:#d8c4a3;}
.beat.direction {color:#a89a8c; font-style:italic; border-left:2px solid #4a3c2c; padding-left:12px; margin:12px 0;}
.cursor {color:#e8c275; font-weight:700; animation:blink 1s steps(2) infinite; margin-left:1px;}
@keyframes blink {0%,50%{opacity:1} 51%,100%{opacity:0}}
.meter-wrap {background:#171210; border:1px solid #34291f; border-radius:12px; padding:13px 15px; margin-top:6px; font-family:'EB Garamond',serif;}
.meter-top,.meter-bottom {display:flex; justify-content:space-between; font-size:.86rem; color:#c3b39a;}
.meter-num {font-variant-numeric:tabular-nums; color:#f0e3cd;}
.meter-bar {height:13px; background:#251c16; border-radius:7px; overflow:hidden; margin:8px 0; box-shadow:inset 0 1px 3px rgba(0,0,0,.5);}
.meter-fill {height:100%; border-radius:7px; transition:width .45s ease;}
.meter-note {color:#e0903c; font-size:.82rem; margin-top:6px; font-style:italic;}
.meter-cap {color:#7d7064; font-size:.72rem; margin-top:6px; font-style:italic; text-align:right;}
#director-input textarea {min-height:74px !important;}
.forgotten {background:#140f0d; border:1px dashed #3c3026; border-radius:12px; padding:11px 15px; font-size:.88rem; font-family:'EB Garamond',serif;}
.forgotten.empty {color:#6f6458; font-style:italic;}
.forgotten .ff-head {color:#b0703a; font-weight:700; margin:5px 0; text-transform:uppercase; letter-spacing:.1em; font-size:.72rem; font-family:'Playfair Display',serif;}
.forgotten ul {margin:2px 0 10px 0; padding-left:16px;}
.just-forgot li {color:#d8b88c;}
.old-forgot li {color:#6f6458;}
#howto {color:#8f8273; font-family:'EB Garamond',serif; font-size:.92rem; text-align:center; margin-top:4px;}
.director-box {border:1px solid #5a4326 !important; background:#1a130e !important; border-radius:14px !important;
  padding:14px 16px 12px !important; margin-top:14px !important;
  box-shadow:0 0 0 1px rgba(232,165,90,.16), 0 6px 22px rgba(0,0,0,.35) !important;}
.director-head {font-family:'Playfair Display',serif; color:#f0c97a; font-size:1.14rem; font-weight:700; letter-spacing:.02em;}
.director-sub {font-family:'EB Garamond',serif; color:#a99a86; font-size:.92rem; margin:1px 0 10px;}
#director-input textarea, #director-input input {font-size:1.05rem !important; background:#120d0a !important;
  border:1px solid #5a4326 !important; color:#f0e3cd !important;}
"""

AUTOSCROLL_JS = """
() => {
  const mount = document.querySelector('#stage-html');
  if (!mount || mount.dataset.bound) return;
  mount.dataset.bound = '1';
  const obs = new MutationObserver(() => {
    const s = mount.querySelector('.stage');
    if (s) s.scrollTop = s.scrollHeight;
  });
  obs.observe(mount, {childList: true, subtree: true, characterData: true});
}
"""

with gr.Blocks(title="Thousand-Token Theater") as demo:
    engine_state = gr.State(value=None)

    gr.HTML(
        "<div id='playbill'>"
        "<div class='ribbon'>Build Small Hackathon · Thousand Token Wood</div>"
        "<h1>🎭 Thousand-Token Theater</h1>"
        "<div class='rule'></div>"
        "<p class='tag'>A troupe of tiny <b>MiniCPM</b> actors improvises a play you direct — "
        "performed live and <b>spoken aloud</b> by <b>VoxCPM</b>. But their entire memory is "
        "capped at <b>1,000 tokens</b>, so the story you build slowly drifts and forgets itself.</p>"
        "</div>"
    )

    with gr.Row():
        with gr.Column(scale=3):
            stage = gr.HTML(render_stage(None), elem_id="stage-html")
            with gr.Group(elem_classes="director-box"):
                gr.HTML("<div class='director-head'>🎬 You are the Director</div>"
                        "<div class='director-sub'>Whisper a stage direction — a line, an entrance, a reveal — "
                        "and the troupe obeys. Tap an idea below, or just let the play roll on.</div>")
                director_box = gr.Textbox(
                    placeholder="e.g. 'a hooded stranger steps into the lantern-light and slowly lowers their hood'…",
                    show_label=False, container=False, lines=3, elem_id="director-input",
                )
                with gr.Row():
                    ex1 = gr.Button("🌩️ A storm breaks", size="sm", variant="secondary")
                    ex2 = gr.Button("🗡️ An enemy returns", size="sm", variant="secondary")
                    ex3 = gr.Button("🤫 Someone confesses", size="sm", variant="secondary")
                    ex4 = gr.Button("👑 Reveal the secret", size="sm", variant="secondary")
                with gr.Row():
                    next_btn = gr.Button("Play it ▶", scale=2, variant="primary")
                    twist_btn = gr.Button("Surprise me 🎲", scale=1)
            status = gr.Markdown(render_status(None))
            tts_audio = gr.Audio(label="🔊 The troupe speaks (VoxCPM)",
                                 autoplay=True, interactive=False, elem_id="tts-audio")
        with gr.Column(scale=2):
            setting = gr.Dropdown(choices=list(SETTING_LABELS.keys()),
                                  value=list(SETTING_LABELS.keys())[0], label="Setting")
            premise = gr.Textbox(label="Premise (optional)",
                                 placeholder="e.g. someone here is secretly a king")
            start_btn = gr.Button("🎬 Raise the curtain", variant="primary", size="lg")
            meter = gr.HTML(render_meter(None))
            forgotten = gr.HTML(render_forgotten(None))

    gr.HTML(
        "<div id='howto'>Runs <b>openbmb/MiniCPM3-4B</b> (script) + <b>openbmb/VoxCPM2</b> "
        "(voices) live on ZeroGPU. The 1,000-token cap is enforced by the model's own tokenizer — "
        "the forgetting is real, not scripted.</div>"
    )

    start_btn.click(on_start, [setting, premise, engine_state],
                    [stage, meter, forgotten, status, engine_state, tts_audio])
    next_btn.click(on_next, [director_box, engine_state],
                   [stage, meter, forgotten, status, engine_state, director_box, tts_audio])
    twist_btn.click(on_twist, [engine_state],
                    [stage, meter, forgotten, status, engine_state, director_box, tts_audio])
    ex1.click(lambda: "A sudden storm breaks over the scene.", None, director_box)
    ex2.click(lambda: "An old enemy strides in from the shadows.", None, director_box)
    ex3.click(lambda: "One of you confesses a long-held secret.", None, director_box)
    ex4.click(lambda: "Reveal who has secretly been hiding the truth.", None, director_box)
    demo.load(None, None, None, js=AUTOSCROLL_JS)

if __name__ == "__main__":
    # Gradio 6: theme and css go to launch(), not the Blocks constructor.
    demo.queue(max_size=24).launch(
        theme=gr.themes.Base(primary_hue="amber", neutral_hue="stone"), css=CSS)

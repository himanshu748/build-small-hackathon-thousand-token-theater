"""
Thousand-Token Theater — Gradio 6 Space for the Build Small Hackathon
(track: Adventure in Thousand Token Wood).

A troupe of tiny-model actors (openbmb/MiniCPM5-1B on ZeroGPU) improvises a
one-act play you direct. Each line is streamed live onto the stage as the model
writes it — and then SPOKEN ALOUD in that character's own voice (openbmb/VoxCPM2).
The troupe's whole shared memory is hard-capped at 1,000 tokens (MiniCPM's own
tokenizer); past the cap the oldest beats are forgotten and the play drifts. The
models are genuinely invoked (see model.py / voice.py); nothing is pre-written and
no voice is faked.

New in this build: every actor has a distinct voice, the memory meter visibly
DRAINS as the scene fills, and you can save favourite lines or whole scenes — they
survive even after the troupe forgets them.
"""

from __future__ import annotations

import os
# Disable torch.compile/Dynamo process-wide BEFORE torch is imported (via model/voice):
# VoxCPM2 torch.compiles a submodule that otherwise crashes TorchDynamo on this stack.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import html
import random
import re
import tempfile
import threading

import gradio as gr

import model  # real MiniCPM text boundary (loads on the Space's GPU)
import voice  # real VoxCPM2 voice boundary (loads on the Space's GPU)
from theater import TheaterEngine, NARRATOR, DEFAULT_CAST

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
    # Strip a leading "Name:" / "Name, the badger says:" / "**Name**:" label so the
    # character's name isn't duplicated next to the stage header while streaming.
    t = t.replace("**", "")  # drop markdown bold; only single * is used (stage actions)
    t = re.sub(rf"^\s*[>\"'“”\[(]*\s*{re.escape(name)}\b[^\n:]{{0,40}}:\s*", "", t, count=1, flags=re.IGNORECASE)
    t = re.sub(rf"^\s*[>\"'“”\[(]*\s*{re.escape(name)}\s*[:,\-–—]\s*", "", t, count=1, flags=re.IGNORECASE)
    return t.strip()


# --------------------------------------------------------------------------- #
# Performance: reveal a line sentence-by-sentence, speaking each as it appears
# --------------------------------------------------------------------------- #

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _split_sentences(text: str):
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    return parts or [text]


def _speak_sentences(speaker_name: str, text: str, voices_on: bool):
    """Yield (audio_value, voice_note) for each speakable sentence, autoplayed in order.

    voice.synthesize is a regular @spaces.GPU call that the warm-up never touches, so
    it gets a clean worker and runs reliably from the handler. We call it once per
    sentence so the voice arrives in chunks (each autoplays as it's ready) instead of
    one long clip after the line. On a real synth error we surface an honest note and
    keep going — never a fake voice. Pure-action sentences (no speakable words) are
    skipped.
    """
    if not voices_on:
        return
    vnote = ""
    for sent in _split_sentences(text):
        clip = None
        try:
            clip = voice.synthesize(sent, speaker_name)
        except Exception as e:  # noqa: BLE001 — report, never fake
            vnote = f" · 🔇 voice unavailable ({str(e)[:60]})"
        if clip is not None:
            yield clip, vnote


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
    """The memory meter, reframed as a DRAINING reservoir of working memory.

    The bar shows how much of the 1,000-token memory is still free. As the play
    grows it drains toward empty; once empty the troupe is at capacity and every
    new line buries an older one — that is the moment the forgetting begins.
    """
    tokens = engine.memory_tokens() if engine else 0
    used = min(BUDGET, tokens)
    remaining = max(0, BUDGET - used)
    rem_pct = max(0, min(100, round(remaining / BUDGET * 100)))
    # Colour by how much memory is LEFT: plenty = green, low = amber, gone = red.
    color = "#5fb56a" if rem_pct > 40 else ("#e0a23c" if rem_pct > 12 else "#d8553f")
    remembered = len(engine.memory) if engine else 0
    forgotten = len(engine.forgotten) if engine else 0
    just = len(engine.last_forgotten) if engine else 0
    at_capacity = bool(engine) and (tokens >= BUDGET * 0.99 or forgotten > 0)
    drain_class = " draining" if at_capacity else ""
    drip = f"<span class='drip'>−{just} just slipped away</span>" if just else ""
    note = ("<div class='meter-note'>Memory's run dry — the troupe is at capacity, and every "
            "new line now buries an older one.</div>" if at_capacity else "")
    return f"""
    <div class='meter-wrap{drain_class}'>
      <div class='meter-top'><span>🧠 Troupe memory <span class='mfree'>· draining</span></span>
        <span class='meter-num'>{remaining} free · {used}/{BUDGET}</span></div>
      <div class='meter-bar'><div class='meter-fill' style='width:{rem_pct}%;background:{color}'></div></div>
      <div class='meter-bottom'><span>🎭 {remembered} on stage</span>
        <span>🍂 {forgotten} forgotten {drip}</span></div>
      {note}
      <div class='meter-cap'>live count from the model's own tokenizer — when it drains to zero, the forgetting begins</div>
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
# Saved highlights
# --------------------------------------------------------------------------- #

def render_saved(saved) -> str:
    if not saved:
        return ("<div class='saved empty'>No highlights yet. When a line lands — or a whole "
                "scene sings — save it here. Saved moments stay even after the troupe forgets them.</div>")
    items = []
    for it in reversed(saved[-12:]):
        head = html.escape(it["title"])
        body = _fmt_text(it["text"])
        cls = "scene" if it["type"] == "scene" else "line"
        glyph = "🎬" if it["type"] == "scene" else "⭐"
        items.append(f"<div class='saved-item {cls}'><div class='si-head'>{glyph} {head}</div>"
                     f"<div class='si-body'>{body}</div></div>")
    return f"<div class='saved'>{''.join(items)}</div>"


def _export_saved(saved):
    """Write saved highlights to a downloadable text file; return its path or None."""
    if not saved:
        return None
    lines = ["Thousand-Token Theater — saved highlights", "=" * 42, ""]
    for it in saved:
        if it["type"] == "scene":
            lines += [f"[SCENE] {it['title']}", it["text"], ""]
        else:
            lines += [f"[LINE]  {it['title']}: {it['text']}"]
    path = os.path.join(tempfile.gettempdir(), "thousand_token_theater_highlights.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _last_spoken_beat(engine):
    if not engine or not engine.memory:
        return None
    for b in reversed(engine.memory):
        if b.kind in ("line", "narration"):
            return b
    return engine.memory[-1]


def on_save_line(engine, saved):
    saved = list(saved or [])
    beat = _last_spoken_beat(engine)
    if beat is None:
        msg = "Nothing to save yet — let the troupe speak first."
    else:
        saved.append({"type": "line", "title": f"{beat.emoji} {beat.speaker}",
                      "text": beat.text})
        msg = f"Saved {beat.speaker}'s line ⭐"
    return (render_saved(saved), saved,
            gr.update(value=_export_saved(saved), visible=bool(saved)),
            render_status(engine, msg))


def on_save_scene(engine, saved):
    saved = list(saved or [])
    if not engine or not engine.memory:
        msg = "Nothing to save yet — raise the curtain first."
    else:
        script = engine.transcript()
        n = sum(1 for it in saved if it["type"] == "scene") + 1
        saved.append({"type": "scene", "title": f"Scene {n} · {len(engine.memory)} beats on stage",
                      "text": script})
        msg = "Saved this scene 🎬"
    return (render_saved(saved), saved,
            gr.update(value=_export_saved(saved), visible=bool(saved)),
            render_status(engine, msg))


def on_clear_saved(_saved):
    return (render_saved([]), [], gr.update(value=None, visible=False),
            "Cleared your saved highlights.")


# --------------------------------------------------------------------------- #
# Event handlers (streaming generators)
# --------------------------------------------------------------------------- #

def on_start(setting_label, premise, voices_on, _engine):
    engine = new_engine()
    key = SETTING_LABELS.get(setting_label, "woodland")
    messages = engine.prepare_opening(key, premise=(premise or "").strip())
    partial = ""
    for partial in model.generate_stream(messages):
        live = (NARRATOR.name, NARRATOR.emoji, _light_clean(partial, NARRATOR.name))
        yield (render_stage(engine, live=live), render_meter(engine),
               render_forgotten(engine), "🎙️ The Narrator sets the scene…", engine, gr.update())
    beat = engine.commit_opening(partial)
    vnote = ""
    for audio, vnote in _speak_sentences(NARRATOR.name, beat.text, voices_on):
        yield (render_stage(engine), render_meter(engine), render_forgotten(engine),
               "🎙️ The Narrator speaks…" + vnote, engine, audio)
    yield (render_stage(engine), render_meter(engine), render_forgotten(engine),
           render_status(engine, "The curtain rises. Press ▶ to let the troupe play on." + vnote),
           engine, gr.update())


def _run_beat(engine, note, voices_on, status_suffix):
    speaker, messages = engine.prepare_beat(note)
    partial = ""
    for partial in model.generate_stream(messages):
        live = (speaker.name, speaker.emoji, _light_clean(partial, speaker.name))
        yield (render_stage(engine, live=live), render_meter(engine),
               render_forgotten(engine), f"{speaker.emoji} **{speaker.name}** is speaking…",
               engine, "", gr.update())
    beat = engine.commit_beat(speaker, partial)
    vnote = ""
    for audio, vnote in _speak_sentences(speaker.name, beat.text, voices_on):
        yield (render_stage(engine), render_meter(engine), render_forgotten(engine),
               f"🎙️ {speaker.name} speaks…" + vnote, engine, "", audio)
    yield (render_stage(engine), render_meter(engine), render_forgotten(engine),
           render_status(engine, (status_suffix + vnote).strip(" ·")), engine, "", gr.update())


def on_next(note, voices_on, engine):
    if engine is None:
        yield (render_stage(None), render_meter(None), render_forgotten(None),
               "Raise the curtain first 🎭", None, "", None)
        return
    yield from _run_beat(engine, note or "", voices_on, "")


def on_twist(voices_on, engine):
    if engine is None:
        yield (render_stage(None), render_meter(None), render_forgotten(None),
               "Raise the curtain first 🎭", None, "", None)
        return
    twist = random.choice(TWISTS)
    yield from _run_beat(engine, twist, voices_on, f"Twist thrown: *{twist}*")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=EB+Garamond:ital@0;1&display=swap');
/* Dark theater background fills the WHOLE Space frame (was only on .gradio-container,
   which left the area outside the 1100px shell white in light browser theme). */
html, body, gradio-app {min-height:100vh; margin:0; color-scheme:dark;
  background:
    radial-gradient(1200px 500px at 50% -10%, rgba(224,151,90,0.10), transparent 60%),
    radial-gradient(900px 600px at 50% 120%, rgba(120,60,30,0.16), transparent 60%),
    #120e0c !important;}
gradio-app {display:block;}
/* Responsive, centered stage: use the available width on desktop instead of a fixed 1100px. */
.gradio-container {max-width: 1440px !important; width:100% !important; margin:0 auto !important;
  background:transparent !important;}
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
.stage-empty {height:320px; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#9a8b78; text-align:center;}
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
.meter-top,.meter-bottom {display:flex; justify-content:space-between; font-size:.86rem; color:#d8c9b2;}
.meter-num {font-variant-numeric:tabular-nums; color:#f0e3cd;}
.mfree {color:#9bbf86; font-style:italic; letter-spacing:.02em;}
.meter-bar {height:13px; background:#251c16; border-radius:7px; overflow:hidden; margin:8px 0; box-shadow:inset 0 1px 3px rgba(0,0,0,.5);}
.meter-fill {height:100%; border-radius:7px; transition:width .5s ease, background .5s ease;}
.meter-wrap.draining {border-color:#5a2f23; box-shadow:0 0 0 1px rgba(216,85,63,.25);}
.meter-wrap.draining .meter-bar {animation:shudder 1.6s ease-in-out infinite;}
@keyframes shudder {0%,100%{transform:translateX(0)} 25%{transform:translateX(-1px)} 75%{transform:translateX(1px)}}
.drip {color:#e0903c; font-style:italic; margin-left:6px; animation:fade 2s ease;}
@keyframes fade {from{opacity:0} 30%{opacity:1} to{opacity:.55}}
.meter-note {color:#e0903c; font-size:.82rem; margin-top:6px; font-style:italic;}
.meter-cap {color:#b3a48f; font-size:.72rem; margin-top:6px; font-style:italic; text-align:right;}
.forgotten {background:#140f0d; border:1px dashed #3c3026; border-radius:12px; padding:11px 15px; font-size:.88rem; font-family:'EB Garamond',serif;}
.forgotten.empty {color:#9a8b78; font-style:italic;}
.forgotten .ff-head {color:#b0703a; font-weight:700; margin:5px 0; text-transform:uppercase; letter-spacing:.1em; font-size:.72rem; font-family:'Playfair Display',serif;}
.forgotten ul {margin:2px 0 10px 0; padding-left:16px;}
.just-forgot li {color:#d8b88c;}
.old-forgot li {color:#9a8b78;}
/* Saved highlights */
.saved {background:#140f0d; border:1px solid #3c3026; border-radius:12px; padding:8px 14px; font-size:.9rem; font-family:'EB Garamond',serif; max-height:230px; overflow-y:auto;}
.saved.empty {color:#9a8b78; font-style:italic; border-style:dashed;}
.saved-item {padding:8px 0; border-bottom:1px solid #28201a;}
.saved-item:last-child {border-bottom:none;}
.si-head {color:#f0c97a; font-family:'Playfair Display',serif; font-weight:700; font-size:.84rem; margin-bottom:2px;}
.saved-item.scene .si-body {color:#cdbfa8; white-space:pre-wrap; font-size:.82rem; max-height:120px; overflow:auto;}
.saved-item.line .si-body {color:#ece1cf;}
#voices-row {margin-top:8px;}
#voice-audio {margin-top:10px;}
#howto {color:#b3a48f; font-family:'EB Garamond',serif; font-size:.92rem; text-align:center; margin-top:4px;}
.director-box {border:1px solid #5a4326 !important; background:#1a130e !important; border-radius:14px !important;
  padding:14px 16px 12px !important; margin-top:14px !important;
  box-shadow:0 0 0 1px rgba(232,165,90,.16), 0 6px 22px rgba(0,0,0,.35) !important;}
.director-head {font-family:'Playfair Display',serif; color:#f0c97a; font-size:1.14rem; font-weight:700; letter-spacing:.02em;}
.director-sub {font-family:'EB Garamond',serif; color:#c2b49e; font-size:.92rem; margin:1px 0 10px;}
#director-input textarea, #director-input input {font-size:1.05rem !important; background:#120d0a !important;
  border:1px solid #5a4326 !important; color:#f0e3cd !important;}
#director-input textarea {min-height:74px !important;}
#director-input textarea::placeholder {color:#8c8070 !important;}

/* --- Readability: keep NATIVE Gradio components legible on the dark stage in BOTH
   light and dark browser themes. These scope Gradio's own theme variables to the
   affected components, so they no longer flip to dark-on-dark in light theme. --- */
#status-line {background:#171210; border:1px solid #34291f; border-radius:10px;
  padding:8px 13px; margin-top:10px; font-family:'EB Garamond',serif;}
#status-line, #status-line * {color:#ece1cf !important;}
#status-line strong {color:#f0c97a !important;}
#setting-dd, #premise-tb, #voices-row {
  --block-background-fill:transparent; --block-border-color:transparent;
  --input-background-fill:#120d0a; --input-border-color:#5a4326;
  --body-text-color:#f0e3cd; --body-text-color-subdued:#c3b39a;
  --block-title-text-color:#f0e3cd; --block-info-text-color:#c3b39a;
  --checkbox-label-text-color:#f0e3cd;}
#premise-tb textarea::placeholder, #premise-tb input::placeholder {color:#8c8070 !important;}

/* Responsive: let the shell go full-width and the stage grow on narrow screens. */
@media (max-width: 860px) {
  .gradio-container {max-width:100% !important;}
  .stage {max-height:none;}
}
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
    saved_state = gr.State(value=[])

    gr.HTML(
        "<div id='playbill'>"
        "<div class='ribbon'>Build Small Hackathon · Thousand Token Wood</div>"
        "<h1>🎭 Thousand-Token Theater</h1>"
        "<div class='rule'></div>"
        "<p class='tag'>A troupe of tiny <b>MiniCPM</b> actors improvises a play you direct — "
        "performed live, line by line, each in <b>their own voice</b>. But their entire memory "
        "is capped at <b>1,000 tokens</b>, so the story you build slowly drifts and forgets itself.</p>"
        "</div>"
    )

    with gr.Row():
        with gr.Column(scale=3):
            stage = gr.HTML(render_stage(None), elem_id="stage-html")
            voice_audio = gr.Audio(label="🔊 Now speaking", autoplay=True, interactive=False,
                                   elem_id="voice-audio")
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
            status = gr.Markdown(render_status(None), elem_id="status-line")
        with gr.Column(scale=2):
            setting = gr.Dropdown(choices=list(SETTING_LABELS.keys()),
                                  value=list(SETTING_LABELS.keys())[0], label="Setting",
                                  elem_id="setting-dd")
            premise = gr.Textbox(label="Premise (optional)",
                                 placeholder="e.g. someone here is secretly a king",
                                 elem_id="premise-tb")
            with gr.Row(elem_id="voices-row"):
                voices_on = gr.Checkbox(value=True, label="🔊 Voices — let each actor speak")
            start_btn = gr.Button("🎬 Raise the curtain", variant="primary", size="lg")
            meter = gr.HTML(render_meter(None))
            forgotten = gr.HTML(render_forgotten(None))
            gr.HTML("<div class='director-head' style='font-size:1.02rem;margin-top:6px'>⭐ Saved highlights</div>"
                    "<div class='director-sub'>Bookmark a favourite line or a whole scene — it survives even "
                    "after the troupe forgets it.</div>")
            with gr.Row():
                save_line_btn = gr.Button("⭐ Save last line", size="sm", variant="secondary")
                save_scene_btn = gr.Button("🎬 Save this scene", size="sm", variant="secondary")
                clear_saved_btn = gr.Button("Clear", size="sm", variant="secondary")
            saved_panel = gr.HTML(render_saved([]))
            saved_file = gr.File(label="⬇️ Download highlights", visible=False, interactive=False)

    gr.HTML(
        "<div id='howto'>Runs <b>openbmb/MiniCPM5-1B</b> (the actors) and <b>openbmb/VoxCPM2</b> "
        "(their voices) live on ZeroGPU. The 1,000-token cap is enforced by the model's own "
        "tokenizer — the forgetting is real, not scripted.</div>"
    )

    start_btn.click(on_start, [setting, premise, voices_on, engine_state],
                    [stage, meter, forgotten, status, engine_state, voice_audio])
    next_btn.click(on_next, [director_box, voices_on, engine_state],
                   [stage, meter, forgotten, status, engine_state, director_box, voice_audio])
    twist_btn.click(on_twist, [voices_on, engine_state],
                    [stage, meter, forgotten, status, engine_state, director_box, voice_audio])
    ex1.click(lambda: "A sudden storm breaks over the scene.", None, director_box)
    ex2.click(lambda: "An old enemy strides in from the shadows.", None, director_box)
    ex3.click(lambda: "One of you confesses a long-held secret.", None, director_box)
    ex4.click(lambda: "Reveal who has secretly been hiding the truth.", None, director_box)

    save_line_btn.click(on_save_line, [engine_state, saved_state],
                        [saved_panel, saved_state, saved_file, status])
    save_scene_btn.click(on_save_scene, [engine_state, saved_state],
                         [saved_panel, saved_state, saved_file, status])
    clear_saved_btn.click(on_clear_saved, [saved_state],
                          [saved_panel, saved_state, saved_file, status])

    demo.load(None, None, None, js=AUTOSCROLL_JS)


def _boot_warmup():
    """Materialize both models at boot so the first user line is quick (best-effort on ZeroGPU)."""
    try:
        model.generate([{"role": "user", "content": "Say the single word: ready."}], max_new_tokens=6)
        print("[warmup] text model warm.", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[warmup] text model: {e}", flush=True)
    try:
        voice.warmup([NARRATOR.name] + [c.name for c in DEFAULT_CAST])
    except Exception as e:  # noqa: BLE001
        print(f"[warmup] voice: {e}", flush=True)


if __name__ == "__main__":
    # Warm the GPU models in the background while Gradio starts serving.
    threading.Thread(target=_boot_warmup, daemon=True).start()
    # Gradio 6: theme and css go to launch(), not the Blocks constructor.
    demo.queue(max_size=24).launch(
        theme=gr.themes.Base(primary_hue="amber", neutral_hue="stone"), css=CSS)

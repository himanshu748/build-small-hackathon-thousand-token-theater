# Field Notes — Thousand-Token Theater

Thousand-Token Theater is a small-model storytelling demo built around one constraint: the troupe can only remember 1,000 tokens of the play at a time.

Instead of hiding the model's context limit, the app turns it into the central mechanic. A MiniCPM-powered cast improvises a scene live, the memory meter counts the remembered script with the model's own tokenizer, and older story beats fall into the **Forgotten** panel once the play crosses the limit.

## What I built

- A custom Gradio stage UI with a playbill header, theater-style script view, director controls, memory meter, and forgotten-story panel.
- A bounded-memory theater engine that keeps only the newest script beats inside a 1,000-token budget.
- A live MiniCPM generation path using `openbmb/MiniCPM4.1-8B` on Hugging Face ZeroGPU.
- Streaming output so each actor's line appears as the model writes it.
- Local engine tests with a deterministic stub so the memory logic can be checked without a GPU.

## Why this fits Thousand Token Wood

The project is not just a chatbot with a short context window. The short context window is the story.

The audience can see the model's memory fill up, see old details disappear, and then watch the cast continue with only the remaining script. Forgotten names, lost secrets, and drifting motives become part of the performance.

## What I learned

Small context is usually treated as a limitation to work around. In this project, it became a creative rule. The hardest part was making the constraint visible and understandable without turning the app into a technical dashboard. The theater metaphor helped: the model forgets, the cast adapts, and the user can direct the chaos.

## What I would improve next

- Add shareable saved scripts for favorite runs.
- Add a public trace gallery if I collect safe, non-sensitive examples.
- Add more theater modes, such as mystery, courtroom, or One Thousand and One Nights inspired scenes.
- Add a downloadable playbill after each session.

## Evidence

- Live Space: https://huggingface.co/spaces/build-small-hackathon/thousand-token-theater
- GitHub repo: https://github.com/himanshu748/build-small-hackathon-thousand-token-theater
- Demo video + social post: https://x.com/i/status/2064354192748110158

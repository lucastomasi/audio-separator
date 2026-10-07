# Audio Separator — agent notes

## Cursor Cloud specific instructions

### Product (hard rule)

Audio Separator is a **local Mac desktop app**. Processing runs on the user’s machine. It is **not** a cloud product, not a website, and not an online service.

- Core flows (Separar, Entrenar, Convertir, Unir) must work **offline** once models are on disk.
- The full standalone zip is **canned**: the destination Mac must not need to download weights.
- Do **not** “fix” problems by adding cloud APIs, hosted inference, Hugging Face runtime pulls into the app venv, or new network-required stacks.
- Do **not** treat Gradio’s localhost URL as the product surface for users or for UI QA.
- Optional network helpers that already exist (YouTube fetch, Edge/ElevenLabs TTS, lite “Completar instalación”) are secondary. Never expand them, never make core flows depend on them, never reintroduce RunPod/GPU hire or similar.
- Completar instalación is the **only** place that may fetch missing UVR/RVC weights. Separar / Entrenar / Convertir must fail with that button — never download mid-job.

### UI testing (hard rule)

Audio Separator is a **native desktop window** (`desktop.py` / pywebview), not a website.

- **Never** validate layout, scroll, freezes, or “how it feels” in Chrome/Firefox/Safari against `http://127.0.0.1:…`.
- Browser chrome, page scrollbars, and click targets are not the product. Screenshots from a browser are invalid evidence for UI polish.
- Manual UI checks must use the native window:
  - Dev: `python desktop.py` (or the venv equivalent)
  - Release: the bundled `.app` / standalone zip on a Mac
- On Cloud Agent Linux VMs, pywebview often cannot start (no GTK/Qt). In that case:
  - Do **not** fall back to the browser for UI judgment
  - Run unit/API checks for status locks and copy
  - Leave native visual QA for a Mac with `desktop.py` or the `.app`

### Product surface

- Gradio only exists as the content inside pywebview.
- Security assumes loopback + local token in that window, not a shared webpage.

### Entry modules (`app.py`, `remix.py`)

Both files are **tracked on `main`** (`desktop.py` imports `app.launch_app`; `app_jobs` imports `remix.remix_to_wav`). If imports fail, refresh the checkout (`git fetch origin main && git checkout main`) before reimplementing anything.

On Linux cloud VMs, pywebview usually lacks GTK/Qt, so **`python desktop.py` is not a reliable smoke test**. After `.cursor/install.sh`, verify with:

```bash
.venv/bin/python -c "from app import build_server, launch_app; from remix import remix_to_wav"
```

Optional headless Gradio check (loopback + token): start `launch_app(prevent_thread_lock=True, …)` in a thread and poll `/` with `x-audio-separator-token` (same as `desktop.py`).

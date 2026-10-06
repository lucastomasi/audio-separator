# Audio Separator — agent notes

## Cursor Cloud specific instructions

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

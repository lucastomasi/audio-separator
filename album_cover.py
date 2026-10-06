"""Album cover for audio-only share. Pillow default; optional artistic pass."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

from exports import copy_to_downloads, unique_path, exports_dir


def _hash_color(text: str, salt: int = 0):
    digest = hashlib.sha256(f"{text}:{salt}".encode("utf-8")).digest()
    return tuple(digest[i] for i in range(3))


def generate_cover(
    title: str,
    artist: str = "",
    size: int = 1400,
    dest_path: str | None = None,
    artistic: bool = False,
) -> str:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont

    title = (title or "Audio Separator").strip() or "Audio Separator"
    artist = (artist or "").strip()
    c1 = _hash_color(title, 1)
    c2 = _hash_color(artist or title, 7)
    img = Image.new("RGB", (size, size), c1)
    draw = ImageDraw.Draw(img)
    for y in range(size):
        t = y / max(size - 1, 1)
        if artistic:
            t = 0.5 + 0.5 * math.sin(t * math.pi)
        r = int(c1[0] * (1 - t) + c2[0] * t)
        g = int(c1[1] * (1 - t) + c2[1] * t)
        b = int(c1[2] * (1 - t) + c2[2] * t)
        draw.line([(0, y), (size, y)], fill=(r, g, b))
    if artistic:
        overlay = Image.new("RGB", (size, size), (0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for i in range(8):
            col = _hash_color(title, 20 + i)
            x0 = (digest_int(title, i) % size) - size // 4
            y0 = (digest_int(artist or title, i + 3) % size) - size // 4
            od.ellipse([x0, y0, x0 + size // 2, y0 + size // 2], fill=col)
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=size // 18))
        img = Image.blend(img, overlay, 0.35)
        draw = ImageDraw.Draw(img)

    font_large = _font(int(size * 0.07))
    font_small = _font(int(size * 0.04))
    margin = int(size * 0.08)
    draw.rectangle(
        [margin, size - int(size * 0.28), size - margin, size - margin],
        fill=(0, 0, 0, ),
    )
    # redraw as dark bar
    bar = Image.new("RGB", (size - 2 * margin, int(size * 0.20)), (12, 12, 14))
    img.paste(bar, (margin, size - int(size * 0.28)))
    draw = ImageDraw.Draw(img)
    draw.text(
        (margin + 24, size - int(size * 0.26)),
        title[:48],
        fill=(250, 250, 250),
        font=font_large,
    )
    if artist:
        draw.text(
            (margin + 24, size - int(size * 0.16)),
            artist[:48],
            fill=(180, 180, 190),
            font=font_small,
        )

    if not dest_path:
        dest_path = unique_path(exports_dir(), "portada.jpg")
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    img.save(dest_path, "JPEG", quality=92, optimize=True)
    return os.path.abspath(dest_path)


def digest_int(text: str, salt: int) -> int:
    return int(hashlib.sha256(f"{text}:{salt}".encode()).hexdigest()[:8], 16)


def _font(size: int):
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def save_cover_with_audio(cover_path: str, audio_path: str | None = None) -> str:
    """Copy cover next to audio in Downloads. Does not re-encode audio."""
    if not cover_path or not os.path.isfile(cover_path):
        raise ValueError("No hay portada para guardar.")
    _, copied = copy_to_downloads([cover_path], ["portada"])
    saved = copied[0] if copied else cover_path
    if audio_path and os.path.isfile(audio_path):
        # Sidecar next to the audio filename in Downloads if possible
        stem = Path(audio_path).stem
        dest = unique_path(exports_dir(), f"{stem}_portada.jpg")
        import shutil

        shutil.copy2(cover_path, dest)
        return dest
    return saved

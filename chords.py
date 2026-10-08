"""Local guitar/bass chord estimate and typical-shape tablature.

No librosa, no network, no UVR. numpy + scipy + audio_io only.
This is an estimate of likely chords, not a transcription of the recording.
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
from scipy.signal import butter, medfilt, sosfilt, stft

import audio_io

SR = 11025
N_FFT = 2048
HOP = 1024
BASS_MAX_HZ = 180.0
BASS_MIN_HZ = 41.0
MIN_SEG_SEC = 0.55
MAX_SONG_SEC = 12 * 60

PC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
GUITAR_STRINGS = ("e", "B", "G", "D", "A", "E")
BASS_STRINGS = ("G", "D", "A", "E")
BASS_OPEN_MIDI = (28, 33, 38, 43)  # E1 A1 D2 G2, low to high

_E_MAJ = (0, 2, 2, 1, 0, 0)
_E_MIN = (0, 2, 2, 0, 0, 0)
_A_MAJ = (None, 0, 2, 2, 2, 0)
_A_MIN = (None, 0, 2, 2, 1, 0)
_E7 = (0, 2, 0, 1, 0, 0)
_A7 = (None, 0, 2, 0, 2, 0)
_EM7 = (0, 2, 0, 0, 0, 0)
_AM7 = (None, 0, 2, 0, 1, 0)

# Cosine templates: root, third, fifth, (optional seventh).
_TEMPLATES = {
    "": np.array([1.0, 0, 0, 0, 0.9, 0, 0, 1.0, 0, 0, 0, 0]),
    "m": np.array([1.0, 0, 0, 0.9, 0, 0, 0, 1.0, 0, 0, 0, 0]),
    "7": np.array([1.0, 0, 0, 0, 0.85, 0, 0, 0.9, 0, 0, 0.7, 0]),
    "m7": np.array([1.0, 0, 0, 0.85, 0, 0, 0, 0.9, 0, 0, 0.7, 0]),
}

DISCLAIMER = (
    "Estimación local (no es la tablatura de la grabación). "
    "Formas abiertas típicas de guitarra y patrón root-5 de bajo."
)


def _shift_shape(shape, n):
    out = []
    for fret in shape:
        out.append(None if fret is None else int(fret) + n)
    return tuple(out)


def _build_guitar_shapes():
    shapes = {
        "C": (None, 3, 2, 0, 1, 0),
        "D": (None, None, 0, 2, 3, 2),
        "E": _E_MAJ,
        "G": (3, 2, 0, 0, 0, 3),
        "A": _A_MAJ,
        "Em": _E_MIN,
        "Am": _A_MIN,
        "Dm": (None, None, 0, 2, 3, 1),
        "C7": (None, 3, 2, 3, 1, 0),
        "D7": (None, None, 0, 2, 1, 2),
        "E7": _E7,
        "G7": (3, 2, 0, 0, 0, 1),
        "A7": _A7,
        "B7": (None, 2, 1, 2, 0, 2),
        "Em7": _EM7,
        "Am7": _AM7,
        "Dm7": (None, None, 0, 2, 1, 1),
        "F": _shift_shape(_E_MAJ, 1),
        "F#": _shift_shape(_E_MAJ, 2),
        "Fm": _shift_shape(_E_MIN, 1),
        "F#m": _shift_shape(_E_MIN, 2),
        "F7": _shift_shape(_E7, 1),
        "F#7": _shift_shape(_E7, 2),
        "Fm7": _shift_shape(_EM7, 1),
        "F#m7": _shift_shape(_EM7, 2),
    }
    a_maj_roots = {
        "Bb": 1,
        "B": 2,
        "C#": 4,
        "Eb": 6,
        "Ab": 11,
    }
    a_min_roots = {
        "Bbm": 1,
        "Bm": 2,
        "Cm": 3,
        "C#m": 4,
        "Ebm": 6,
        "Gm": 10,
        "Abm": 11,
    }
    for name, fret in a_maj_roots.items():
        shapes[name] = _shift_shape(_A_MAJ, fret)
        shapes[name + "7"] = _shift_shape(_A7, fret)
    for name, fret in a_min_roots.items():
        shapes[name] = _shift_shape(_A_MIN, fret)
        shapes[name.replace("m", "m7", 1)] = _shift_shape(_AM7, fret)
    shapes["Gm7"] = _shift_shape(_AM7, 10)
    return shapes


GUITAR_SHAPES = _build_guitar_shapes()


def _chroma_filterbank(freqs, midi_lo=36, midi_hi=88):
    bank = np.zeros((12, len(freqs)), dtype=np.float64)
    for midi in range(midi_lo, midi_hi + 1):
        center = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        lower = 440.0 * (2.0 ** ((midi - 1 - 69) / 12.0))
        upper = 440.0 * (2.0 ** ((midi + 1 - 69) / 12.0))
        pc = midi % 12
        for i, freq in enumerate(freqs):
            if freq <= lower or freq >= upper:
                continue
            if freq < center:
                weight = (freq - lower) / (center - lower)
            else:
                weight = (upper - freq) / (upper - center)
            if weight > 0:
                bank[pc, i] += weight
    norms = np.linalg.norm(bank, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return bank / norms


_FREQS = np.fft.rfftfreq(N_FFT, 1.0 / SR)
_CHROMA_FB = _chroma_filterbank(_FREQS)
for _name, _vec in list(_TEMPLATES.items()):
    _TEMPLATES[_name] = _vec / np.linalg.norm(_vec)


def parse_chord(name):
    text = (name or "").strip()
    if text in {"N.C.", "NC", ""}:
        return None, ""
    if len(text) >= 2 and text[1] in "#b":
        return text[:2], text[2:]
    return text[:1], text[1:]


def guitar_shape(name):
    if name in GUITAR_SHAPES:
        return GUITAR_SHAPES[name]
    root, qual = parse_chord(name)
    if not root:
        return (None, None, None, None, None, None)
    fallback = GUITAR_SHAPES.get(root + (qual if qual in {"m", "7", "m7"} else ""))
    if fallback:
        return fallback
    return GUITAR_SHAPES.get(root, (None, None, None, None, None, None))


def shape_code(shape):
    chars = []
    for fret in shape:
        chars.append("x" if fret is None else str(int(fret)))
    return "".join(chars)


def bass_position(pc, max_fret=10):
    """Lowest E/A-string fret for a pitch class. Returns (string_index E=0, fret)."""
    pc = int(pc) % 12
    best = None
    for string in (0, 1):
        open_pc = BASS_OPEN_MIDI[string] % 12
        fret = (pc - open_pc) % 12
        if fret > max_fret:
            continue
        midi = BASS_OPEN_MIDI[string] + fret
        if best is None or midi < best[2] or (midi == best[2] and fret < best[1]):
            best = (string, fret, midi)
    if best is None:
        return 0, (pc - (BASS_OPEN_MIDI[0] % 12)) % 12
    return best[0], best[1]


def _lowpass(wave, sr, cutoff=BASS_MAX_HZ):
    nyq = sr * 0.49
    freq = min(float(cutoff), nyq)
    sos = butter(4, freq, btype="low", fs=sr, output="sos")
    return sosfilt(sos, wave)


def _f0_acf(frame, sr, fmin=BASS_MIN_HZ, fmax=BASS_MAX_HZ):
    if frame.size < 32:
        return None
    frame = frame.astype(np.float64)
    frame = frame - frame.mean()
    energy = np.dot(frame, frame)
    if energy < 1e-8:
        return None
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2 :]
    min_lag = max(1, int(sr / fmax))
    max_lag = min(len(corr) - 1, int(sr / fmin))
    if max_lag <= min_lag:
        return None
    region = corr[min_lag:max_lag]
    lag = min_lag + int(np.argmax(region))
    if corr[lag] < 0.2 * corr[0]:
        return None
    return sr / float(lag)


def hz_to_pc(freq):
    if not freq or freq <= 0:
        return None
    midi = 69.0 + 12.0 * np.log2(float(freq) / 440.0)
    return int(np.round(midi)) % 12


def _chroma_frames(wave, sr):
    _freq, _times, spec = stft(
        wave,
        fs=sr,
        nperseg=N_FFT,
        noverlap=N_FFT - HOP,
        window="hann",
        boundary=None,
        padded=False,
    )
    mag = np.abs(spec)
    chroma = _CHROMA_FB @ mag
    energy = chroma.sum(axis=0)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    chroma = chroma / norms
    hop_sec = HOP / float(sr)
    return chroma, energy, hop_sec


def _score_chords(chroma_col):
    best_name = "N.C."
    best_score = -1.0
    for root in range(12):
        rotated = np.roll(chroma_col, -root)
        for qual, template in _TEMPLATES.items():
            score = float(np.dot(rotated, template))
            if qual in {"7", "m7"}:
                score *= 0.92
            if score > best_score:
                best_score = score
                best_name = PC_NAMES[root] + qual
    if best_score < 0.55:
        return "N.C.", best_score
    return best_name, best_score


def _smooth_labels(labels, scores, hop_sec):
    ids = []
    catalog = []
    index = {}
    for name in labels:
        if name not in index:
            index[name] = len(catalog)
            catalog.append(name)
        ids.append(index[name])
    width = 5 if hop_sec * 5 < 1.2 else 3
    if width % 2 == 0:
        width += 1
    if len(ids) >= width:
        ids = medfilt(np.asarray(ids, dtype=np.int32), kernel_size=width)
    return [catalog[int(i)] for i in ids]


def _merge_segments(labels, hop_sec, min_sec=MIN_SEG_SEC):
    if not labels:
        return []
    raw = []
    start = 0
    current = labels[0]
    for i, name in enumerate(labels[1:], start=1):
        if name == current:
            continue
        raw.append((start * hop_sec, i * hop_sec, current))
        start = i
        current = name
    raw.append((start * hop_sec, len(labels) * hop_sec, current))

    merged = []
    for start, end, name in raw:
        if merged and (end - start) < min_sec:
            prev_start, prev_end, prev_name = merged[-1]
            merged[-1] = (prev_start, end, prev_name)
            continue
        if merged and merged[-1][2] == name:
            prev_start, _prev_end, prev_name = merged[-1]
            merged[-1] = (prev_start, end, prev_name)
            continue
        merged.append((start, end, name))
    cleaned = [item for item in merged if item[2] != "N.C." and (item[1] - item[0]) >= min_sec]
    if not cleaned:
        cleaned = [item for item in merged if item[2] != "N.C."]
    return cleaned


def _bass_pc_for_span(wave, sr, start, end):
    i0 = max(0, int(start * sr))
    i1 = min(wave.size, int(end * sr))
    if i1 - i0 < int(0.2 * sr):
        return None
    chunk = _lowpass(wave[i0:i1], sr)
    step = max(1, int(0.25 * sr))
    win = max(step, int(0.4 * sr))
    votes = []
    for pos in range(0, max(1, chunk.size - win), step):
        freq = _f0_acf(chunk[pos : pos + win], sr)
        pc = hz_to_pc(freq) if freq else None
        if pc is not None:
            votes.append(pc)
    if not votes:
        freq = _f0_acf(chunk, sr)
        return hz_to_pc(freq) if freq else None
    return Counter(votes).most_common(1)[0][0]


def detect_chords(wave, sr):
    if wave.ndim > 1:
        wave = np.mean(wave, axis=0)
    wave = np.asarray(wave, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(wave)))) if wave.size else 0.0
    if wave.size < int(sr * 0.6) or rms < 1e-4:
        raise ValueError("No se pudieron estimar los acordes.")
    chroma, energy, hop_sec = _chroma_frames(wave, sr)
    if chroma.shape[1] < 3:
        raise ValueError("No se pudieron estimar los acordes.")
    energy_gate = max(float(np.median(energy)) * 0.15, 1e-8)
    labels = []
    scores = []
    for i in range(chroma.shape[1]):
        if energy[i] < energy_gate:
            labels.append("N.C.")
            scores.append(0.0)
            continue
        name, score = _score_chords(chroma[:, i])
        labels.append(name)
        scores.append(score)
    labels = _smooth_labels(labels, scores, hop_sec)
    spans = _merge_segments(labels, hop_sec)
    if not spans:
        raise ValueError("No se pudieron estimar los acordes.")
    segments = []
    for start, end, name in spans:
        root, _qual = parse_chord(name)
        root_pc = PC_NAMES.index(root) if root in PC_NAMES else None
        bass_pc = _bass_pc_for_span(wave, sr, start, end)
        if bass_pc is None:
            bass_pc = root_pc
        segments.append(
            {
                "start": float(start),
                "end": float(end),
                "name": name,
                "root_pc": root_pc,
                "bass_pc": bass_pc,
            }
        )
    return segments


def _stamp(seconds):
    total = max(0, int(round(float(seconds))))
    return f"{total // 60}:{total % 60:02d}"


def _tab_cell(fret, width=4):
    if fret is None:
        body = "-"
    else:
        body = str(int(fret))
    pad = width - len(body)
    left = pad // 2
    right = pad - left
    return ("-" * left) + body + ("-" * right)


def guitar_tab_lines(segments, per_line=6):
    if not segments:
        return []
    lines = []
    for offset in range(0, len(segments), per_line):
        chunk = segments[offset : offset + per_line]
        shapes = [guitar_shape(item["name"]) for item in chunk]
        rows = []
        for string_i, label in enumerate(GUITAR_STRINGS):
            cells = []
            for shape in shapes:
                fret = shape[5 - string_i]
                cells.append(_tab_cell(fret, width=6))
            rows.append(f"{label}|{''.join(cells)}|")
        names = "".join(f"{item['name']:^{6}}" for item in chunk)
        times = "".join(f"{_stamp(item['start']):^{6}}" for item in chunk)
        rows.append(f"  {names}")
        rows.append(f"  {times}")
        lines.extend(rows)
        lines.append("")
    return lines


def bass_tab_lines(segments, per_line=6):
    if not segments:
        return []
    lines = []
    for offset in range(0, len(segments), per_line):
        chunk = segments[offset : offset + per_line]
        positions = []
        for item in chunk:
            root_pc = item.get("bass_pc")
            if root_pc is None:
                root_pc = item.get("root_pc") or 0
            fifth_pc = (int(root_pc) + 7) % 12
            positions.append((bass_position(root_pc), bass_position(fifth_pc)))
        rows = []
        for display_i, label in enumerate(BASS_STRINGS):
            string_i = 3 - display_i
            cells = []
            for (root_s, root_f), (fifth_s, fifth_f) in positions:
                # Two slots per chord: root then fifth (root-5).
                root_fret = root_f if root_s == string_i else None
                fifth_fret = fifth_f if fifth_s == string_i else None
                cells.append(_tab_cell(root_fret) + _tab_cell(fifth_fret))
            rows.append(f"{label}|{''.join(cells)}|")
        names = "".join(f"{item['name']:^{8}}" for item in chunk)
        times = "".join(f"{_stamp(item['start']):^{8}}" for item in chunk)
        rows.append(f"  {names}")
        rows.append(f"  {times}")
        lines.extend(rows)
        lines.append("")
    return lines


def unique_chords(segments):
    seen = []
    found = set()
    for item in segments:
        name = item["name"]
        if name in found or name == "N.C.":
            continue
        found.add(name)
        seen.append(name)
    return seen


def render_text(source_name, segments, duration, truncated=False):
    used = unique_chords(segments)
    lines = [
        "Audio Separator — estimación de acordes",
        f"Canción: {source_name}",
        f"Duración: {_stamp(duration)}",
        DISCLAIMER,
    ]
    if truncated:
        lines.append("Solo se usaron los primeros 12 minutos.")
    lines.extend(
        [
        "",
        "Acordes (línea de tiempo)",
        ]
    )
    for item in segments:
        bass_name = PC_NAMES[item["bass_pc"]] if item.get("bass_pc") is not None else "-"
        lines.append(
            f"{_stamp(item['start'])}–{_stamp(item['end'])}  {item['name']}  "
            f"(bajo {bass_name})"
        )
    lines.extend(["", "Guitarra — formas típicas"])
    for name in used:
        shape = guitar_shape(name)
        lines.append(f"{name:<4}  {shape_code(shape)}")
    lines.extend(["", "Tablatura de guitarra"])
    lines.extend(guitar_tab_lines(segments) or ["(sin acordes)"])
    lines.extend(["Tablatura de bajo (root-5)"])
    lines.extend(bass_tab_lines(segments) or ["(sin acordes)"])
    return "\n".join(lines).rstrip() + "\n"


def write_export(text, source_path):
    from app_env import data_dir
    from exports import unique_path

    stem = os.path.splitext(os.path.basename(source_path))[0] or "cancion"
    out_dir = os.path.join(data_dir(), "Trabajos", "Acordes")
    os.makedirs(out_dir, exist_ok=True)
    dest = unique_path(out_dir, f"{stem}_acordes.txt")
    with open(dest, "w", encoding="utf-8") as handle:
        handle.write(text)
    return dest


def estimate_song(path):
    path = audio_io._as_path(path)
    if not path or not os.path.isfile(str(path)):
        raise ValueError("Elegí una canción en Canción.")
    wave, sr = audio_io.load(path, mono=True, sr=SR)
    truncated = wave.size > int(SR * MAX_SONG_SEC)
    if truncated:
        wave = wave[: int(SR * MAX_SONG_SEC)]
    duration = wave.size / float(sr)
    segments = detect_chords(wave, sr)
    name = os.path.basename(str(path))
    text = render_text(name, segments, duration, truncated=truncated)
    export_path = write_export(text, path)
    return text, export_path, segments

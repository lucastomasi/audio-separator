"""Empty and very short buffers for the RVC converter. Numpy and scipy only."""
import numpy as np
from scipy import signal


def require_samples(audio):
    audio = np.asarray(audio)
    if audio.size == 0:
        raise ValueError("El audio está vacío.")
    return audio


def highpass_audio(audio, numerator, denominator):
    """High-pass that still returns audio shorter than filtfilt's padlen.

    scipy.signal.filtfilt rejects inputs whose length is not greater than
    ``3 * max(len(b), len(a))`` (18 samples for the order-5 48 Hz high-pass).
    """
    audio = np.asarray(audio)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    padlen = 3 * max(len(numerator), len(denominator))
    if audio.shape[-1] <= padlen:
        extra = padlen + 1
        padded = np.pad(audio, (extra, extra), mode="constant")
        filtered = signal.filtfilt(numerator, denominator, padded)
        return filtered[extra:-extra]
    return signal.filtfilt(numerator, denominator, audio)


def use_split(chunks, audio):
    """Keep a silence split, or process the whole buffer when it found nothing."""
    if len(chunks) == 0:
        return [audio], False
    return list(chunks), True

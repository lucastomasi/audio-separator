"""RVC guards for empty and very short audio. No faiss, librosa, or weights."""
import os
import sys
import unittest

import numpy as np
from scipy import signal

_VC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "vc")
if _VC not in sys.path:
    sys.path.insert(0, _VC)

from rvc.infer.audio_guard import highpass_audio, require_samples, use_split
from rvc.lib.tools.split_audio import merge_audio


def _butter():
    return signal.butter(N=5, Wn=48, btype="high", fs=16000)


class RvcShortAudioTests(unittest.TestCase):
    def test_empty_audio_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            require_samples(np.zeros(0, dtype=np.float32))
        self.assertIn("vacío", str(ctx.exception))

    def test_highpass_keeps_short_and_empty_lengths(self):
        numerator, denominator = _butter()
        empty = highpass_audio(np.zeros(0, dtype=np.float32), numerator, denominator)
        self.assertEqual(empty.shape, (0,))
        for length in (1, 10, 18, 19, 100):
            source = np.linspace(-0.2, 0.2, length, dtype=np.float32)
            filtered = highpass_audio(source, numerator, denominator)
            self.assertEqual(len(filtered), length)
            self.assertTrue(np.isfinite(filtered).all())

    def test_long_highpass_matches_filtfilt(self):
        numerator, denominator = _butter()
        source = np.linspace(-0.2, 0.2, 100, dtype=np.float32)
        direct = signal.filtfilt(numerator, denominator, source)
        wrapped = highpass_audio(source, numerator, denominator)
        np.testing.assert_allclose(wrapped, direct)

    def test_empty_split_uses_the_whole_buffer(self):
        audio = np.zeros(16000, dtype=np.float32)
        chunks, split = use_split([], audio)
        self.assertFalse(split)
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(chunks[0], audio)

    def test_merge_without_chunks_is_empty(self):
        merged = merge_audio([], [], [], 16000, 40000)
        self.assertEqual(merged.size, 0)


if __name__ == "__main__":
    unittest.main()

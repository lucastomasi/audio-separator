import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from remix import align_lag_ms, delay_ms, mix_tracks, remix_to_wav, stretch_to_length, to_stereo


class RemixTests(unittest.TestCase):
    def test_to_stereo_from_mono(self):
        mono = np.ones(8, dtype=np.float32)
        stereo = to_stereo(mono)
        self.assertEqual(stereo.shape, (2, 8))

    def test_align_finds_late_voice(self):
        sr = 1000
        inst = np.zeros((2, 400), dtype=np.float32)
        inst[:, 50:80] = 1
        voice = np.zeros((2, 400), dtype=np.float32)
        voice[:, 150:180] = 1
        lag = align_lag_ms(voice, inst, sr, max_ms=500)
        self.assertAlmostEqual(lag, -100, delta=15)

    def test_delay_positive_pads(self):
        wave = np.ones((2, 10), dtype=np.float32)
        delayed = delay_ms(wave, 1000, 5)
        self.assertEqual(delayed.shape[1], 15)
        np.testing.assert_array_equal(delayed[:, :5], 0)

    def test_delay_negative_trims(self):
        wave = np.arange(10, dtype=np.float32)
        wave = np.stack([wave, wave])
        delayed = delay_ms(wave, 1000, -3)
        self.assertEqual(delayed.shape[1], 7)
        self.assertEqual(delayed[0, 0], 3)

    def test_mix_adds_and_clips(self):
        voice = np.ones((2, 4), dtype=np.float32)
        inst = np.ones((2, 6), dtype=np.float32)
        mixed = mix_tracks(voice, inst, voice_db=0, instrumental_db=0)
        self.assertEqual(mixed.shape, (2, 6))
        self.assertLessEqual(mixed.max(), 1.0)

    def test_stretch_shortens(self):
        wave = np.random.randn(2, 4000).astype(np.float32) * 0.1
        out = stretch_to_length(wave, 44100, 2000, high_quality=False)
        self.assertEqual(out.shape[1], 2000)

    def test_remix_writes_wav(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        sr = 44100
        t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False, dtype=np.float32)
        voice = np.stack([0.2 * np.sin(2 * np.pi * 440 * t)] * 2)
        inst = np.stack([0.1 * np.sin(2 * np.pi * 220 * t)] * 2)
        vpath = os.path.join(tmp.name, "v.wav")
        ipath = os.path.join(tmp.name, "i.wav")
        out = os.path.join(tmp.name, "mix.wav")
        sf.write(vpath, voice.T, sr)
        sf.write(ipath, inst.T, sr)
        remix_to_wav(vpath, ipath, out, delay_milliseconds=10, match_duration=True)
        self.assertTrue(os.path.isfile(out))
        data, file_sr = sf.read(out)
        self.assertEqual(file_sr, sr)
        self.assertGreater(len(data), 0)


if __name__ == "__main__":
    unittest.main()

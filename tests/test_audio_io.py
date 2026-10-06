import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

import audio_io


def write_wav(path, channels, sr=44100, seconds=0.1):
    n = int(sr * seconds)
    t = np.linspace(0, seconds, n, endpoint=False, dtype=np.float32)
    if channels == 1:
        data = 0.2 * np.sin(2 * np.pi * 440 * t)
    else:
        left = 0.2 * np.sin(2 * np.pi * 440 * t)
        right = 0.1 * np.sin(2 * np.pi * 660 * t)
        data = np.stack([left, right], axis=1)
    sf.write(path, data, sr)
    return n, sr


class AudioIoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_duration_filename_and_path(self):
        wav = self.dir / "tone.wav"
        n, sr = write_wav(wav, channels=2)
        expected = n / sr
        self.assertAlmostEqual(
            audio_io.get_duration(filename=str(wav)), expected, places=4
        )
        self.assertAlmostEqual(
            audio_io.get_duration(path=str(wav)), expected, places=4
        )

    def test_load_stereo_shape(self):
        wav = self.dir / "stereo.wav"
        write_wav(wav, channels=2)
        wave, sr = audio_io.load(str(wav), mono=False, sr=44100)
        self.assertEqual(sr, 44100)
        self.assertEqual(wave.ndim, 2)
        self.assertEqual(wave.shape[0], 2)
        self.assertEqual(wave.dtype, np.float32)

    def test_load_accepts_gradio_dict(self):
        wav = self.dir / "stereo.wav"
        write_wav(wav, channels=2)
        payload = {
            "path": str(wav),
            "orig_name": "stereo.wav",
            "meta": {"_type": "gradio.FileData"},
        }
        wave, sr = audio_io.load(payload, mono=False, sr=44100)
        self.assertEqual(sr, 44100)
        self.assertEqual(wave.shape[0], 2)

    def test_load_mono_file_is_1d(self):
        wav = self.dir / "mono.wav"
        write_wav(wav, channels=1)
        wave, sr = audio_io.load(str(wav), mono=False, sr=44100)
        self.assertEqual(wave.ndim, 1)
        self.assertEqual(sr, 44100)

    def test_load_mono_true_averages_channels(self):
        wav = self.dir / "stereo.wav"
        write_wav(wav, channels=2)
        wave, _ = audio_io.load(str(wav), mono=True, sr=44100)
        self.assertEqual(wave.ndim, 1)

    def test_resample_identity_same_rate(self):
        y = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
        out = audio_io.resample(y, orig_sr=44100, target_sr=44100)
        np.testing.assert_array_equal(out, y)

    def test_resample_stereo_float32(self):
        y = np.stack(
            [np.ones(200, dtype=np.float32), np.zeros(200, dtype=np.float32)]
        )
        out = audio_io.resample(y, orig_sr=22050, target_sr=44100)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], 2)
        self.assertGreater(out.shape[1], 200)

    def test_duration_ffprobe_fallback(self):
        payload = json.dumps({"format": {"duration": "12.5"}}).encode()
        with mock.patch.object(audio_io.sf, "SoundFile", side_effect=OSError("no")):
            with mock.patch.object(
                audio_io.subprocess, "check_output", return_value=payload
            ) as probe:
                duration = audio_io.get_duration(filename="missing.m4a")
        self.assertEqual(duration, 12.5)
        probe.assert_called_once()
        self.assertIn("ffprobe", probe.call_args[0][0])


if __name__ == "__main__":
    unittest.main()

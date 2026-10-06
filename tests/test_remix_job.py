import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import soundfile as sf

import app_jobs


def _short_wav(path):
    sr = 44100
    t = np.linspace(0, 0.05, int(sr * 0.05), endpoint=False, dtype=np.float32)
    wave = np.stack([0.2 * np.sin(2 * np.pi * 440 * t)] * 2)
    sf.write(path, wave.T, sr)


class RemixJobTests(unittest.TestCase):
    def test_remix_job_unwraps_gradio_audio_dicts(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        vpath = os.path.join(tmp.name, "v.wav")
        ipath = os.path.join(tmp.name, "i.wav")
        _short_wav(vpath)
        _short_wav(ipath)
        voice = {
            "path": vpath,
            "orig_name": "v.wav",
            "meta": {"_type": "gradio.FileData"},
        }
        inst = {
            "path": ipath,
            "orig_name": "i.wav",
            "meta": {"_type": "gradio.FileData"},
        }
        with mock.patch(
            "app_jobs.copy_to_downloads",
            side_effect=lambda files, labels: (tmp.name, list(files)),
        ):
            audio, archivo, status = app_jobs.remix_job(
                voice, inst, 0, False, 0, 0, "WAV"
            )
        self.assertTrue(audio)
        self.assertTrue(os.path.isfile(audio))
        self.assertIn("Pistas unidas", str(status))
        self.assertNotIn("No se pudo armar el remix", str(status))


if __name__ == "__main__":
    unittest.main()

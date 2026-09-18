import gc
import os
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import soundfile as sf


class ConvertWavTests(unittest.TestCase):
    def test_stereo_44100_wav_skips_ffmpeg(self):
        import uvr_runtime

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "ready.wav")
        audio = np.zeros((2048, 2), dtype=np.float32)
        sf.write(path, audio, 44100)
        with mock.patch.object(uvr_runtime.subprocess, "Popen") as popen:
            out = uvr_runtime.convert_to_stereo_and_wav(path)
        self.assertEqual(out, path)
        popen.assert_not_called()

    def test_youtube_48k_wav_is_resampled(self):
        import uvr_runtime

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "yt.wav")
        audio = np.zeros((4800, 2), dtype=np.float32)
        sf.write(path, audio, 48000)
        fake = mock.Mock()
        fake.returncode = 0
        fake.communicate.return_value = (b"", b"")

        def popen(cmd, **kwargs):
            dest = cmd[-1]
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            sf.write(dest, np.zeros((4410, 2), dtype=np.float32), 44100)
            return fake

        with mock.patch.object(uvr_runtime.subprocess, "Popen", side_effect=popen):
            out = uvr_runtime.convert_to_stereo_and_wav(path)
        self.assertNotEqual(out, path)
        self.assertIn("44100", os.path.basename(out))

    def test_model_hash_is_cached(self):
        import uvr_runtime

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "model.bin")
        with open(path, "wb") as handle:
            handle.write(b"x" * 100)
        uvr_runtime._MODEL_HASHES.clear()
        first = uvr_runtime.MDX.get_hash(path)
        second = uvr_runtime.MDX.get_hash(path)
        self.assertEqual(first, second)
        self.assertEqual(uvr_runtime._MODEL_HASHES[os.path.abspath(path)], first)

    def test_small_model_hash_does_not_leak_file(self):
        import uvr_runtime

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "tiny.bin")
        with open(path, "wb") as handle:
            handle.write(b"x" * 100)
        uvr_runtime._MODEL_HASHES.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            digest = uvr_runtime.MDX.get_hash(path)
            gc.collect()
        self.assertEqual(len(digest), 32)

    def test_separate_hides_torch_nameerror(self):
        import uvr_runtime

        msg = uvr_runtime._separate_error_message(
            NameError("name 'torch' is not defined")
        )
        self.assertNotIn("torch", msg.lower())
        self.assertIn("Cerrá la app", msg)

    def test_mdx_model_imports_soundfile(self):
        import mdx_model

        self.assertTrue(hasattr(mdx_model, "sf"))
        self.assertTrue(callable(mdx_model.sf.write))

    def test_ensure_ml_returns_torch(self):
        import mdx_model

        torch_mod, ort_mod, _ = mdx_model._ensure_ml()
        self.assertTrue(hasattr(torch_mod, "cuda"))
        self.assertTrue(hasattr(ort_mod, "get_device"))


if __name__ == "__main__":
    unittest.main()

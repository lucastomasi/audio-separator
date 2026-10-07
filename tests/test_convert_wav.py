import gc
import os
import shutil
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import soundfile as sf


class ConvertWavTests(unittest.TestCase):
    def test_convert_uses_ffmpeg_binary_not_path_ffmpeg(self):
        import inspect

        import uvr_runtime

        src = inspect.getsource(uvr_runtime.convert_to_stereo_and_wav)
        self.assertIn("ffmpeg_binary", src)
        self.assertNotIn('"ffmpeg"', src)

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

        with mock.patch("youtube_lib.ffmpeg_binary", return_value="ffmpeg"):
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

    def test_separate_drops_work_dir_when_download_copy_ok(self):
        import uvr_runtime

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = tmp.name
        song = os.path.join(root, "songmdx")
        os.makedirs(song)
        vocal = os.path.join(song, "clip_Vocals.wav")
        with open(vocal, "wb") as handle:
            handle.write(b"wav-vocal")
        src = os.path.join(root, "in.wav")
        with open(src, "wb") as handle:
            handle.write(b"source!!")
        downloads = os.path.join(root, "dl")

        def fake_copy(paths, labels):
            os.makedirs(downloads, exist_ok=True)
            copied = []
            for path, label in zip(paths, labels):
                dest = os.path.join(downloads, f"{label}.wav")
                shutil.copy(path, dest)
                copied.append(dest)
            return downloads, copied

        with mock.patch.object(uvr_runtime, "output_dir", root):
            with mock.patch.object(
                uvr_runtime,
                "process_uvr_task",
                return_value=(vocal, None, None, vocal, vocal),
            ):
                with mock.patch("uvr_runtime.copy_to_downloads", side_effect=fake_copy):
                    with mock.patch.object(
                        uvr_runtime.librosa, "get_duration", return_value=1.0
                    ):
                        with mock.patch("library.set_session_meta"):
                            out_v, _bg, _files, _status, _btn = uvr_runtime._sound_separate(
                                src,
                                "solo_voz",
                                False,
                                False,
                                False,
                                False,
                                0, 0, 0, 0,
                                0, 0,
                                0, 0, 0, 0,
                                0,
                                0, 0,
                                0, 0, 0,
                                0, 0, 0,
                                0,
                                0,
                                "WAV",
                            )
        self.assertFalse(os.path.isdir(song))
        self.assertTrue(os.path.isfile(out_v))
        self.assertTrue(os.path.abspath(out_v).startswith(os.path.abspath(downloads)))

    def test_separate_hides_torch_nameerror(self):
        import uvr_runtime

        msg = uvr_runtime._separate_error_message(
            NameError("name 'torch' is not defined")
        )
        self.assertNotIn("torch", msg.lower())
        self.assertEqual(msg, uvr_runtime.MSG_SEPARATE)

    def test_mdx_model_imports_soundfile(self):
        import mdx_model

        self.assertTrue(hasattr(mdx_model, "sf"))
        self.assertTrue(callable(mdx_model.sf.write))

    def test_ensure_ml_returns_torch(self):
        import mdx_model

        torch_mod, ort_mod, _ = mdx_model._ensure_ml()
        self.assertTrue(hasattr(torch_mod, "cuda"))
        self.assertTrue(hasattr(ort_mod, "get_device"))

    def test_ensure_uvr_model_uses_existing_file(self):
        import uvr_runtime

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        name = "UVR-MDX-NET-Voc_FT.onnx"
        path = os.path.join(tmp.name, name)
        with open(path, "wb") as handle:
            handle.write(b"onnx")
        with mock.patch.object(uvr_runtime, "mdxnet_models_dir", tmp.name):
            self.assertEqual(uvr_runtime.ensure_uvr_model(name), path)

    def test_ensure_uvr_model_copies_from_can(self):
        import uvr_runtime

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest_dir = os.path.join(tmp.name, "mdx")
        can = os.path.join(tmp.name, "can")
        os.makedirs(dest_dir)
        os.makedirs(os.path.join(can, "mdx_models"))
        name = "UVR-MDX-NET-Voc_FT.onnx"
        with open(os.path.join(can, "mdx_models", name), "wb") as handle:
            handle.write(b"from-can")
        with mock.patch.object(uvr_runtime, "mdxnet_models_dir", dest_dir):
            with mock.patch.dict(os.environ, {"AUDIO_SEPARATOR_CAN": can}):
                out = uvr_runtime.ensure_uvr_model(name)
        self.assertTrue(os.path.isfile(out))
        with open(out, "rb") as handle:
            self.assertEqual(handle.read(), b"from-can")

    def test_ensure_uvr_model_does_not_download(self):
        import inspect

        import uvr_runtime

        src = inspect.getsource(uvr_runtime.ensure_uvr_model)
        self.assertNotIn("download_manager", src)
        self.assertNotIn("Bajando modelo", src)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest_dir = os.path.join(tmp.name, "mdx")
        can = os.path.join(tmp.name, "can")
        os.makedirs(dest_dir)
        os.makedirs(can)
        with mock.patch.object(uvr_runtime, "mdxnet_models_dir", dest_dir):
            with mock.patch.dict(os.environ, {"AUDIO_SEPARATOR_CAN": can}):
                with self.assertRaises(ValueError) as ctx:
                    uvr_runtime.ensure_uvr_model("UVR-MDX-NET-Voc_FT.onnx")
        self.assertIn("Completar instalación", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

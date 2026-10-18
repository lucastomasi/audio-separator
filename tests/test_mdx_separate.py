"""Separar: vocal/instrumental residual scale, and inference stays on one thread."""
import inspect
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np


class PeakScaleTests(unittest.TestCase):
    def test_peak_norm_does_not_mutate_mix(self):
        import mdx_model

        mix = np.array([[0.25, -0.1], [0.05, 0.2]], dtype=np.float32)
        original = mix.copy()
        norm, peak = mdx_model._peak_norm(mix)
        np.testing.assert_array_equal(mix, original)
        self.assertAlmostEqual(peak, 0.25)
        self.assertAlmostEqual(float(np.max(np.abs(norm))), 1.0, places=5)
        restored = mdx_model._restore_peak(norm, peak)
        np.testing.assert_allclose(restored, original, atol=1e-6)

    def test_as_stereo_duplicates_mono(self):
        import mdx_model

        mono = np.array([0.2, -0.2], dtype=np.float32)
        stereo = mdx_model._as_stereo(mono)
        self.assertEqual(stereo.shape, (2, 2))
        np.testing.assert_array_equal(stereo[0], stereo[1])

    def test_run_mdx_residual_uses_original_mix(self):
        """Voice subtracted from the full-scale mix, not the normalized copy."""
        import mdx_model

        mix = np.zeros((2, 64), dtype=np.float32)
        mix[0, 0] = 0.25
        mix[1, 3] = -0.1
        seen = {}

        def fake_process(self, wave, mt_threads=1, on_chunk=None):
            seen["peak"] = float(np.max(np.abs(wave)))
            if on_chunk:
                on_chunk(1, 4)
                on_chunk(4, 4)
            return wave.copy()

        written = []

        def fake_write(path, audio, sr):
            written.append((path, np.array(audio), sr))

        progress = mock.Mock()
        params = {
            "abc": {
                "mdx_dim_f_set": 8,
                "mdx_dim_t_set": 2,
                "mdx_n_fft_scale_set": 16,
                "primary_stem": "Vocals",
                "compensate": 1.0,
            }
        }
        fake_model = mock.Mock(stem_name="Vocals", compensation=1.0)
        with mock.patch.object(mdx_model.MDX, "__init__", lambda self, *a, **k: None):
            with mock.patch.object(mdx_model.MDX, "process_wave", fake_process):
                with mock.patch.object(mdx_model, "MDXModel", return_value=fake_model):
                    with mock.patch.object(
                        mdx_model.MDX, "get_hash", return_value="abc"
                    ):
                        with mock.patch.object(
                            mdx_model.librosa,
                            "load",
                            return_value=(mix.copy(), 44100),
                        ):
                            with mock.patch.object(mdx_model.sf, "write", fake_write):
                                mdx_model.run_mdx(
                                    params,
                                    tempfile.gettempdir(),
                                    "model.onnx",
                                    "song.wav",
                                    device_base="cpu",
                                    progress=progress,
                                    progress_span=(0.12, 0.96),
                                    progress_desc="Separando voz e instrumental…",
                                )
        self.assertAlmostEqual(seen["peak"], 1.0, places=5)
        self.assertEqual(len(written), 2)
        vocal = written[0][1]
        instrumental = written[1][1]
        np.testing.assert_allclose(vocal, mix.T, atol=1e-5)
        np.testing.assert_allclose(instrumental, np.zeros_like(mix.T), atol=1e-5)
        fractions = [call.args[0] for call in progress.call_args_list]
        self.assertTrue(fractions)
        self.assertGreater(fractions[-1], fractions[0])
        self.assertLess(fractions[-1], 1.0)
        self.assertIn("4/4", progress.call_args_list[-1].kwargs.get("desc", ""))


class ProcessWaveTests(unittest.TestCase):
    def test_cpu_inference_stays_on_caller_and_finishes(self):
        import mdx_model

        torch, _ort, _tqdm = mdx_model._ensure_ml()
        model = mdx_model.MDXModel(
            torch.device("cpu"),
            dim_f=8,
            dim_t=4,
            n_fft=512,
            hop=256,
            stem_name="Vocals",
            compensation=1.0,
        )
        sess = mdx_model.MDX.__new__(mdx_model.MDX)
        sess.model = model
        sess.device = torch.device("cpu")

        def fake_process(spec):
            return np.zeros(spec.detach().cpu().numpy().shape, dtype=np.float32)

        sess.process = fake_process
        rng = np.random.default_rng(0)
        wave = (rng.standard_normal((2, 1200)).astype(np.float32)) * 0.25
        ticks = []
        caller = threading.get_ident()
        threads = []

        def on_chunk(done, total):
            threads.append(threading.get_ident())
            ticks.append((done, total))

        started = time.monotonic()
        with mock.patch.object(mdx_model.threading, "Thread") as thread_cls:
            out = sess.process_wave(wave, mt_threads=1, on_chunk=on_chunk)
        elapsed = time.monotonic() - started
        thread_cls.assert_not_called()
        self.assertLess(elapsed, 20.0)
        self.assertEqual(out.shape, wave.shape)
        self.assertTrue(ticks)
        self.assertEqual(ticks[-1][0], ticks[-1][1])
        self.assertTrue(all(ident == caller for ident in threads))


class SeparateProgressTests(unittest.TestCase):
    def test_sound_separate_does_not_track_tqdm(self):
        import uvr_runtime

        default = inspect.signature(uvr_runtime.sound_separate).parameters[
            "progress"
        ].default
        self.assertFalse(default.track_tqdm)


if __name__ == "__main__":
    unittest.main()

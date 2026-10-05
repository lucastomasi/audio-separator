"""Voice training prep: slices, pitch bins, and the inference checkpoint."""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

import app
import rvc_engine
import rvc_train


class SliceTests(unittest.TestCase):
    def test_a_sung_take_is_kept_and_silence_is_not(self):
        sample_rate = 40000
        sung = (0.2 * np.sin(2 * np.pi * 220 * np.arange(sample_rate * 4) / sample_rate)).astype(
            np.float32
        )
        pieces = rvc_train.slice_voice(sung, sample_rate)
        self.assertGreaterEqual(sum(piece.size for piece in pieces), sample_rate * 2)
        silence = np.zeros(sample_rate * 4, dtype=np.float32)
        self.assertEqual(rvc_train.slice_voice(silence, sample_rate), [])

    def test_name_and_pitch_bins(self):
        self.assertEqual(rvc_train.voice_name("Mi voz"), "Mi_voz")
        with self.assertRaises(ValueError):
            rvc_train.voice_name("   ")
        coarse = rvc_train.quantize_f0([0, 50, 1100, 440])
        self.assertEqual(int(coarse[0]), 1)
        self.assertEqual(int(coarse[1]), 1)
        self.assertEqual(int(coarse[2]), 255)
        self.assertGreater(int(coarse[3]), 1)
        self.assertLess(int(coarse[3]), 255)


class ExportTests(unittest.TestCase):
    def test_exported_checkpoint_is_ready_to_convert(self):
        state = {
            "enc_p.emb_phone.weight": torch.zeros(192, 768),
            "enc_p.emb_pitch.weight": torch.zeros(256, 192),
            "emb_g.weight": torch.zeros(109, 256),
            "enc_q.mean": torch.zeros(1),
        }
        with tempfile.TemporaryDirectory() as folder:
            dest = os.path.join(folder, "mi-voz.pth")
            rvc_train.export_inference_checkpoint(state, dest, 20)
            loaded = torch.load(dest, map_location="cpu", weights_only=False)
        packed, note, complete = rvc_engine.build_inference_checkpoint(loaded, dest)
        self.assertTrue(complete)
        self.assertEqual(packed["version"], "v2")
        self.assertEqual(packed["f0"], 1)
        self.assertEqual(packed["config"][-1], 40000)
        self.assertNotIn("enc_q.mean", packed["weight"])
        self.assertIn("v2", note)

    def test_a_base_without_the_speaker_embedding_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            generator = os.path.join(folder, "f0G40k.pth")
            discriminator = os.path.join(folder, "f0D40k.pth")
            torch.save({"weight": {"enc_p.emb_phone.weight": torch.zeros(1)}}, generator)
            torch.save({"model": {"conv.weight": torch.zeros(1)}}, discriminator)
            with self.assertRaises(ValueError) as caught:
                rvc_train._load_pretrained(object(), object(), generator, discriminator)
        self.assertIn("modelo base", str(caught.exception))


class EngineImportTests(unittest.TestCase):
    def setUp(self):
        self._modules = {
            key: sys.modules[key]
            for key in list(sys.modules)
            if key == "infer_rvc_python" or key.startswith("infer_rvc_python.")
        }

    def tearDown(self):
        for key in list(sys.modules):
            if key == "infer_rvc_python" or key.startswith("infer_rvc_python."):
                sys.modules.pop(key, None)
        sys.modules.update(self._modules)

    def test_training_models_import_without_the_inference_entrypoint(self):
        rvc_train._ensure_rvc_namespace()
        from infer_rvc_python.lib.infer_pack.models import (
            MultiPeriodDiscriminatorV2,
            SynthesizerTrnMs768NSFsid,
        )
        from infer_rvc_python.lib.rmvpe import RMVPE

        self.assertTrue(callable(SynthesizerTrnMs768NSFsid))
        self.assertTrue(callable(MultiPeriodDiscriminatorV2))
        self.assertTrue(callable(RMVPE))


class TrainUiTests(unittest.TestCase):
    def test_training_without_takes_says_so(self):
        with self.assertRaises(app.gr.Error) as caught:
            app.on_train("mi-voz", None, None, False, 1, progress=None)
        self.assertIn("tomas", str(caught.exception).lower())

"""Smoke tests for local RVC conversion. No network and no real voice models."""
import os
import tempfile
import unittest
import wave

import numpy as np

import rvc_engine


def _wav(path, seconds=0.2, rate=16000):
    frames = int(seconds * rate)
    with wave.open(path, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)


class ConvertVoiceTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = self._tmpdir.name
        self._old_dir = rvc_engine.RVC_DIR
        rvc_engine.RVC_DIR = os.path.join(self.root, "rvc_models")
        os.makedirs(rvc_engine.RVC_DIR, exist_ok=True)
        self.voice = os.path.join(self.root, "voz.wav")
        _wav(self.voice)
        self.pth = os.path.join(rvc_engine.RVC_DIR, "voz.pth")

    def tearDown(self):
        rvc_engine.RVC_DIR = self._old_dir
        rvc_engine._converter = None
        rvc_engine._converter_key = None
        self._tmpdir.cleanup()

    def test_missing_audio(self):
        with open(self.pth, "wb") as handle:
            handle.write(b"not-empty")
        with self.assertRaises(ValueError) as caught:
            rvc_engine.convert_voice(os.path.join(self.root, "no.wav"), self.pth)
        self.assertIn("voz", str(caught.exception).lower())

    def test_missing_pth(self):
        with self.assertRaises(ValueError) as caught:
            rvc_engine.convert_voice(self.voice, os.path.join(self.root, "no.pth"))
        self.assertIn(".pth", str(caught.exception))

    def test_missing_support_models(self):
        with open(self.pth, "wb") as handle:
            handle.write(b"not-empty")
        with self.assertRaises(ValueError) as caught:
            rvc_engine.convert_voice(self.voice, self.pth)
        message = str(caught.exception)
        self.assertIn("Faltan los modelos locales de RVC", message)
        self.assertIn("hubert_base", message)
        self.assertIn("rmvpe.pt", message)

    def test_hubert_folder_needs_config(self):
        os.makedirs(os.path.join(rvc_engine.RVC_DIR, "hubert_base"), exist_ok=True)
        with open(os.path.join(rvc_engine.RVC_DIR, "rmvpe.pt"), "wb") as handle:
            handle.write(b"weights")
        with open(self.pth, "wb") as handle:
            handle.write(b"weights")
        with self.assertRaises(ValueError) as caught:
            rvc_engine.convert_voice(self.voice, self.pth)
        self.assertIn("config.json", str(caught.exception))

    def test_pt_hubert_is_not_enough(self):
        with open(self.pth, "wb") as handle:
            handle.write(b"not-empty")
        with open(os.path.join(rvc_engine.RVC_DIR, "hubert_base.pt"), "wb") as handle:
            handle.write(b"weights")
        with open(os.path.join(rvc_engine.RVC_DIR, "rmvpe.pt"), "wb") as handle:
            handle.write(b"weights")
        with self.assertRaises(ValueError) as caught:
            rvc_engine.convert_voice(self.voice, self.pth)
        self.assertIn("hubert_base", str(caught.exception))

    def test_existing_files_do_not_echo_input(self):
        hubert = os.path.join(rvc_engine.RVC_DIR, "hubert_base")
        os.makedirs(hubert, exist_ok=True)
        with open(os.path.join(hubert, "config.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        with open(os.path.join(rvc_engine.RVC_DIR, "rmvpe.pt"), "wb") as handle:
            handle.write(b"not-a-real-rmvpe")
        with open(self.pth, "wb") as handle:
            handle.write(b"not-a-real-rvc-model")
        try:
            result = rvc_engine.convert_voice(self.voice, self.pth)
        except ValueError as exc:
            self.assertTrue(str(exc))
            self.assertNotIn(self.voice, str(exc))
            self.assertNotIn("/Users/", str(exc))
            return
        self.assertNotEqual(os.path.abspath(result), os.path.abspath(self.voice))
        self.assertTrue(os.path.isfile(result))
        self.assertGreater(os.path.getsize(result), 0)

    def test_tiny_local_models_write_a_new_wav(self):
        try:
            import torch
            from torch import nn
            from transformers import HubertConfig, HubertModel
        except Exception as exc:
            self.skipTest(f"RVC stack is not installed: {exc}")
        rvc_engine._stub_pyworld()
        rvc_engine._ensure_torchcrepe()
        try:
            from infer_rvc_python.lib.rmvpe import E2E
            from infer_rvc_python.lib.infer_pack.models import SynthesizerTrnMs256NSFsid
        except Exception as exc:
            self.skipTest(f"infer-rvc-python is not installed: {exc}")

        class HubertModelWithFinalProj(HubertModel):
            def __init__(self, config):
                super().__init__(config)
                self.final_proj = nn.Linear(
                    config.hidden_size, config.classifier_proj_size
                )

        hubert = os.path.join(rvc_engine.RVC_DIR, "hubert_base")
        os.makedirs(hubert, exist_ok=True)
        config = HubertConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            conv_dim=(32, 32, 32, 32, 32, 32, 32),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_bias=(False,) * 7,
            num_conv_pos_embeddings=128,
            num_conv_pos_embedding_groups=8,
            vocab_size=32,
        )
        config.classifier_proj_size = 256
        HubertModelWithFinalProj(config).save_pretrained(hubert)

        rmvpe = os.path.join(rvc_engine.RVC_DIR, "rmvpe.pt")
        torch.save(E2E(4, 1, (2, 2)).state_dict(), rmvpe)

        cfg = [
            32, 32, 16, 32, 64, 2, 1, 3, 0.0, "1",
            [3], [[1, 3, 5]], [10, 4, 4], 32, [16, 8, 8], 1, 16, 16000,
        ]
        net = SynthesizerTrnMs256NSFsid(*cfg, is_half=False)
        del net.enc_q
        pth = os.path.join(rvc_engine.RVC_DIR, "tiny.pth")
        torch.save(
            {"weight": net.state_dict(), "config": cfg, "f0": 1, "version": "v1"},
            pth,
        )
        rate = 16000
        count = int(0.5 * rate)
        tone = (
            0.2
            * np.sin(2 * np.pi * 220 * np.arange(count, dtype=np.float64) / rate)
            * 32767
        ).astype("<i2")
        with wave.open(self.voice, "w") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(tone.tobytes())

        out_dir = os.path.join(self.root, "rvc_output")
        old_out = rvc_engine.OUTPUT_DIR
        rvc_engine.OUTPUT_DIR = out_dir
        try:
            result = rvc_engine.convert_voice(self.voice, pth)
        finally:
            rvc_engine.OUTPUT_DIR = old_out

        self.assertNotEqual(os.path.abspath(result), os.path.abspath(self.voice))
        self.assertTrue(result.startswith(out_dir))
        with wave.open(result) as handle:
            converted = handle.readframes(handle.getnframes())
        with wave.open(self.voice) as handle:
            original = handle.readframes(handle.getnframes())
        self.assertGreater(len(converted), 1000)
        self.assertNotEqual(converted, original)
        self.assertTrue(any(converted))


if __name__ == "__main__":
    unittest.main()

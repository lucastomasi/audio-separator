"""Checkpoint recognition, index pairing, names, and CPU retry. No real weights."""
import json
import os
import tempfile
import unittest
import wave
from unittest import mock

import rvc_engine


class _Shape:
    def __init__(self, *shape):
        self.shape = shape


def _weights(dim):
    return {
        "enc_p.emb_phone.weight": _Shape(32, dim),
        "enc_p.emb_pitch.weight": _Shape(256, 32),
        "emb_g.weight": _Shape(1, 16),
        "enc_q.mean": _Shape(1),
    }


def _train_config():
    return {
        "data": {"filter_length": 2048, "sampling_rate": 40000},
        "model": {
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.0,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [10, 10, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "spk_embed_dim": 109,
            "gin_channels": 256,
        },
        "train": {"if_f0": 1},
    }


def _safetensors(names):
    header = {
        name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for name in names
    }
    payload = json.dumps(header).encode()
    return len(payload).to_bytes(8, "little") + payload + b"\0\0\0\0"


class CheckpointTests(unittest.TestCase):
    def test_ready_checkpoint_keeps_its_path(self):
        weight = _weights(256)
        loaded = {
            "weight": weight,
            "config": [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3], [[1]], [10], 512, [16], 1, 256, 40000],
            "f0": 1,
            "version": "v1",
        }
        del weight["enc_q.mean"]
        packed, note, complete = rvc_engine.build_inference_checkpoint(loaded, "voz.pth")
        self.assertTrue(complete)
        self.assertEqual(packed["version"], "v1")
        self.assertIn("v1", note)

    def test_missing_version_follows_the_phone_size(self):
        loaded = {
            "weight": _weights(768),
            "config": [1] * 18,
            "f0": 1,
        }
        packed, _note, complete = rvc_engine.build_inference_checkpoint(loaded, "voz.pth")
        self.assertFalse(complete)
        self.assertEqual(packed["version"], "v2")
        self.assertNotIn("enc_q.mean", packed["weight"])

    def test_generator_without_config_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            rvc_engine.build_inference_checkpoint(_weights(768), "G_23333.pth")
        message = str(caught.exception)
        self.assertIn("entrenamiento", message)
        self.assertIn("weights", message)
        self.assertIn("config.json", message)

    def test_generator_with_sibling_config_is_packed(self):
        with tempfile.TemporaryDirectory() as folder:
            config_path = os.path.join(folder, "config.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(_train_config(), handle)
            packed, note, complete = rvc_engine.build_inference_checkpoint(
                _weights(768), os.path.join(folder, "G_100.pth")
            )
        self.assertFalse(complete)
        self.assertEqual(packed["version"], "v2")
        self.assertEqual(packed["f0"], 1)
        self.assertEqual(packed["config"][0], 1025)
        self.assertEqual(packed["config"][1], 32)
        self.assertEqual(packed["config"][-1], 40000)
        self.assertIn("config.json", note)
        self.assertNotIn("enc_q.mean", packed["weight"])

    def test_phone_size_wins_over_the_stated_version(self):
        loaded = {
            "weight": _weights(768),
            "config": [1] * 18,
            "f0": 1,
            "version": "v1",
        }
        packed, _note, complete = rvc_engine.build_inference_checkpoint(loaded, "voz.pth")
        self.assertFalse(complete)
        self.assertEqual(packed["version"], "v2")

    def test_a_voice_without_the_speaker_embedding_is_refused(self):
        weight = _weights(256)
        del weight["emb_g.weight"]
        del weight["enc_q.mean"]
        loaded = {"weight": weight, "config": [1] * 18, "f0": 1, "version": "v1"}
        with self.assertRaises(ValueError) as caught:
            rvc_engine.build_inference_checkpoint(loaded, "voz.pth")
        self.assertIn("hablante", str(caught.exception))

    def test_hubert_config_is_not_a_training_config(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as handle:
                json.dump({"model_type": "hubert"}, handle)
            with self.assertRaises(ValueError):
                rvc_engine.build_inference_checkpoint(
                    _weights(256), os.path.join(folder, "G_1.pth")
                )


class HubertAndIndexTests(unittest.TestCase):
    def test_v1_requires_final_proj_and_v2_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "model.safetensors"), "wb") as handle:
                handle.write(_safetensors(["encoder.weight"]))
            with self.assertRaises(ValueError) as caught:
                rvc_engine._require_hubert_weights(folder, "v1")
            self.assertIn("final_proj", str(caught.exception))
            self.assertTrue(rvc_engine._require_hubert_weights(folder, "v2"))
            with open(os.path.join(folder, "model.safetensors"), "wb") as handle:
                handle.write(_safetensors(["final_proj.weight", "encoder.weight"]))
            self.assertTrue(
                rvc_engine._require_hubert_weights(folder, "v1").endswith(".safetensors")
            )

    def test_index_follows_the_voice_name(self):
        paths = [
            "/voices/otra.index",
            "/voices/added_IVF494_Flat_nprobe_1_mi-voz_v2.index",
        ]
        self.assertEqual(
            rvc_engine.match_index("/voices/mi-voz.pth", paths),
            paths[1],
        )
        self.assertEqual(rvc_engine.match_index("/voices/sola.pth", paths), "")
        self.assertEqual(
            rvc_engine.match_index("/voices/mi-voz.pth", paths + ["/voices/mi-voz.index"]),
            "/voices/mi-voz.index",
        )

    def test_index_dimension_must_match_the_version(self):
        class Index:
            d = 256

        with mock.patch.dict("sys.modules", {"faiss": mock.Mock(read_index=lambda _path: Index())}):
            path, note = rvc_engine._index_for("/voices/mi-voz.index", "v2")
        self.assertIsNone(path)
        self.assertIn("v2", note)
        self.assertIn("sin índice", note.lower())


class SafetyTests(unittest.TestCase):
    def test_an_ordinary_checkpoint_is_allowed(self):
        class Item:
            class safety:
                value = "innocuous"

        class Result:
            globals = [Item()]
            scan_err = False
            issues_count = 2

        with mock.patch("rvc_engine.scan_file_path", return_value=Result()):
            rvc_engine._reject_unsafe_checkpoint("voz.pth")

    def test_executable_pickle_is_refused(self):
        class Item:
            class safety:
                value = "dangerous"

        class Result:
            globals = [Item()]
            scan_err = False
            issues_count = 1

        with mock.patch("rvc_engine.scan_file_path", return_value=Result()):
            with self.assertRaises(ValueError) as caught:
                rvc_engine._reject_unsafe_checkpoint("voz.pth")
        self.assertIn("seguro", str(caught.exception))


class SeparationConfigTests(unittest.TestCase):
    def test_missing_n_fft_for_3072_is_6144(self):
        import app

        class Input:
            shape = [1, 4, 3072, 256]

        class Session:
            def get_inputs(self):
                return [Input()]

        with tempfile.TemporaryDirectory() as folder:
            config = app._mdx_config(os.path.join(folder, "modelo.onnx"), Session())
        self.assertEqual(int(config["n_fft"]), 6144)
        self.assertEqual(app._export_name("/canciones/tema-voz-rvc.wav"), "tema-voz-rvc")


class NameAndDeviceTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_out = rvc_engine.OUTPUT_DIR
        rvc_engine.OUTPUT_DIR = self._tmpdir.name
        rvc_engine._cpu_only = False

    def tearDown(self):
        rvc_engine.OUTPUT_DIR = self._old_out
        rvc_engine._cpu_only = False
        rvc_engine.last_note = ""
        self._tmpdir.cleanup()

    def test_converted_name_keeps_the_song(self):
        self.assertEqual(rvc_engine.song_stem("tema-voz-123456.wav"), "tema")
        self.assertEqual(rvc_engine.song_stem("tema-voz-rvc (2).wav"), "tema")
        self.assertEqual(rvc_engine.song_stem("Mi tema-remix.wav"), "Mi_tema")
        first = rvc_engine._output_path("tema-voz-123456.wav")
        self.assertTrue(first.endswith("tema-voz-rvc.wav"))
        with open(first, "wb") as handle:
            handle.write(b"x")
        second = rvc_engine._output_path("tema-voz-123456.wav")
        self.assertTrue(second.endswith("tema-voz-rvc (2).wav"))

    def test_mps_runs_in_full_precision(self):
        class Config:
            device = "mps"
            is_half = True
            x_pad = 3

        class Converter:
            config = Config()

        rvc_engine._full_precision_on_mps(Converter())
        self.assertFalse(Converter.config.is_half)
        self.assertEqual(Converter.config.x_pad, 1)

    def test_chip_failure_retries_on_cpu(self):
        calls = []

        def callback(only_cpu):
            calls.append(only_cpu)
            if not only_cpu:
                raise RuntimeError("mps")
            return "listo"

        with mock.patch.object(rvc_engine, "accelerator_available", return_value=True):
            value, used_cpu = rvc_engine.run_with_device_fallback(callback)
        self.assertEqual(value, "listo")
        self.assertTrue(used_cpu)
        self.assertEqual(calls, [False, True])
        self.assertTrue(rvc_engine._cpu_only)

    def test_a_voice_error_does_not_retry(self):
        calls = []

        def callback(only_cpu):
            calls.append(only_cpu)
            raise ValueError("Ese archivo es del entrenamiento")

        with mock.patch.object(rvc_engine, "accelerator_available", return_value=True):
            with self.assertRaises(ValueError):
                rvc_engine.run_with_device_fallback(callback)
        self.assertEqual(calls, [False])
        self.assertFalse(rvc_engine._cpu_only)

    def test_convert_reports_the_cpu_retry(self):
        root = self._tmpdir.name
        voice = os.path.join(root, "tema-voz-101010.wav")
        with wave.open(voice, "w") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 160)
        model = os.path.join(root, "mi-voz.pth")
        with open(model, "wb") as handle:
            handle.write(b"checkpoint")
        hubert = os.path.join(root, "hubert_base")
        os.makedirs(hubert)
        with open(os.path.join(hubert, "config.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        rmvpe = os.path.join(root, "rmvpe.pt")
        with open(rmvpe, "wb") as handle:
            handle.write(b"pitch")
        old_rvc = rvc_engine.RVC_DIR
        rvc_engine.RVC_DIR = root

        def execute(*args):
            if not args[-1]:
                raise RuntimeError("mps")
            return voice

        try:
            with mock.patch.object(rvc_engine, "accelerator_available", return_value=True), \
                mock.patch.object(rvc_engine, "prepare_voice_model", return_value=(model, "v2", "Modelo v2.")), \
                mock.patch.object(rvc_engine, "_require_hubert_weights"), \
                mock.patch.object(rvc_engine, "_index_for", return_value=(None, "Sin índice.")), \
                mock.patch.object(rvc_engine, "_execute", side_effect=execute):
                result = rvc_engine.convert_voice(voice, model, pitch=3, index_influence=0.66)
        finally:
            rvc_engine.RVC_DIR = old_rvc
        self.assertEqual(result, voice)
        self.assertIn("Modelo v2", rvc_engine.last_note)
        self.assertIn("Seguí en CPU", rvc_engine.last_note)

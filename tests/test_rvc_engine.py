import os
import tempfile
import unittest
from unittest import mock

import rvc_engine


class RvcEngineTests(unittest.TestCase):
    def test_missing_audio(self):
        with self.assertRaises(ValueError):
            rvc_engine.convert_voice("", "model.pth")

    def test_missing_model_file(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(b"x")
        tmp.close()
        self.addCleanup(lambda: os.remove(tmp.name))
        with self.assertRaises(ValueError):
            rvc_engine.convert_voice(tmp.name, "/no/existe.pth")

    def test_missing_support_models(self):
        with mock.patch.object(rvc_engine, "local_hubert_path", return_value=None):
            with mock.patch.object(rvc_engine, "local_rmvpe_path", return_value=None):
                with self.assertRaises(ValueError) as ctx:
                    rvc_engine.require_support_models()
        self.assertIn("Faltan los modelos locales", str(ctx.exception))
        self.assertIn("Completar instalación", str(ctx.exception))

    def test_convert_blocked_during_train(self):
        import occupancy

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        audio = os.path.join(tmp.name, "v.wav")
        model = os.path.join(tmp.name, "m.pth")
        for path in (audio, model):
            with open(path, "wb") as handle:
                handle.write(b"data")
        occ = occupancy.Occupancy("train", exp="voz")
        with mock.patch("occupancy._snapshot_unlocked", return_value=occ):
            with self.assertRaises(ValueError) as ctx:
                rvc_engine.convert_voice(audio, model, copy_downloads=False)
        self.assertIn("Convertir", str(ctx.exception))

    def test_convert_calls_vc_runner(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        audio = os.path.join(tmp.name, "pibe-voz.wav")
        model = os.path.join(tmp.name, "Palandri.pth")
        out = os.path.join(tmp.name, "out.wav")
        for path in (audio, model, out):
            with open(path, "wb") as handle:
                handle.write(b"data")
        with mock.patch("occupancy._snapshot_unlocked", return_value=None):
            with mock.patch.object(rvc_engine, "require_support_models"):
                with mock.patch.object(rvc_engine, "sync_support_into_applio"):
                    with mock.patch.object(rvc_engine, "_scan_model"):
                        with mock.patch.object(rvc_engine, "run_vc_infer", return_value=out) as infer:
                            with mock.patch.object(
                                rvc_engine,
                                "copy_to_downloads",
                                return_value=(tmp.name, [out]),
                            ) as copied:
                                result = rvc_engine.convert_voice(
                                    audio, model, index_path="/tmp/model.index"
                                )
        self.assertEqual(result, out)
        infer.assert_called_once()
        self.assertEqual(infer.call_args[0][0], audio)
        self.assertEqual(copied.call_args[0][1], ["pibe-voz-Palandri"])

    def test_convert_passes_pitch_and_index_rate(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        audio = os.path.join(tmp.name, "v.wav")
        model = os.path.join(tmp.name, "m.pth")
        out = os.path.join(tmp.name, "out.wav")
        for path in (audio, model, out):
            with open(path, "wb") as handle:
                handle.write(b"data")
        with mock.patch("occupancy._snapshot_unlocked", return_value=None):
            with mock.patch.object(rvc_engine, "require_support_models"):
                with mock.patch.object(rvc_engine, "sync_support_into_applio"):
                    with mock.patch.object(rvc_engine, "_scan_model"):
                        with mock.patch.object(rvc_engine, "run_vc_infer", return_value=out) as infer:
                            rvc_engine.convert_voice(
                                audio,
                                model,
                                pitch=2,
                                index_rate=0.8,
                                f0_method="rmvpe",
                                copy_downloads=False,
                            )
        kwargs = infer.call_args.kwargs
        self.assertEqual(kwargs["pitch"], 2)
        self.assertEqual(kwargs["index_rate"], 0.8)
        self.assertEqual(kwargs["filter_radius"], 0)

    def test_hubert_pth_upload_is_ignored(self):
        from app_jobs import load_support_weights

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = os.path.join(tmp.name, "voice.pth")
        with open(fake, "wb") as handle:
            handle.write(b"x")
        with mock.patch("library.register") as register:
            with mock.patch("library.place_voice_file") as place:
                status = load_support_weights(fake, None, None, None)
        register.assert_not_called()
        place.assert_not_called()
        self.assertNotIsInstance(status, tuple)
        text = status.get("value", status) if isinstance(status, dict) else status
        self.assertIn("hubert ignorado", str(text))

    def test_voice_load_does_not_register_support_files(self):
        from app_jobs import load_voice_into_library

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        voice = os.path.join(tmp.name, "Maria.pth")
        with open(voice, "wb") as handle:
            handle.write(b"x")
        installed = os.path.join(tmp.name, "Maria", "Maria.pth")
        choices = [("Maria", installed)]
        with mock.patch("library.place_voice_file", return_value=installed) as place:
            with mock.patch("library.place_beside") as beside:
                with mock.patch("library.register") as register:
                    with mock.patch("library.list_rvc_voices", return_value=[]):
                        with mock.patch("library.dropdown_choices", return_value=choices):
                            pick, status, model_upd, index_upd = load_voice_into_library(
                                voice, None, "otra.pth"
                            )
        place.assert_called_once()
        beside.assert_not_called()
        register.assert_not_called()
        self.assertEqual(pick["value"], installed)
        self.assertIsNone(model_upd["value"])
        self.assertIsNone(index_upd["value"])
        text = status.get("value", status) if isinstance(status, dict) else status
        self.assertIn("Maria.pth", str(text))
        self.assertNotIn("hubert", str(text))

    def test_convert_uses_the_dropdown_model(self):
        from app_jobs import rvc_job

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(b"x")
        tmp.close()
        self.addCleanup(lambda: os.remove(tmp.name))
        model = os.path.join(os.path.dirname(tmp.name), "Elegida.pth")
        with mock.patch("library.is_installed_voice", return_value=True):
            with mock.patch("library.find_index_for_model", return_value=None):
                with mock.patch("library.place_voice_file") as place:
                    with mock.patch("rvc_engine.convert_voice", return_value=tmp.name) as convert:
                        rvc_job(tmp.name, model)
        place.assert_not_called()
        self.assertEqual(convert.call_args.args[1], model)

    def test_sync_support_into_applio_links_local_files(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        hubert = os.path.join(tmp.name, "hubert_base")
        os.makedirs(hubert)
        with open(os.path.join(hubert, "config.json"), "w", encoding="utf-8") as handle:
            handle.write("{}")
        with open(os.path.join(hubert, "model.safetensors"), "wb") as handle:
            handle.write(b"w")
        rmvpe = os.path.join(tmp.name, "rmvpe.pt")
        with open(rmvpe, "wb") as handle:
            handle.write(b"r")
        vc = os.path.join(tmp.name, "vc")
        with mock.patch.object(rvc_engine, "local_hubert_path", return_value=hubert):
            with mock.patch.object(rvc_engine, "local_rmvpe_path", return_value=rmvpe):
                with mock.patch.object(rvc_engine, "vc_root", return_value=vc):
                    rvc_engine.sync_support_into_applio()
        dest = os.path.join(vc, "rvc", "models", "embedders", "contentvec", "config.json")
        self.assertTrue(os.path.lexists(dest))
        pred = os.path.join(vc, "rvc", "models", "predictors", "rmvpe.pt")
        self.assertTrue(os.path.lexists(pred))

    def test_convert_source_requires_local_weights(self):
        import inspect

        src = inspect.getsource(rvc_engine.convert_voice)
        self.assertIn("require_support_models", src)
        self.assertIn("sync_support_into_applio", src)


if __name__ == "__main__":
    unittest.main()

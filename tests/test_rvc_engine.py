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

    def test_convert_calls_vc_runner(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        audio = os.path.join(tmp.name, "v.wav")
        model = os.path.join(tmp.name, "m.pth")
        out = os.path.join(tmp.name, "out.wav")
        for path in (audio, model, out):
            with open(path, "wb") as handle:
                handle.write(b"data")
        with mock.patch.object(rvc_engine, "_scan_model"):
            with mock.patch.object(rvc_engine, "run_vc_infer", return_value=out) as infer:
                with mock.patch.object(
                    rvc_engine,
                    "copy_to_downloads",
                    return_value=(tmp.name, [out]),
                ):
                    result = rvc_engine.convert_voice(
                        audio, model, index_path="/tmp/model.index"
                    )
        self.assertEqual(result, out)
        infer.assert_called_once()
        self.assertEqual(infer.call_args[0][0], audio)

    def test_convert_passes_pitch_and_index_rate(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        audio = os.path.join(tmp.name, "v.wav")
        model = os.path.join(tmp.name, "m.pth")
        out = os.path.join(tmp.name, "out.wav")
        for path in (audio, model, out):
            with open(path, "wb") as handle:
                handle.write(b"data")
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

    def test_hubert_pth_upload_is_ignored(self):
        from app_jobs import load_rvc_into_library

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fake = os.path.join(tmp.name, "voice.pth")
        with open(fake, "wb") as handle:
            handle.write(b"x")
        with mock.patch("app_jobs.refresh_library_ui", return_value=([], [])):
            with mock.patch("library.register") as register:
                _a, _b, status = load_rvc_into_library(
                    fake, None, None, None
                )
        register.assert_not_called()
        text = status.get("value", status) if isinstance(status, dict) else status
        self.assertIn("hubert ignorado", str(text))


if __name__ == "__main__":
    unittest.main()

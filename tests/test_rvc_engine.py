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

    def test_convert_calls_local_loader(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        audio = os.path.join(tmp.name, "v.wav")
        model = os.path.join(tmp.name, "m.pth")
        out = os.path.join(tmp.name, "out.wav")
        for path in (audio, model, out):
            with open(path, "wb") as handle:
                handle.write(b"data")

        fake = mock.Mock()
        fake.apply_conf.return_value = "ok"
        fake.return_value = [out]

        with mock.patch.object(rvc_engine, "_scan_model"):
            with mock.patch.object(rvc_engine, "get_converter", return_value=fake):
                with mock.patch.object(
                    rvc_engine, "copy_to_downloads", return_value=(tmp.name, [out])
                ):
                    result = rvc_engine.convert_voice(
                        audio, model, index_path="/tmp/model.index"
                    )

        self.assertEqual(result, out)
        fake.apply_conf.assert_called_once()
        kwargs = fake.apply_conf.call_args.kwargs
        self.assertEqual(kwargs["pitch_algo"], "rmvpe")
        self.assertEqual(kwargs["pitch_lvl"], 0)
        self.assertEqual(kwargs["file_index"], "/tmp/model.index")
        fake.assert_called_once()


if __name__ == "__main__":
    unittest.main()

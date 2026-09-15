import os
import tempfile
import unittest
from unittest import mock

import clone_engine


class CloneEngineTests(unittest.TestCase):
    def test_empty_text(self):
        with self.assertRaises(ValueError):
            clone_engine.clone_voice("  ", "/tmp/ref.wav")

    def test_missing_speaker(self):
        with self.assertRaises(ValueError):
            clone_engine.clone_voice("Hola", "/no/existe.wav")

    def test_missing_xtts_files(self):
        with mock.patch.object(
            clone_engine,
            "missing_xtts_files",
            return_value=["xtts_models/model.pth"],
        ):
            with self.assertRaises(ValueError) as ctx:
                clone_engine.require_xtts_models()
        self.assertIn("Faltan los modelos locales de XTTS", str(ctx.exception))

    def test_clone_calls_full_xtts(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        speaker = os.path.join(tmp.name, "ref.wav")
        with open(speaker, "wb") as handle:
            handle.write(b"wav")
        fake = mock.Mock()

        def write_file(**kwargs):
            path = kwargs["file_path"]
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(b"out")

        fake.synthesize.return_value = {"wav": [0.0, 0.1, 0.0]}
        with mock.patch.object(clone_engine, "get_tts", return_value=(fake, object(), "cpu")):
            with mock.patch.object(
                clone_engine,
                "copy_to_downloads",
                side_effect=lambda paths, labels: (tmp.name, paths),
            ):
                result = clone_engine.clone_voice("Hola mundo", speaker)
        fake.synthesize.assert_called_once()
        kwargs = fake.synthesize.call_args.kwargs
        self.assertEqual(kwargs["language"], "es")
        self.assertTrue(str(result).endswith("voz_clon.wav"))


if __name__ == "__main__":
    unittest.main()

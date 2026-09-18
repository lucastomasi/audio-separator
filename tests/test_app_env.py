import os
import unittest
from unittest import mock

import app_env


class AppEnvTests(unittest.TestCase):
    def test_data_dir_respects_env(self):
        with mock.patch.dict(os.environ, {"AUDIO_SEPARATOR_DATA": "/tmp/as-data-test"}):
            path = app_env.data_dir()
        self.assertTrue(path.endswith("as-data-test"))

    def test_host_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUDIO_SEPARATOR_HOST", None)
            self.assertEqual(app_env.host(), "127.0.0.1")

    def test_no_user_path_in_module(self):
        import inspect

        src = inspect.getsource(app_env)
        self.assertNotIn("lucastomasi", src)

    def test_ui_copy_admits_youtube_and_edge(self):
        from pathlib import Path

        src = Path(__file__).resolve().parents[1] / "app.py"
        text = src.read_text(encoding="utf-8")
        self.assertIn("YouTube", text)
        self.assertIn("Edge", text)
        self.assertNotIn("sin subir audio", text.lower())


if __name__ == "__main__":
    unittest.main()

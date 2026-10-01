"""Writable data follows AUDIO_SEPARATOR_HOME when the Mac app sets it."""
import os
import tempfile
import unittest

import app_paths


class AppPathsTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("AUDIO_SEPARATOR_HOME", None)

    def test_source_dir_contains_desktop(self):
        self.assertTrue(os.path.isfile(os.path.join(app_paths.source_dir(), "desktop.py")))

    def test_data_dir_defaults_to_source(self):
        os.environ.pop("AUDIO_SEPARATOR_HOME", None)
        self.assertEqual(app_paths.data_dir(), app_paths.source_dir())

    def test_data_dir_follows_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["AUDIO_SEPARATOR_HOME"] = tmp
            self.assertEqual(app_paths.data_dir(), os.path.abspath(tmp))
            self.assertTrue(os.path.isdir(tmp))

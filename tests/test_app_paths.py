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
            self.assertEqual(app_paths.data_dir(), os.path.realpath(tmp))
            self.assertTrue(os.path.isdir(tmp))

    def test_data_dir_rejects_tmp_root(self):
        os.environ["AUDIO_SEPARATOR_HOME"] = "/tmp"
        with self.assertRaises(ValueError):
            app_paths.data_dir()

    def test_data_dir_rejects_slash(self):
        os.environ["AUDIO_SEPARATOR_HOME"] = "/"
        with self.assertRaises(ValueError):
            app_paths.data_dir()

    def test_is_inside_rejects_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "root")
            outside = os.path.join(tmp, "outside.txt")
            os.makedirs(root)
            with open(outside, "w", encoding="utf-8") as handle:
                handle.write("secret")
            link = os.path.join(root, "escape.txt")
            os.symlink(outside, link)
            self.assertTrue(app_paths.is_inside(os.path.join(root, "ok"), root))
            self.assertFalse(app_paths.is_inside(link, root))

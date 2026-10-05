"""Exports only copy files that live in this app's output folders."""
import os
import tempfile
import unittest

import exports


class ExportableTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("AUDIO_SEPARATOR_HOME", None)

    def test_rejects_symlink_outside(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["AUDIO_SEPARATOR_HOME"] = tmp
            clean = os.path.join(tmp, "clean_song_output")
            os.makedirs(clean)
            secret = os.path.join(tmp, "secret.wav")
            with open(secret, "wb") as handle:
                handle.write(b"RIFF")
            link = os.path.join(clean, "voz.wav")
            os.symlink(secret, link)
            self.assertFalse(exports.is_exportable(link))

    def test_accepts_real_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["AUDIO_SEPARATOR_HOME"] = tmp
            clean = os.path.join(tmp, "clean_song_output")
            os.makedirs(clean)
            wav = os.path.join(clean, "voz.wav")
            with open(wav, "wb") as handle:
                handle.write(b"RIFF")
            self.assertTrue(exports.is_exportable(wav))

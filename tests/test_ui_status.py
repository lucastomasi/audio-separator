import os
import tempfile
import unittest
from unittest import mock

import ui_status


class UiStatusTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log = os.path.join(tmp.name, "audio-separator.log")
        self.patch = mock.patch.object(ui_status, "log_file", return_value=self.log)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_fail_hides_nameerror(self):
        msg = ui_status.fail(
            "test", NameError("name 'sf' is not defined"), ui_status.MSG_SEPARATE
        )
        self.assertEqual(msg, ui_status.MSG_SEPARATE)
        self.assertNotIn("sf", msg)

    def test_fail_keeps_short_valueerror(self):
        msg = ui_status.fail("test", ValueError("Falta el archivo de audio."), "x")
        self.assertEqual(msg, "Falta el archivo de audio.")

    def test_fail_drops_ffmpeg_blob(self):
        raw = ValueError("No pude pasar el audio a WAV 44.1 kHz estéreo. /tmp/x.wav")
        msg = ui_status.fail("test", raw, ui_status.MSG_SEPARATE)
        self.assertEqual(msg, ui_status.MSG_SEPARATE)

    def test_status_error_sets_class(self):
        upd = ui_status.status_update(ui_status.KIND_ERROR, "boom")
        self.assertIn("is-error", upd.get("elem_classes") or upd["elem_classes"])

    def test_append_log_writes(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "audio-separator.log")
        with mock.patch.object(ui_status, "log_file", return_value=path):
            ui_status.append_log("hola")
        with open(path, encoding="utf-8") as handle:
            self.assertIn("hola", handle.read())


if __name__ == "__main__":
    unittest.main()

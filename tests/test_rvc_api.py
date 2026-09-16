import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import rvc_api


class RvcApiTests(unittest.TestCase):
    def test_resolve_voice(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        pth = root / "demo.pth"
        index = root / "demo.index"
        pth.write_bytes(b"x")
        index.write_bytes(b"y")
        with mock.patch.object(rvc_api, "VOICE_DIR", root):
            with mock.patch.object(
                rvc_api, "find_index_for_model", return_value=str(index)
            ):
                got_pth, got_index = rvc_api.resolve_voice("demo")
        self.assertEqual(got_pth, pth)
        self.assertEqual(got_index, index)

    def test_resolve_missing(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        with mock.patch.object(rvc_api, "VOICE_DIR", Path(tmp.name)):
            with mock.patch.object(rvc_api, "list_voices", return_value=[]):
                with self.assertRaises(FileNotFoundError):
                    rvc_api.resolve_voice("nope")

    def test_require_exclusive_cli_aborts(self):
        fake = "12345 /Users/lucastomasi/grok/Audio_separator/.venv/bin/python -u desktop.py"
        with mock.patch.object(rvc_api, "_conflicting_rvc_processes", return_value=[fake]):
            with self.assertRaises(SystemExit) as ctx:
                rvc_api.require_exclusive_cli()
        self.assertEqual(ctx.exception.code, 1)

    def test_require_exclusive_cli_ok_when_alone(self):
        with mock.patch.object(rvc_api, "_conflicting_rvc_processes", return_value=[]):
            rvc_api.require_exclusive_cli()


if __name__ == "__main__":
    unittest.main()

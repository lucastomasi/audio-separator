import os
import tempfile
import unittest
from unittest import mock

import exports


class ExportsTests(unittest.TestCase):
    def test_unique_path_adds_index(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        first = os.path.join(tmp.name, "voz.wav")
        open(first, "w").close()
        second = exports.unique_path(tmp.name, "voz.wav")
        self.assertEqual(os.path.basename(second), "voz (2).wav")

    def test_copy_to_downloads(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = os.path.join(tmp.name, "src.wav")
        with open(src, "wb") as handle:
            handle.write(b"abc")
        dest_root = os.path.join(tmp.name, "out")
        with mock.patch.object(exports, "exports_dir", return_value=dest_root):
            directory, copied = exports.copy_to_downloads([src], ["voz"])
        self.assertEqual(directory, dest_root)
        self.assertEqual(len(copied), 1)
        self.assertTrue(copied[0].endswith("voz.wav"))
        with open(copied[0], "rb") as handle:
            self.assertEqual(handle.read(), b"abc")


if __name__ == "__main__":
    unittest.main()

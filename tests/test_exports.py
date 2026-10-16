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

    def test_unique_path_uses_basename_only(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = exports.unique_path(tmp.name, "../secret.wav")
        self.assertEqual(os.path.dirname(dest), tmp.name)
        self.assertEqual(os.path.basename(dest), "secret.wav")

    def test_export_label_uses_song_and_role(self):
        self.assertEqual(exports.export_stem("/tmp/gradio/x/pibe-voz.wav"), "pibe-voz")
        self.assertEqual(exports.song_stem("/tmp/pibe-voz.wav"), "pibe")
        self.assertEqual(exports.song_stem("/tmp/pibe-instrumental.wav"), "pibe")
        self.assertEqual(
            exports.export_label(exports.song_stem("pibe-voz.wav"), "voz", "Palandri"),
            "pibe-voz-Palandri",
        )
        self.assertEqual(
            exports.export_label(exports.song_stem("pibe-instrumental.wav"), "unir"),
            "pibe-unir",
        )
        self.assertEqual(
            exports.export_stem({"orig_name": "El Pibe.mov", "path": "/tmp/hash"}),
            "El-Pibe",
        )

    def test_copy_to_downloads_only_exportable(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        allowed_root = os.path.join(tmp.name, "Trabajos")
        os.makedirs(allowed_root)
        src = os.path.join(allowed_root, "src.wav")
        with open(src, "wb") as handle:
            handle.write(b"abc")
        outside = os.path.join(tmp.name, "secret.wav")
        with open(outside, "wb") as handle:
            handle.write(b"no")
        dest_root = os.path.join(tmp.name, "out")
        with mock.patch.object(exports, "exports_dir", return_value=dest_root), mock.patch.object(
            exports, "_export_roots", return_value=[allowed_root, dest_root]
        ):
            directory, copied = exports.copy_to_downloads([src, outside], ["voz", "leak"])
        self.assertEqual(directory, dest_root)
        self.assertEqual(len(copied), 1)
        self.assertTrue(copied[0].endswith("voz.wav"))
        with open(copied[0], "rb") as handle:
            self.assertEqual(handle.read(), b"abc")

    def test_copy_to_downloads_uses_song_subfolder(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        allowed_root = os.path.join(tmp.name, "Trabajos")
        os.makedirs(allowed_root)
        src = os.path.join(allowed_root, "src.wav")
        with open(src, "wb") as handle:
            handle.write(b"abc")
        dest_root = os.path.join(tmp.name, "out")
        with mock.patch.object(exports, "exports_dir", return_value=dest_root), mock.patch.object(
            exports, "_export_roots", return_value=[allowed_root, dest_root]
        ):
            directory, copied = exports.copy_to_downloads(
                [src], ["pibe-voz"], song="pibe"
            )
        self.assertEqual(
            os.path.realpath(directory),
            os.path.realpath(os.path.join(dest_root, "pibe")),
        )
        self.assertEqual(len(copied), 1)
        self.assertEqual(os.path.basename(copied[0]), "pibe-voz.wav")
        self.assertTrue(os.path.isfile(copied[0]))

    def test_song_stem_strips_video_and_acordes(self):
        self.assertEqual(exports.song_stem("tema-video.mp4"), "tema")
        self.assertEqual(exports.song_stem("tema-acordes.txt"), "tema")


if __name__ == "__main__":
    unittest.main()

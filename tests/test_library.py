import json
import os
import tempfile
import unittest
from unittest import mock

import library


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "library")
        self.patch = mock.patch.multiple(
            library,
            ROOT=self.root,
            INDEX=os.path.join(self.root, "library.json"),
            PATHS={
                "uvr": os.path.join(self.root, "models", "uvr"),
                "rvc": os.path.join(self.root, "models", "rvc"),
                "rvc_voices": os.path.join(self.root, "models", "rvc_voices"),
                "voices": os.path.join(self.root, "voices"),
            },
        )
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_empty_lists(self):
        self.assertEqual(library.list_rvc_voices(), [])
        self.assertEqual(library.list_voices(), [])

    def test_paths_have_no_xtts(self):
        self.assertNotIn("xtts", library.PATHS)
        self.assertFalse(hasattr(library, "xtts_dir"))

    def test_ensure_dirs_prunes_legacy_xtts(self):
        leftover = os.path.join(self.root, "models", "xtts")
        os.makedirs(leftover)
        with open(os.path.join(leftover, "old.bin"), "wb") as handle:
            handle.write(b"x")
        with mock.patch.object(library, "data_home", return_value=self.tmp.name):
            with mock.patch.object(library, "seed_support_weights"):
                library.ensure_dirs()
        self.assertFalse(os.path.isdir(leftover))
        self.assertTrue(os.path.isdir(library.PATHS["rvc"]))
        self.assertTrue(os.path.isdir(library.PATHS["uvr"]))
        self.assertTrue(os.path.isdir(library.PATHS["voices"]))

    def test_ensure_dirs_drops_xtts_index_items(self):
        os.makedirs(self.root, exist_ok=True)
        with open(library.INDEX, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "items": [
                        {"kind": "xtts", "name": "old", "path": "/tmp/models/xtts/a.pth"},
                        {"kind": "voices", "name": "ok", "path": "/tmp/voices/ok.wav"},
                    ]
                },
                handle,
            )
        with mock.patch.object(library, "data_home", return_value=self.tmp.name):
            with mock.patch.object(library, "seed_support_weights"):
                library.ensure_dirs()
        with open(library.INDEX, encoding="utf-8") as handle:
            data = json.load(handle)
        self.assertEqual([item["kind"] for item in data["items"]], ["voices"])

    def test_register_and_list_voice(self):
        src = os.path.join(self.tmp.name, "clip.wav")
        with open(src, "wb") as handle:
            handle.write(b"wav")
        item = library.register("voices", src, "clip.wav")
        self.assertTrue(os.path.isfile(item["path"]))
        names = [row["name"] for row in library.list_voices()]
        self.assertIn("clip", names)

    def test_recent_media_lists_newest_download_first(self):
        downloads = os.path.join(self.tmp.name, "downloads")
        os.makedirs(downloads)
        older = os.path.join(downloads, "viejo.wav")
        newer = os.path.join(downloads, "nuevo.mp3")
        for path in (older, newer):
            with open(path, "wb") as handle:
                handle.write(b"x")
        os.utime(older, (1_000, 1_000))
        os.utime(newer, (2_000, 2_000))
        with mock.patch.object(library, "data_home", return_value=self.tmp.name):
            choices = library.recent_media()
        self.assertEqual(choices[0][1], os.path.abspath(newer))
        self.assertIn(os.path.abspath(older), [path for _label, path in choices])

    def test_register_rvc(self):
        src = os.path.join(self.tmp.name, "voice.pth")
        with open(src, "wb") as handle:
            handle.write(b"pth")
        library.register("rvc_voices", src)
        self.assertEqual(len(library.list_rvc_voices()), 1)

    def test_find_index_for_model(self):
        folder = library.PATHS["rvc_voices"]
        os.makedirs(folder, exist_ok=True)
        model = os.path.join(folder, "singer.pth")
        index = os.path.join(folder, "singer.index")
        open(model, "wb").close()
        open(index, "wb").close()
        self.assertEqual(library.find_index_for_model(model), index)
        voices = library.list_rvc_voices()
        self.assertEqual(voices[0]["index"], index)
        choices = library.dropdown_choices(voices)
        self.assertIn("+index", choices[0][0])

    def test_voice_folder_wins_over_flat_copy(self):
        flat_dir = library.PATHS["rvc_voices"]
        os.makedirs(flat_dir, exist_ok=True)
        flat = os.path.join(flat_dir, "ana.pth")
        with open(flat, "wb") as handle:
            handle.write(b"flat")
        owned = library.place_voice_file(flat, "ana")
        with open(owned, "wb") as handle:
            handle.write(b"folder")
        voices = library.list_rvc_voices()
        self.assertEqual([row["name"] for row in voices], ["ana"])
        self.assertEqual(os.path.abspath(voices[0]["path"]), os.path.abspath(owned))
        self.assertTrue(library.is_installed_voice(owned))
        outside = os.path.join(self.tmp.name, "other.pth")
        with open(outside, "wb") as handle:
            handle.write(b"x")
        self.assertFalse(library.is_installed_voice(outside))

    def test_register_rejects_path_escape(self):
        src = os.path.join(self.tmp.name, "clip.wav")
        with open(src, "wb") as handle:
            handle.write(b"wav")
        with self.assertRaises(ValueError):
            library.register("voices", src, "../escape.wav")
        escaped = os.path.join(self.tmp.name, "escape.wav")
        self.assertFalse(os.path.isfile(escaped))

    def test_session_meta(self):
        library.set_session_meta(last_video_path="/tmp/a.mp4")
        self.assertEqual(library.get_session_meta()["last_video_path"], "/tmp/a.mp4")
        library.set_session_meta(last_video_path=None)
        self.assertNotIn("last_video_path", library.get_session_meta())


if __name__ == "__main__":
    unittest.main()

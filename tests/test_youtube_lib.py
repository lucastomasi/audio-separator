import os
import tempfile
import unittest
from unittest import mock

from youtube_lib import (
    download_audio,
    extract_youtube_id,
    existing_audio,
    ydl_options,
)


class YoutubeIdTests(unittest.TestCase):
    def test_watch_and_short_urls(self):
        self.assertEqual(
            extract_youtube_id("https://www.youtube.com/watch?v=jNQXAC9IVRw"),
            "jNQXAC9IVRw",
        )
        self.assertEqual(extract_youtube_id("https://youtu.be/jNQXAC9IVRw"), "jNQXAC9IVRw")
        self.assertEqual(
            extract_youtube_id("www.youtube.com/watch?v=jNQXAC9IVRw"),
            "jNQXAC9IVRw",
        )

    def test_empty(self):
        self.assertIsNone(extract_youtube_id(""))
        self.assertIsNone(extract_youtube_id(None))


class DownloadAudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_url(self):
        with self.assertRaises(ValueError):
            download_audio("  ", directory=self.dir)

    def test_skips_redownload_if_mp3_exists(self):
        path = os.path.join(self.dir, "jNQXAC9IVRw.mp3")
        with open(path, "wb") as handle:
            handle.write(b"fake-audio")
        with mock.patch("yt_dlp.YoutubeDL") as ydl_cls:
            result, reused, note = download_audio(
                "https://youtu.be/jNQXAC9IVRw", directory=self.dir
            )
        self.assertTrue(reused)
        self.assertIsNone(note)
        self.assertEqual(result, os.path.abspath(path))
        ydl_cls.assert_not_called()

    def test_download_writes_mp3(self):
        video_id = "abcdefghijk"

        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=True):
                dest = os.path.join(self.dir, f"{video_id}.mp3")
                with open(dest, "wb") as handle:
                    handle.write(b"mp3-bytes")
                return {"id": video_id, "ext": "mp3"}

        FakeYDL.dir = self.dir

        with mock.patch("yt_dlp.YoutubeDL", FakeYDL):
            path, reused, note = download_audio(
                "https://youtu.be/abcdefghijk", directory=self.dir
            )
        self.assertFalse(reused)
        self.assertIsNone(note)
        self.assertTrue(path.endswith(".mp3"))
        self.assertTrue(os.path.isfile(path))

    def test_options_are_audio_only(self):
        opts = ydl_options(self.dir)
        self.assertIn("bestaudio", opts["format"])
        self.assertTrue(opts["noplaylist"])
        self.assertEqual(opts["postprocessors"][0]["preferredcodec"], "mp3")


if __name__ == "__main__":
    unittest.main()

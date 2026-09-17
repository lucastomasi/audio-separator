import os
import tempfile
import unittest
from unittest import mock

from youtube_lib import (
    clip_audio,
    download_audio,
    extract_youtube_id,
    parse_seconds,
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


class ParseSecondsTests(unittest.TestCase):
    def test_mm_ss_and_number(self):
        self.assertEqual(parse_seconds("0:15"), 15)
        self.assertEqual(parse_seconds("1:02"), 62)
        self.assertEqual(parse_seconds(25), 25)
        self.assertIsNone(parse_seconds(""))

    def test_clip_requires_times(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(b"x")
        tmp.close()
        self.addCleanup(lambda: os.remove(tmp.name))
        with self.assertRaises(ValueError):
            clip_audio(tmp.name, None, None)

    def test_clip_calls_ffmpeg(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = os.path.join(tmp.name, "song.mp3")
        with open(src, "wb") as handle:
            handle.write(b"abc")

        def fake_run(cmd, capture_output=True, text=True):
            dest = cmd[-1]
            with open(dest, "wb") as handle:
                handle.write(b"clip")
            return mock.Mock(returncode=0, stderr="")

        with mock.patch("youtube_lib.ffmpeg_binary", return_value="ffmpeg"):
            with mock.patch("youtube_lib.subprocess.run", side_effect=fake_run):
                path = clip_audio(src, "0:10", "0:20", directory=tmp.name)
        self.assertTrue(path.endswith("song_10-20.wav"))
        self.assertTrue(os.path.isfile(path))


class DownloadAudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_url(self):
        with self.assertRaises(ValueError):
            download_audio("  ", directory=self.dir)

    def test_skips_redownload_if_wav_exists(self):
        path = os.path.join(self.dir, "jNQXAC9IVRw.wav")
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

    def test_download_writes_wav(self):
        video_id = "abcdefghijk"

        class FakeYDL:
            def __init__(self, opts):
                self.opts = opts

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=True):
                dest = os.path.join(self.dir, f"{video_id}.wav")
                with open(dest, "wb") as handle:
                    handle.write(b"RIFF" + b"\x00" * 12 + b"WAVEfmt ")
                return {"id": video_id, "ext": "wav"}

        FakeYDL.dir = self.dir

        with mock.patch("yt_dlp.YoutubeDL", FakeYDL):
            path, reused, note = download_audio(
                "https://youtu.be/abcdefghijk", directory=self.dir
            )
        self.assertFalse(reused)
        self.assertIsNone(note)
        self.assertTrue(path.endswith(".wav"))
        self.assertTrue(os.path.isfile(path))

    def test_options_are_audio_only(self):
        opts = ydl_options(self.dir)
        self.assertIn("bestaudio", opts["format"])
        self.assertTrue(opts["noplaylist"])
        self.assertEqual(opts["postprocessors"][0]["preferredcodec"], "wav")
        self.assertEqual(opts.get("concurrent_fragment_downloads"), 4)


if __name__ == "__main__":
    unittest.main()

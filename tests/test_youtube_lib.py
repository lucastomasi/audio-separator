"""YouTube downloads accept only reconstructed youtube.com watch URLs."""
import unittest
from unittest import mock

from audio_text import extract_youtube_id, youtube_watch_url
import youtube_lib


class YoutubeWatchUrlTests(unittest.TestCase):
    def test_watch_url(self):
        url, video_id = youtube_watch_url(
            "https://www.youtube.com/watch?v=dQw4w9wgWcQ"
        )
        self.assertEqual(video_id, "dQw4w9wgWcQ")
        self.assertEqual(url, "https://www.youtube.com/watch?v=dQw4w9wgWcQ")

    def test_short_url(self):
        url, video_id = youtube_watch_url("https://youtu.be/dQw4w9wgWcQ")
        self.assertEqual(video_id, "dQw4w9wgWcQ")
        self.assertEqual(url, "https://www.youtube.com/watch?v=dQw4w9wgWcQ")

    def test_rejects_localhost(self):
        with self.assertRaises(ValueError):
            youtube_watch_url("http://127.0.0.1/watch?v=dQw4w9wgWcQ")

    def test_rejects_lan(self):
        with self.assertRaises(ValueError):
            youtube_watch_url("http://192.168.1.8/watch?v=dQw4w9wgWcQ")

    def test_rejects_file(self):
        with self.assertRaises(ValueError):
            youtube_watch_url("file:///etc/passwd")

    def test_rejects_other_host(self):
        with self.assertRaises(ValueError):
            youtube_watch_url("https://example.com/watch?v=dQw4w9wgWcQ")

    def test_rejects_userinfo(self):
        with self.assertRaises(ValueError):
            youtube_watch_url("https://u:p@www.youtube.com/watch?v=dQw4w9wgWcQ")

    def test_id_in_evil_url_is_not_enough(self):
        self.assertEqual(
            extract_youtube_id("http://127.0.0.1/watch?v=dQw4w9wgWcQ"),
            "dQw4w9wgWcQ",
        )
        with self.assertRaises(ValueError):
            youtube_watch_url("http://127.0.0.1/watch?v=dQw4w9wgWcQ")


class YdlOptionsTests(unittest.TestCase):
    def test_no_remote_javascript(self):
        opts = youtube_lib.ydl_options("/tmp")
        self.assertNotIn("remote_components", opts)
        self.assertNotIn("js_runtimes", opts)


class DownloadAudioTests(unittest.TestCase):
    def test_bad_url_does_not_import_ydl(self):
        with mock.patch("builtins.__import__", side_effect=ImportError("yt_dlp")) as importer:
            with self.assertRaises(ValueError) as caught:
                youtube_lib.download_audio("http://127.0.0.1/secret")
            self.assertIn("YouTube", str(caught.exception))
            self.assertFalse(
                any(call.args and call.args[0] == "yt_dlp" for call in importer.call_args_list)
            )

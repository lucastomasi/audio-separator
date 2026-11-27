import os
import tempfile
import unittest
from unittest import mock

from youtube_lib import (
    clip_audio,
    download_audio,
    extract_audio_from_media,
    extract_youtube_id,
    ffmpeg_binary,
    identity_line,
    open_downloads_folder,
    parse_seconds,
    probe_youtube,
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

    def test_extract_audio_from_video(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = os.path.join(tmp.name, "clip.mp4")
        dest = os.path.join(tmp.name, "clip.wav")
        with open(src, "wb") as handle:
            handle.write(b"mp4")

        def fake_run(cmd, capture_output=True):
            with open(cmd[-1], "wb") as handle:
                handle.write(b"RIFF")
            return mock.Mock(returncode=0)

        with mock.patch("youtube_lib.ffmpeg_binary", return_value="ffmpeg"):
            with mock.patch("youtube_lib.subprocess.run", side_effect=fake_run):
                path = extract_audio_from_media(src, dest)
        self.assertEqual(path, os.path.abspath(dest))
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

    def test_rejects_non_youtube_url(self):
        with mock.patch("yt_dlp.YoutubeDL") as ydl:
            with self.assertRaises(ValueError) as ctx:
                download_audio("https://example.com/secret.wav", directory=self.dir)
        self.assertIn("YouTube", str(ctx.exception))
        ydl.assert_not_called()

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

    def test_probe_does_not_download(self):
        seen = {}

        class FakeYDL:
            def __init__(self, opts):
                seen["opts"] = opts

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def extract_info(self, url, download=True):
                seen["download"] = download
                seen["url"] = url
                return {
                    "id": "abcdefghijk",
                    "channel": "DotDager",
                    "title": "CFK ERA DE DERECHA",
                }

        with mock.patch("yt_dlp.YoutubeDL", FakeYDL):
            ident = probe_youtube("https://youtu.be/abcdefghijk")
        self.assertFalse(seen["download"])
        self.assertEqual(ident["channel"], "DotDager")
        self.assertEqual(ident["title"], "CFK ERA DE DERECHA")
        self.assertEqual(ident["id"], "abcdefghijk")
        self.assertEqual(
            identity_line(ident["channel"], ident["title"], ident["id"]),
            "DotDager — CFK ERA DE DERECHA (abcdefghijk)",
        )

    def test_ffmpeg_binary_falls_back_to_imageio(self):
        fake = "/tmp/imageio-ffmpeg-bin"
        imageio = mock.Mock()
        imageio.get_ffmpeg_exe.return_value = fake
        with mock.patch("youtube_lib.shutil.which", return_value=None):
            with mock.patch(
                "youtube_lib.os.path.isfile", side_effect=lambda p: p == fake
            ):
                with mock.patch.dict("sys.modules", {"imageio_ffmpeg": imageio}):
                    self.assertEqual(ffmpeg_binary(), fake)

    def test_options_are_audio_only(self):
        opts = ydl_options(self.dir)
        self.assertIn("bestaudio", opts["format"])
        self.assertTrue(opts["noplaylist"])
        self.assertEqual(opts["postprocessors"][0]["preferredcodec"], "wav")
        self.assertEqual(opts.get("concurrent_fragment_downloads"), 4)


class OpenDownloadsFolderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ok = mock.Mock(returncode=0, stderr="", stdout="")

    def _data_env(self):
        data = os.path.join(self.tmp.name, "data")
        return mock.patch.dict(os.environ, {"AUDIO_SEPARATOR_DATA": data}), data

    def test_creates_and_opens_empty_folder(self):
        env, data = self._data_env()
        with env:
            with mock.patch("youtube_lib.subprocess.run", return_value=self.ok) as run:
                shown, revealed = open_downloads_folder()
        folder = os.path.realpath(os.path.join(data, "downloads"))
        self.assertFalse(revealed)
        self.assertEqual(shown, folder)
        self.assertTrue(os.path.isdir(folder))
        self.assertEqual(run.call_args.args[0], ["open", folder])
        self.assertEqual(run.call_args.kwargs.get("timeout"), 15)
        self.assertNotIn(
            os.path.join(os.path.expanduser("~"), "Downloads", "Audio Separator"),
            shown,
        )

    def test_reveals_newest_finished_file(self):
        env, data = self._data_env()
        with env:
            folder = os.path.join(os.path.abspath(data), "downloads")
            os.makedirs(folder)
            older = os.path.join(folder, "old.wav")
            newer = os.path.join(folder, "new.mp4")
            partial = os.path.join(folder, "busy.mp4.part")
            hidden = os.path.join(folder, ".secret.wav")
            empty = os.path.join(folder, "empty.wav")
            for path, mtime, blob in (
                (older, 1_000, b"old"),
                (newer, 2_000, b"new"),
                (partial, 3_000, b"part"),
                (hidden, 4_000, b"hid"),
                (empty, 5_000, b""),
            ):
                with open(path, "wb") as handle:
                    handle.write(blob)
                os.utime(path, (mtime, mtime))
            with mock.patch("youtube_lib.subprocess.run", return_value=self.ok) as run:
                shown, revealed = open_downloads_folder()
        self.assertTrue(revealed)
        self.assertEqual(shown, os.path.realpath(newer))
        self.assertEqual(
            run.call_args.args[0], ["open", "-R", os.path.realpath(newer)]
        )

    def test_skips_symlink_that_leaves_the_folder(self):
        env, data = self._data_env()
        outside = os.path.join(self.tmp.name, "secret.wav")
        with open(outside, "wb") as handle:
            handle.write(b"secret")
        os.utime(outside, (9_000, 9_000))
        with env:
            folder = os.path.join(os.path.abspath(data), "downloads")
            os.makedirs(folder)
            local = os.path.join(folder, "song.wav")
            with open(local, "wb") as handle:
                handle.write(b"song")
            os.utime(local, (1_000, 1_000))
            os.symlink(outside, os.path.join(folder, "leak.wav"))
            with mock.patch("youtube_lib.subprocess.run", return_value=self.ok) as run:
                shown, revealed = open_downloads_folder()
        self.assertTrue(revealed)
        self.assertEqual(shown, os.path.realpath(local))
        self.assertEqual(run.call_args.args[0][0:2], ["open", "-R"])
        self.assertNotIn("secret.wav", run.call_args.args[0][2])

    def test_falls_back_to_folder_when_reveal_fails(self):
        env, data = self._data_env()
        with env:
            folder = os.path.join(os.path.abspath(data), "downloads")
            os.makedirs(folder)
            media = os.path.join(folder, "song.wav")
            with open(media, "wb") as handle:
                handle.write(b"song")

            def fake_run(args, capture_output=True, text=True, timeout=15):
                if "-R" in args:
                    return mock.Mock(returncode=1, stderr="no", stdout="")
                return self.ok

            with mock.patch("youtube_lib.subprocess.run", side_effect=fake_run) as run:
                shown, revealed = open_downloads_folder()
        folder = os.path.realpath(folder)
        self.assertFalse(revealed)
        self.assertEqual(shown, folder)
        self.assertEqual(run.call_args_list[0].args[0][1], "-R")
        self.assertEqual(run.call_args_list[1].args[0], ["open", folder])

    def test_missing_finder_command_raises(self):
        env, _data = self._data_env()
        with env:
            with mock.patch(
                "youtube_lib.subprocess.run", side_effect=FileNotFoundError("open")
            ):
                with self.assertRaises(FileNotFoundError):
                    open_downloads_folder()


if __name__ == "__main__":
    unittest.main()

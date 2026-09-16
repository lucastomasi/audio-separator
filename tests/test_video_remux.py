import os
import tempfile
import unittest
from unittest import mock

from video_remux import remux_audio_onto_video


class RemuxTests(unittest.TestCase):
    def test_missing_files(self):
        with self.assertRaises(ValueError):
            remux_audio_onto_video("/no.mp4", "/no.wav", "/out.mp4")

    def test_ffmpeg_command(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        video = os.path.join(tmp.name, "v.mp4")
        audio = os.path.join(tmp.name, "a.wav")
        out = os.path.join(tmp.name, "out.mp4")
        for path in (video, audio):
            with open(path, "wb") as handle:
                handle.write(b"x")

        def fake_run(cmd, capture_output=True):
            self.assertIn("-c:v", cmd)
            self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")
            self.assertIn("320k", cmd)
            with open(cmd[-1], "wb") as handle:
                handle.write(b"mp4")
            return mock.Mock(returncode=0, stderr=b"")

        with mock.patch("video_remux.ffmpeg_binary", return_value="ffmpeg"):
            with mock.patch("video_remux.subprocess.run", side_effect=fake_run):
                path = remux_audio_onto_video(video, audio, out)
        self.assertTrue(os.path.isfile(path))


if __name__ == "__main__":
    unittest.main()

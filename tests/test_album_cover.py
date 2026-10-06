import os
import tempfile
import unittest
from unittest import mock

from album_cover import generate_cover


class CoverTests(unittest.TestCase):
    def test_writes_jpeg(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = os.path.join(tmp.name, "c.jpg")
        path = generate_cover("Hola", "Artista", size=200, dest_path=dest)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 100)

    def test_cover_job_reports_local_when_gemini_fails(self):
        from app_jobs import cover_job

        with mock.patch("gemini_cover.available", return_value=True):
            with mock.patch(
                "gemini_cover.generate_png", side_effect=RuntimeError("no")
            ):
                with mock.patch(
                    "album_cover.generate_cover", return_value="/tmp/local.jpg"
                ):
                    with mock.patch(
                        "album_cover.save_cover_with_audio",
                        return_value="/tmp/saved.jpg",
                    ):
                        saved, status = cover_job("T", "A", "/tmp/a.wav", True)
        self.assertEqual(saved, "/tmp/saved.jpg")
        text = status.get("value", status) if isinstance(status, dict) else str(status)
        self.assertIn("local", str(text).lower())
        self.assertIn("Gemini no anduvo", str(text))


if __name__ == "__main__":
    unittest.main()

import os
import unittest

import app


class DemoTests(unittest.TestCase):
    def test_demo_song_exists(self):
        path = app.demo_song_path()
        self.assertTrue(path)
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith("test.mp3"))

    def test_demo_song_is_in_allowed_paths(self):
        path = os.path.realpath(app.demo_song_path())
        allowed = [os.path.realpath(item) for item in app._allowed_paths()]
        self.assertTrue(
            any(path == item or path.startswith(item + os.sep) for item in allowed),
            path,
        )

    def test_servable_media_drops_repo_files(self):
        from app_jobs import DEMO_SONG, servable_media

        self.assertIsNone(servable_media(DEMO_SONG))
        self.assertEqual(servable_media(app.demo_song_path()), app.demo_song_path())


if __name__ == "__main__":
    unittest.main()

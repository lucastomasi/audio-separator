import os
import unittest

import app


class DemoTests(unittest.TestCase):
    def test_demo_song_exists(self):
        path = app.demo_song_path()
        self.assertTrue(path)
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith("test.mp3"))


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest

from album_cover import generate_cover


class CoverTests(unittest.TestCase):
    def test_writes_jpeg(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        dest = os.path.join(tmp.name, "c.jpg")
        path = generate_cover("Hola", "Artista", size=200, dest_path=dest)
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 100)


if __name__ == "__main__":
    unittest.main()

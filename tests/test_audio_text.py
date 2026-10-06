import unittest

from audio_text import (
    STEM_AMBAS,
    STEM_SOLO_INST,
    STEM_SOLO_VOZ,
    normalize_media_url,
    stem_choice_to_list,
)


class StemChoiceTests(unittest.TestCase):
    def test_default_is_vocal(self):
        self.assertEqual(stem_choice_to_list(None), ["vocal"])
        self.assertEqual(stem_choice_to_list(STEM_SOLO_VOZ), ["vocal"])
        self.assertEqual(stem_choice_to_list(""), ["vocal"])

    def test_instrumental_and_both(self):
        self.assertEqual(stem_choice_to_list(STEM_SOLO_INST), ["background"])
        self.assertEqual(stem_choice_to_list(STEM_AMBAS), ["vocal", "background"])

    def test_list_passthrough_drops_empty(self):
        self.assertEqual(
            stem_choice_to_list(["vocal", "", "background"]),
            ["vocal", "background"],
        )
        self.assertEqual(stem_choice_to_list(()), [])


class NormalizeUrlTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(normalize_media_url(""), "")
        self.assertEqual(normalize_media_url(None), "")
        self.assertEqual(normalize_media_url("   "), "")

    def test_strips_quotes_and_extracts_http(self):
        self.assertEqual(
            normalize_media_url('mira esto https://youtu.be/jNQXAC9IVRw?si=abc hola'),
            "https://youtu.be/jNQXAC9IVRw?si=abc",
        )
        self.assertEqual(
            normalize_media_url('"https://www.youtube.com/watch?v=abc"'),
            "https://www.youtube.com/watch?v=abc",
        )

    def test_adds_https_for_www_and_host(self):
        self.assertEqual(
            normalize_media_url("www.youtube.com/watch?v=abc"),
            "https://www.youtube.com/watch?v=abc",
        )
        self.assertEqual(
            normalize_media_url("youtu.be/abc"),
            "https://youtu.be/abc",
        )
        self.assertEqual(
            normalize_media_url("music.youtube.com/watch?v=abc"),
            "https://music.youtube.com/watch?v=abc",
        )


if __name__ == "__main__":
    unittest.main()

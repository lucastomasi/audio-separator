import unittest

import numpy as np

from diarize import _cluster, parse_silence_times, speech_regions


class DiarizeTests(unittest.TestCase):
    def test_parse_silence(self):
        log = """
[silencedetect @ 0x] silence_start: 1.2
[silencedetect @ 0x] silence_end: 2.0 | silence_duration: 0.8
[silencedetect @ 0x] silence_start: 5.5
"""
        starts, ends = parse_silence_times(log)
        self.assertEqual(starts, [1.2, 5.5])
        self.assertEqual(ends, [2.0])

    def test_speech_regions_two_turns(self):
        regions = speech_regions(10.0, starts=[1.0, 6.0], ends=[2.0], min_turn=0.3)
        self.assertTrue(any(abs(a - 0.0) < 1e-6 for a, _ in regions))
        self.assertTrue(any(abs(b - 6.0) < 1e-6 for _, b in regions))

    def test_cluster_two_speakers(self):
        a = np.zeros(32, dtype=np.float32)
        a[:16] = 1.0
        b = np.zeros(32, dtype=np.float32)
        b[16:] = 1.0
        labels = _cluster([a, a, b, b])
        self.assertEqual(len(set(labels)), 2)


if __name__ == "__main__":
    unittest.main()

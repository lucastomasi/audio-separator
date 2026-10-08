import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

import chords


def _tone(freq, seconds, sr, amp=0.25):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    return amp * np.sin(2 * np.pi * freq * t).astype(np.float32)


def _triad(root_hz, seconds, sr):
    third = root_hz * (2.0 ** (4.0 / 12.0))
    fifth = root_hz * (2.0 ** (7.0 / 12.0))
    bass = root_hz / 4.0
    return (
        _tone(root_hz, seconds, sr, 0.28)
        + _tone(third, seconds, sr, 0.24)
        + _tone(fifth, seconds, sr, 0.24)
        + _tone(bass, seconds, sr, 0.4)
    )


class ChordDspTests(unittest.TestCase):
    def test_no_librosa_or_network(self):
        src = inspect.getsource(chords)
        self.assertNotIn("librosa", src)
        self.assertNotIn("huggingface", src.lower())
        self.assertNotIn("requests", src)
        self.assertNotIn("urllib", src)

    def test_guitar_open_shapes(self):
        self.assertEqual(chords.shape_code(chords.guitar_shape("C")), "x32010")
        self.assertEqual(chords.shape_code(chords.guitar_shape("G")), "320003")
        self.assertEqual(chords.shape_code(chords.guitar_shape("Am")), "x02210")
        self.assertEqual(chords.guitar_shape("F")[0], 1)

    def test_bass_c_on_a_string(self):
        string, fret = chords.bass_position(0)
        self.assertEqual(string, 1)
        self.assertEqual(fret, 3)

    def test_detects_c_major_triad(self):
        sr = chords.SR
        wave = _triad(261.63, 2.4, sr)
        segments = chords.detect_chords(wave, sr)
        names = {item["name"] for item in segments}
        self.assertTrue(any(name.startswith("C") for name in names), names)
        self.assertGreaterEqual(segments[0]["end"] - segments[0]["start"], 0.5)

    def test_two_chord_sequence_c_then_g(self):
        sr = chords.SR
        wave = np.concatenate([_triad(261.63, 2.6, sr), _triad(196.00, 2.6, sr)])
        segments = chords.detect_chords(wave, sr)
        names = [item["name"] for item in segments]
        self.assertGreaterEqual(len(names), 2)
        self.assertTrue(names[0].startswith("C"), names)
        self.assertTrue(any(name.startswith("G") for name in names[1:]), names)

    def test_silent_audio_fails(self):
        sr = chords.SR
        with self.assertRaises(ValueError) as ctx:
            chords.detect_chords(np.zeros(sr, dtype=np.float32), sr)
        self.assertIn("acordes", str(ctx.exception))

    def test_render_includes_tab_and_disclaimer(self):
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "name": "C",
                "root_pc": 0,
                "bass_pc": 0,
            },
            {
                "start": 2.0,
                "end": 4.0,
                "name": "G",
                "root_pc": 7,
                "bass_pc": 7,
            },
        ]
        text = chords.render_text("demo.wav", segments, 4.0)
        self.assertIn(chords.DISCLAIMER, text)
        self.assertIn("Tablatura de guitarra", text)
        self.assertIn("Tablatura de bajo", text)
        self.assertIn("e|", text)
        self.assertIn("E|", text)
        self.assertIn("C", text)
        self.assertIn("G", text)
        self.assertIn("x32010", text)

    def test_estimate_song_writes_exportable_txt(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        wav = Path(tmp.name) / "tema.wav"
        sr = 22050
        wave = _triad(261.63, 2.2, sr)
        sf.write(wav, wave, sr)
        data = Path(tmp.name) / "data"
        with mock.patch("app_env.data_dir", return_value=str(data)):
            text, export_path, segments = chords.estimate_song(str(wav))
        self.assertTrue(export_path.endswith("_acordes.txt"))
        self.assertTrue(export_path.startswith(str(data)))
        self.assertIn("Acordes", export_path)
        self.assertTrue(os.path.isfile(export_path))
        self.assertIn("tema.wav", text)
        self.assertTrue(segments)
        from exports import is_exportable

        with mock.patch("app_env.data_dir", return_value=str(data)):
            self.assertTrue(is_exportable(export_path))


class ChordsJobTests(unittest.TestCase):
    def test_missing_audio_asks_for_cancion(self):
        import app_jobs

        text, path, status, unlock = app_jobs.chords_job(None)
        self.assertEqual(text, "")
        self.assertIsNone(path)
        self.assertIn("Canción", str(status.get("value", "")))
        self.assertEqual(unlock.get("value"), "Estimar acordes")
        self.assertTrue(unlock.get("interactive", False))

    def test_job_exports_and_takes_occupancy(self):
        import app_jobs
        import occupancy

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        wav = os.path.join(tmp.name, "c.wav")
        sr = 22050
        sf.write(wav, _triad(261.63, 2.2, sr), sr)
        data = os.path.join(tmp.name, "data")
        downloads = os.path.join(tmp.name, "dl")
        os.makedirs(downloads)
        acquired = []

        def fake_acquire(holder, exp=None):
            acquired.append(holder)
            return occupancy.Occupancy(holder, pid=os.getpid())

        with mock.patch("app_env.data_dir", return_value=data), mock.patch(
            "occupancy.acquire", side_effect=fake_acquire
        ), mock.patch("occupancy.release") as release, mock.patch(
            "app_jobs.copy_to_downloads",
            side_effect=lambda files, labels: (downloads, list(files)),
        ), mock.patch.object(occupancy, "_ps_commands", return_value=""):
            text, saved, status, unlock = app_jobs.chords_job(wav)
        self.assertIn(occupancy.HOLD_CHORDS, acquired)
        release.assert_called_with(occupancy.HOLD_CHORDS)
        self.assertTrue(saved.endswith("_acordes.txt"))
        self.assertIn("Listo", str(status.get("value", "")))
        self.assertIn("Tablatura", text)
        self.assertEqual(unlock.get("value"), "Estimar acordes")


if __name__ == "__main__":
    unittest.main()

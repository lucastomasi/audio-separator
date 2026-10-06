import os
import tempfile
import unittest
from unittest import mock

import train_prep


class SpeechTests(unittest.TestCase):
    def test_full_silence_is_not_speech(self):
        with self.assertRaises(ValueError) as ctx:
            train_prep.speech_regions_for_train(100.0, [0.0], [100.0])
        self.assertIn("No hay habla", str(ctx.exception))

    def test_eta_uses_half_of_the_paid_run(self):
        self.assertAlmostEqual(train_prep.eta_hours(21 * 60), 2.7, places=5)

    def test_names(self):
        self.assertTrue(train_prep.names_match("Dot Dager", "DotDager"))
        self.assertFalse(train_prep.names_match("gordopablo", "DotDager"))


class PrepareTests(unittest.TestCase):
    def test_eta_is_reported_before_uvr(self):
        order = []
        src = self._touch("mix.wav")

        def separate(path, progress=None):
            order.append("uvr")
            vocal = os.path.join(os.path.dirname(src), "voz.wav")
            with open(vocal, "wb") as handle:
                handle.write(b"wav")
            return vocal

        def progress(frac, desc=None):
            order.append(desc)

        with mock.patch("train_prep.audio_io.get_duration", return_value=30.0):
            with mock.patch("train_prep._is_last_vocal", return_value=False):
                with mock.patch("train_prep.separate_voice_only", side_effect=separate):
                    out = train_prep.prepare_for_train([src], progress=progress)
        self.assertTrue(any(text and text.startswith("ETA estimada") for text in order))
        self.assertLess(order.index(next(t for t in order if t and str(t).startswith("ETA"))), order.index("uvr"))
        self.assertTrue(out[0].endswith("voz.wav"))

    def test_no_speech_skips_uvr(self):
        src = self._touch("long.wav")
        uvr = mock.Mock()
        with mock.patch("train_prep.audio_io.get_duration", return_value=400.0):
            with mock.patch(
                "train_prep.trim_if_long",
                side_effect=ValueError(train_prep.NO_SPEECH),
            ):
                with mock.patch("train_prep.separate_voice_only", uvr):
                    with self.assertRaises(ValueError):
                        train_prep.prepare_for_train([src], progress=mock.Mock())
        uvr.assert_not_called()

    def test_already_separated_skips_uvr(self):
        src = self._touch("voz.wav")
        uvr = mock.Mock()
        with mock.patch("train_prep.audio_io.get_duration", return_value=400.0):
            with mock.patch("train_prep.trim_if_long", return_value=(src, 20.0)):
                with mock.patch("train_prep._is_last_vocal", return_value=True):
                    with mock.patch("train_prep.separate_voice_only", uvr):
                        out = train_prep.prepare_for_train([src], progress=mock.Mock())
        uvr.assert_not_called()
        self.assertEqual(out, [os.path.abspath(src)])

    def test_missing_vocal_wav_raises(self):
        src = self._touch("mix.wav")
        with mock.patch("train_prep.audio_io.get_duration", return_value=10.0):
            with mock.patch("train_prep._is_last_vocal", return_value=False):
                with mock.patch(
                    "train_prep.separate_voice_only",
                    side_effect=ValueError(train_prep.NO_VOCAL),
                ):
                    with self.assertRaises(ValueError) as ctx:
                        train_prep.prepare_for_train([src], progress=mock.Mock())
        self.assertIn("voz separada", str(ctx.exception))

    def _touch(self, name):
        if not hasattr(self, "tmp"):
            self.tmp = tempfile.TemporaryDirectory()
            self.addCleanup(self.tmp.cleanup)
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as handle:
            handle.write(b"wav")
        return path


class TrainJobGateTests(unittest.TestCase):
    def _job(self, name, path, meta):
        import app_jobs

        with mock.patch("library.get_session_meta", return_value=meta):
            with mock.patch("train_prep.library.get_session_meta", return_value=meta):
                with mock.patch(
                    "train_prep.prepare_for_train", return_value=[path]
                ) as prep:
                    with mock.patch(
                        "rvc_train.train_voice", return_value=("/tmp/m.pth", None)
                    ) as train:
                        with mock.patch(
                            "app_jobs.refresh_library_ui",
                            return_value=(mock.Mock(), mock.Mock()),
                        ):
                            with mock.patch("library.dropdown_choices", return_value=[]):
                                with mock.patch("library.list_rvc_voices", return_value=[]):
                                    bar = app_jobs.train_rvc_job(
                                        name, [path], epochs=5, progress=mock.Mock()
                                    )
        return bar, prep, train

    def test_channel_mismatch_does_not_train(self):
        path = "/tmp/yt-audio.wav"
        with mock.patch("train_prep.os.path.isfile", return_value=True):
            bar, prep, train = self._job(
                "gordopablo",
                path,
                {
                    "last_youtube_channel": "DotDager",
                    "last_audio_path": path,
                },
            )
        train.assert_not_called()
        prep.assert_not_called()
        text = bar[1]["value"] if isinstance(bar[1], dict) else str(bar[1])
        self.assertIn("DotDager", text)
        self.assertIn("Paro", text)

    def test_matching_channel_trains(self):
        path = "/tmp/yt-audio.wav"
        with mock.patch("train_prep.os.path.isfile", return_value=True):
            _bar, prep, train = self._job(
                "Dot Dager",
                path,
                {
                    "last_youtube_channel": "DotDager",
                    "last_audio_path": path,
                },
            )
        prep.assert_not_called()
        train.assert_called_once()

    def test_local_file_without_channel_still_trains(self):
        path = "/tmp/local.wav"
        with mock.patch("train_prep.os.path.isfile", return_value=True):
            _bar, prep, train = self._job("mi_voz", path, {})
        prep.assert_not_called()
        train.assert_called_once()


class DownloadIdentityTests(unittest.TestCase):
    def test_status_shows_identity_before_download(self):
        import app_jobs

        order = []

        def probe(url):
            order.append("probe")
            return {
                "channel": "DotDager",
                "title": "CFK ERA DE DERECHA",
                "id": "abcdefghijk",
                "note": None,
            }

        def download(url, with_video=True):
            order.append("download")
            return ("/tmp/a.wav", None, False, None)

        notes = []

        def progress(frac, desc=None):
            notes.append(desc)

        with mock.patch("youtube_lib.probe_youtube", side_effect=probe):
            with mock.patch("youtube_lib.download_media", side_effect=download):
                with mock.patch("library.set_session_meta") as meta:
                    with mock.patch("app_jobs.unlock_run_button", return_value="go"):
                        result = app_jobs.audio_downloader(
                            "https://youtu.be/abcdefghijk",
                            with_video=False,
                            progress=progress,
                        )
        self.assertEqual(order, ["probe", "download"])
        self.assertEqual(notes[0], "DotDager — CFK ERA DE DERECHA (abcdefghijk)")
        text = result[3]["value"] if isinstance(result[3], dict) else str(result[3])
        self.assertIn("DotDager — CFK ERA DE DERECHA (abcdefghijk)", text)
        saved = meta.call_args.kwargs
        self.assertEqual(saved["last_youtube_channel"], "DotDager")
        self.assertEqual(saved["last_youtube_id"], "abcdefghijk")
